from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import inspect
import math
import textwrap
from typing import Any, Callable, get_type_hints

from .errors import ComplexityError, DefinitionError
from .runtime import QRegistry, QuantumDefinition
from .types import ComplexityBreakdown


_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_SAFE_BUILTINS = {"bool", "int", "len", "max", "min", "range", "str", "tuple"}


@dataclass(frozen=True)
class ValidationResult:
    complexity: tuple[ComplexityBreakdown, ...]
    source_by_quantum: dict[str, str]
    source_by_primitive: dict[str, str]


def validate_registry(registry: QRegistry) -> ValidationResult:
    reports = []
    sources = {}
    primitive_sources = {}
    primitive_trees: dict[str, ast.FunctionDef] = {}
    zero_costs = {name: 0 for name in registry.primitive_definitions}
    for primitive_name in sorted(registry.primitive_definitions):
        primitive = registry.primitive_definitions[primitive_name]
        primitive_source, tree = _function_tree(primitive.function)
        primitive_sources[primitive_name] = primitive_source
        primitive_trees[primitive_name] = tree
        pseudo_definition = QuantumDefinition(
            id=f"primitive:{primitive_name}",
            label=primitive_name,
            function=primitive.function,
            wrapper=primitive.wrapper,
            declared_source_reads=(),
            output_cardinality=None,
        )
        _RestrictionValidator(registry, pseudo_definition, validating_primitive=True).visit(tree)
        observed_loop_bound = max(
            (
                _static_range_iterations(node.iter) or 0
                for node in ast.walk(tree)
                if isinstance(node, ast.For)
            ),
            default=0,
        )
        if (
            primitive.max_iterations is not None
            and observed_loop_bound > primitive.max_iterations
        ):
            raise DefinitionError(
                f"primitive {primitive_name!r} has loop bound {observed_loop_bound}, exceeding "
                f"declared max_iterations={primitive.max_iterations}"
            )
        local_components = _cost_components(tree, registry, zero_costs, zero_costs)
        local_cost = sum(local_components.values())
        if primitive.cost < local_cost:
            details = ", ".join(
                f"{name}={cost}"
                for name, cost in sorted(local_components.items())
                if cost
            )
            raise ComplexityError(
                f"primitive {primitive_name!r} declares cost={primitive.cost}, below its "
                f"analyzed local-body cost {local_cost}: {details}"
            )
        observed_cardinality = max(
            _literal_lookup_cardinality(tree),
            _global_table_cardinality(primitive.function, tree),
        )
        if observed_cardinality > primitive.table_cardinality:
            raise DefinitionError(
                f"primitive {primitive_name!r} embeds {observed_cardinality} table entries but declares "
                f"table_cardinality={primitive.table_cardinality}"
            )
    effective_primitive_costs, effective_primitive_tables = _expanded_primitive_costs(
        registry,
        primitive_trees,
    )
    for quantum_id in sorted(registry.quantum_definitions):
        definition = registry.quantum_definitions[quantum_id]
        source, tree = _function_tree(definition.function)
        sources[quantum_id] = source
        _RestrictionValidator(registry, definition).visit(tree)
        components = _cost_components(
            tree,
            registry,
            effective_primitive_costs,
            effective_primitive_tables,
        )
        if definition.output_cardinality is not None:
            components["output_cardinality"] += max(1, math.ceil(math.log2(max(2, definition.output_cardinality))))
        total = sum(components.values())
        report = ComplexityBreakdown(
            quantum_id=quantum_id,
            total=total,
            budget=registry.unit_complexity,
            components=tuple(sorted((name, cost) for name, cost in components.items() if cost)),
            output_cardinality=definition.output_cardinality,
        )
        reports.append(report)
        if total > registry.unit_complexity:
            details = ", ".join(f"{name}={cost}" for name, cost in report.components)
            raise ComplexityError(
                f"quantum {quantum_id!r} costs {total}, exceeding unit budget "
                f"{registry.unit_complexity}: {details}"
            )
    return ValidationResult(
        complexity=tuple(reports),
        source_by_quantum=sources,
        source_by_primitive=primitive_sources,
    )


def validate_entrypoint(entrypoint: Callable[..., Any], registry: QRegistry) -> str:
    source, tree = _function_tree(entrypoint)
    signature = inspect.signature(entrypoint)
    hints = get_type_hints(entrypoint)
    if len(signature.parameters) != 1 or "return" not in hints or any(
        name not in hints for name in signature.parameters
    ):
        raise DefinitionError("the prediction entrypoint must have one typed state input and a typed return")
    state_name = next(iter(signature.parameters))
    if getattr(hints[state_name], "__name__", None) != "PredictiveState":
        raise DefinitionError("the prediction entrypoint input must be PredictiveState")
    quantum_names = {definition.wrapper.__name__ for definition in registry.quantum_definitions.values()}
    allowed = quantum_names | {"when", "when_item", "unwrap"} | _SAFE_BUILTINS
    local_names = set(signature.parameters)
    for statement in tree.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_names.add(node.id)
    for statement in tree.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == state_name:
                raise DefinitionError(
                    f"entrypoint {entrypoint.__name__!r} reads predictive state directly; "
                    "all exogenous reads must occur inside an annotated quantum"
                )
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local_names:
                if node.id in allowed:
                    continue
                value = entrypoint.__globals__.get(node.id, _MISSING)
                if isinstance(value, type) and issubclass(value, Enum):
                    continue
                raise DefinitionError(
                    f"entrypoint {entrypoint.__name__!r} contains hidden global access {node.id!r}"
                )
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda)):
                raise DefinitionError(f"entrypoint {entrypoint.__name__!r} contains forbidden {type(node).__name__}")
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name not in allowed:
                    raise DefinitionError(
                        f"entrypoint {entrypoint.__name__!r} calls unregistered helper {name!r}; "
                        "the reachable call graph may contain only annotated quanta and Q-program control helpers"
                    )
    return source


class _RestrictionValidator(ast.NodeVisitor):
    def __init__(
        self,
        registry: QRegistry,
        definition: QuantumDefinition,
        *,
        validating_primitive: bool = False,
    ) -> None:
        self.registry = registry
        self.definition = definition
        self.local_names = set(inspect.signature(definition.function).parameters)
        self.primitive_names = {
            primitive.wrapper.__name__ for primitive in registry.primitive_definitions.values()
        }
        self.quantum_names = {
            quantum.wrapper.__name__ for quantum in registry.quantum_definitions.values()
        }
        self.validating_primitive = validating_primitive
        hints = get_type_hints(definition.function)
        self.state_parameters = {
            name
            for name in inspect.signature(definition.function).parameters
            if getattr(hints.get(name), "__name__", None) == "PredictiveState"
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for statement in node.body:
            self.visit(statement)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _FORBIDDEN_NODES):
            self._fail(node, f"forbidden syntax {type(node).__name__}")
        super().generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, (ast.Name, ast.Tuple)):
                self._fail(node, "only immutable local assignment is permitted")
            for name in _assigned_names(target):
                if name in self.local_names:
                    self._fail(node, f"local {name!r} is assigned more than once")
                self.local_names.add(name)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            self._fail(node, "only immutable local assignment is permitted")
        if node.target.id in self.local_names:
            self._fail(node, f"local {node.target.id!r} is assigned more than once")
        self.local_names.add(node.target.id)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._fail(node, "mutation through augmented assignment is forbidden")

    def visit_For(self, node: ast.For) -> None:
        iterations = _static_range_iterations(node.iter)
        if iterations is None:
            self._fail(node, "loops must use a statically bounded range")
        if iterations > 32:
            self._fail(node, f"loop bound {iterations} exceeds the compiler maximum of 32")
        if not isinstance(node.target, (ast.Name, ast.Tuple)):
            self._fail(node, "loop targets must be local names")
        for name in _assigned_names(node.target):
            if name in self.local_names:
                self._fail(node, f"loop variable {name!r} shadows an existing local")
            self.local_names.add(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name == self.definition.function.__name__:
            self._fail(node, "recursion is forbidden")
        if name in self.quantum_names:
            self._fail(node, "quanta must be composed by the prediction entrypoint, not hidden inside another quantum")
        if name == "read" and isinstance(node.func, ast.Attribute):
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                self._fail(node, "PredictiveState.read() requires one literal field name")
        elif name not in self.primitive_names and name not in _SAFE_BUILTINS:
            self._fail(node, f"call to unregistered helper {name!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Store):
            self._fail(node, "attribute mutation is forbidden")
        if isinstance(node.value, ast.Name) and node.value.id in self.state_parameters and node.attr != "read":
            self._fail(
                node,
                f"predictive state field {node.attr!r} bypasses provenance; use {node.value.id}.read({node.attr!r})",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load) or node.id in self.local_names:
            return
        if node.id in self.primitive_names or node.id in _SAFE_BUILTINS:
            return
        value = self.definition.function.__globals__.get(node.id, _MISSING)
        if isinstance(value, type) and issubclass(value, Enum):
            return
        if self.validating_primitive and isinstance(value, (dict, tuple, frozenset, str, bytes)):
            return
        self._fail(node, f"hidden global access {node.id!r} is forbidden")

    def _fail(self, node: ast.AST, message: str) -> None:
        raise DefinitionError(
            f"quantum {self.definition.id!r}, line {getattr(node, 'lineno', '?')}: {message}"
        )


def _cost_components(
    tree: ast.FunctionDef,
    registry: QRegistry,
    effective_costs: dict[str, int],
    effective_tables: dict[str, int],
) -> Counter[str]:
    primitive_by_name = {
        definition.wrapper.__name__: definition for definition in registry.primitive_definitions.values()
    }

    def cost_nodes(nodes: list[ast.stmt]) -> Counter[str]:
        result: Counter[str] = Counter()
        for node in nodes:
            if isinstance(node, ast.If):
                result["branches"] += 1
                result.update(cost_expression(node.test))
                result.update(cost_nodes(node.body))
                result.update(cost_nodes(node.orelse))
            elif isinstance(node, ast.For):
                iterations = _static_range_iterations(node.iter) or 0
                body = cost_nodes(node.body)
                for key, value in body.items():
                    result[f"loop:{key}"] += iterations * value
                result["loop_iterations"] += iterations
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.expr):
                        result.update(cost_expression(child))
        return result

    def cost_expression(expression: ast.AST) -> Counter[str]:
        result: Counter[str] = Counter()
        for node in ast.walk(expression):
            if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                result["arithmetic"] += 1
            elif isinstance(node, (ast.BoolOp, ast.Compare)):
                result["logic"] += max(1, len(getattr(node, "values", getattr(node, "ops", ()))) - 1)
            elif isinstance(node, ast.Subscript):
                result["indexing"] += 1
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in primitive_by_name:
                    primitive = primitive_by_name[name]
                    result[f"primitive:{primitive.name}"] += effective_costs[primitive.name]
                    result[f"table:{primitive.name}"] += effective_tables[primitive.name]
                elif name == "read":
                    result["source_read"] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > 16:
                result["large_constants"] += math.ceil(len(node.value) / 16)
            elif isinstance(node, ast.Constant) and isinstance(node.value, int) and abs(node.value).bit_length() > 16:
                result["large_constants"] += math.ceil(abs(node.value).bit_length() / 16)
            elif isinstance(node, ast.IfExp):
                result["branches"] += 1
        return result

    return cost_nodes(tree.body)


def _function_tree(function: Callable[..., Any]) -> tuple[str, ast.FunctionDef]:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise DefinitionError(f"source is unavailable for {function.__qualname__!r}") from exc
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise DefinitionError(f"expected one ordinary function definition for {function.__qualname__!r}")
    functions[0].decorator_list = []
    return source, functions[0]


def _call_name(function: ast.expr) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return "<dynamic>"


def _assigned_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple):
        return [name for element in target.elts for name in _assigned_names(element)]
    return []


def _static_range_iterations(expression: ast.expr) -> int | None:
    if not isinstance(expression, ast.Call) or _call_name(expression.func) != "range":
        return None
    values = []
    for argument in expression.args:
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, int):
            return None
        values.append(int(argument.value))
    try:
        return len(range(*values))
    except TypeError:
        return None


def _literal_lookup_cardinality(tree: ast.AST) -> int:
    cardinalities = []
    roots = tree.body if isinstance(tree, ast.FunctionDef) else (tree,)
    for root in roots:
        for node in ast.walk(root):
            if not isinstance(node, ast.Subscript):
                continue
            table = node.value
            if isinstance(table, ast.Dict):
                cardinalities.append(len(table.keys))
            elif isinstance(table, (ast.List, ast.Tuple)):
                cardinalities.append(len(table.elts))
    return max(cardinalities, default=0)


def _global_table_cardinality(function: Callable[..., Any], tree: ast.FunctionDef) -> int:
    cardinalities = []
    for statement in tree.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            value = function.__globals__.get(node.id, _MISSING)
            if isinstance(value, (dict, tuple, frozenset)):
                cardinalities.append(len(value))
            elif isinstance(value, (str, bytes)) and len(value) > 1:
                cardinalities.append(len(value))
    return max(cardinalities, default=0)


_MISSING = object()


def _expanded_primitive_costs(
    registry: QRegistry,
    trees: dict[str, ast.FunctionDef],
) -> tuple[dict[str, int], dict[str, int]]:
    definition_by_call_name = {
        definition.wrapper.__name__: definition
        for definition in registry.primitive_definitions.values()
    }
    calls: dict[str, Counter[str]] = {}
    for primitive_name, tree in trees.items():
        counts: Counter[str] = Counter()
        for statement in tree.body:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                called = definition_by_call_name.get(_call_name(node.func))
                if called is not None:
                    counts[called.name] += 1
        calls[primitive_name] = counts

    costs: dict[str, int] = {}
    tables: dict[str, int] = {}
    active: list[str] = []

    def expand(name: str) -> tuple[int, int]:
        if name in active:
            cycle = " -> ".join((*active[active.index(name) :], name))
            raise DefinitionError(f"registered primitive helper cycle is forbidden: {cycle}")
        if name in costs:
            return costs[name], tables[name]
        active.append(name)
        definition = registry.primitive_definitions[name]
        total_cost = definition.cost
        total_table = definition.table_cardinality
        for child, count in calls[name].items():
            child_cost, child_table = expand(child)
            total_cost += count * child_cost
            total_table += count * child_table
        active.pop()
        costs[name] = total_cost
        tables[name] = total_table
        return total_cost, total_table

    for name in sorted(registry.primitive_definitions):
        expand(name)
    return costs, tables
