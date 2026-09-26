from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping


COMPILER_VERSION = "5"


def canonical_value(value: Any) -> Any:
    """Convert a semantic value to deterministic, JSON-compatible data."""
    if isinstance(value, Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "name": value.name}
    if is_dataclass(value):
        return {
            item.name: canonical_value(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("canonical", True)
        }
    if isinstance(value, Mapping):
        return {str(key): canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"semantic value {value!r} of type {type(value).__name__} is not serializable")


def canonical_key(value: Any) -> str:
    """Return the stable compact JSON key used for semantic-value counters."""
    return json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    payload = canonical_key(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PredictiveState:
    """Immutable exogenous state available to one next-token prediction."""

    digits: tuple[int, ...]
    prefix: tuple[str, ...] = ()
    separator: str | None = None
    position: int | None = None
    extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "digits", tuple(int(digit) for digit in self.digits))
        object.__setattr__(self, "prefix", tuple(str(token) for token in self.prefix))
        object.__setattr__(self, "extra", tuple(sorted(self.extra, key=lambda item: item[0])))

    def read(self, field_name: str) -> Any:
        from .runtime import record_source_read

        name = str(field_name)
        values = {
            "digits": self.digits,
            "prefix": self.prefix,
            "separator": self.separator,
            "position": len(self.prefix) if self.position is None else self.position,
            **dict(self.extra),
        }
        if name not in values:
            raise KeyError(f"unknown predictive-state field: {name!r}")
        record_source_read(name)
        return values[name]

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True)
class PredictionEvent:
    id: str
    state: PredictiveState
    target: Any


@dataclass(frozen=True)
class Dependency:
    parent: str
    child: str
    kinds: tuple[str, ...]
    witness_event: str


@dataclass(frozen=True)
class InvocationTrace:
    quantum_id: str
    inputs: Any
    output: Any
    data_parents: tuple[str, ...]
    control_parents: tuple[str, ...]
    source_reads: tuple[str, ...]
    emitted_token: Any | None = None


@dataclass(frozen=True, slots=True)
class InvocationSchema:
    """Event-independent part of an invocation, interned across compact traces."""

    inputs: Any
    data_parent_indices: tuple[int, ...]
    control_parent_indices: tuple[int, ...]
    source_reads: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TraceSchemaCatalog:
    """Shared node alignment and invocation schemas for compact event traces."""

    nodes: tuple[str, ...]
    parent_masks: tuple[int, ...]
    invocation_schemas: tuple[InvocationSchema, ...]


@dataclass(frozen=True)
class EventTrace:
    event_id: str
    state: Any
    target: Any
    produced: Any
    invocations: tuple[InvocationTrace, ...]
    active_nodes: tuple[str, ...]
    primitive_calls: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    activity_targets: tuple[int, ...] = ()
    activity_mask: tuple[int, ...] = ()
    semantic_targets: tuple[Any | None, ...] = ()
    parent_context: tuple[Any, ...] = ()

    @property
    def signature(self) -> tuple[str, ...]:
        return self.active_nodes

    def derive_parent_context(self, structure: PosetStructure) -> tuple[Any, ...]:
        invocation_by_node = {item.quantum_id: item for item in self.invocations}
        return tuple(
            {
                "node": node,
                "active_parents": [
                    parent for parent in structure.parents[node] if parent in invocation_by_node
                ],
                "semantic_parents": {
                    parent: invocation_by_node[parent].output
                    for parent in structure.parents[node]
                    if parent in invocation_by_node
                },
                "control_parents": list(invocation_by_node[node].control_parents)
                if node in invocation_by_node
                else [],
            }
            for node in structure.nodes
        )

    def resolve_invocation_inputs(self, invocation: InvocationTrace | str) -> Any:
        """Resolve compact state/producer references to exact semantic values."""
        quantum_id = invocation if isinstance(invocation, str) else invocation.quantum_id
        invocation_by_node = {item.quantum_id: item for item in self.invocations}
        try:
            encoded = invocation_by_node[quantum_id].inputs
        except KeyError as exc:
            raise KeyError(f"quantum {quantum_id!r} is inactive in trace {self.event_id!r}") from exc
        outputs = {node: item.output for node, item in invocation_by_node.items()}
        return _resolve_trace_input_references(encoded, state=self.state, outputs=outputs)

    def expand_invocations(self) -> tuple[InvocationTrace, ...]:
        """Return the already-materialized validation invocations."""
        return self.invocations


@dataclass(frozen=True, slots=True)
class AlignedEventTrace:
    """Compact, node-aligned trace used for training and evaluation domains.

    Semantic outputs are stored once in node order. Everything else belonging to
    an invocation is interned in a shared catalog and expanded only on request.
    """

    event_id: str
    state: Any
    target: Any
    produced: Any
    active_mask: int
    semantic_targets: tuple[Any | None, ...]
    invocation_schema_ids: tuple[int, ...]
    invocation_order: tuple[int, ...]
    emitter_index: int
    catalog: TraceSchemaCatalog = field(repr=False, metadata={"canonical": False})

    def __post_init__(self) -> None:
        width = len(self.catalog.nodes)
        if len(self.semantic_targets) != width or len(self.invocation_schema_ids) != width:
            raise ValueError("compact trace rows must align exactly with the schema catalog nodes")
        if self.active_mask < 0 or self.active_mask >= (1 << width):
            raise ValueError("compact trace active_mask exceeds the schema catalog width")
        if not 0 <= self.emitter_index < width or not self.is_active_index(self.emitter_index):
            raise ValueError("compact trace emitter must identify one active node")
        active_indices = {
            index for index in range(width) if self.is_active_index(index)
        }
        if len(self.invocation_order) != len(active_indices) or set(self.invocation_order) != active_indices:
            raise ValueError("compact trace invocation order must contain every active node exactly once")
        for index, schema_id in enumerate(self.invocation_schema_ids):
            active = self.is_active_index(index)
            if active != (schema_id >= 0):
                raise ValueError("compact trace schema IDs must be present exactly for active nodes")
            if schema_id >= len(self.catalog.invocation_schemas):
                raise ValueError("compact trace references an unknown invocation schema")

    def is_active_index(self, index: int) -> bool:
        return bool(self.active_mask & (1 << int(index)))

    @property
    def active_nodes(self) -> tuple[str, ...]:
        return tuple(
            node for index, node in enumerate(self.catalog.nodes) if self.is_active_index(index)
        )

    @property
    def signature(self) -> tuple[str, ...]:
        return self.active_nodes

    @property
    def activity_targets(self) -> tuple[int, ...]:
        return tuple(int(self.is_active_index(index)) for index in range(len(self.catalog.nodes)))

    @property
    def activity_mask(self) -> tuple[int, ...]:
        return tuple(
            int(self.active_mask & parent_mask == parent_mask)
            for parent_mask in self.catalog.parent_masks
        )

    @property
    def parent_context(self) -> tuple[Any, ...]:
        return self.derive_parent_context()

    @property
    def invocations(self) -> tuple[InvocationTrace, ...]:
        """Lazily expand full invocation objects for diagnostic compatibility."""
        return self.expand_invocations()

    def invocation(self, invocation: InvocationTrace | str) -> InvocationTrace:
        quantum_id = invocation if isinstance(invocation, str) else invocation.quantum_id
        try:
            index = self.catalog.nodes.index(quantum_id)
        except ValueError as exc:
            raise KeyError(f"unknown quantum identity {quantum_id!r}") from exc
        if not self.is_active_index(index):
            raise KeyError(f"quantum {quantum_id!r} is inactive in trace {self.event_id!r}")
        schema = self.catalog.invocation_schemas[self.invocation_schema_ids[index]]
        return InvocationTrace(
            quantum_id=quantum_id,
            inputs=schema.inputs,
            output=self.semantic_targets[index],
            data_parents=tuple(self.catalog.nodes[item] for item in schema.data_parent_indices),
            control_parents=tuple(self.catalog.nodes[item] for item in schema.control_parent_indices),
            source_reads=schema.source_reads,
            emitted_token=self.produced if index == self.emitter_index else None,
        )

    def expand_invocations(self) -> tuple[InvocationTrace, ...]:
        return tuple(
            self.invocation(self.catalog.nodes[index])
            for index in self.invocation_order
        )

    def resolve_invocation_inputs(self, invocation: InvocationTrace | str) -> Any:
        expanded = self.invocation(invocation)
        outputs = {
            node: self.semantic_targets[index]
            for index, node in enumerate(self.catalog.nodes)
            if self.is_active_index(index)
        }
        return _resolve_trace_input_references(expanded.inputs, state=self.state, outputs=outputs)

    def derive_parent_context(self, structure: PosetStructure | None = None) -> tuple[Any, ...]:
        if structure is not None and structure.nodes != self.catalog.nodes:
            raise ValueError("parent context structure does not match the compact trace node alignment")
        invocations = {
            item.quantum_id: item for item in self.expand_invocations()
        }
        if structure is None:
            parents = {
                node: tuple(
                    self.catalog.nodes[index]
                    for index in range(len(self.catalog.nodes))
                    if self.catalog.parent_masks[node_index] & (1 << index)
                )
                for node_index, node in enumerate(self.catalog.nodes)
            }
        else:
            parents = structure.parents
        return tuple(
            {
                "node": node,
                "active_parents": [parent for parent in parents[node] if parent in invocations],
                "semantic_parents": {
                    parent: invocations[parent].output
                    for parent in parents[node]
                    if parent in invocations
                },
                "control_parents": list(invocations[node].control_parents)
                if node in invocations
                else [],
            }
            for node in self.catalog.nodes
        )


@dataclass(frozen=True)
class ComplexityBreakdown:
    quantum_id: str
    total: int
    budget: int
    components: tuple[tuple[str, int], ...]
    output_cardinality: int | None


@dataclass(frozen=True)
class PosetStructure:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    dependencies: tuple[Dependency, ...]
    unreduced_edges: tuple[tuple[str, str], ...]
    unreduced_dependencies: tuple[Dependency, ...]
    parents: dict[str, tuple[str, ...]]
    children: dict[str, tuple[str, ...]]
    ancestors: dict[str, tuple[str, ...]]
    descendants: dict[str, tuple[str, ...]]
    topological_order: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    depth: int
    source_adjacent: tuple[str, ...]
    leaves: tuple[str, ...]
    joins: tuple[str, ...]


@dataclass(frozen=True)
class CoverageReport:
    event_count: int
    activity_counts: dict[str, int]
    negative_eligible_counts: dict[str, int]
    semantic_outcome_counts: dict[str, dict[str, int]]
    semantic_pair_counts: dict[str, int]
    parent_semantic_context_counts: dict[str, int]
    trace_signature_counts: dict[str, int]
    parent_only_counts: dict[str, int]
    parent_plus_child_counts: dict[str, int]
    pairwise_coactivation: dict[str, int]
    higher_order_coactivation: dict[str, int]
    incidence_rank: int
    incidence_condition: float | None
    one_class_activity_nodes: tuple[str, ...]
    indistinguishable_activity_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class CoverageComparison:
    train: CoverageReport
    audit: CoverageReport
    trace_overlap: int
    predictive_state_overlap: int
    audit_only_quanta: tuple[str, ...]
    audit_only_semantic_outcomes: dict[str, tuple[str, ...]]
    audit_only_semantic_pairs: tuple[str, ...]
    audit_only_parent_semantic_contexts: dict[str, tuple[str, ...]]
    training_absent_semantic_outcomes: dict[str, tuple[str, ...]]
    rare_training_semantic_outcomes: dict[str, tuple[tuple[str, int], ...]]


@dataclass(frozen=True)
class ReuseStat:
    activation_count: int
    distinct_semantic_input_contexts: int
    distinct_semantic_outputs: int
    downstream_consumers: tuple[str, ...]
    distinct_roles: int
    appears_in_multiple_roles: bool


@dataclass(frozen=True)
class RepeatedPrimitiveComputation:
    parent: str
    child: str
    primitive: str
    coactive_event_count: int
    repeated_event_count: int


@dataclass(frozen=True)
class NonIdentifyingSummary:
    node: str
    consumer: str
    activation_count: int
    distinct_input_contexts: int
    distinct_outputs: int


@dataclass(frozen=True)
class CompilerAuditReport:
    repeated_primitive_computations: tuple[RepeatedPrimitiveComputation, ...]
    non_identifying_summaries: tuple[NonIdentifyingSummary, ...]


@dataclass(frozen=True)
class CompilationMetadata:
    compiler_version: str
    program_fingerprint: str
    primitive_cost_fingerprint: str
    validation_domain_fingerprint: str
    training_distribution_fingerprint: str
    evaluation_distribution_fingerprint: str
    node_parameter_budget: int | None


@dataclass(frozen=True)
class ExhaustiveValidationReport:
    """Bounded-memory evidence from an exhaustive predictive-state domain."""

    compiler_version: str
    program_id: str
    program_fingerprint: str
    primitive_cost_fingerprint: str
    validation_domain_fingerprint: str
    event_count: int
    invocation_count: int
    first_event_id: str
    last_event_id: str
    active_nodes: tuple[str, ...]
    activity_counts: dict[str, int]
    semantic_outcome_counts: dict[str, dict[str, int]]
    emitted_token_counts: dict[str, dict[str, int]]
    dependencies: tuple[Dependency, ...]
    source_reads: dict[str, tuple[str, ...]]
    expected_structure_validated: bool

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True)
class CompiledQProgram:
    id: str
    structure: PosetStructure
    node_labels: dict[str, str]
    source_reads: dict[str, tuple[str, ...]]
    semantic_types: dict[str, str]
    semantic_outcomes: dict[str, tuple[Any, ...]]
    complexity: tuple[ComplexityBreakdown, ...]
    trace_schema_catalog: TraceSchemaCatalog
    validation_traces: tuple[EventTrace, ...]
    training_traces: tuple[AlignedEventTrace, ...]
    evaluation_traces: tuple[AlignedEventTrace, ...]
    coverage: CoverageComparison
    reuse: dict[str, ReuseStat]
    audits: CompilerAuditReport
    warnings: tuple[str, ...]
    metadata: CompilationMetadata

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.structure.nodes

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return self.structure.edges

    def expand_trace(
        self,
        trace: EventTrace | AlignedEventTrace,
    ) -> tuple[InvocationTrace, ...]:
        """Expand either retained validation or compact distribution invocations."""
        return trace.expand_invocations()

    def resolve_invocation_inputs(
        self,
        trace: EventTrace | AlignedEventTrace,
        invocation: InvocationTrace | str,
    ) -> Any:
        """Resolve exact semantic invocation inputs from either trace representation."""
        return trace.resolve_invocation_inputs(invocation)

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)

    def write(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        trace_counts = {
            "validation": len(self.validation_traces),
            "training": len(self.training_traces),
            "evaluation": len(self.evaluation_traces),
        }
        summary = canonical_value(
            replace(
                self,
                validation_traces=(),
                training_traces=(),
                evaluation_traces=(),
            )
        )
        for trace_field in ("validation_traces", "training_traces", "evaluation_traces"):
            summary.pop(trace_field, None)
        summary["trace_storage"] = {
            "format": "python_pickle",
            "path": "compiled_qprogram.pkl",
            "counts": trace_counts,
        }
        (path / "compiled_qprogram.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (path / "structure.json").write_text(
            json.dumps(canonical_value(self.structure), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (path / "coverage.json").write_text(
            json.dumps(canonical_value(self.coverage), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (path / "complexity.json").write_text(
            json.dumps(canonical_value(self.complexity), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (path / "audits.json").write_text(
            json.dumps(canonical_value(self.audits), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (path / "poset.mmd").write_text(self.mermaid(), encoding="utf-8")
        (path / "report.txt").write_text(self.report(), encoding="utf-8")
        with (path / "compiled_qprogram.pkl").open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def read(cls, path: str | Path) -> CompiledQProgram:
        artifact = Path(path)
        if artifact.is_dir():
            artifact = artifact / "compiled_qprogram.pkl"
        if artifact.suffix != ".pkl":
            raise ValueError("compiled Q-program loading requires the compiler-generated .pkl artifact.")
        with artifact.open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, cls):
            raise TypeError(f"artifact {artifact} does not contain a CompiledQProgram.")
        if value.metadata.compiler_version != COMPILER_VERSION:
            raise ValueError(
                f"compiled artifact version {value.metadata.compiler_version!r} does not match {COMPILER_VERSION!r}."
            )
        return value

    def mermaid(self) -> str:
        lines = ["graph TD\n"]
        for node in self.nodes:
            if not self.structure.parents[node] and not self.structure.children[node]:
                lines.append(f'    {_mermaid_id(node)}["{self.node_labels[node]}"]\n')
        for parent, child in self.edges:
            dependency = next(
                item
                for item in self.structure.dependencies
                if item.parent == parent and item.child == child
            )
            kinds = "+".join(dependency.kinds)
            lines.append(
                f'    {_mermaid_id(parent)}["{self.node_labels[parent]}"] '
                f'-->|"{kinds}"| {_mermaid_id(child)}["{self.node_labels[child]}"]\n'
            )
        return "".join(lines)

    def report(self) -> str:
        audit_only_outcome_counts = {
            node: len(outcomes)
            for node, outcomes in self.coverage.audit_only_semantic_outcomes.items()
        }
        audit_only_outcome_sample = {
            node: list(outcomes[:8])
            for node, outcomes in self.coverage.audit_only_semantic_outcomes.items()
        }
        audit_only_pair_sample = list(
            self.coverage.audit_only_semantic_pairs[:16]
        )
        absent_training_outcome_counts = {
            node: len(outcomes)
            for node, outcomes in self.coverage.training_absent_semantic_outcomes.items()
        }
        absent_training_outcome_sample = {
            node: list(outcomes[:8])
            for node, outcomes in self.coverage.training_absent_semantic_outcomes.items()
        }
        rare_training_outcome_counts = {
            node: len(outcomes)
            for node, outcomes in self.coverage.rare_training_semantic_outcomes.items()
        }
        rare_training_outcome_sample = {
            node: dict(outcomes[:8])
            for node, outcomes in self.coverage.rare_training_semantic_outcomes.items()
        }
        lines = [
            f"Q-program: {self.id}",
            f"Compiler version: {self.metadata.compiler_version}",
            f"Nodes: {len(self.nodes)}",
            f"Direct edges: {len(self.edges)}",
            f"Depth: {self.structure.depth}",
            "",
            "Structure:",
        ]
        lines.extend(
            f"- {node}: parents={list(self.structure.parents[node])}, "
            f"source_reads={list(self.source_reads[node])}, "
            f"outputs={len(self.semantic_outcomes[node])} "
            f"sample={list(self.semantic_outcomes[node][:8])}, "
            f"activations={self.reuse[node].activation_count}, "
            f"input_contexts={self.reuse[node].distinct_semantic_input_contexts}, "
            f"consumers={list(self.reuse[node].downstream_consumers)}"
            for node in self.nodes
        )
        lines.extend(["", "Complexity:"])
        lines.extend(
            f"- {item.quantum_id}: {item.total}/{item.budget} ({dict(item.components)})"
            for item in self.complexity
        )
        lines.extend(
            [
                "",
                f"Training events: {self.coverage.train.event_count}",
                f"Audit events: {self.coverage.audit.event_count}",
                f"Training trace signatures: {len(self.coverage.train.trace_signature_counts)}",
                f"Audit trace signatures: {len(self.coverage.audit.trace_signature_counts)}",
                f"Train/audit trace-signature overlap: {self.coverage.trace_overlap}",
                f"Training incidence rank: {self.coverage.train.incidence_rank}/{len(self.nodes)}",
                f"Audit-only quanta: {list(self.coverage.audit_only_quanta)}",
                f"Audit-only semantic outcome counts: {audit_only_outcome_counts}",
                f"Audit-only semantic outcome sample: {audit_only_outcome_sample}",
                f"Audit-only compact semantic pairs: {len(self.coverage.audit_only_semantic_pairs)}",
                f"Audit-only compact semantic-pair sample: {audit_only_pair_sample}",
                f"Audit-only parent semantic contexts: {self.coverage.audit_only_parent_semantic_contexts}",
                f"Audit-domain semantic outcomes absent from training: {absent_training_outcome_counts}",
                f"Absent training semantic-outcome sample: {absent_training_outcome_sample}",
                f"Rare training semantic outcome counts: {rare_training_outcome_counts}",
                f"Rare training semantic-outcome sample: {rare_training_outcome_sample}",
                f"Predictive-state train/audit overlap: {self.coverage.predictive_state_overlap}",
                f"One-class eligible activity: {list(self.coverage.train.one_class_activity_nodes)}",
                f"Indistinguishable activity groups: {list(self.coverage.train.indistinguishable_activity_groups)}",
                "",
                "Training node coverage:",
            ]
        )
        lines.extend(
            f"- {node}: positive={self.coverage.train.activity_counts[node]}, "
            f"eligible_negative={self.coverage.train.negative_eligible_counts[node]}, "
            f"semantic_outcomes={len(self.coverage.train.semantic_outcome_counts[node])}, "
            f"semantic_sample={dict(list(self.coverage.train.semantic_outcome_counts[node].items())[:8])}"
            for node in self.nodes
        )
        lines.extend(
            [
                "",
                "Compiler audits:",
                f"- repeated primitive computations: {list(self.audits.repeated_primitive_computations)}",
                f"- non-identifying summaries: {list(self.audits.non_identifying_summaries)}",
                "",
                "Warnings:",
            ]
        )
        lines.extend(f"- {warning}" for warning in self.warnings)
        if not self.warnings:
            lines.append("- none")
        return "\n".join(lines) + "\n"


def domain_fingerprint(events: Iterable[PredictionEvent]) -> str:
    events = sorted(events, key=lambda event: str(event.id))
    hasher = hashlib.sha256()
    hasher.update(b"[")
    for index, event in enumerate(events):
        if index:
            hasher.update(b",")
        payload = canonical_value(
            {
                "id": event.id,
                "state": event.state.to_dict(),
                "target": canonical_value(event.target),
            }
        )
        hasher.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    hasher.update(b"]")
    return hasher.hexdigest()


def _mermaid_id(node: str) -> str:
    return "q_" + "".join(character if character.isalnum() else "_" for character in node)


def _resolve_trace_input_references(
    value: Any,
    *,
    state: Any,
    outputs: Mapping[str, Any],
) -> Any:
    if isinstance(value, Mapping):
        if value == {"$qprogram_ref": "state"}:
            return state
        if set(value) == {"$qprogram_ref", "quantum_id"} and value["$qprogram_ref"] == "producer":
            producer = str(value["quantum_id"])
            try:
                return outputs[producer]
            except KeyError as exc:
                raise KeyError(f"trace input references unknown producer {producer!r}") from exc
        return {
            key: _resolve_trace_input_references(item, state=state, outputs=outputs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_trace_input_references(item, state=state, outputs=outputs)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_trace_input_references(item, state=state, outputs=outputs)
            for item in value
        )
    return value
