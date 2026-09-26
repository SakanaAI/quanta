from __future__ import annotations

from contextlib import AbstractContextManager
import contextvars
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import inspect
from typing import Any, Callable, Generic, Iterable, Mapping, ParamSpec, TypeVar, get_args, get_origin, get_type_hints

from .errors import DefinitionError, TraceError
from .types import InvocationTrace, PredictionEvent, PredictiveState, canonical_key, canonical_value


P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class QuantumDefinition:
    id: str
    label: str
    function: Callable[..., Any]
    wrapper: Callable[..., Any]
    declared_source_reads: tuple[str, ...]
    output_cardinality: int | None


@dataclass(frozen=True)
class PrimitiveDefinition:
    name: str
    function: Callable[..., Any]
    wrapper: Callable[..., Any]
    cost: int
    table_cardinality: int
    max_iterations: int | None


@dataclass(frozen=True)
class SemanticValue(Generic[T]):
    value: T
    producer: str

    def unwrap(self) -> T:
        return self.value

    def __bool__(self) -> bool:
        raise TraceError("semantic values cannot be used as implicit guards; use 'with when(value, expected) as active'")


@dataclass
class _InvocationFrame:
    quantum_id: str
    source_reads: set[str] = field(default_factory=set)
    primitive_calls: set[tuple[str, str]] = field(default_factory=set)


class QRegistry:
    """Owns all definitions for one exact functional Q-program."""

    def __init__(self, program_id: str, *, unit_complexity: int = 16) -> None:
        if int(unit_complexity) <= 0:
            raise ValueError("unit_complexity must be positive")
        self.program_id = str(program_id)
        self.unit_complexity = int(unit_complexity)
        self.quantum_definitions: dict[str, QuantumDefinition] = {}
        self.primitive_definitions: dict[str, PrimitiveDefinition] = {}

    def primitive(
        self,
        *,
        cost: int,
        table_cardinality: int = 0,
        max_iterations: int | None = None,
        name: str | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        if int(cost) < 0 or int(table_cardinality) < 0:
            raise DefinitionError("primitive costs and table cardinalities must be non-negative")
        if max_iterations is not None and int(max_iterations) <= 0:
            raise DefinitionError("primitive max_iterations must be positive when provided")

        def decorate(function: Callable[P, T]) -> Callable[P, T]:
            _require_typed_signature(function)
            primitive_name = str(name or function.__name__)
            if primitive_name in self.primitive_definitions:
                raise DefinitionError(f"duplicate primitive identity: {primitive_name}")

            @wraps(function)
            def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
                frame = _ACTIVE_INVOCATION.get()
                recorder = _ACTIVE_RECORDER.get()
                if frame is not None and recorder is not None and recorder.record_primitive_calls:
                    frame.primitive_calls.add(
                        (
                            primitive_name,
                            canonical_key({"args": args, "kwargs": kwargs}),
                        )
                    )
                return function(*args, **kwargs)

            definition = PrimitiveDefinition(
                name=primitive_name,
                function=function,
                wrapper=wrapped,
                cost=int(cost),
                table_cardinality=int(table_cardinality),
                max_iterations=None if max_iterations is None else int(max_iterations),
            )
            self.primitive_definitions[primitive_name] = definition
            setattr(wrapped, "__qprimitive__", definition)
            return wrapped

        return decorate

    def quantum(
        self,
        quantum_id: str,
        *,
        label: str | None = None,
        source_reads: Iterable[str] = (),
        output_cardinality: int | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T | SemanticValue[T]]]:
        stable_id = str(quantum_id)
        if not stable_id or stable_id.startswith("<"):
            raise DefinitionError(f"invalid quantum identity: {stable_id!r}")
        if stable_id in self.quantum_definitions:
            raise DefinitionError(f"duplicate quantum identity: {stable_id}")

        def decorate(function: Callable[P, T]) -> Callable[P, T | SemanticValue[T]]:
            _require_typed_signature(function)

            @wraps(function)
            def wrapped(*args: P.args, **kwargs: P.kwargs) -> T | SemanticValue[T]:
                recorder = _ACTIVE_RECORDER.get()
                if recorder is None:
                    return function(*args, **kwargs)
                return recorder.call(stable_id, function, args, kwargs)

            definition = QuantumDefinition(
                id=stable_id,
                label=str(label or stable_id),
                function=function,
                wrapper=wrapped,
                declared_source_reads=tuple(sorted({str(item) for item in source_reads})),
                output_cardinality=None if output_cardinality is None else int(output_cardinality),
            )
            self.quantum_definitions[stable_id] = definition
            setattr(wrapped, "__quantum__", definition)
            return wrapped

        return decorate


class TraceRecorder:
    def __init__(
        self,
        registry: QRegistry,
        event: PredictionEvent,
        *,
        output_overrides: Mapping[str, Any] | None = None,
        record_primitive_calls: bool = True,
    ) -> None:
        self.registry = registry
        self.event = event
        self.invocations: list[InvocationTrace] = []
        self.called: set[str] = set()
        self.primitive_calls: dict[str, tuple[tuple[str, str], ...]] = {}
        self.output_overrides = dict(output_overrides or {})
        self.record_primitive_calls = bool(record_primitive_calls)

    def call(
        self,
        quantum_id: str,
        function: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> SemanticValue[T]:
        if quantum_id in self.called:
            raise TraceError(f"event {self.event.id!r} calls quantum {quantum_id!r} more than once")
        self.called.add(quantum_id)
        data_parents = tuple(sorted(_producers((args, kwargs))))
        control_parents = tuple(sorted({guard.producer for guard in _ACTIVE_GUARDS.get()}))
        clean_args = _unwrap(args)
        clean_kwargs = _unwrap(kwargs)
        frame = _InvocationFrame(quantum_id)
        frame_token = _ACTIVE_INVOCATION.set(frame)
        try:
            output = function(*clean_args, **clean_kwargs)
        finally:
            _ACTIVE_INVOCATION.reset(frame_token)
        if isinstance(output, SemanticValue):
            output = output.value
        if quantum_id in self.output_overrides:
            output = self.output_overrides[quantum_id]
        definition = self.registry.quantum_definitions[quantum_id]
        _validate_runtime_type(function, output)
        observed_reads = tuple(sorted(frame.source_reads))
        self.primitive_calls[quantum_id] = tuple(sorted(frame.primitive_calls))
        declared = set(definition.declared_source_reads)
        if declared and not set(observed_reads).issubset(declared):
            undeclared = sorted(set(observed_reads) - declared)
            raise TraceError(f"quantum {quantum_id!r} read undeclared source fields {undeclared}")
        self.invocations.append(
            InvocationTrace(
                quantum_id=quantum_id,
                inputs=canonical_value(
                    {
                        "args": _trace_input_references(args),
                        "kwargs": _trace_input_references(kwargs),
                    }
                ),
                output=canonical_value(output),
                data_parents=data_parents,
                control_parents=control_parents,
                source_reads=observed_reads,
            )
        )
        return SemanticValue(output, quantum_id)


class _When(AbstractContextManager[bool]):
    def __init__(self, value: Any, expected: Any) -> None:
        self.value = value
        self.expected = expected
        self.active = (value.value if isinstance(value, SemanticValue) else value) == expected
        self.token: contextvars.Token[tuple[SemanticValue[Any], ...]] | None = None

    def __enter__(self) -> bool:
        if self.active and isinstance(self.value, SemanticValue):
            self.token = _ACTIVE_GUARDS.set((*_ACTIVE_GUARDS.get(), self.value))
        return self.active

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.token is not None:
            _ACTIVE_GUARDS.reset(self.token)


def when(value: Any, expected: Any) -> _When:
    """Declare a branch guard so calls in the active branch gain a control edge."""
    return _When(value, expected)


class _WhenItem(_When):
    def __init__(self, value: Any, index: int, expected: Any) -> None:
        semantic = value.value if isinstance(value, SemanticValue) else value
        self.value = value
        self.expected = expected
        self.active = semantic[int(index)] == expected
        self.token = None


def when_item(value: Any, index: int, expected: Any) -> _WhenItem:
    """Guard on one item of a compound semantic value while preserving its provenance."""
    if not isinstance(index, int):
        raise TypeError("when_item index must be an integer")
    return _WhenItem(value, index, expected)


def unwrap(value: T | SemanticValue[T]) -> T:
    return value.value if isinstance(value, SemanticValue) else value


def record_source_read(field_name: str) -> None:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        return
    frame = _ACTIVE_INVOCATION.get()
    if frame is None:
        raise TraceError("predictive-state fields may only be read inside an annotated quantum")
    frame.source_reads.add(str(field_name))


_ACTIVE_RECORDER: contextvars.ContextVar[TraceRecorder | None] = contextvars.ContextVar(
    "qprogram_recorder", default=None
)
_ACTIVE_INVOCATION: contextvars.ContextVar[_InvocationFrame | None] = contextvars.ContextVar(
    "qprogram_invocation", default=None
)
_ACTIVE_GUARDS: contextvars.ContextVar[tuple[SemanticValue[Any], ...]] = contextvars.ContextVar(
    "qprogram_guards", default=()
)


def tracing(recorder: TraceRecorder) -> contextvars.Token[TraceRecorder | None]:
    return _ACTIVE_RECORDER.set(recorder)


def stop_tracing(token: contextvars.Token[TraceRecorder | None]) -> None:
    _ACTIVE_RECORDER.reset(token)


def _require_typed_signature(function: Callable[..., Any]) -> None:
    signature = inspect.signature(function)
    hints = _resolved_type_hints(function)
    missing = [name for name in signature.parameters if name not in hints]
    if "return" not in hints:
        missing.append("return")
    if missing:
        raise DefinitionError(f"quantum {function.__name__!r} is missing type annotations for {missing}")


def _validate_runtime_type(function: Callable[..., Any], output: Any) -> None:
    expected = _resolved_type_hints(function)["return"]
    if not _matches_runtime_type(output, expected):
        raise TraceError(
            f"quantum {function.__name__!r} returned {type(output).__name__}, expected {expected!r}"
        )


@lru_cache(maxsize=None)
def _resolved_type_hints(function: Callable[..., Any]) -> dict[str, Any]:
    return get_type_hints(function)


def _matches_runtime_type(value: Any, expected: Any) -> bool:
    if expected is Any:
        return True
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_matches_runtime_type(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _matches_runtime_type(item, item_type)
            for item, item_type in zip(value, arguments)
        )
    if origin is list:
        return isinstance(value, list) and all(
            _matches_runtime_type(item, arguments[0]) for item in value
        )
    if origin is not None and arguments:
        return any(_matches_runtime_type(value, argument) for argument in arguments)
    return isinstance(value, expected) if isinstance(expected, type) else True


def _producers(value: Any) -> set[str]:
    if isinstance(value, SemanticValue):
        return {value.producer}
    if isinstance(value, Mapping):
        return set().union(*(_producers(item) for item in value.values()), set())
    if isinstance(value, (tuple, list, set, frozenset)):
        return set().union(*(_producers(item) for item in value), set())
    return set()


def _unwrap(value: Any) -> Any:
    if isinstance(value, SemanticValue):
        return value.value
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap(item) for key, item in value.items()}
    if isinstance(value, set):
        return {_unwrap(item) for item in value}
    return value


def _trace_input_references(value: Any) -> Any:
    """Encode immutable trace inputs without copying state or parent outputs."""
    if isinstance(value, PredictiveState):
        return {"$qprogram_ref": "state"}
    if isinstance(value, SemanticValue):
        return {"$qprogram_ref": "producer", "quantum_id": value.producer}
    if isinstance(value, tuple):
        return tuple(_trace_input_references(item) for item in value)
    if isinstance(value, list):
        return [_trace_input_references(item) for item in value]
    if isinstance(value, dict):
        return {key: _trace_input_references(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_trace_input_references(item) for item in value]
        return sorted(items, key=repr)
    return value
