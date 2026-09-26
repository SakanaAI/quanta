from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import math
from typing import Any, Callable, Iterable, get_args, get_origin, get_type_hints

from .coverage import compare_coverage
from .errors import CompilationError, TraceError
from .graph import build_poset
from .runtime import QRegistry, SemanticValue, TraceRecorder, stop_tracing, tracing
from .types import (
    COMPILER_VERSION,
    AlignedEventTrace,
    CompilerAuditReport,
    CompilationMetadata,
    CompiledQProgram,
    Dependency,
    EventTrace,
    ExhaustiveValidationReport,
    InvocationSchema,
    NonIdentifyingSummary,
    PosetStructure,
    PredictionEvent,
    RepeatedPrimitiveComputation,
    ReuseStat,
    TraceSchemaCatalog,
    canonical_key,
    canonical_value,
    domain_fingerprint,
    fingerprint,
)
from .validation import validate_entrypoint, validate_registry


@dataclass(frozen=True)
class FunctionalProgram:
    registry: QRegistry
    predict: Callable[[Any], Any]

    @property
    def id(self) -> str:
        return self.registry.program_id


@dataclass(frozen=True)
class CompileOptions:
    check_determinism: bool = True
    retain_validation_traces: bool = True
    node_parameter_budget: int | None = None
    rare_activation_warning: int = 2
    rare_semantic_outcome_warning: int = 2
    semantic_cardinality_warning: int | None = 20


class _TraceSchemaInterner:
    """Intern the event-independent portions of distribution invocations."""

    def __init__(self, structure: Any) -> None:
        self.structure = structure
        self.node_to_index = {node: index for index, node in enumerate(structure.nodes)}
        self.schemas: list[InvocationSchema] = []
        self.schema_ids: dict[str, int] = {}

    def intern(self, invocation: Any) -> int:
        schema = InvocationSchema(
            inputs=invocation.inputs,
            data_parent_indices=tuple(
                self.node_to_index[parent] for parent in invocation.data_parents
            ),
            control_parent_indices=tuple(
                self.node_to_index[parent] for parent in invocation.control_parents
            ),
            source_reads=invocation.source_reads,
        )
        key = canonical_key(schema)
        schema_id = self.schema_ids.get(key)
        if schema_id is None:
            schema_id = len(self.schemas)
            self.schemas.append(schema)
            self.schema_ids[key] = schema_id
        return schema_id

    def catalog(self) -> TraceSchemaCatalog:
        node_to_index = self.node_to_index
        return TraceSchemaCatalog(
            nodes=self.structure.nodes,
            parent_masks=tuple(
                sum(1 << node_to_index[parent] for parent in self.structure.parents[node])
                for node in self.structure.nodes
            ),
            invocation_schemas=tuple(self.schemas),
        )


@dataclass(frozen=True, slots=True)
class _PendingAlignedTrace:
    event_id: str
    state: Any
    target: Any
    produced: Any
    active_mask: int
    semantic_targets: tuple[Any | None, ...]
    invocation_schema_ids: tuple[int, ...]
    invocation_order: tuple[int, ...]
    emitter_index: int

    def finalize(self, catalog: TraceSchemaCatalog) -> AlignedEventTrace:
        return AlignedEventTrace(
            event_id=self.event_id,
            state=self.state,
            target=self.target,
            produced=self.produced,
            active_mask=self.active_mask,
            semantic_targets=self.semantic_targets,
            invocation_schema_ids=self.invocation_schema_ids,
            invocation_order=self.invocation_order,
            emitter_index=self.emitter_index,
            catalog=catalog,
        )


def compile_qprogram(
    program: FunctionalProgram,
    *,
    validation_events: Iterable[PredictionEvent],
    training_events: Iterable[PredictionEvent] = (),
    evaluation_events: Iterable[PredictionEvent] = (),
    options: CompileOptions = CompileOptions(),
) -> CompiledQProgram:
    validation_events = _canonical_events(validation_events, domain_name="validation")
    training_events = _canonical_events(training_events, domain_name="training")
    evaluation_events = _canonical_events(evaluation_events, domain_name="evaluation")
    if not validation_events:
        raise CompilationError("the compiler audit domain must contain at least one prediction event")
    if options.node_parameter_budget is not None and options.node_parameter_budget <= 0:
        raise CompilationError("node_parameter_budget must be positive when provided")
    if options.rare_activation_warning < 0:
        raise CompilationError("rare_activation_warning must be non-negative")
    if options.rare_semantic_outcome_warning < 0:
        raise CompilationError("rare_semantic_outcome_warning must be non-negative")
    if (
        options.semantic_cardinality_warning is not None
        and options.semantic_cardinality_warning < 2
    ):
        raise CompilationError("semantic_cardinality_warning must be at least 2 or None")

    # Compute distribution fingerprints before retaining any traces. The canonical
    # fingerprint builder needs temporary JSON-shaped rows, so doing this at the
    # end would multiply peak memory by combining those rows with the compiled
    # training/evaluation supervision.
    validation_domain_fingerprint = domain_fingerprint(validation_events)
    training_distribution_fingerprint = domain_fingerprint(training_events)
    evaluation_distribution_fingerprint = domain_fingerprint(evaluation_events)

    static = validate_registry(program.registry)
    entrypoint_source = validate_entrypoint(program.predict, program.registry)
    # Structural compilation deliberately precedes distribution tracing. Full
    # validation invocations are needed for graph/effect/reuse audits, whereas
    # training and evaluation rows can be compacted immediately once node order
    # and parent alignment are known.
    validation_traces = tuple(
        _trace_event(program, event, check_determinism=options.check_determinism)
        for event in validation_events
    )

    reachable = {node for trace in validation_traces for node in trace.active_nodes}
    dependency_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    witnesses: dict[tuple[str, str], str] = {}
    observed_source_reads: dict[str, set[str]] = defaultdict(set)
    for trace in validation_traces:
        for invocation in trace.invocations:
            observed_source_reads[invocation.quantum_id].update(invocation.source_reads)
            for parent in invocation.data_parents:
                dependency_kinds[(parent, invocation.quantum_id)].add("data")
                witnesses.setdefault((parent, invocation.quantum_id), trace.event_id)
            for parent in invocation.control_parents:
                dependency_kinds[(parent, invocation.quantum_id)].add("control")
                witnesses.setdefault((parent, invocation.quantum_id), trace.event_id)

    source_reads = {
        node: tuple(
            sorted(
                observed_source_reads[node]
                | set(program.registry.quantum_definitions[node].declared_source_reads)
            )
        )
        for node in sorted(reachable)
    }
    structure = build_poset(reachable, dependency_kinds, witnesses, source_reads)
    _validate_structural_effect(validation_traces)
    _validate_counterfactual_effect(program, validation_events, validation_traces)
    _validate_stable_parent_signatures(validation_traces, structure)
    _validate_trace_closure(validation_traces, structure.ancestors)
    validation_traces = tuple(_add_supervision(trace, structure) for trace in validation_traces)

    schema_interner = _TraceSchemaInterner(structure)
    training_rows = _compact_distribution(
        program,
        training_events,
        structure=structure,
        reachable=reachable,
        domain_name="training",
        schema_interner=schema_interner,
    )
    training_events = ()
    evaluation_rows = _compact_distribution(
        program,
        evaluation_events,
        structure=structure,
        reachable=reachable,
        domain_name="evaluation",
        schema_interner=schema_interner,
    )
    evaluation_events = ()
    trace_schema_catalog = schema_interner.catalog()
    training_traces = tuple(trace.finalize(trace_schema_catalog) for trace in training_rows)
    evaluation_traces = tuple(trace.finalize(trace_schema_catalog) for trace in evaluation_rows)

    warnings = []
    unreachable = sorted(set(program.registry.quantum_definitions) - reachable)
    if unreachable:
        warnings.append(f"unreachable quantum definitions: {', '.join(unreachable)}")
    semantic_outcomes = {}
    for node in structure.nodes:
        values = {
            json.dumps(invocation.output, sort_keys=True, separators=(",", ":"))
            for trace in validation_traces
            for invocation in trace.invocations
            if invocation.quantum_id == node
        }
        semantic_outcomes[node] = tuple(json.loads(value) for value in sorted(values))
        declared_cardinality = program.registry.quantum_definitions[node].output_cardinality
        if declared_cardinality is not None and len(values) > declared_cardinality:
            raise CompilationError(
                f"quantum {node!r} declares output_cardinality={declared_cardinality} but the full "
                f"validation domain contains {len(values)} semantic outcomes"
            )
        cardinality = len(values)
        if cardinality == 1:
            warnings.append(
                f"quantum {node} has one reachable semantic outcome; "
                "categorical semantic supervision will be omitted"
            )
        elif (
            options.semantic_cardinality_warning is not None
            and cardinality > options.semantic_cardinality_warning
        ):
            warnings.append(
                f"quantum {node} has {cardinality} reachable semantic outcomes, exceeding "
                f"semantic_cardinality_warning={options.semantic_cardinality_warning}; "
                "categorical semantic supervision may be omitted and this may indicate an "
                "over-broad quantum"
            )
        count = sum(node in trace.active_nodes for trace in training_traces)
        if training_traces and options.rare_activation_warning and count <= options.rare_activation_warning:
            warnings.append(f"quantum {node} is active in only {count} training prediction events")

    primitive_costs = _primitive_cost_spec(program.registry)
    program_fingerprint = _program_fingerprint(
        entrypoint_source=entrypoint_source,
        quantum_sources=static.source_by_quantum,
        primitive_sources=static.source_by_primitive,
        primitive_costs=primitive_costs,
    )
    complexity = _finalize_complexity(
        tuple(report for report in static.complexity if report.quantum_id in reachable),
        semantic_outcomes,
        program.registry,
    )
    metadata = CompilationMetadata(
        compiler_version=COMPILER_VERSION,
        program_fingerprint=program_fingerprint,
        primitive_cost_fingerprint=fingerprint(primitive_costs),
        validation_domain_fingerprint=validation_domain_fingerprint,
        training_distribution_fingerprint=training_distribution_fingerprint,
        evaluation_distribution_fingerprint=evaluation_distribution_fingerprint,
        node_parameter_budget=options.node_parameter_budget,
    )
    coverage = compare_coverage(
        training_traces,
        validation_traces,
        structure,
        validation_semantic_outcomes=semantic_outcomes,
        rare_semantic_threshold=options.rare_semantic_outcome_warning,
    )
    for node, outcomes in coverage.audit_only_semantic_outcomes.items():
        sample = list(outcomes[:8])
        warnings.append(
            f"quantum {node} has {len(outcomes)} audit-only semantic outcomes; sample={sample}"
        )
    if coverage.audit_only_semantic_pairs:
        warnings.append(
            f"audit sample contains {len(coverage.audit_only_semantic_pairs)} compact semantic pairs absent from training"
        )
    for node, contexts in coverage.audit_only_parent_semantic_contexts.items():
        warnings.append(
            f"quantum {node} has {len(contexts)} audit parent-semantic contexts absent from training; "
            f"sample={list(contexts[:8])}"
        )
    for node, outcomes in coverage.training_absent_semantic_outcomes.items():
        warnings.append(
            f"quantum {node} has {len(outcomes)} audit-domain semantic outcomes absent from training; "
            f"sample={list(outcomes[:8])}"
        )
    for node, outcomes in coverage.rare_training_semantic_outcomes.items():
        warnings.append(
            f"quantum {node} has {len(outcomes)} semantic outcomes active in at most "
            f"{options.rare_semantic_outcome_warning} training events; sample={dict(outcomes[:8])}"
        )
    for node in coverage.train.one_class_activity_nodes:
        positive = coverage.train.activity_counts[node]
        negative = coverage.train.negative_eligible_counts[node]
        warnings.append(
            f"quantum {node} has one-class eligible training activity: positive={positive}, negative={negative}"
        )
    for group in coverage.train.indistinguishable_activity_groups:
        warnings.append(f"training activity columns are indistinguishable: {list(group)}")
    reuse = _reuse_report(validation_traces, structure)
    audits = _compiler_audits(
        validation_traces,
        structure=structure,
        source_reads=source_reads,
        reuse=reuse,
        registry=program.registry,
    )
    for item in audits.repeated_primitive_computations:
        warnings.append(
            f"direct dependency {item.parent}->{item.child} repeats primitive {item.primitive!r} "
            f"on identical inputs in {item.repeated_event_count}/{item.coactive_event_count} "
            "coactive audit events; consider extracting the shared computation as a quantum"
        )
    for item in audits.non_identifying_summaries:
        warnings.append(
            f"quantum {item.node} is an injective one-consumer summary of "
            f"{item.distinct_input_contexts} input contexts into {item.distinct_outputs} outputs, "
            f"is always coactive with {item.consumer}, and reads no source; consider merging it "
            "or replacing it with a compressive reusable factor"
        )
    return CompiledQProgram(
        id=program.id,
        structure=structure,
        node_labels={node: program.registry.quantum_definitions[node].label for node in structure.nodes},
        source_reads=source_reads,
        semantic_types={
            node: _semantic_type_name(program.registry.quantum_definitions[node].function)
            for node in structure.nodes
        },
        semantic_outcomes=semantic_outcomes,
        complexity=complexity,
        trace_schema_catalog=trace_schema_catalog,
        validation_traces=validation_traces if options.retain_validation_traces else (),
        training_traces=training_traces,
        evaluation_traces=evaluation_traces,
        coverage=coverage,
        reuse=reuse,
        audits=audits,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def validate_exhaustive_domain(
    program: FunctionalProgram,
    events: Iterable[PredictionEvent],
    *,
    check_determinism: bool = True,
    expected_structure: PosetStructure | None = None,
) -> ExhaustiveValidationReport:
    """Validate a canonical exhaustive domain without retaining events or traces.

    Event identities must be unique and arrive in strictly increasing string order.
    This ordering contract makes the incremental domain fingerprint identical to
    :func:`domain_fingerprint` without sorting or materializing the iterable. Exact
    semantic and emitted-token counters retain one key per distinct canonical value,
    so report memory is independent of the number of prediction events. When an
    expected structure is supplied, every streamed event is also checked online for
    known nodes, downward closure, and the compiled conjunctive parent signature.
    """
    static = validate_registry(program.registry)
    entrypoint_source = validate_entrypoint(program.predict, program.registry)
    primitive_costs = _primitive_cost_spec(program.registry)
    program_fingerprint = _program_fingerprint(
        entrypoint_source=entrypoint_source,
        quantum_sources=static.source_by_quantum,
        primitive_sources=static.source_by_primitive,
        primitive_costs=primitive_costs,
    )

    domain_hasher = hashlib.sha256()
    domain_hasher.update(b"[")
    event_count = 0
    invocation_count = 0
    first_event_id: str | None = None
    previous_event_id: str | None = None
    activity_counts: Counter[str] = Counter()
    semantic_outcome_counts: dict[str, Counter[str]] = defaultdict(Counter)
    emitted_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dependency_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    witnesses: dict[tuple[str, str], str] = {}
    source_reads: dict[str, set[str]] = defaultdict(set)

    for event in events:
        event_id = str(event.id)
        if previous_event_id is not None and event_id <= previous_event_id:
            relation = "duplicate" if event_id == previous_event_id else "out-of-order"
            raise CompilationError(
                f"{relation} exhaustive validation event identity {event_id!r}; "
                "events must arrive in strictly increasing string-ID order"
            )
        if first_event_id is None:
            first_event_id = event_id
        previous_event_id = event_id

        event_payload = {
            "id": event.id,
            "state": event.state.to_dict(),
            "target": canonical_value(event.target),
        }
        if event_count:
            domain_hasher.update(b",")
        domain_hasher.update(
            json.dumps(
                canonical_value(event_payload),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        trace = _trace_event(
            program,
            event,
            check_determinism=bool(check_determinism),
            record_primitive_calls=False,
        )
        _validate_structural_effect((trace,))
        if expected_structure is not None:
            _validate_distribution_nodes(
                (trace,), set(expected_structure.nodes), "exhaustive validation"
            )
            _validate_stable_parent_signatures((trace,), expected_structure)
            _validate_trace_closure((trace,), expected_structure.ancestors)
        event_count += 1
        invocation_count += len(trace.invocations)
        activity_counts.update(trace.active_nodes)
        for invocation in trace.invocations:
            semantic_outcome_counts[invocation.quantum_id][canonical_key(invocation.output)] += 1
            if invocation.emitted_token is not None:
                emitted_token_counts[invocation.quantum_id][
                    canonical_key(invocation.emitted_token)
                ] += 1
            source_reads[invocation.quantum_id].update(invocation.source_reads)
            for parent in invocation.data_parents:
                dependency_kinds[(parent, invocation.quantum_id)].add("data")
                witnesses.setdefault((parent, invocation.quantum_id), trace.event_id)
            for parent in invocation.control_parents:
                dependency_kinds[(parent, invocation.quantum_id)].add("control")
                witnesses.setdefault((parent, invocation.quantum_id), trace.event_id)

    if event_count == 0:
        raise CompilationError("the exhaustive validation domain must contain at least one prediction event")
    domain_hasher.update(b"]")
    active_nodes = tuple(sorted(activity_counts))
    dependencies = tuple(
        Dependency(
            parent=parent,
            child=child,
            kinds=tuple(sorted(dependency_kinds[(parent, child)])),
            witness_event=witnesses[(parent, child)],
        )
        for parent, child in sorted(dependency_kinds)
    )
    return ExhaustiveValidationReport(
        compiler_version=COMPILER_VERSION,
        program_id=program.id,
        program_fingerprint=program_fingerprint,
        primitive_cost_fingerprint=fingerprint(primitive_costs),
        validation_domain_fingerprint=domain_hasher.hexdigest(),
        event_count=event_count,
        invocation_count=invocation_count,
        first_event_id=first_event_id or "",
        last_event_id=previous_event_id or "",
        active_nodes=active_nodes,
        activity_counts={node: activity_counts[node] for node in active_nodes},
        semantic_outcome_counts={
            node: dict(sorted(semantic_outcome_counts[node].items()))
            for node in active_nodes
        },
        emitted_token_counts={
            node: dict(sorted(emitted_token_counts[node].items()))
            for node in active_nodes
        },
        dependencies=dependencies,
        source_reads={node: tuple(sorted(source_reads[node])) for node in active_nodes},
        expected_structure_validated=expected_structure is not None,
    )


def _primitive_cost_spec(registry: QRegistry) -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "cost": item.cost,
            "table_cardinality": item.table_cardinality,
            "max_iterations": item.max_iterations,
        }
        for item in sorted(registry.primitive_definitions.values(), key=lambda item: item.name)
    ]


def _program_fingerprint(
    *,
    entrypoint_source: str,
    quantum_sources: dict[str, str],
    primitive_sources: dict[str, str],
    primitive_costs: list[dict[str, Any]],
) -> str:
    return fingerprint(
        {
            "entrypoint": entrypoint_source,
            "quanta": quantum_sources,
            "primitive_sources": primitive_sources,
            "primitives": primitive_costs,
        }
    )


def _trace_event(
    program: FunctionalProgram,
    event: PredictionEvent,
    *,
    check_determinism: bool,
    record_primitive_calls: bool = True,
) -> EventTrace:
    trace = _execute_once(program, event, record_primitive_calls=record_primitive_calls)
    if check_determinism:
        repeated = _execute_once(program, event, record_primitive_calls=record_primitive_calls)
        if trace != repeated:
            raise CompilationError(
                f"program is nondeterministic for event {event.id!r}: first trace differs from repeated trace"
            )
    return trace


def _canonical_events(
    events: Iterable[PredictionEvent],
    *,
    domain_name: str,
) -> tuple[PredictionEvent, ...]:
    rows = tuple(events)
    ids = [str(event.id) for event in rows]
    duplicates = sorted(event_id for event_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise CompilationError(f"duplicate {domain_name} event identities: {duplicates}")
    return tuple(sorted(rows, key=lambda event: str(event.id)))


def _validate_distribution_nodes(
    traces: tuple[EventTrace, ...],
    validation_nodes: set[str],
    domain_name: str,
) -> None:
    for trace in traces:
        unknown = set(trace.active_nodes) - validation_nodes
        if unknown:
            raise CompilationError(
                f"{domain_name} event {trace.event_id!r} activates nodes absent from the compiled "
                f"audit domain: {sorted(unknown)}"
            )


def _compact_distribution(
    program: FunctionalProgram,
    events: tuple[PredictionEvent, ...],
    *,
    structure: Any,
    reachable: set[str],
    domain_name: str,
    schema_interner: _TraceSchemaInterner,
) -> tuple[_PendingAlignedTrace, ...]:
    """Trace and compact one distribution without retaining full invocations."""
    rows: list[_PendingAlignedTrace] = []
    for event in events:
        trace = _trace_event(
            program,
            event,
            check_determinism=False,
            record_primitive_calls=False,
        )
        _validate_distribution_nodes((trace,), reachable, domain_name)
        _validate_stable_parent_signatures((trace,), structure)
        _validate_trace_closure((trace,), structure.ancestors)
        rows.append(_align_trace(trace, structure, schema_interner))
    return tuple(rows)


def _align_trace(
    trace: EventTrace,
    structure: Any,
    schema_interner: _TraceSchemaInterner,
) -> _PendingAlignedTrace:
    node_to_index = schema_interner.node_to_index
    width = len(structure.nodes)
    semantics: list[Any | None] = [None] * width
    schema_ids = [-1] * width
    active_mask = 0
    emitter_index = -1
    invocation_order: list[int] = []
    for invocation in trace.invocations:
        index = node_to_index[invocation.quantum_id]
        invocation_order.append(index)
        active_mask |= 1 << index
        semantics[index] = invocation.output
        schema_ids[index] = schema_interner.intern(invocation)
        if invocation.emitted_token is not None:
            if emitter_index >= 0:
                raise CompilationError(
                    f"event {trace.event_id!r} has multiple emitting quanta while compacting"
                )
            emitter_index = index
    if emitter_index < 0:
        raise CompilationError(f"event {trace.event_id!r} has no emitting quantum while compacting")
    return _PendingAlignedTrace(
        event_id=trace.event_id,
        state=trace.state,
        target=trace.target,
        produced=trace.produced,
        active_mask=active_mask,
        semantic_targets=tuple(semantics),
        invocation_schema_ids=tuple(schema_ids),
        invocation_order=tuple(invocation_order),
        emitter_index=emitter_index,
    )


def _execute_once(
    program: FunctionalProgram,
    event: PredictionEvent,
    *,
    output_overrides: dict[str, Any] | None = None,
    enforce_target: bool = True,
    record_primitive_calls: bool = True,
) -> EventTrace:
    recorder = TraceRecorder(
        program.registry,
        event,
        output_overrides=output_overrides,
        record_primitive_calls=record_primitive_calls,
    )
    token = tracing(recorder)
    try:
        produced = program.predict(event.state)
    except Exception as exc:
        if isinstance(exc, (CompilationError, TraceError)):
            raise
        raise CompilationError(f"event {event.id!r} failed during exact functional execution: {exc}") from exc
    finally:
        stop_tracing(token)
    if not isinstance(produced, SemanticValue):
        raise CompilationError(
            f"event {event.id!r} returned an untraced token; the next token must be produced by an annotated quantum"
        )
    actual = canonical_value(produced.value)
    expected = canonical_value(event.target)
    if enforce_target and actual != expected:
        semantic = {invocation.quantum_id: invocation.output for invocation in recorder.invocations}
        raise CompilationError(
            f"event {event.id!r} target mismatch: expected {expected!r}, produced {actual!r}; "
            f"prefix={event.state.prefix!r}; active={tuple(item.quantum_id for item in recorder.invocations)!r}; "
            f"semantic={semantic!r}"
        )
    invocations = tuple(
        replace(invocation, emitted_token=actual)
        if invocation.quantum_id == produced.producer
        else invocation
        for invocation in recorder.invocations
    )
    return EventTrace(
        event_id=str(event.id),
        state=event.state.to_dict(),
        target=expected,
        produced=actual,
        invocations=invocations,
        active_nodes=tuple(sorted(item.quantum_id for item in invocations)),
        primitive_calls=tuple(sorted(recorder.primitive_calls.items())),
    )


def _validate_structural_effect(traces: tuple[EventTrace, ...]) -> None:
    for trace in traces:
        invocation_by_node = {item.quantum_id: item for item in trace.invocations}
        emitters = [item.quantum_id for item in trace.invocations if item.emitted_token is not None]
        if len(emitters) != 1:
            raise CompilationError(f"event {trace.event_id!r} must have exactly one emitting quantum")
        effectful = set(emitters)
        stack = list(emitters)
        while stack:
            child = stack.pop()
            invocation = invocation_by_node[child]
            for parent in (*invocation.data_parents, *invocation.control_parents):
                if parent not in effectful:
                    effectful.add(parent)
                    stack.append(parent)
        dead = set(trace.active_nodes) - effectful
        if dead:
            raise CompilationError(
                f"event {trace.event_id!r} executes structurally dead quanta {sorted(dead)}; "
                "every active quantum must influence the emitted token through data or control provenance"
            )


def _validate_counterfactual_effect(
    program: FunctionalProgram,
    events: tuple[PredictionEvent, ...],
    traces: tuple[EventTrace, ...],
) -> None:
    """Audit observed semantic alternatives without claiming a formal proof."""
    outcomes: dict[str, dict[str, Any]] = defaultdict(dict)
    for trace in traces:
        for invocation in trace.invocations:
            key = json.dumps(invocation.output, sort_keys=True, separators=(",", ":"))
            outcomes[invocation.quantum_id][key] = invocation.output
    event_by_id = {event.id: event for event in events}
    for quantum_id, values_by_key in outcomes.items():
        if len(values_by_key) < 2:
            continue
        changed = False
        for baseline in traces:
            invocation = next(
                (item for item in baseline.invocations if item.quantum_id == quantum_id),
                None,
            )
            if invocation is None:
                continue
            baseline_key = json.dumps(invocation.output, sort_keys=True, separators=(",", ":"))
            alternative = next(value for key, value in values_by_key.items() if key != baseline_key)
            restored = _restore_semantic_value(
                alternative,
                program.registry.quantum_definitions[quantum_id].function,
            )
            try:
                perturbed = _execute_once(
                    program,
                    event_by_id[baseline.event_id],
                    output_overrides={quantum_id: restored},
                    enforce_target=False,
                    record_primitive_calls=False,
                )
            except (CompilationError, TraceError, KeyError, ValueError):
                changed = True
                break
            baseline_downstream = tuple(
                (item.quantum_id, item.output) for item in baseline.invocations if item.quantum_id != quantum_id
            )
            perturbed_downstream = tuple(
                (item.quantum_id, item.output) for item in perturbed.invocations if item.quantum_id != quantum_id
            )
            if baseline.produced != perturbed.produced or baseline_downstream != perturbed_downstream:
                changed = True
                break
        if not changed:
            raise CompilationError(
                f"quantum {quantum_id!r} fails the counterfactual effect audit: replacing its output "
                "with other observed semantic outcomes never changes a downstream value, branch, or token"
            )


def _validate_trace_closure(
    traces: tuple[EventTrace, ...],
    ancestors: dict[str, tuple[str, ...]],
) -> None:
    for trace in traces:
        active = set(trace.active_nodes)
        for node in trace.active_nodes:
            missing = set(ancestors[node]) - active
            if missing:
                raise CompilationError(
                    f"event {trace.event_id!r} is not downward closed: {node!r} is active without "
                    f"ancestors {sorted(missing)}"
                )


def _validate_stable_parent_signatures(
    traces: tuple[EventTrace, ...],
    structure: PosetStructure,
) -> None:
    """Require every active child to receive all compiled direct cover parents."""
    for trace in traces:
        for invocation in trace.invocations:
            required = set(structure.parents[invocation.quantum_id])
            observed = set(invocation.data_parents) | set(invocation.control_parents)
            missing = required - observed
            if missing:
                raise CompilationError(
                    f"event {trace.event_id!r} gives quantum {invocation.quantum_id!r} an unstable "
                    f"parent signature: missing compiled direct parents {sorted(missing)}; "
                    f"observed data={list(invocation.data_parents)}, "
                    f"control={list(invocation.control_parents)}. Refactor alternative producer "
                    "sets into distinct quantum identities or a stable merge quantum."
                )


def _add_supervision(trace: EventTrace, structure: Any) -> EventTrace:
    active = set(trace.active_nodes)
    invocation_by_node = {item.quantum_id: item for item in trace.invocations}
    targets = tuple(int(node in active) for node in structure.nodes)
    mask = tuple(int(all(parent in active for parent in structure.parents[node])) for node in structure.nodes)
    semantics = tuple(invocation_by_node[node].output if node in active else None for node in structure.nodes)
    return replace(
        trace,
        activity_targets=targets,
        activity_mask=mask,
        semantic_targets=semantics,
        parent_context=(),
    )


def _semantic_type_name(function: Callable[..., Any]) -> str:
    annotation = get_type_hints(function)["return"]
    return getattr(annotation, "__qualname__", str(annotation))


def _reuse_report(
    traces: tuple[EventTrace, ...],
    structure: Any,
) -> dict[str, ReuseStat]:
    result = {}
    for node in structure.nodes:
        invocations = tuple(
            invocation
            for trace in traces
            for invocation in trace.invocations
            if invocation.quantum_id == node
        )
        inputs = {
            json.dumps(
                trace.resolve_invocation_inputs(invocation),
                sort_keys=True,
                separators=(",", ":"),
            )
            for trace in traces
            for invocation in trace.invocations
            if invocation.quantum_id == node
        }
        outputs = {
            json.dumps(invocation.output, sort_keys=True, separators=(",", ":"))
            for invocation in invocations
        }
        roles = {
            (invocation.data_parents, invocation.control_parents, invocation.source_reads)
            for invocation in invocations
        }
        consumers = tuple(
            sorted(child for parent, child in structure.unreduced_edges if parent == node)
        )
        result[node] = ReuseStat(
            activation_count=len(invocations),
            distinct_semantic_input_contexts=len(inputs),
            distinct_semantic_outputs=len(outputs),
            downstream_consumers=consumers,
            distinct_roles=len(roles),
            appears_in_multiple_roles=len(roles) > 1,
        )
    return result


def _compiler_audits(
    traces: tuple[EventTrace, ...],
    *,
    structure: PosetStructure,
    source_reads: dict[str, tuple[str, ...]],
    reuse: dict[str, ReuseStat],
    registry: QRegistry,
) -> CompilerAuditReport:
    coactive_counts: Counter[tuple[str, str]] = Counter()
    repeated_counts: Counter[tuple[str, str, str]] = Counter()
    for trace in traces:
        primitive_calls = {
            node: set(calls)
            for node, calls in trace.primitive_calls
        }
        active = set(trace.active_nodes)
        for parent, child in structure.unreduced_edges:
            if parent not in active or child not in active:
                continue
            coactive_counts[(parent, child)] += 1
            shared = primitive_calls.get(parent, set()) & primitive_calls.get(child, set())
            for primitive in {primitive for primitive, _ in shared}:
                definition = registry.primitive_definitions[primitive]
                if definition.cost > 1 or definition.table_cardinality > 0:
                    repeated_counts[(parent, child, primitive)] += 1
    repeated = tuple(
        RepeatedPrimitiveComputation(
            parent=parent,
            child=child,
            primitive=primitive,
            coactive_event_count=coactive_counts[(parent, child)],
            repeated_event_count=count,
        )
        for (parent, child, primitive), count in sorted(repeated_counts.items())
    )

    active_events = {
        node: {trace.event_id for trace in traces if node in trace.active_nodes}
        for node in structure.nodes
    }
    summaries = []
    for node in structure.nodes:
        consumers = reuse[node].downstream_consumers
        if node in structure.leaves or len(consumers) != 1 or source_reads[node]:
            continue
        consumer = consumers[0]
        stat = reuse[node]
        if (
            stat.distinct_semantic_input_contexts <= 1
            or stat.distinct_semantic_input_contexts != stat.distinct_semantic_outputs
            or active_events[node] != active_events[consumer]
        ):
            continue
        summaries.append(
            NonIdentifyingSummary(
                node=node,
                consumer=consumer,
                activation_count=stat.activation_count,
                distinct_input_contexts=stat.distinct_semantic_input_contexts,
                distinct_outputs=stat.distinct_semantic_outputs,
            )
        )
    return CompilerAuditReport(
        repeated_primitive_computations=repeated,
        non_identifying_summaries=tuple(summaries),
    )


def _restore_semantic_value(value: Any, function: Callable[..., Any]) -> Any:
    annotation = get_type_hints(function)["return"]
    return _restore_annotated_value(value, annotation)


def _restore_annotated_value(value: Any, annotation: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"enum", "name"} and isinstance(annotation, type):
        return annotation[value["name"]]
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is tuple and isinstance(value, list):
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_restore_annotated_value(item, arguments[0]) for item in value)
        return tuple(
            _restore_annotated_value(item, item_type)
            for item, item_type in zip(value, arguments)
        )
    if origin is list and isinstance(value, list):
        return [
            _restore_annotated_value(item, arguments[0])
            for item in value
        ]
    return value


def _finalize_complexity(
    reports: tuple[Any, ...],
    semantic_outcomes: dict[str, tuple[Any, ...]],
    registry: QRegistry,
) -> tuple[Any, ...]:
    finalized = []
    for report in reports:
        definition = registry.quantum_definitions[report.quantum_id]
        if definition.output_cardinality is not None:
            finalized.append(report)
            continue
        cardinality = len(semantic_outcomes.get(report.quantum_id, ()))
        cardinality_cost = max(1, math.ceil(math.log2(max(2, cardinality))))
        components = dict(report.components)
        components["output_cardinality"] = cardinality_cost
        total = report.total + cardinality_cost
        inferred = replace(
            report,
            total=total,
            components=tuple(sorted(components.items())),
            output_cardinality=cardinality,
        )
        if total > report.budget:
            details = ", ".join(f"{name}={cost}" for name, cost in inferred.components)
            raise CompilationError(
                f"quantum {report.quantum_id!r} costs {total}, exceeding unit budget "
                f"{report.budget} after inferring semantic output cardinality: {details}"
            )
        finalized.append(inferred)
    return tuple(finalized)
