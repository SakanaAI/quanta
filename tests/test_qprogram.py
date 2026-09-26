from __future__ import annotations

from dataclasses import replace
from enum import Enum
import json
import math

import pytest
import torch

from quanta.qprogram import (
    AlignedEventTrace,
    CompilationError,
    CompiledQProgram,
    CompileOptions,
    ComplexityError,
    DefinitionError,
    FunctionalProgram,
    PredictionEvent,
    PredictiveState,
    QRegistry,
    SequenceExample,
    TraceError,
    compile_qprogram,
    activity_tensors,
    canonical_key,
    prediction_events,
    unwrap,
    validate_exhaustive_domain,
    validate_greedy,
    when,
    when_item,
)
from quanta.experiments.quanta_net.quantanet import (
    ParentIntervention,
    QComputer,
    SemanticClassifierSuite,
    balanced_activity_loss,
    qcore_training_loss,
    semantic_classification_loss,
)
from quanta.experiments.quanta_net.steering import (
    FixedQInterface,
    FrozenQTeacher,
    LayerwiseQAlignment,
    alignment_objective,
    layerwise_alignment_loss,
    rms_depth_scales,
)
from quanta.experiments.number_naming.compile_qprogram import _artifact_name
from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.qprogram.graph import build_poset


class Group(Enum):
    HIGH = "high"
    LOW = "low"


_REGISTERED_TABLE = ("zero", "one")


def _branching_program() -> tuple[FunctionalProgram, tuple[PredictionEvent, ...]]:
    registry = QRegistry("synthetic_branch", unit_complexity=32)

    @registry.primitive(cost=1)
    def first_digit(digits: tuple[int, ...]) -> int:
        return digits[0]

    @registry.quantum("CONTROL_GROUP", source_reads=("digits",), output_cardinality=2)
    def control_group(state: PredictiveState) -> Group:
        digit = first_digit(state.read("digits"))
        return Group.HIGH if digit >= 5 else Group.LOW

    @registry.quantum("EMIT_HIGH", output_cardinality=1)
    def emit_high(group: Group) -> str:
        return "high"

    @registry.quantum("EMIT_LOW", output_cardinality=1)
    def emit_low(group: Group) -> str:
        return "low"

    @registry.quantum("UNREACHABLE", output_cardinality=1)
    def unreachable(group: Group) -> str:
        return "never"

    def predict(state: PredictiveState) -> str:
        group = control_group(state)
        with when(group, Group.HIGH) as high:
            if high:
                return emit_high(group)
        with when(group, Group.LOW) as low:
            if low:
                return emit_low(group)
        return unreachable(group)

    events = (
        PredictionEvent("low", PredictiveState((2,), ()), "low"),
        PredictionEvent("high", PredictiveState((8,), ()), "high"),
    )
    return FunctionalProgram(registry, predict), events


def test_compiler_traces_data_control_sources_and_supervision() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(
        program,
        validation_events=events,
        training_events=(events[0],),
        evaluation_events=(events[1],),
    )

    assert compiled.nodes == ("CONTROL_GROUP", "EMIT_HIGH", "EMIT_LOW")
    assert compiled.edges == (("CONTROL_GROUP", "EMIT_HIGH"), ("CONTROL_GROUP", "EMIT_LOW"))
    assert compiled.source_reads == {
        "CONTROL_GROUP": ("digits",),
        "EMIT_HIGH": (),
        "EMIT_LOW": (),
    }
    dependency = next(item for item in compiled.structure.dependencies if item.child == "EMIT_HIGH")
    assert dependency.kinds == ("control", "data")
    low_trace = compiled.training_traces[0]
    assert low_trace.activity_targets == (1, 0, 1)
    assert low_trace.activity_mask == (1, 1, 1)
    assert low_trace.semantic_targets[0]["name"] == "LOW"
    control_invocation = next(
        invocation for invocation in low_trace.invocations if invocation.quantum_id == "CONTROL_GROUP"
    )
    emit_invocation = next(
        invocation for invocation in low_trace.invocations if invocation.quantum_id == "EMIT_LOW"
    )
    assert control_invocation.inputs["args"] == [{"$qprogram_ref": "state"}]
    assert emit_invocation.inputs["args"] == [
        {"$qprogram_ref": "producer", "quantum_id": "CONTROL_GROUP"}
    ]
    assert low_trace.resolve_invocation_inputs("CONTROL_GROUP")["args"][0] == low_trace.state
    assert low_trace.resolve_invocation_inputs(emit_invocation)["args"][0]["name"] == "LOW"
    assert compiled.coverage.audit_only_quanta == ("EMIT_HIGH",)
    assert compiled.coverage.train.negative_eligible_counts["EMIT_HIGH"] == 1
    assert compiled.coverage.training_absent_semantic_outcomes == {
        "CONTROL_GROUP": (canonical_key(Group.HIGH),),
        "EMIT_HIGH": (canonical_key("high"),),
    }
    assert compiled.coverage.rare_training_semantic_outcomes == {
        "CONTROL_GROUP": ((canonical_key(Group.LOW), 1),),
        "EMIT_LOW": ((canonical_key("low"), 1),),
    }
    assert compiled.structure.depth == 2
    assert compiled.structure.leaves == ("EMIT_HIGH", "EMIT_LOW")
    assert compiled.reuse["CONTROL_GROUP"].activation_count == 2
    assert compiled.reuse["CONTROL_GROUP"].distinct_semantic_input_contexts == 2
    assert compiled.reuse["CONTROL_GROUP"].downstream_consumers == ("EMIT_HIGH", "EMIT_LOW")
    assert compiled.warnings[0] == "unreachable quantum definitions: UNREACHABLE"
    assert any("EMIT_HIGH is active in only 0" in warning for warning in compiled.warnings)
    assert any("audit-domain semantic outcomes absent from training" in warning for warning in compiled.warnings)
    assert any("semantic outcomes active in at most 2" in warning for warning in compiled.warnings)


def test_evaluation_distribution_cannot_change_compiler_conclusions() -> None:
    program, events = _branching_program()
    low_eval = compile_qprogram(
        program,
        validation_events=events,
        training_events=(events[0],),
        evaluation_events=(events[0],),
    )
    high_eval = compile_qprogram(
        program,
        validation_events=events,
        training_events=(events[0],),
        evaluation_events=(events[1],),
    )

    assert low_eval.structure == high_eval.structure
    assert low_eval.semantic_outcomes == high_eval.semantic_outcomes
    assert low_eval.coverage == high_eval.coverage
    assert low_eval.audits == high_eval.audits
    assert low_eval.warnings == high_eval.warnings


def test_compiler_artifacts_use_program_names_and_reject_older_schemas(tmp_path) -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events)

    artifact_name = _artifact_name(compiled, "compact")
    assert artifact_name.startswith("compact-")
    assert "compiler" not in artifact_name

    older = replace(
        compiled,
        metadata=replace(compiled.metadata, compiler_version="4"),
    )
    older.write(tmp_path)
    with pytest.raises(ValueError, match="does not match '5'"):
        CompiledQProgram.read(tmp_path)


def test_compiler_audits_repeated_primitives_and_injective_summaries() -> None:
    registry = QRegistry("factoring_audit", unit_complexity=32)

    @registry.primitive(cost=2)
    def parity(value: int) -> int:
        return value % 2

    @registry.quantum("SOURCE_VALUE", source_reads=("digits",), output_cardinality=2)
    def source_value(state: PredictiveState) -> int:
        return parity(state.read("digits")[0])

    @registry.quantum("SUMMARY", output_cardinality=2)
    def summary(value: int) -> tuple[int]:
        return (value,)

    @registry.quantum("EMIT", source_reads=("digits",), output_cardinality=2)
    def emit(state: PredictiveState, value: int, packed: tuple[int]) -> str:
        repeated = parity(state.read("digits")[0])
        return str(value + packed[0] + repeated)

    def predict(state: PredictiveState) -> str:
        value = source_value(state)
        packed = summary(value)
        return emit(state, value, packed)

    events = (
        PredictionEvent("even", PredictiveState((2,), ()), "0"),
        PredictionEvent("odd", PredictiveState((3,), ()), "3"),
    )
    compiled = compile_qprogram(
        FunctionalProgram(registry, predict),
        validation_events=events,
        training_events=events,
    )

    assert [item.primitive for item in compiled.audits.repeated_primitive_computations] == ["parity"]
    assert [item.node for item in compiled.audits.non_identifying_summaries] == ["SUMMARY"]


def test_compact_distribution_trace_expands_losslessly_from_shared_schema() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(
        program,
        validation_events=events,
        training_events=events,
        evaluation_events=events,
    )

    for validation, compact in zip(compiled.validation_traces, compiled.training_traces):
        assert isinstance(compact, AlignedEventTrace)
        assert "invocations" not in AlignedEventTrace.__slots__
        assert compact.active_nodes == validation.active_nodes
        assert compact.activity_targets == validation.activity_targets
        assert compact.activity_mask == validation.activity_mask
        assert compact.semantic_targets == validation.semantic_targets
        assert compact.expand_invocations() == validation.invocations
        assert compact.derive_parent_context(compiled.structure) == validation.derive_parent_context(
            compiled.structure
        )
        for invocation in validation.invocations:
            assert compiled.resolve_invocation_inputs(compact, invocation.quantum_id) == (
                validation.resolve_invocation_inputs(invocation)
            )
    assert all(
        trace.catalog is compiled.trace_schema_catalog
        for trace in (*compiled.training_traces, *compiled.evaluation_traces)
    )


def test_coverage_warning_thresholds_must_be_non_negative() -> None:
    program, events = _branching_program()

    with pytest.raises(CompilationError, match="rare_activation_warning"):
        compile_qprogram(
            program,
            validation_events=events,
            options=CompileOptions(rare_activation_warning=-1),
        )
    with pytest.raises(CompilationError, match="rare_semantic_outcome_warning"):
        compile_qprogram(
            program,
            validation_events=events,
            options=CompileOptions(rare_semantic_outcome_warning=-1),
        )
    with pytest.raises(CompilationError, match="semantic_cardinality_warning"):
        compile_qprogram(
            program,
            validation_events=events,
            options=CompileOptions(semantic_cardinality_warning=1),
        )


def test_semantic_cardinality_warnings_are_advisory() -> None:
    registry = QRegistry("cardinality_warnings", unit_complexity=16)

    @registry.quantum("VALUE", source_reads=("digits",), output_cardinality=3)
    def value(state: PredictiveState) -> int:
        return state.read("digits")[0]

    @registry.quantum("CONSTANT", output_cardinality=1)
    def constant() -> str:
        return ":"

    @registry.quantum("EMIT", output_cardinality=3)
    def emit(item: int, suffix: str) -> str:
        return str(item) + suffix

    def predict(state: PredictiveState) -> str:
        return emit(value(state), constant())

    events = tuple(
        PredictionEvent(str(item), PredictiveState((item,)), f"{item}:")
        for item in (1, 2, 3)
    )
    compiled = compile_qprogram(
        FunctionalProgram(registry, predict),
        validation_events=events,
        options=CompileOptions(semantic_cardinality_warning=2),
    )

    assert any("CONSTANT has one reachable semantic outcome" in item for item in compiled.warnings)
    assert any(
        "VALUE has 3 reachable semantic outcomes" in item and "over-broad quantum" in item
        for item in compiled.warnings
    )


def test_when_item_guards_compound_semantics_with_direct_provenance() -> None:
    registry = QRegistry("compound_guard", unit_complexity=16)

    @registry.quantum("CONTEXT", output_cardinality=2)
    def context(value: int) -> tuple[int, str]:
        return (value, "token")

    @registry.quantum("EMIT", output_cardinality=1)
    def emit(value: tuple[int, str]) -> str:
        return value[1]

    def predict(state: PredictiveState) -> str:
        value = context(1)
        with when_item(value, 0, 1) as active:
            if active:
                return emit(value)
        return emit(value)

    event = PredictionEvent("compound", PredictiveState((1,)), "token")
    compiled = compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))
    dependency = compiled.structure.dependencies[0]
    assert (dependency.parent, dependency.child) == ("CONTEXT", "EMIT")
    assert dependency.kinds == ("control", "data")


def test_compiler_is_deterministic_and_writes_canonical_artifacts(tmp_path) -> None:
    program, events = _branching_program()
    left = compile_qprogram(program, validation_events=events, training_events=events)
    right = compile_qprogram(program, validation_events=reversed(events), training_events=events)

    assert left.nodes == right.nodes
    assert left.edges == right.edges
    assert left.metadata.program_fingerprint == right.metadata.program_fingerprint
    assert left.metadata.validation_domain_fingerprint == right.metadata.validation_domain_fingerprint
    assert left.validation_traces == right.validation_traces
    left.write(tmp_path)
    assert CompiledQProgram.read(tmp_path) == left
    payload = json.loads((tmp_path / "compiled_qprogram.json").read_text())
    assert payload["structure"]["nodes"] == list(left.nodes)
    assert payload["trace_storage"]["counts"] == {
        "validation": len(left.validation_traces),
        "training": len(left.training_traces),
        "evaluation": len(left.evaluation_traces),
    }
    assert "validation_traces" not in payload
    assert (tmp_path / "structure.json").exists()
    assert (tmp_path / "coverage.json").exists()
    assert (tmp_path / "complexity.json").exists()
    assert "graph TD" in (tmp_path / "poset.mmd").read_text()
    assert "Q-program: synthetic_branch" in (tmp_path / "report.txt").read_text()


def test_compiler_retains_transitive_data_provenance_outside_cover_graph() -> None:
    registry = QRegistry("transitive", unit_complexity=32)

    @registry.quantum("A", source_reads=("position",))
    def node_a(state: PredictiveState) -> int:
        return int(state.read("position"))

    @registry.quantum("B")
    def node_b(value: int) -> int:
        return value + 1

    @registry.quantum("C")
    def node_c(original: int, incremented: int) -> str:
        return str(original + incremented)

    def predict(state: PredictiveState) -> str:
        original = node_a(state)
        incremented = node_b(original)
        return node_c(original, incremented)

    event = PredictionEvent("chain", PredictiveState((1,), position=1), "3")
    compiled = compile_qprogram(
        FunctionalProgram(registry, predict), validation_events=(event,)
    )

    assert compiled.edges == (("A", "B"), ("B", "C"))
    assert compiled.structure.parents["C"] == ("B",)
    assert compiled.structure.unreduced_edges == (("A", "B"), ("A", "C"), ("B", "C"))
    transitive = next(
        dependency
        for dependency in compiled.structure.unreduced_dependencies
        if (dependency.parent, dependency.child) == ("A", "C")
    )
    assert transitive.kinds == ("data",)
    assert transitive.witness_event == "chain"


def test_compiler_may_reduce_a_control_only_edge_inherited_through_a_parent() -> None:
    registry = QRegistry("control_reduction", unit_complexity=16)

    @registry.quantum("SELECT", source_reads=("position",), output_cardinality=1)
    def select(state: PredictiveState) -> int:
        return int(state.read("position"))

    @registry.quantum("PARENT", output_cardinality=1)
    def parent() -> int:
        return 1

    @registry.quantum("EMIT", output_cardinality=1)
    def emit(value: int) -> str:
        return str(value)

    def predict(state: PredictiveState) -> str:
        selection = select(state)
        with when(selection, 0) as selected:
            if selected:
                return emit(parent())
        assert False, "unreachable"

    event = PredictionEvent("only", PredictiveState((1,), position=0), "1")
    compiled = compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))

    assert compiled.structure.unreduced_edges == (
        ("PARENT", "EMIT"),
        ("SELECT", "EMIT"),
        ("SELECT", "PARENT"),
    )
    assert compiled.edges == (("PARENT", "EMIT"), ("SELECT", "PARENT"))
    removed = next(
        dependency
        for dependency in compiled.structure.unreduced_dependencies
        if (dependency.parent, dependency.child) == ("SELECT", "EMIT")
    )
    assert removed.kinds == ("control",)
    assert removed.witness_event == "only"


def test_compiler_rejects_alternative_direct_parent_signatures() -> None:
    registry = QRegistry("alternative_parents", unit_complexity=32)

    @registry.quantum("SELECT", source_reads=("digits",), output_cardinality=2)
    def select(state: PredictiveState) -> Group:
        return Group.HIGH if state.read("digits")[0] > 5 else Group.LOW

    @registry.quantum("HIGH_VALUE", output_cardinality=1)
    def high_value(group: Group) -> str:
        return "high"

    @registry.quantum("LOW_VALUE", output_cardinality=1)
    def low_value(group: Group) -> str:
        return "low"

    @registry.quantum("EMIT", output_cardinality=2)
    def emit(value: str) -> str:
        return value

    def predict(state: PredictiveState) -> str:
        group = select(state)
        with when(group, Group.HIGH) as high:
            if high:
                return emit(high_value(group))
        with when(group, Group.LOW) as low:
            if low:
                return emit(low_value(group))
        assert False, "unreachable"

    events = (
        PredictionEvent("high", PredictiveState((8,)), "high"),
        PredictionEvent("low", PredictiveState((2,)), "low"),
    )
    with pytest.raises(
        CompilationError,
        match=r"EMIT.*unstable parent signature.*LOW_VALUE.*alternative producer",
    ):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=events)


def test_exhaustive_validation_streams_once_and_matches_compiler_fingerprints() -> None:
    program, unsorted_events = _branching_program()
    events = tuple(sorted(unsorted_events, key=lambda event: str(event.id)))

    class OnePassEvents:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("exhaustive events were iterated more than once")
            yield from events

    stream = OnePassEvents()
    report = validate_exhaustive_domain(program, stream)
    compiled = compile_qprogram(program, validation_events=events)

    assert stream.iterations == 1
    assert report.event_count == 2
    assert report.invocation_count == 4
    assert report.first_event_id == "high"
    assert report.last_event_id == "low"
    assert report.active_nodes == compiled.nodes

    structured = validate_exhaustive_domain(
        program,
        events,
        expected_structure=compiled.structure,
    )
    assert structured.expected_structure_validated is True
    assert report.activity_counts == {"CONTROL_GROUP": 2, "EMIT_HIGH": 1, "EMIT_LOW": 1}
    assert report.semantic_outcome_counts == {
        "CONTROL_GROUP": {
            canonical_key(Group.HIGH): 1,
            canonical_key(Group.LOW): 1,
        },
        "EMIT_HIGH": {canonical_key("high"): 1},
        "EMIT_LOW": {canonical_key("low"): 1},
    }
    assert report.emitted_token_counts == {
        "CONTROL_GROUP": {},
        "EMIT_HIGH": {canonical_key("high"): 1},
        "EMIT_LOW": {canonical_key("low"): 1},
    }
    assert sum(
        count
        for node_counts in report.emitted_token_counts.values()
        for count in node_counts.values()
    ) == report.event_count
    assert report.program_fingerprint == compiled.metadata.program_fingerprint
    assert report.primitive_cost_fingerprint == compiled.metadata.primitive_cost_fingerprint
    assert report.validation_domain_fingerprint == compiled.metadata.validation_domain_fingerprint
    assert report.source_reads["CONTROL_GROUP"] == ("digits",)
    assert report.to_dict()["event_count"] == 2


def test_exhaustive_validation_checks_each_trace_against_expected_structure() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events)
    stricter = replace(
        compiled.structure,
        ancestors={
            **compiled.structure.ancestors,
            "EMIT_HIGH": ("CONTROL_GROUP", "EMIT_LOW"),
        },
    )

    high = next(event for event in events if event.id == "high")
    with pytest.raises(CompilationError, match="not downward closed.*EMIT_LOW"):
        validate_exhaustive_domain(
            program,
            (high,),
            expected_structure=stricter,
        )

    low = next(event for event in events if event.id == "low")
    low_only = compile_qprogram(program, validation_events=(low,))
    with pytest.raises(CompilationError, match="nodes absent from the compiled audit domain.*EMIT_HIGH"):
        validate_exhaustive_domain(
            program,
            (high,),
            expected_structure=low_only.structure,
        )


def test_exhaustive_validation_requires_canonical_unique_event_order() -> None:
    program, events = _branching_program()

    with pytest.raises(CompilationError, match="out-of-order exhaustive validation event identity"):
        validate_exhaustive_domain(program, events)
    with pytest.raises(CompilationError, match="duplicate exhaustive validation event identity"):
        validate_exhaustive_domain(program, (events[0], events[0]))


def test_exhaustive_validation_aggregates_repeated_outcomes_without_retaining_events() -> None:
    program, _ = _branching_program()
    events = tuple(
        PredictionEvent(
            f"{index:03d}",
            PredictiveState((8 if index % 2 else 2,), ()),
            "high" if index % 2 else "low",
        )
        for index in range(100)
    )

    report = validate_exhaustive_domain(program, iter(events), check_determinism=False)

    assert report.semantic_outcome_counts["CONTROL_GROUP"] == {
        canonical_key(Group.HIGH): 50,
        canonical_key(Group.LOW): 50,
    }
    assert report.emitted_token_counts["EMIT_HIGH"] == {canonical_key("high"): 50}
    assert report.emitted_token_counts["EMIT_LOW"] == {canonical_key("low"): 50}
    assert sum(len(counts) for counts in report.semantic_outcome_counts.values()) == 4
    assert sum(len(counts) for counts in report.emitted_token_counts.values()) == 2


def test_exhaustive_validation_detects_nondeterministic_semantic_traces() -> None:
    registry = QRegistry("nondeterministic_stream", unit_complexity=16)

    class Counter:
        def __init__(self) -> None:
            self.value = 0

        def int(self) -> int:
            result = self.value
            self.value += 1
            return result

    counter = Counter()

    @registry.quantum("VALUE", output_cardinality=2)
    def value(source: object = counter) -> int:
        return source.int()

    @registry.quantum("EMIT", output_cardinality=1)
    def emit(observed: int) -> str:
        return "same"

    def predict(state: PredictiveState) -> str:
        return emit(value())

    event = PredictionEvent("only", PredictiveState((1,)), "same")
    with pytest.raises(CompilationError, match="nondeterministic"):
        validate_exhaustive_domain(FunctionalProgram(registry, predict), (event,))


def test_repeated_quantum_call_in_one_event_is_rejected() -> None:
    registry = QRegistry("repeated", unit_complexity=8)

    @registry.quantum("Q", source_reads=("position",))
    def quantum(state: PredictiveState) -> str:
        return str(state.read("position"))

    def predict(state: PredictiveState) -> str:
        quantum(state)
        return quantum(state)

    event = PredictionEvent("repeat", PredictiveState((1,), position=0), "0")
    with pytest.raises(TraceError, match="more than once"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))


def test_duplicate_event_identities_are_rejected() -> None:
    program, events = _branching_program()
    duplicate = PredictionEvent(events[0].id, events[1].state, events[1].target)
    with pytest.raises(CompilationError, match="duplicate validation event identities"):
        compile_qprogram(program, validation_events=(events[0], duplicate))


def test_untraced_output_and_target_mismatch_are_rejected() -> None:
    registry = QRegistry("bad_output", unit_complexity=8)

    @registry.quantum("Q")
    def quantum(value: int) -> str:
        return str(value)

    def untraced(state: PredictiveState) -> str:
        return "0"

    event = PredictionEvent("bad", PredictiveState((1,)), "1")
    with pytest.raises(CompilationError, match="untraced token"):
        compile_qprogram(FunctionalProgram(registry, untraced), validation_events=(event,))

    def wrong(state: PredictiveState) -> str:
        return quantum(0)

    with pytest.raises(CompilationError, match="target mismatch"):
        compile_qprogram(FunctionalProgram(registry, wrong), validation_events=(event,))
    with pytest.raises(CompilationError, match="target mismatch"):
        validate_exhaustive_domain(FunctionalProgram(registry, wrong), (event,))


def test_dead_active_quantum_is_rejected() -> None:
    registry = QRegistry("dead", unit_complexity=8)

    @registry.quantum("DEAD")
    def dead(value: int) -> int:
        return value + 1

    @registry.quantum("EMIT")
    def emit(value: int) -> str:
        return str(value)

    def predict(state: PredictiveState) -> str:
        dead(1)
        return emit(1)

    event = PredictionEvent("dead", PredictiveState((1,)), "1")
    with pytest.raises(CompilationError, match="structurally dead"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))
    with pytest.raises(CompilationError, match="structurally dead"):
        validate_exhaustive_domain(FunctionalProgram(registry, predict), (event,))


def test_counterfactual_effect_audit_rejects_ignored_semantic_inputs() -> None:
    registry = QRegistry("ignored", unit_complexity=16)

    @registry.quantum("VALUE", source_reads=("position",), output_cardinality=2)
    def value(state: PredictiveState) -> int:
        return int(state.read("position"))

    @registry.quantum("EMIT")
    def emit(ignored: int) -> str:
        return "same"

    def predict(state: PredictiveState) -> str:
        return emit(value(state))

    events = (
        PredictionEvent("zero", PredictiveState((1,), position=0), "same"),
        PredictionEvent("one", PredictiveState((1,), position=1), "same"),
    )
    with pytest.raises(CompilationError, match="counterfactual effect audit"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=events)


def test_restricted_python_rejects_mutation_and_unregistered_helpers() -> None:
    registry = QRegistry("restricted", unit_complexity=32)

    @registry.quantum("MUTATE")
    def mutate(values: list[int]) -> int:
        values.append(1)
        return len(values)

    def predict(state: PredictiveState) -> int:
        return mutate([])

    event = PredictionEvent("mutation", PredictiveState((1,)), 1)
    with pytest.raises(DefinitionError, match="unregistered helper 'append'"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))


def test_restricted_python_rejects_source_bypass_and_hidden_globals() -> None:
    source_registry = QRegistry("source_bypass", unit_complexity=16)

    @source_registry.quantum("Q")
    def direct_source(state: PredictiveState) -> str:
        return str(state.digits[0])

    def source_predict(state: PredictiveState) -> str:
        return direct_source(state)

    event = PredictionEvent("source", PredictiveState((1,)), "1")
    with pytest.raises(DefinitionError, match="bypasses provenance"):
        compile_qprogram(FunctionalProgram(source_registry, source_predict), validation_events=(event,))

    hidden_table = ("zero", "one")
    global_registry = QRegistry("global_bypass", unit_complexity=16)

    @global_registry.quantum("Q")
    def global_source(index: int) -> str:
        return hidden_table[index]

    def global_predict(state: PredictiveState) -> str:
        return global_source(1)

    with pytest.raises(DefinitionError, match="hidden global access 'hidden_table'"):
        compile_qprogram(FunctionalProgram(global_registry, global_predict), validation_events=(event,))


def test_entrypoint_cannot_bypass_annotated_source_reads() -> None:
    registry = QRegistry("entrypoint_bypass", unit_complexity=16)

    @registry.quantum("Q")
    def emit(value: int) -> str:
        return str(value)

    def predict(state: PredictiveState) -> str:
        return emit(state.digits[0])

    event = PredictionEvent("entrypoint", PredictiveState((1,)), "1")
    with pytest.raises(DefinitionError, match="reads predictive state directly"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))


def test_registered_global_primitive_table_is_counted_and_permitted() -> None:
    registry = QRegistry("registered_table", unit_complexity=16)

    @registry.primitive(cost=1, table_cardinality=2)
    def lookup(index: int) -> str:
        return _REGISTERED_TABLE[index]

    @registry.quantum("Q")
    def lexicalize(index: int) -> str:
        return lookup(index)

    def predict(state: PredictiveState) -> str:
        return lexicalize(1)

    event = PredictionEvent("table", PredictiveState((1,)), "one")
    compiled = compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))
    assert dict(compiled.complexity[0].components)["table:lookup"] == 2


def test_registered_primitive_helper_costs_are_expanded() -> None:
    registry = QRegistry("helper_expansion", unit_complexity=32)

    @registry.primitive(cost=2, table_cardinality=2)
    def inner(index: int) -> str:
        return {0: "zero", 1: "one"}[index]

    @registry.primitive(cost=1, table_cardinality=0)
    def outer(index: int) -> str:
        return inner(index)

    @registry.quantum("Q")
    def lexicalize(index: int) -> str:
        return outer(index)

    def predict(state: PredictiveState) -> str:
        return lexicalize(1)

    event = PredictionEvent("helper", PredictiveState((1,)), "one")
    compiled = compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))
    components = dict(compiled.complexity[0].components)
    assert components["primitive:outer"] == 3
    assert components["table:outer"] == 2


def test_union_poset_closure_violation_is_rejected() -> None:
    registry = QRegistry("closure", unit_complexity=32)

    @registry.quantum("SELECT", source_reads=("position",), output_cardinality=2)
    def select(state: PredictiveState) -> int:
        return int(state.read("position"))

    @registry.quantum("A")
    def node_a(value: int) -> int:
        return value

    @registry.quantum("B")
    def node_b(value: int) -> int:
        return value + 1

    @registry.quantum("EMIT")
    def emit(value: int, selection: int) -> str:
        return str(value)

    def predict(state: PredictiveState) -> str:
        selection = select(state)
        if unwrap(selection) == 0:
            value = node_b(node_a(1))
        else:
            value = node_b(1)
        return emit(value, selection)

    events = (
        PredictionEvent("with-a", PredictiveState((1,), position=0), "2"),
        PredictionEvent("without-a", PredictiveState((1,), position=1), "2"),
    )
    with pytest.raises(CompilationError, match="not downward closed|unstable parent signature"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=events)


def test_cycle_diagnostic_includes_dependency_kinds_and_witness() -> None:
    with pytest.raises(CompilationError, match=r"A -\[data, event=left\]-> B"):
        build_poset(
            ("A", "B"),
            {("A", "B"): {"data"}, ("B", "A"): {"control"}},
            {("A", "B"): "left", ("B", "A"): "right"},
            {"A": (), "B": ()},
        )


def test_hidden_lookup_complexity_and_oversized_quantum_fail() -> None:
    registry = QRegistry("complexity", unit_complexity=2)

    @registry.primitive(cost=1, table_cardinality=3)
    def lookup(index: int) -> str:
        return {0: "zero", 1: "one", 2: "two"}[index]

    @registry.quantum("LEX", output_cardinality=3)
    def lexicalize(index: int) -> str:
        return lookup(index)

    def predict(state: PredictiveState) -> str:
        return lexicalize(1)

    event = PredictionEvent("lookup", PredictiveState((1,)), "one")
    with pytest.raises(ComplexityError, match="exceeding unit budget"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))


def test_semantic_output_cardinality_is_inferred_and_declared_bounds_are_checked() -> None:
    inferred_registry = QRegistry("inferred_cardinality", unit_complexity=16)

    @inferred_registry.quantum("Q", source_reads=("position",))
    def inferred(state: PredictiveState) -> int:
        return int(state.read("position"))

    def inferred_predict(state: PredictiveState) -> int:
        return inferred(state)

    events = tuple(
        PredictionEvent(str(index), PredictiveState((1,), position=index), index)
        for index in range(4)
    )
    compiled = compile_qprogram(
        FunctionalProgram(inferred_registry, inferred_predict),
        validation_events=events,
    )
    report = compiled.complexity[0]
    assert report.output_cardinality == 4
    assert dict(report.components)["output_cardinality"] == 2

    declared_registry = QRegistry("declared_cardinality", unit_complexity=16)

    @declared_registry.quantum("Q", source_reads=("position",), output_cardinality=1)
    def declared(state: PredictiveState) -> int:
        return int(state.read("position"))

    def declared_predict(state: PredictiveState) -> int:
        return declared(state)

    with pytest.raises(CompilationError, match="declares output_cardinality=1"):
        compile_qprogram(
            FunctionalProgram(declared_registry, declared_predict),
            validation_events=events[:2],
        )


def test_primitive_must_declare_embedded_table_cardinality() -> None:
    registry = QRegistry("hidden_table", unit_complexity=32)

    @registry.primitive(cost=1, table_cardinality=1)
    def hidden(index: int) -> str:
        return {0: "zero", 1: "one"}[index]

    @registry.quantum("Q")
    def quantum(index: int) -> str:
        return hidden(index)

    def predict(state: PredictiveState) -> str:
        return quantum(0)

    event = PredictionEvent("hidden", PredictiveState((1,)), "zero")
    with pytest.raises(DefinitionError, match="embeds 2 table entries"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))


def test_primitive_tuple_record_is_not_misclassified_as_a_lookup_table() -> None:
    registry = QRegistry("tuple_record", unit_complexity=16)

    @registry.primitive(cost=1)
    def record(left: int, right: int) -> tuple[int, int]:
        return (left, right)

    @registry.quantum("Q", output_cardinality=1)
    def quantum(left: int, right: int) -> tuple[int, int]:
        return record(left, right)

    @registry.quantum("EMIT", output_cardinality=1)
    def emit(value: tuple[int, int]) -> str:
        return str(value[0] + value[1])

    def predict(state: PredictiveState) -> str:
        value = quantum(1, 2)
        return emit(value)

    event = PredictionEvent("record", PredictiveState((1,)), "3")
    compiled = compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))
    assert compiled.nodes == ("EMIT", "Q")


def test_primitive_cost_must_cover_analyzed_local_body() -> None:
    registry = QRegistry("underdeclared_primitive", unit_complexity=32)

    @registry.primitive(cost=1)
    def arithmetic(value: int) -> int:
        return value * 10 + 1

    @registry.quantum("Q", output_cardinality=1)
    def quantum(value: int) -> str:
        return str(arithmetic(value))

    def predict(state: PredictiveState) -> str:
        return quantum(1)

    event = PredictionEvent("cost", PredictiveState((1,)), "11")
    with pytest.raises(ComplexityError, match="below its analyzed local-body cost"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))


def test_primitive_declared_loop_bound_is_enforced() -> None:
    registry = QRegistry("primitive_loop_bound", unit_complexity=32)

    @registry.primitive(cost=8, max_iterations=3)
    def bounded(value: int) -> int:
        for index in range(4):
            int(index)
        return value

    @registry.quantum("Q", output_cardinality=1)
    def quantum(value: int) -> str:
        return str(bounded(value))

    def predict(state: PredictiveState) -> str:
        return quantum(1)

    event = PredictionEvent("loop", PredictiveState((1,)), "1")
    with pytest.raises(DefinitionError, match="exceeding declared max_iterations=3"):
        compile_qprogram(FunctionalProgram(registry, predict), validation_events=(event,))

    with pytest.raises(DefinitionError, match="max_iterations must be positive"):
        registry.primitive(cost=1, max_iterations=0)


def test_validation_traces_can_be_omitted_without_changing_compilation() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(
        program,
        validation_events=events,
        options=CompileOptions(retain_validation_traces=False, node_parameter_budget=128),
    )
    assert compiled.validation_traces == ()
    assert compiled.metadata.node_parameter_budget == 128
    assert compiled.nodes == ("CONTROL_GROUP", "EMIT_HIGH", "EMIT_LOW")


def test_compiled_metadata_instantiates_homogeneous_quantanet_and_both_routing_modes() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(
        program,
        validation_events=events,
        training_events=events,
        options=CompileOptions(node_parameter_budget=128),
    )
    model = QComputer(
        compiled=compiled,
        vocab_size=5,
        max_seq_len=4,
        d_source=4,
        d_quantum=4,
        pad_id=0,
        sep_id=1,
        read_heads=1,
        activation="layernorm_gelu",
        add_initial_state=False,
    )
    targets, masks = activity_tensors(compiled, split="train")
    targets = targets.unsqueeze(1)
    masks = masks.unsqueeze(1)
    input_ids = torch.tensor([[2, 1], [3, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, 2], [-100, 3]])

    predicted = model(input_ids=input_ids, attention_mask=attention_mask, routing="predicted")
    target_routed = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        routing="oracle",
        activity_targets=targets,
    )
    assert model.core.nodes == ("CONTROL_GROUP", "EMIT_HIGH", "EMIT_LOW")
    assert target_routed.effective_activity.equal(targets)
    assert predicted.effective_activity.dtype == predicted.local_gate_probabilities.dtype
    expected_effective = {}
    for node in model.core.topological_order:
        index = model.core.node_to_index[node]
        activity = predicted.local_gate_probabilities[..., index]
        for parent in model.core.parents[node]:
            activity = torch.minimum(activity, expected_effective[parent])
        expected_effective[node] = activity
        assert torch.equal(predicted.effective_activity[..., index], activity)
    assert masks.shape == targets.shape
    assert torch.equal(target_routed.read_attention[..., 1], torch.zeros_like(target_routed.read_attention[..., 1]))
    assert torch.equal(
        target_routed.ancestor_aggregates[..., 0, :],
        torch.zeros_like(target_routed.ancestor_aggregates[..., 0, :]),
    )
    assert target_routed.final_state.equal(target_routed.messages.sum(dim=-2))
    assert target_routed.depth_updates.sum(dim=-2).equal(target_routed.final_state)
    assert all(
        isinstance(module.hidden_normalization, torch.nn.LayerNorm)
        for module in model.core.node_modules.values()
    )

    losses = qcore_training_loss(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        activity_targets=targets,
        eligibility_mask=masks,
        activity_weight=1.0,
        semantic_weight=1.0,
    )
    assert torch.isfinite(losses["ce_loss"])
    assert torch.isfinite(losses["activity_loss"])

    model.zero_grad(set_to_none=True)
    losses["ce_loss"].backward(retain_graph=True)
    assert all(module.gate.weight.grad is None for module in model.core.node_modules.values())
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()
    assert all(module.gate.weight.grad is not None for module in model.core.node_modules.values())

    audit = model.parameter_audit()
    assert len({item.total for item in audit.per_node}) == 1
    assert audit.per_node_parameter_count > 0
    assert audit.quantum_parameters == audit.per_node_parameter_count * len(compiled.nodes)
    assert audit.total_parameters == sum(parameter.numel() for parameter in model.parameters())


def test_add_initial_state_controls_only_the_final_q_readout_state() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events, training_events=events)
    common = {
        "compiled": compiled,
        "vocab_size": 5,
        "max_seq_len": 4,
        "d_source": 4,
        "d_quantum": 4,
        "pad_id": 0,
        "sep_id": 1,
    }
    without_source = QComputer(**common, add_initial_state=False)
    with_source = QComputer(**common, add_initial_state=True)
    with_source.load_state_dict(without_source.state_dict())
    targets, _ = activity_tensors(compiled, split="train")
    targets = targets.unsqueeze(1)
    inputs = {
        "input_ids": torch.tensor([[2, 1], [3, 1]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "routing": "oracle",
        "activity_targets": targets,
    }

    excluded = without_source(**inputs)
    included = with_source(**inputs)

    assert torch.equal(excluded.messages, included.messages)
    assert torch.equal(excluded.final_state, excluded.messages.sum(dim=-2))
    assert torch.allclose(included.final_state, excluded.final_state + included.source_query_state)


def test_all_quanta_output_controls_only_which_messages_reach_readout() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events, training_events=events)
    common = {
        "compiled": compiled,
        "vocab_size": 5,
        "max_seq_len": 4,
        "d_source": 4,
        "d_quantum": 4,
        "pad_id": 0,
        "sep_id": 1,
        "add_initial_state": False,
    }
    all_quanta = QComputer(**common, all_quanta_output=True)
    emitters_only = QComputer(**common, all_quanta_output=False)
    emitters_only.load_state_dict(all_quanta.state_dict())
    targets, _ = activity_tensors(compiled, split="train")
    inputs = {
        "input_ids": torch.tensor([[2, 1], [3, 1]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "routing": "oracle",
        "activity_targets": targets.unsqueeze(1),
    }

    full = all_quanta(**inputs)
    terminal = emitters_only(**inputs)
    leaf_indices = [emitters_only.core.node_to_index[node] for node in compiled.structure.leaves]

    assert torch.equal(full.messages, terminal.messages)
    assert torch.equal(full.final_state, full.messages.sum(dim=-2))
    assert torch.equal(terminal.final_state, terminal.messages[..., leaf_indices, :].sum(dim=-2))


def test_target_activity_execution_rejects_non_closed_routes() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events)
    model = QComputer(
        compiled=compiled,
        vocab_size=5,
        max_seq_len=4,
        d_source=4,
        d_quantum=4,
        pad_id=0,
        sep_id=1,
    )
    invalid = torch.tensor([[[0.0, 1.0, 0.0]]])
    with pytest.raises(ValueError, match="not downward closed"):
        model(
            input_ids=torch.tensor([[2, 1]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            routing="oracle",
            activity_targets=invalid,
        )


def test_activity_loss_macro_averages_nodes_instead_of_sibling_groups() -> None:
    logits = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
    targets = torch.tensor([[[1.0, 1.0]], [[0.0, 1.0]]])
    eligible = torch.ones_like(targets)
    result = balanced_activity_loss(logits, targets, eligible, nodes=("A", "B"), warn=False)
    assert result.observed_nodes == 2
    assert result.single_class_nodes == ("B",)
    assert torch.allclose(result.loss, torch.tensor(math.log(2.0)))


def test_semantic_classifiers_supervise_only_small_nonconstant_compiler_domains() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events, training_events=events)
    classifiers = SemanticClassifierSuite(
        compiled=compiled,
        d_quantum=4,
        max_supervised_classes=2,
    )
    activity, _ = activity_tensors(compiled, split="train")
    activity = activity.unsqueeze(1)
    semantics = tuple((trace.semantic_targets,) for trace in compiled.training_traces)
    targets = classifiers.encode_targets(semantics, activity, device="cpu")
    deltas = torch.randn(2, 1, len(compiled.nodes), 4, requires_grad=True)

    result = semantic_classification_loss(deltas, targets, classifiers)
    result.loss.backward()

    assert classifiers.supervised_nodes == ("CONTROL_GROUP",)
    assert classifiers.constant_nodes == ("EMIT_HIGH", "EMIT_LOW")
    assert classifiers.excluded_by_cardinality == {}
    assert result.observed_nodes == 1
    assert result.supervised_examples == 2
    assert deltas.grad is not None
    assert deltas.grad[..., classifiers.node_to_index["CONTROL_GROUP"], :].abs().sum() > 0

    excluded = SemanticClassifierSuite(
        compiled=compiled,
        d_quantum=4,
        max_supervised_classes=1,
    )
    assert excluded.supervised_nodes == ()
    assert excluded.excluded_by_cardinality == {"CONTROL_GROUP": 2}


def test_semantic_loss_uses_the_ordinary_active_example_mean() -> None:
    class Classifiers:
        nodes = ("Q",)
        supervised_nodes = ("Q",)
        node_to_index = {"Q": 0}

        @staticmethod
        def logits(node: str, deltas: torch.Tensor) -> torch.Tensor:
            assert node == "Q"
            return deltas[..., 0, :2]

    deltas = torch.tensor(
        [
            [[[3.0, 0.0]]],
            [[[0.0, 3.0]]],
            [[[0.0, 3.0]]],
            [[[3.0, 0.0]]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[[0]], [[0]], [[0]], [[1]]])

    result = semantic_classification_loss(deltas, targets, Classifiers())
    logits = deltas[:, 0, 0, :2]
    labels = targets[:, 0, 0]
    example_losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    assert torch.allclose(result.loss, example_losses.mean())


def test_semantic_loss_trains_raw_quantum_delta_but_not_gate_head() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events, training_events=events)
    model = QComputer(
        compiled=compiled,
        vocab_size=5,
        max_seq_len=4,
        d_source=4,
        d_quantum=4,
        pad_id=0,
        sep_id=1,
    )
    classifiers = SemanticClassifierSuite(
        compiled=compiled,
        d_quantum=4,
        max_supervised_classes=2,
    )
    activity, eligibility = activity_tensors(compiled, split="train")
    activity = activity.unsqueeze(1)
    eligibility = eligibility.unsqueeze(1)
    semantics = tuple((trace.semantic_targets,) for trace in compiled.training_traces)
    semantic_targets = classifiers.encode_targets(semantics, activity, device="cpu")
    losses = qcore_training_loss(
        model,
        input_ids=torch.tensor([[2, 1], [3, 1]]),
        attention_mask=torch.ones(2, 2, dtype=torch.long),
        labels=torch.tensor([[-100, 2], [-100, 3]]),
        activity_targets=activity,
        eligibility_mask=eligibility,
        activity_weight=1.0,
        semantic_weight=0.25,
        semantic_classifiers=classifiers,
        semantic_targets=semantic_targets,
    )
    expected_total = losses["ce_loss"] + losses["activity_loss"] + 0.25 * losses["semantic_loss"]
    assert torch.allclose(losses["loss"], expected_total)

    losses["semantic_loss"].backward()

    control = model.core.node_modules["CONTROL_GROUP"]
    assert control.mlp_output.weight.grad is not None
    assert control.mlp_output.weight.grad.abs().sum() > 0
    assert control.gate.weight.grad is None
    assert classifiers.classifiers["node_0"].weight.grad is not None


def test_parent_zeroing_is_an_evaluation_only_message_intervention() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events, training_events=events)
    model = QComputer(
        compiled=compiled,
        vocab_size=5,
        max_seq_len=4,
        d_source=4,
        d_quantum=4,
        pad_id=0,
        sep_id=1,
    )
    targets, _ = activity_tensors(compiled, split="train")
    targets = targets.unsqueeze(1)
    inputs = {"input_ids": torch.tensor([[2, 1], [3, 1]]), "attention_mask": torch.ones(2, 2, dtype=torch.long)}
    normal = model(**inputs, routing="oracle", activity_targets=targets)
    zeroed = model(
        **inputs,
        routing="oracle",
        activity_targets=targets,
        parent_intervention=ParentIntervention(kind="zero"),
    )
    control_index = model.core.node_to_index["CONTROL_GROUP"]
    assert normal.ancestor_aggregates[..., control_index, :].abs().sum() == 0
    assert zeroed.ancestor_aggregates.abs().sum() == 0


def test_layerwise_alignment_uses_complete_blocks_frozen_teacher_and_identity_interface() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events, training_events=events)
    q_model = QComputer(
        compiled=compiled,
        vocab_size=5,
        max_seq_len=4,
        d_source=4,
        d_quantum=4,
        pad_id=0,
        sep_id=1,
    )
    transformer = DecoderTransformerLM(
        vocab_size=5,
        max_seq_len=4,
        d_model=4,
        n_layers=2,
        n_heads=1,
        dropout=0.0,
        pad_id=0,
        mlp_ratio=1.0,
    )
    model = LayerwiseQAlignment(
        transformer=transformer,
        teacher=FrozenQTeacher(q_model),
        interface_seed=7,
    )
    assert model.interface.is_identity
    assert torch.equal(model.interface.matrix, torch.eye(4))
    targets, _ = activity_tensors(compiled, split="train")
    targets = targets.unsqueeze(1)
    input_ids = torch.tensor([[2, 1], [3, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, 2], [-100, 3]])
    trace = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        activity_targets=targets,
        valid_prediction_mask=torch.tensor([[True, False], [True, False]]),
    )
    losses = alignment_objective(trace, labels=labels, lambda_align=1.0)
    baseline_losses = alignment_objective(trace, labels=labels, lambda_align=0.0)
    assert torch.equal(baseline_losses["loss"], baseline_losses["task_loss"])
    losses["loss"].backward()
    assert torch.isfinite(trace.alignment_loss)
    assert trace.alignment_loss_by_depth.shape == (2,)
    assert trace.cumulative_state_mse_by_depth.shape == (2,)
    assert torch.allclose(trace.alignment_loss, trace.alignment_loss_by_depth.mean())
    assert torch.allclose(
        trace.cumulative_state_mse,
        trace.cumulative_state_mse_by_depth.mean(),
    )
    assert len(trace.transformer.block_boundaries) == 3
    expected_updates = torch.stack(
        [
            trace.transformer.block_boundaries[1] - trace.transformer.block_boundaries[0],
            trace.transformer.block_boundaries[2] - trace.transformer.block_boundaries[1],
        ],
        dim=-2,
    )
    assert torch.allclose(trace.projected_block_updates, expected_updates)
    assert trace.teacher_depth_updates.requires_grad is False
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert any(parameter.grad is not None for parameter in model.transformer.parameters())
    assert model.deployment_model() is transformer


def test_layerwise_alignment_requires_one_block_per_q_depth() -> None:
    program, events = _branching_program()
    compiled = compile_qprogram(program, validation_events=events)
    q_model = QComputer(
        compiled=compiled,
        vocab_size=5,
        max_seq_len=4,
        d_source=4,
        d_quantum=4,
        pad_id=0,
        sep_id=1,
    )
    transformer = DecoderTransformerLM(
        vocab_size=5,
        max_seq_len=4,
        d_model=4,
        n_layers=3,
        n_heads=1,
        dropout=0.0,
        pad_id=0,
    )
    with pytest.raises(ValueError, match="one complete transformer block per Q depth"):
        LayerwiseQAlignment(
            transformer=transformer,
            teacher=FrozenQTeacher(q_model),
            interface_seed=0,
        )


def test_fixed_q_interface_is_semi_orthogonal_and_rms_scales_use_message_norms() -> None:
    interface = FixedQInterface(3, 5, seed=9)
    assert torch.allclose(interface.matrix.T @ interface.matrix, torch.eye(3), atol=1.0e-6)
    sums = torch.tensor([8.0, 18.0])
    counts = torch.tensor([2.0, 2.0])
    assert torch.allclose(rms_depth_scales(sums, counts), torch.tensor([2.0, 3.0]))


def test_layerwise_alignment_loss_is_mean_depth_of_scaled_squared_l2_norms() -> None:
    predicted = torch.zeros(1, 1, 2, 3)
    target = torch.ones_like(predicted)
    loss = layerwise_alignment_loss(
        predicted,
        target,
        valid_mask=torch.tensor([[True]]),
        depth_scales=torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(loss, torch.tensor((3.0 + 3.0 / 4.0) / 2.0))



def test_sequence_domain_expands_prediction_events_and_validates_greedy() -> None:
    registry = QRegistry("sequence", unit_complexity=16)

    @registry.quantum("NEXT", source_reads=("position",), output_cardinality=3)
    def next_token(state: PredictiveState) -> str:
        position = int(state.read("position"))
        return ("one", "two", "EOS")[position]

    def predict(state: PredictiveState) -> str:
        return next_token(state)

    program = FunctionalProgram(registry, predict)
    examples = (SequenceExample("example", (1, 2), ("one", "two")),)
    events = prediction_events(examples, eos_token="EOS")
    assert tuple(event.id for event in events) == ("example:0", "example:1", "example:2")
    assert tuple(event.state.prefix for event in events) == ((), ("one",), ("one", "two"))
    compile_qprogram(program, validation_events=events)
    validate_greedy(program, examples, eos_token="EOS", max_steps=4)
