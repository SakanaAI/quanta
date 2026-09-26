from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

from quanta.qprogram import CompiledQProgram
from quanta.qprogram import compile_qprogram as compile_functional_program
from quanta.qprogram.types import canonical_value
from quanta.experiments.number_naming.compile_qprogram import (
    compile_number_naming_program,
    number_naming_events,
    sampled_audit_examples,
    validate_number_naming_domain_bundle,
)
from quanta.experiments.number_naming.data import NumberNamingExample
from quanta.experiments.number_naming.names import english_number_name
from quanta.experiments.number_naming import qprogram
from quanta.experiments.number_naming.qprogram import LexicalForm


def _examples(numbers: tuple[int, ...]) -> tuple[NumberNamingExample, ...]:
    return tuple(NumberNamingExample(number, english_number_name(number)) for number in numbers)


def test_audit_sampling_is_seeded_and_independent_of_evaluation() -> None:
    first = sampled_audit_examples(max_number=500, sample_size=20, seed=7)
    second = sampled_audit_examples(max_number=500, sample_size=20, seed=7)
    changed = sampled_audit_examples(max_number=500, sample_size=20, seed=8)

    assert first == second
    assert first != changed
    assert len(first) == 20


def test_functional_qprogram_module_does_not_import_the_oracle() -> None:
    source = inspect.getsource(qprogram)
    assert "english_number_name" not in source
    assert qprogram.Group.__module__ == "quanta.experiments.number_naming.qprogram"


def test_compiled_semantic_identities_are_canonical_and_artifact_loads_fresh(
    tmp_path: Path,
) -> None:
    examples = _examples((1, 10, 19, 20, 21, 100, 101, 110, 111, 121, 1000, 1001, 1111))
    compiled = compile_number_naming_program(
        training_examples=examples,
        evaluation_examples=examples,
        output_dir=tmp_path,
    )
    semantic_payload = json.dumps(canonical_value(compiled.semantic_outcomes), sort_keys=True)
    assert "__main__." not in semantic_payload
    assert "quanta.experiments.number_naming.qprogram.Group" in semantic_payload

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from quanta.qprogram import CompiledQProgram; "
                f"value = CompiledQProgram.read({str(tmp_path)!r}); "
                "assert value.id == 'number_naming'; "
                "assert all('__main__.' not in repr(v) for v in value.semantic_outcomes.values())"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert CompiledQProgram.read(tmp_path).metadata == compiled.metadata


def test_full_domain_greedy_certificate_is_exact_and_shard_independent() -> None:
    single = validate_number_naming_domain_bundle(max_number=125, workers=1)
    sharded = validate_number_naming_domain_bundle(max_number=125, workers=3)

    assert single.greedy == sharded.greedy
    assert single.greedy["method"] == "induction_over_exhaustively_validated_prefixes"
    assert single.greedy["sequence_count"] == 125
    assert single.greedy["exact_sequence_count"] == 125
    assert single.greedy["terminal_eos_count"] == 125
    assert single.greedy["exact"] is True
    assert single.greedy["deterministic"] is True
    assert single.greedy["terminates"] is True
    assert single.greedy["prediction_event_count"] == single.exhaustive["event_count"]
    assert (
        single.exhaustive["validation_domain_fingerprint"]
        == sharded.exhaustive["validation_domain_fingerprint"]
    )
    assert sum(
        count
        for counts in single.exhaustive["emitted_token_counts"].values()
        for count in counts.values()
    ) == single.exhaustive["event_count"]


def test_exact_lexical_program_separates_unit_ten_teen_and_tens_roles() -> None:
    examples = _examples((7, 10, 13, 21, 700))
    events = tuple(number_naming_events(examples))
    program = qprogram.build_number_naming_program()
    for event in events:
        assert program.predict(event.state) == event.target

    compiled = compile_functional_program(program, validation_events=events)
    traces = {trace.event_id: trace for trace in compiled.validation_traces}

    expected = {
        "000007:00": (LexicalForm.UNIT, "LEX_UNIT"),
        "000010:00": (LexicalForm.TEN, "EMIT_TEN"),
        "000013:00": (LexicalForm.TEEN, "LEX_TEEN"),
        "000021:00": (LexicalForm.TENS, "LEX_TENS"),
        "000021:01": (LexicalForm.UNIT, "LEX_UNIT"),
        "000700:00": (LexicalForm.UNIT, "LEX_UNIT"),
    }
    for event_id, (form, lexical_node) in expected.items():
        trace = traces[event_id]
        outputs = {item.quantum_id: item.output for item in trace.invocations}
        assert outputs["LEXICAL_FORM"]["name"] == form.name
        assert lexical_node in trace.active_nodes

    assert "LEXICAL_VALUE" not in traces["000010:00"].active_nodes
    assert "LEX_SMALL" not in compiled.nodes


def test_targeted_boundary_refactor_has_stable_direct_poset_and_unit_costs() -> None:
    examples = _examples(
        (1, 10, 11, 20, 21, 100, 101, 110, 111, 121, 1000, 1001, 999999)
    )
    compiled = compile_functional_program(
        qprogram.build_number_naming_program(),
        validation_events=number_naming_events(examples),
    )

    assert compiled.nodes == (
        "BOUNDARY_ACTION",
        "CHUNK_LENGTH",
        "CONTENT_SLOT",
        "CONTROL_GROUP",
        "EMIT_EOS",
        "EMIT_HUNDRED",
        "EMIT_TEN",
        "EMIT_THOUSAND",
        "GROUP_PROGRESS",
        "HIGH_CONTENT_LENGTH",
        "HUNDREDS_COMPONENT",
        "LEXICAL_FORM",
        "LEXICAL_VALUE",
        "LEX_TEEN",
        "LEX_TENS",
        "LEX_UNIT",
        "SELECT_CHUNK",
        "TAIL_KIND",
        "TAIL_TENS_DIGIT",
        "TAIL_UNIT_DIGIT",
    )
    assert compiled.edges == (
        ("BOUNDARY_ACTION", "CONTENT_SLOT"),
        ("BOUNDARY_ACTION", "EMIT_EOS"),
        ("BOUNDARY_ACTION", "EMIT_THOUSAND"),
        ("CHUNK_LENGTH", "BOUNDARY_ACTION"),
        ("CONTENT_SLOT", "EMIT_HUNDRED"),
        ("CONTENT_SLOT", "LEXICAL_FORM"),
        ("CONTENT_SLOT", "LEXICAL_VALUE"),
        ("CONTROL_GROUP", "GROUP_PROGRESS"),
        ("CONTROL_GROUP", "SELECT_CHUNK"),
        ("GROUP_PROGRESS", "BOUNDARY_ACTION"),
        ("HIGH_CONTENT_LENGTH", "CONTROL_GROUP"),
        ("HUNDREDS_COMPONENT", "CHUNK_LENGTH"),
        ("LEXICAL_FORM", "EMIT_TEN"),
        ("LEXICAL_FORM", "LEX_TEEN"),
        ("LEXICAL_FORM", "LEX_TENS"),
        ("LEXICAL_FORM", "LEX_UNIT"),
        ("LEXICAL_VALUE", "LEX_TEEN"),
        ("LEXICAL_VALUE", "LEX_TENS"),
        ("LEXICAL_VALUE", "LEX_UNIT"),
        ("SELECT_CHUNK", "HUNDREDS_COMPONENT"),
        ("SELECT_CHUNK", "TAIL_TENS_DIGIT"),
        ("SELECT_CHUNK", "TAIL_UNIT_DIGIT"),
        ("TAIL_KIND", "CHUNK_LENGTH"),
        ("TAIL_TENS_DIGIT", "TAIL_KIND"),
        ("TAIL_UNIT_DIGIT", "TAIL_KIND"),
    )
    assert compiled.structure.depth == 10
    assert all(report.total <= report.budget == 80 for report in compiled.complexity)
    assert max(compiled.complexity, key=lambda report: report.total).quantum_id == "SELECT_CHUNK"
    assert compiled.audits.non_identifying_summaries == ()
