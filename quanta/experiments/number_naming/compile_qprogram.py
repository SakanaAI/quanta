from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import random
from typing import Iterable, Iterator

from quanta.qprogram import (
    CompileOptions,
    CompiledQProgram,
    ExhaustiveValidationReport,
    FunctionalProgram,
    PredictionEvent,
    PredictiveState,
    SequenceExample,
    canonical_key,
    compile_qprogram as compile_functional_program,
    validate_exhaustive_domain,
)
from quanta.qprogram.types import PosetStructure, canonical_value, fingerprint

from .data import NumberNamingExample
from .names import english_number_name
from .qprogram import build_number_naming_program
from .qprogram_compact import build_number_naming_program as build_compact_number_naming_program
from .tokenizer import EOS


PROGRAM_VARIANTS = ("factorized", "compact")


def build_program(variant: str = "factorized") -> FunctionalProgram:
    if variant == "factorized":
        return build_number_naming_program()
    if variant == "compact":
        return build_compact_number_naming_program()
    raise ValueError(
        f"unknown NumberNaming Q-program variant {variant!r}; expected one of {PROGRAM_VARIANTS}."
    )


@dataclass(frozen=True)
class NumberNamingDomainValidation:
    exhaustive: dict[str, object]
    greedy: dict[str, object]


@dataclass(frozen=True)
class _ShardValidation:
    report: ExhaustiveValidationReport
    minimum: int
    maximum: int
    number_count: int
    event_count: int
    eos_event_count: int
    max_rollout_steps: int


def sequence_example(example: NumberNamingExample) -> SequenceExample:
    return SequenceExample(
        id=str(int(example.number)),
        digits=tuple(int(digit) for digit in str(int(example.number))),
        tokens=tuple(example.text.split()),
    )


def number_naming_events(examples: Iterable[NumberNamingExample]) -> Iterator[PredictionEvent]:
    for example in examples:
        sequence = (*example.text.split(), EOS)
        digits = tuple(int(digit) for digit in str(int(example.number)))
        for position, target in enumerate(sequence):
            yield PredictionEvent(
                id=f"{int(example.number):06d}:{position:02d}",
                state=PredictiveState(digits=digits, prefix=sequence[:position], position=position),
                target=target,
            )


def exhaustive_number_naming_events(start: int, stop: int) -> Iterator[PredictionEvent]:
    _validate_number_range(start, stop)
    examples = (
        NumberNamingExample(number, english_number_name(number))
        for number in range(int(start), int(stop))
    )
    yield from number_naming_events(examples)


def structural_validation_examples() -> Iterator[NumberNamingExample]:
    numbers = set(range(1, 1000))
    numbers.update(high * 1000 for high in range(1, 1000))
    numbers.update(high * 1000 + 1 for high in range(1, 1000))
    numbers.update(1000 + low for low in range(1, 1000))
    for number in sorted(numbers):
        yield NumberNamingExample(number, english_number_name(number))


def sampled_audit_examples(
    *, max_number: int, sample_size: int, seed: int
) -> tuple[NumberNamingExample, ...]:
    if int(max_number) < 1 or int(max_number) > 999_999:
        raise ValueError("max_number must be in 1..999999.")
    if int(sample_size) < 0:
        raise ValueError("audit sample size must be non-negative.")
    count = min(int(sample_size), int(max_number))
    numbers = sorted(random.Random(int(seed)).sample(range(1, int(max_number) + 1), count))
    return tuple(
        NumberNamingExample(number, english_number_name(number))
        for number in numbers
    )


def compile_number_naming_program(
    *,
    training_examples: Iterable[NumberNamingExample],
    evaluation_examples: Iterable[NumberNamingExample],
    audit_examples: Iterable[NumberNamingExample] = (),
    output_dir: str | Path | None = None,
    node_parameter_budget: int | None = None,
    program_variant: str = "factorized",
) -> CompiledQProgram:
    training_examples = tuple(training_examples)
    evaluation_examples = tuple(evaluation_examples)
    audit_examples = tuple(audit_examples)
    validation_examples = {
        int(example.number): example
        for example in (*training_examples, *audit_examples)
    }
    if not validation_examples:
        raise ValueError("compilation requires training examples or an explicit audit sample.")
    compiled = compile_functional_program(
        build_program(program_variant),
        validation_events=number_naming_events(
            validation_examples[number] for number in sorted(validation_examples)
        ),
        training_events=number_naming_events(training_examples),
        evaluation_events=number_naming_events(evaluation_examples),
        options=CompileOptions(
            check_determinism=True,
            retain_validation_traces=False,
            node_parameter_budget=node_parameter_budget,
        ),
    )
    _validate_canonical_semantic_identities(compiled)
    if output_dir is not None:
        compiled.write(output_dir)
    return compiled


def validate_number_naming_domain_bundle(
    *,
    max_number: int = 999_999,
    workers: int = 1,
    check_determinism: bool = True,
    expected_structure: PosetStructure | None = None,
    program_variant: str = "factorized",
) -> NumberNamingDomainValidation:
    """Validate every target prefix and certify full-domain greedy exactness.

    The greedy certificate reuses exhaustive next-token validation. Once the
    empty-prefix prediction is exact, each generated prefix equals the next
    teacher-forced prefix; induction therefore proves the entire rollout,
    including EOS, without a redundant second execution over the domain.
    """
    if int(max_number) < 1 or int(max_number) > 999_999:
        raise ValueError("max_number must be in 1..999999.")
    worker_count = max(1, min(int(workers), int(max_number)))
    ranges = _number_ranges(int(max_number), worker_count)
    arguments = [
        (start, stop, bool(check_determinism), expected_structure, program_variant)
        for start, stop in ranges
    ]
    if worker_count == 1:
        shards = [_validate_number_range_worker(arguments[0])]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            shards = list(executor.map(_validate_number_range_worker, arguments))
    exhaustive = _merge_exhaustive_reports(
        shards,
        max_number=int(max_number),
        workers=worker_count,
        check_determinism=bool(check_determinism),
    )
    greedy = _greedy_certificate(
        shards,
        exhaustive=exhaustive,
        max_number=int(max_number),
        check_determinism=bool(check_determinism),
    )
    return NumberNamingDomainValidation(exhaustive=exhaustive, greedy=greedy)


def validate_number_naming_domain(
    *,
    max_number: int = 999_999,
    workers: int = 1,
    check_determinism: bool = True,
    expected_structure: PosetStructure | None = None,
    program_variant: str = "factorized",
) -> dict[str, object]:
    """Return the concise full-domain validation evidence."""
    return validate_number_naming_domain_bundle(
        max_number=max_number,
        workers=workers,
        check_determinism=check_determinism,
        expected_structure=expected_structure,
        program_variant=program_variant,
    ).exhaustive


def _number_ranges(max_number: int, workers: int) -> list[tuple[int, int]]:
    count = int(max_number)
    base, remainder = divmod(count, int(workers))
    ranges = []
    start = 1
    for index in range(int(workers)):
        size = base + (1 if index < remainder else 0)
        stop = start + size
        ranges.append((start, stop))
        start = stop
    return ranges


def _validate_number_range(start: int, stop: int) -> None:
    if int(start) < 1 or int(stop) <= int(start) or int(stop) > 1_000_000:
        raise ValueError("exhaustive NumberNaming ranges must satisfy 1 <= start < stop <= 1000000.")


def _validate_number_range_worker(
    arguments: tuple[int, int, bool, PosetStructure | None, str],
) -> _ShardValidation:
    start, stop, check_determinism, expected_structure, program_variant = arguments
    _validate_number_range(start, stop)
    stats = {
        "number_count": 0,
        "event_count": 0,
        "eos_event_count": 0,
        "max_rollout_steps": 0,
    }

    def events() -> Iterator[PredictionEvent]:
        for number in range(start, stop):
            example = NumberNamingExample(number, english_number_name(number))
            sequence = (*example.text.split(), EOS)
            stats["number_count"] += 1
            stats["event_count"] += len(sequence)
            stats["eos_event_count"] += int(sequence[-1] == EOS)
            stats["max_rollout_steps"] = max(stats["max_rollout_steps"], len(sequence))
            digits = tuple(int(digit) for digit in str(number))
            for position, target in enumerate(sequence):
                yield PredictionEvent(
                    id=f"{number:06d}:{position:02d}",
                    state=PredictiveState(
                        digits=digits,
                        prefix=sequence[:position],
                        position=position,
                    ),
                    target=target,
                )

    report = validate_exhaustive_domain(
        build_program(program_variant),
        events(),
        check_determinism=check_determinism,
        expected_structure=expected_structure,
    )
    if report.event_count != stats["event_count"]:
        raise RuntimeError("exhaustive validator did not consume the complete NumberNaming shard.")
    return _ShardValidation(
        report=report,
        minimum=start,
        maximum=stop - 1,
        number_count=stats["number_count"],
        event_count=stats["event_count"],
        eos_event_count=stats["eos_event_count"],
        max_rollout_steps=stats["max_rollout_steps"],
    )


def _merge_exhaustive_reports(
    shards: list[_ShardValidation],
    *,
    max_number: int,
    workers: int,
    check_determinism: bool,
) -> dict[str, object]:
    reports = [shard.report for shard in shards]
    program_fingerprints = {report.program_fingerprint for report in reports}
    primitive_fingerprints = {report.primitive_cost_fingerprint for report in reports}
    if len(program_fingerprints) != 1 or len(primitive_fingerprints) != 1:
        raise RuntimeError("exhaustive validation shards used inconsistent functional programs.")
    structure_certificates = {report.expected_structure_validated for report in reports}
    if len(structure_certificates) != 1:
        raise RuntimeError("exhaustive validation shards used inconsistent expected structures.")
    activity: Counter[str] = Counter()
    semantic_outcomes: dict[str, Counter[str]] = {}
    emitted_tokens: dict[str, Counter[str]] = {}
    nodes = set()
    dependency_rows = {}
    source_reads: dict[str, set[str]] = {}
    for report in reports:
        activity.update(report.activity_counts)
        nodes.update(report.active_nodes)
        for node, counts in report.semantic_outcome_counts.items():
            semantic_outcomes.setdefault(node, Counter()).update(counts)
        for node, counts in report.emitted_token_counts.items():
            emitted_tokens.setdefault(node, Counter()).update(counts)
        for dependency in report.dependencies:
            key = (dependency.parent, dependency.child)
            dependency_rows.setdefault(key, canonical_value(dependency))
        for node, fields in report.source_reads.items():
            source_reads.setdefault(node, set()).update(fields)
    oracle_module = inspect.getmodule(english_number_name)
    oracle_source = inspect.getsource(oracle_module) if oracle_module is not None else ""
    return {
        "domain": {"minimum": 1, "maximum": max_number},
        "check_determinism": check_determinism,
        "expected_structure_validated": next(iter(structure_certificates)),
        "workers": workers,
        "program_fingerprint": next(iter(program_fingerprints)),
        "primitive_cost_fingerprint": next(iter(primitive_fingerprints)),
        "validation_domain_fingerprint": fingerprint(
            {
                "task": "number_naming",
                "range": [1, max_number],
                "target_oracle_source": oracle_source,
                "eos_token": EOS,
            }
        ),
        "event_count": sum(report.event_count for report in reports),
        "invocation_count": sum(report.invocation_count for report in reports),
        "active_nodes": sorted(nodes),
        "activity_counts": dict(sorted(activity.items())),
        "semantic_outcome_counts": {
            node: dict(sorted(counts.items()))
            for node, counts in sorted(semantic_outcomes.items())
        },
        "emitted_token_counts": {
            node: dict(sorted(counts.items()))
            for node, counts in sorted(emitted_tokens.items())
        },
        "dependencies": [dependency_rows[key] for key in sorted(dependency_rows)],
        "source_reads": {node: sorted(fields) for node, fields in sorted(source_reads.items())},
        "shards": [report.to_dict() for report in reports],
    }


def _greedy_certificate(
    shards: list[_ShardValidation],
    *,
    exhaustive: dict[str, object],
    max_number: int,
    check_determinism: bool,
) -> dict[str, object]:
    number_count = sum(shard.number_count for shard in shards)
    event_count = sum(shard.event_count for shard in shards)
    eos_count = sum(shard.eos_event_count for shard in shards)
    if number_count != max_number:
        raise RuntimeError(f"greedy certificate covers {number_count} numbers, expected {max_number}.")
    if event_count != int(exhaustive["event_count"]):
        raise RuntimeError("greedy certificate and exhaustive validation event counts differ.")
    if eos_count != max_number:
        raise RuntimeError("the target domain does not contain exactly one terminal EOS per number.")
    if not check_determinism:
        raise RuntimeError("greedy certification requires deterministic exhaustive validation.")
    payload: dict[str, object] = {
        "domain": {"minimum": 1, "maximum": max_number},
        "method": "induction_over_exhaustively_validated_prefixes",
        "exact": True,
        "deterministic": True,
        "terminates": True,
        "sequence_count": number_count,
        "exact_sequence_count": number_count,
        "prediction_event_count": event_count,
        "terminal_eos_count": eos_count,
        "max_rollout_steps": max(shard.max_rollout_steps for shard in shards),
        "program_fingerprint": exhaustive["program_fingerprint"],
        "validation_domain_fingerprint": exhaustive["validation_domain_fingerprint"],
        "proof_obligations": {
            "every_valid_prefix_executed": True,
            "every_next_token_matches_oracle": True,
            "repeated_execution_matches": True,
            "downward_closure_and_parent_signatures": bool(
                exhaustive["expected_structure_validated"]
            ),
            "one_terminal_eos_per_sequence": True,
            "base_case_empty_prefix": True,
            "inductive_prefix_extension": True,
        },
    }
    payload["certificate_fingerprint"] = fingerprint(payload)
    return payload


def _validate_canonical_semantic_identities(compiled: CompiledQProgram) -> None:
    payload = canonical_key(
        {
            "semantic_types": compiled.semantic_types,
            "semantic_outcomes": compiled.semantic_outcomes,
            "trace_schema_catalog": compiled.trace_schema_catalog,
        }
    )
    if "__main__." in payload:
        raise RuntimeError(
            "compiled semantic identities contain __main__; run the canonical compile_qprogram module."
        )


def _artifact_name(compiled: CompiledQProgram, program_variant: str) -> str:
    return "-".join(
        (
            program_variant,
            compiled.metadata.program_fingerprint[:12],
            compiled.metadata.training_distribution_fingerprint[:8],
            compiled.metadata.evaluation_distribution_fingerprint[:8],
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile and validate the exact NumberNaming Q-program.")
    parser.add_argument("config", help="QuantaNet config defining the training distribution and trace payloads.")
    parser.add_argument("--output-root", default=".artifacts/number_naming_qprogram")
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 10)))
    parser.add_argument("--program", choices=PROGRAM_VARIANTS, default="factorized")
    parser.add_argument(
        "--audit-samples",
        type=int,
        default=4096,
        help="independent task-domain examples used for compiler audits; 0 uses training only",
    )
    parser.add_argument("--audit-seed", type=int, default=0)
    parser.add_argument("--skip-exhaustive", action="store_true")
    args = parser.parse_args(argv)

    from quanta.config import load_quanta_net_config

    from .task import NumberNamingTask

    config = load_quanta_net_config(args.config)
    output_root = Path(args.output_root)
    config.save_dir = str(output_root / "_compile_cache" / "run")
    task = NumberNamingTask(config)
    compiled = compile_number_naming_program(
        training_examples=task.train,
        evaluation_examples=task.eval_examples,
        audit_examples=sampled_audit_examples(
            max_number=int(config.max_number),
            sample_size=int(args.audit_samples),
            seed=int(args.audit_seed),
        ),
        program_variant=args.program,
    )
    artifact_dir = output_root / _artifact_name(compiled, args.program)
    validation: NumberNamingDomainValidation | None = None
    if not args.skip_exhaustive:
        validation = validate_number_naming_domain_bundle(
            max_number=int(config.max_number),
            workers=int(args.workers),
            check_determinism=True,
            expected_structure=compiled.structure,
            program_variant=args.program,
        )
        exhaustive = validation.exhaustive
        if exhaustive["program_fingerprint"] != compiled.metadata.program_fingerprint:
            raise RuntimeError("compiled and exhaustive program fingerprints differ.")
        if tuple(exhaustive["active_nodes"]) != compiled.nodes:
            raise RuntimeError("structural validation did not discover the exhaustive node set.")
        exhaustive_semantics = exhaustive["semantic_outcome_counts"]
        for node in compiled.nodes:
            compiled_keys = {canonical_key(value) for value in compiled.semantic_outcomes[node]}
            exhaustive_keys = set(exhaustive_semantics[node])
            if not compiled_keys.issubset(exhaustive_keys):
                raise RuntimeError(
                    f"audit semantic outcomes are absent from exhaustive execution for {node}: "
                    f"{sorted(compiled_keys - exhaustive_keys)}"
                )
        emitted_count = sum(
            count
            for counts in exhaustive["emitted_token_counts"].values()
            for count in counts.values()
        )
        if emitted_count != int(exhaustive["event_count"]):
            raise RuntimeError("exhaustive validation did not record exactly one emitter per event.")
        emitter_nodes = {
            node
            for node, counts in exhaustive["emitted_token_counts"].items()
            if counts
        }
        if emitter_nodes != set(compiled.structure.leaves):
            raise RuntimeError(
                "exhaustive emitter nodes differ from compiled terminal nodes: "
                f"emitters={sorted(emitter_nodes)}, leaves={sorted(compiled.structure.leaves)}"
            )
        for node in sorted(emitter_nodes):
            emitted_keys = set(exhaustive["emitted_token_counts"][node])
            semantic_keys = set(exhaustive_semantics[node])
            if emitted_keys != semantic_keys:
                raise RuntimeError(
                    f"terminal quantum {node} has non-emitted semantic outcomes: "
                    f"{sorted(semantic_keys - emitted_keys)}"
                )
        exhaustive_dependencies = {
            (row["parent"], row["child"]): tuple(row["kinds"])
            for row in exhaustive["dependencies"]
        }
        compiled_dependencies = {
            (row.parent, row.child): tuple(row.kinds)
            for row in compiled.structure.unreduced_dependencies
        }
        if exhaustive_dependencies != compiled_dependencies:
            raise RuntimeError(
                "structural validation dependency kinds differ from exhaustive execution."
            )
        exhaustive_source_reads = {
            node: tuple(fields)
            for node, fields in exhaustive["source_reads"].items()
        }
        if exhaustive_source_reads != compiled.source_reads:
            raise RuntimeError(
                "structural validation source reads differ from exhaustive execution."
            )
    compiled.write(artifact_dir)
    if validation is not None:
        exhaustive = validation.exhaustive
        (artifact_dir / "exhaustive_validation.json").write_text(
            json.dumps(exhaustive, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "greedy_validation.json").write_text(
            json.dumps(validation.greedy, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(artifact_dir)


__all__ = [
    "NumberNamingDomainValidation",
    "PROGRAM_VARIANTS",
    "build_program",
    "compile_number_naming_program",
    "exhaustive_number_naming_events",
    "main",
    "number_naming_events",
    "sampled_audit_examples",
    "sequence_example",
    "structural_validation_examples",
    "validate_number_naming_domain",
    "validate_number_naming_domain_bundle",
]


if __name__ == "__main__":
    main()
