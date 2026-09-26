from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from quanta.metrics.utils import loss_nats_to_bits

from .factorized_ar import (
    QuantumInstance,
    build_factorized_nodes,
    build_factorized_plan,
    build_token_labels,
    schema_edges,
)
from .functional_ar import (
    FUNCTIONAL_PROBE_NODES,
    FUNCTIONAL_SCHEMA_EDGES,
    FUNCTIONAL_SCHEMA_LABELS,
    build_functional_token_labels,
)
from .names import english_number_name

PROBE_CACHE_VERSION = 50
EOS_TARGET = "EOS"


@dataclass(frozen=True)
class FrameProbeExample:
    role: str
    number: int
    prefix: tuple[str, ...]
    target: str
    instance_id: str | None = None
    context: str | None = None
    subtype: str | None = None


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    label: str
    examples: list[FrameProbeExample]
    target_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaggedProbeEvalExample:
    number: int
    text: str
    tags: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class FactorizedNodeOccurrence:
    number: int
    text: str
    node_id: str
    node_type: str
    value: int | None
    role: str
    token_positions: tuple[int, ...]
    children: tuple[str, ...]
    quantum_id: str | None = None
    instance_id: str | None = None
    args: tuple[str, ...] = ()
    context: str | None = None
    schema_children: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorizedExampleTree:
    number: int
    text: str
    root_node_id: str
    occurrences: tuple[FactorizedNodeOccurrence, ...]


@dataclass(frozen=True)
class ProbeSuite:
    id: str
    nodes: dict[str, str]
    edges: list[tuple[str, str]]
    probes: list[ProbeSpec]
    tagged_examples: list[TaggedProbeEvalExample]
    mermaid: str
    factorized_occurrences: list[FactorizedNodeOccurrence] = field(default_factory=list)
    factorized_trees: list[FactorizedExampleTree] = field(default_factory=list)


def load_probe_suites(config, eval_examples: Iterable) -> list[ProbeSuite]:
    suite_ids = _probe_suite_ids(getattr(config, "probe_quanta_poset", None))
    if not suite_ids:
        return []
    eval_examples = list(eval_examples)
    return [_load_probe_suite_by_id(config, suite_id, eval_examples) for suite_id in suite_ids]


def load_probe_suite_by_id(config, suite_id: str, eval_examples: Iterable) -> ProbeSuite:
    return _load_probe_suite_by_id(config, str(suite_id), list(eval_examples))


def load_probe_suite(config, eval_examples: Iterable) -> ProbeSuite | None:
    suites = load_probe_suites(config, eval_examples)
    if not suites:
        return None
    if len(suites) != 1:
        raise ValueError("load_probe_suite expects exactly one configured probe suite.")
    return suites[0]


def _probe_suite_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _load_probe_suite_by_id(config, suite_id: str, eval_examples: list) -> ProbeSuite:
    builders = {
        "eng_virtual_tokens": _build_eng_virtual_tokens_suite,
        "eng_tokens": _build_eng_tokens_suite,
        "eng_contextual_tokens": _build_eng_contextual_tokens_suite,
    }
    if suite_id not in builders:
        if suite_id not in {"eng_factorized", "eng_functional"}:
            raise ValueError(f"Unsupported probe_quanta_poset: {suite_id!r}")
    probe_max_size = getattr(config, "probe_max_size", None)
    cache_path = _probe_cache_path(config, suite_id, eval_examples, probe_max_size)
    if cache_path is not None and cache_path.exists():
        cached_suite = _read_probe_cache(cache_path)
        if cached_suite is not None:
            return cached_suite
    suite = (
        _build_structured_suite(config, suite_id, eval_examples, probe_max_size)
        if suite_id in {"eng_factorized", "eng_functional"}
        else builders[suite_id](eval_examples, probe_max_size)
    )
    if cache_path is not None:
        _write_probe_cache(cache_path, suite)
    return suite


def probe_words(suite: ProbeSuite | None) -> list[str]:
    if suite is None:
        return []
    words = []
    for probe in suite.probes:
        words.extend(probe.target_tokens)
        for example in probe.examples:
            words.extend(example.prefix)
            if example.target != EOS_TARGET:
                words.append(example.target)
    return words


def _even_sample(items: list, count: int) -> list:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def nats_to_bits(value: float) -> float:
    return float(loss_nats_to_bits([value])[0])


def _build_eng_virtual_tokens_suite(eval_examples: list, probe_max_size: int | None) -> ProbeSuite:
    occurrences_by_role: dict[str, list[tuple[int, int, FrameProbeExample]]] = {}
    token_counts_by_example: list[int] = []
    for example_index, example in enumerate(eval_examples):
        number = int(example.number)
        words = str(example.text).split()
        roles = _virtual_token_roles(number, words)
        if len(words) != len(roles):
            raise ValueError(f"Virtual token role mismatch for {number}: {words} vs {roles}")
        token_counts_by_example.append(len(words) + 1)
        for index, (word, role) in enumerate(zip(words, roles)):
            virtual_token = _virtual_token_id(word, role)
            occurrences_by_role.setdefault(virtual_token, []).append(
                (example_index, index,
                FrameProbeExample(
                    role=virtual_token,
                    number=number,
                    prefix=tuple(words[:index]),
                    target=word,
                ))
            )
    examples_by_role, tagged_examples = _select_tagged_probe_examples(
        eval_examples,
        token_counts_by_example,
        occurrences_by_role,
        probe_max_size,
    )

    nodes = {
        role: _virtual_token_label(role)
        for role in sorted(examples_by_role)
    }
    mermaid = "graph TD\n" + "".join(
        f"    vt_{index:03d}[\"{_mermaid_label(label)}\"]\n"
        for index, label in enumerate(nodes.values())
    )
    return ProbeSuite(
        id="eng_virtual_tokens",
        nodes=nodes,
        edges=[],
        mermaid=mermaid,
        probes=[
            ProbeSpec(
                id=role,
                label=nodes[role],
                examples=examples_by_role[role],
            )
            for role in sorted(examples_by_role)
        ],
        tagged_examples=tagged_examples,
    )


def _build_eng_tokens_suite(eval_examples: list, probe_max_size: int | None) -> ProbeSuite:
    occurrences_by_role: dict[str, list[tuple[int, int, FrameProbeExample]]] = {}
    token_counts_by_example: list[int] = []
    for example_index, example in enumerate(eval_examples):
        number = int(example.number)
        words = str(example.text).split()
        token_counts_by_example.append(len(words) + 1)
        for index, word in enumerate(words):
            token_id = word.replace(" ", "_")
            occurrences_by_role.setdefault(token_id, []).append(
                (example_index, index,
                FrameProbeExample(
                    role=token_id,
                    number=number,
                    prefix=tuple(words[:index]),
                    target=word,
                ))
            )
    examples_by_role, tagged_examples = _select_tagged_probe_examples(
        eval_examples,
        token_counts_by_example,
        occurrences_by_role,
        probe_max_size,
    )
    nodes = {role: role.replace("_", " ") for role in sorted(examples_by_role)}
    mermaid = "graph TD\n" + "".join(
        f"    token_{index:03d}[\"{_mermaid_label(label)}\"]\n"
        for index, label in enumerate(nodes.values())
    )
    return ProbeSuite(
        id="eng_tokens",
        nodes=nodes,
        edges=[],
        mermaid=mermaid,
        probes=[
            ProbeSpec(
                id=role,
                label=nodes[role],
                examples=examples_by_role[role],
            )
            for role in sorted(examples_by_role)
        ],
        tagged_examples=tagged_examples,
    )


def _build_eng_contextual_tokens_suite(eval_examples: list, probe_max_size: int | None) -> ProbeSuite:
    occurrences_by_role: dict[str, list[tuple[int, int, FrameProbeExample]]] = {}
    token_counts_by_example: list[int] = []
    value_nodes: set[str] = set()
    scale_nodes: set[str] = set()
    role_nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()

    for example_index, example in enumerate(eval_examples):
        number = int(example.number)
        words = str(example.text).split()
        roles = _virtual_token_roles(number, words)
        if len(words) != len(roles):
            raise ValueError(f"Contextual token role mismatch for {number}: {words} vs {roles}")
        token_counts_by_example.append(len(words) + 1)
        for index, (word, role) in enumerate(zip(words, roles)):
            marker_id = _surface_marker_id(word)
            role_id = _contextual_role_id(word, role)
            role_nodes.add(role_id)
            if _is_structural_word(word):
                edges.add((marker_id, role_id))
            else:
                value_id = _value_id(word)
                value_nodes.add(value_id)
                edges.add((marker_id, value_id))
                edges.add((value_id, role_id))
                scale_id = _scale_id_for_role(role)
                if scale_id is not None and index + 1 < len(words):
                    scale_nodes.add(scale_id)
                    edges.add((scale_id, role_id))
                    occurrences_by_role.setdefault(scale_id, []).append(
                        (example_index, index + 1,
                        FrameProbeExample(
                            role=scale_id,
                            number=number,
                            prefix=tuple(words[: index + 1]),
                            target=words[index + 1],
                        ))
                    )
            for probe_id in _contextual_probe_ids_for_occurrence(word, role):
                occurrences_by_role.setdefault(probe_id, []).append(
                    (example_index, index,
                    FrameProbeExample(
                        role=probe_id,
                        number=number,
                        prefix=tuple(words[:index]),
                        target=word,
                    ))
                )

    examples_by_role, tagged_examples = _select_tagged_probe_examples(
        eval_examples,
        token_counts_by_example,
        occurrences_by_role,
        probe_max_size,
    )
    role_nodes = {probe_id for probe_id in examples_by_role if probe_id.startswith("ROLE(")}
    value_nodes = {probe_id for probe_id in examples_by_role if probe_id.startswith("VALUE(")}
    scale_nodes = {probe_id for probe_id in examples_by_role if probe_id.startswith("SCALE(")}
    edges = {
        edge
        for edge in edges
        if edge[0] in examples_by_role and edge[1] in examples_by_role
    }
    contextual_labels: dict[str, str] = {}
    nodes: dict[str, str] = {}
    for probe_id in sorted(examples_by_role):
        if probe_id.startswith("SCALE("):
            nodes[probe_id] = _scale_label(probe_id)
        elif probe_id.startswith("VALUE("):
            nodes[probe_id] = _value_label(probe_id)
        elif probe_id.startswith("ROLE("):
            nodes[probe_id] = _contextual_role_label(probe_id)
        elif probe_id.startswith("MARKER("):
            nodes[probe_id] = _marker_label(probe_id)
        else:
            nodes[probe_id] = contextual_labels[probe_id]
    mermaid_edges = sorted(
        (parent, child)
        for parent, child in edges
        if parent in nodes and child in nodes
    )
    mermaid = _contextual_tokens_mermaid(nodes, mermaid_edges, scale_nodes, value_nodes, role_nodes)
    return ProbeSuite(
        id="eng_contextual_tokens",
        nodes=nodes,
        edges=mermaid_edges,
        mermaid=mermaid,
        probes=[
            ProbeSpec(
                id=role,
                label=nodes[role],
                examples=examples_by_role[role],
                target_tokens=_probe_target_tokens(occurrences_by_role[role]),
            )
            for role in sorted(examples_by_role)
        ],
        tagged_examples=tagged_examples,
    )


def _build_eng_factorized_suite(config, eval_examples: list, probe_max_size: int | None) -> ProbeSuite:
    return _build_structured_suite(config, "eng_factorized", eval_examples, probe_max_size)


def _build_structured_suite(config, suite_id: str, eval_examples: list, probe_max_size: int | None) -> ProbeSuite:
    del eval_examples
    if suite_id == "eng_factorized":
        generated_examples = _factorized_probe_examples(config)
        trees = [_factorized_tree_for_example(example) for example in generated_examples]
        nodes = dict(FACTORIZED_SCHEMA_LABELS)
        edges = set(FACTORIZED_SCHEMA_EDGES)
        probe_examples_by_node = _factorized_schema_probe_examples(config)
    elif suite_id == "eng_functional":
        generated_examples = _functional_probe_examples(config)
        trees = []
        nodes = dict(FUNCTIONAL_SCHEMA_LABELS)
        edges = set(FUNCTIONAL_SCHEMA_EDGES)
        probe_examples_by_node = _functional_schema_probe_examples(config)
    else:
        raise ValueError(f"Unsupported structured probe suite: {suite_id!r}")
    occurrences = [occurrence for tree in trees for occurrence in tree.occurrences]
    if probe_max_size is not None:
        probe_examples_by_node = {
            node_id: _even_sample(examples, min(int(probe_max_size), len(examples)))
            for node_id, examples in probe_examples_by_node.items()
        }

    probes = [
        ProbeSpec(
            id=node_id,
            label=nodes.get(node_id, node_id),
            examples=examples,
            target_tokens=tuple(sorted({example.target for example in examples if example.target != EOS_TARGET})),
        )
        for node_id, examples in sorted(probe_examples_by_node.items())
    ]
    mermaid_edges = sorted(edges)
    mermaid = "graph TD\n" + "".join(
        f"    {_mermaid_node_id(child)}[\"{_mermaid_label(child)}\"] --> {_mermaid_node_id(parent)}[\"{_mermaid_label(parent)}\"]\n"
        for child, parent in mermaid_edges
    )
    return ProbeSuite(
        id=suite_id,
        nodes=nodes,
        edges=mermaid_edges,
        probes=probes,
        tagged_examples=[
            TaggedProbeEvalExample(
                number=int(example.number),
                text=str(example.text),
                tags=tuple(() for _ in range(len(str(example.text).split()) + 1)),
            )
            for example in generated_examples
        ],
        mermaid=mermaid,
        factorized_occurrences=occurrences,
        factorized_trees=trees,
    )


FACTORIZED_SCHEMA_LABELS = {
    "UNIT_LEX": "Unit lexicon",
    "TEEN_LEX": "Teen lexicon",
    "TEN": "Ten multiplier",
    "ONE_HUNDRED": "One hundred multiplier",
    "ONE_THOUSAND": "One thousand multiplier",
    "TENS": "Tens composition",
    "HUNDREDS": "Hundreds composition",
    "THOUSANDS": "Thousands composition",
    "TEEN_THOUSANDS": "Teen-thousands composition",
    "TEN_THOUSANDS": "Ten-thousands composition",
    "TENS_THOUSANDS": "Tens-thousands composition",
    "HUNDRED_THOUSANDS": "Hundred-thousands composition",
    "EOS_DECISION": "EOS decision",
    "EMIT": "Emission decoder",
}


FACTORIZED_SCHEMA_EDGES = [(parent.value, child.value) for parent, child in schema_edges()]


def _factorized_probe_examples(config) -> list:
    max_number = min(int(getattr(config, "max_number", 999_999)), 999_999)
    if max_number < 1:
        return []
    numbers = set(_factorized_base_numbers(max_number))
    chunk_values = _factorized_chunk_values(max_number)
    for chunk in chunk_values:
        _add_factorized_number(numbers, chunk, max_number)
        _add_factorized_number(numbers, chunk * 1_000, max_number)
        _add_factorized_number(numbers, chunk * 1_000 + 1, max_number)
        _add_factorized_number(numbers, 1_000 + chunk, max_number)
    return [
        _example_from_number(number)
        for number in sorted(numbers)
        if 1 <= number <= max_number
    ]


def _factorized_base_numbers(max_number: int) -> list[int]:
    return [
        number
        for number in [
            1, 2, 5, 9, 10, 11, 18, 19, 20, 21, 25, 80, 82, 99,
            100, 101, 105, 110, 118, 120, 125, 300, 382, 485,
            1_000, 1_001, 1_005, 5_000, 5_382, 100_000, 100_005,
            225_000, 225_005, 225_485, 405_000, 405_005, 485_000,
        ]
        if number <= max_number
    ]


def _factorized_chunk_values(max_number: int) -> list[int]:
    del max_number
    values = set(range(1, 20))
    values.update(range(20, 100, 10))
    values.update(tens + unit for tens in range(20, 100, 10) for unit in range(1, 10))
    for hundreds in range(1, 10):
        values.add(hundreds * 100)
        for remainder in [1, 5, 10, 18, 20, 25, 80, 82, 99]:
            values.add(hundreds * 100 + remainder)
    return sorted(value for value in values if 1 <= value <= 999)


def _add_factorized_number(numbers: set[int], number: int, max_number: int) -> None:
    if 1 <= int(number) <= int(max_number):
        numbers.add(int(number))


def _factorized_schema_probe_examples(config) -> dict[str, list[FrameProbeExample]]:
    max_number = min(int(getattr(config, "max_number", 999_999)), 999_999)
    if max_number < 1:
        return {}
    examples_by_node: dict[str, list[FrameProbeExample]] = {}
    for number in _factorized_schema_probe_numbers(max_number):
        for example in _factorized_owner_probe_examples_from_labels(number, include_emit=True):
            examples_by_node.setdefault(example.role, []).append(example)
    examples_by_node = {
        node_id: _factorized_select_probe_examples(config, _dedupe_probe_examples(examples))
        for node_id, examples in examples_by_node.items()
    }
    return {
        node_id: examples
        for node_id, examples in examples_by_node.items()
        if examples
    }


def _functional_probe_examples(config) -> list:
    return _factorized_probe_examples(config)


def _functional_schema_probe_examples(config) -> dict[str, list[FrameProbeExample]]:
    max_number = min(int(getattr(config, "max_number", 999_999)), 999_999)
    if max_number < 1:
        return {}
    examples_by_node: dict[str, list[FrameProbeExample]] = {}
    for number in _factorized_schema_probe_numbers(max_number):
        for example in _functional_probe_examples_from_labels(number):
            examples_by_node.setdefault(example.role, []).append(example)
    examples_by_node = {
        node_id: _factorized_select_probe_examples(config, _dedupe_probe_examples(examples))
        for node_id, examples in examples_by_node.items()
    }
    return {
        node_id: examples
        for node_id, examples in examples_by_node.items()
        if examples
    }


def _factorized_schema_probe_numbers(max_number: int) -> list[int]:
    numbers: set[int] = set()
    for value in range(1, 10):
        _add_factorized_number(numbers, value, max_number)
        _add_factorized_number(numbers, value * 100, max_number)
        _add_factorized_number(numbers, value * 1_000, max_number)
        _add_factorized_number(numbers, value * 100_000, max_number)
    _add_factorized_number(numbers, 10, max_number)
    _add_factorized_number(numbers, 10_000, max_number)
    for value in range(11, 20):
        _add_factorized_number(numbers, value, max_number)
        _add_factorized_number(numbers, value * 1_000, max_number)
    for tens in range(20, 100, 10):
        _add_factorized_number(numbers, tens, max_number)
        _add_factorized_number(numbers, tens * 1_000, max_number)
    for number in [101, 105, 110, 118, 125, 382, 485, 1_001, 1_005, 5_382, 225_485, 530_802]:
        _add_factorized_number(numbers, number, max_number)
    return sorted(numbers)


def _factorized_select_probe_examples(config, examples: list[FrameProbeExample]) -> list[FrameProbeExample]:
    strategy = str(getattr(config, "eval_strategy", "digits_wise"))
    examples = sorted(examples, key=_factorized_probe_example_sort_key)
    if strategy == "digits_wise":
        return _factorized_digit_uniform_probe_examples(config, examples)
    if strategy == "uniform":
        return examples
    return examples


def _factorized_digit_uniform_probe_examples(config, examples: list[FrameProbeExample]) -> list[FrameProbeExample]:
    requested = getattr(config, "probe_max_size", None)
    if requested is None:
        requested = getattr(config, "eval_samples_per_digit", None)
    if requested is None:
        return examples
    buckets: dict[int, list[FrameProbeExample]] = {}
    for example in examples:
        buckets.setdefault(len(str(int(example.number))), []).append(example)
    digit_lengths = sorted(buckets)
    if not digit_lengths:
        return []
    per_digit = max(1, int(requested) // len(digit_lengths))
    selected: list[FrameProbeExample] = []
    for digits in digit_lengths:
        selected.extend(_even_sample(buckets[digits], min(per_digit, len(buckets[digits]))))
    if len(selected) < int(requested):
        selected_keys = {_factorized_probe_example_key(example) for example in selected}
        remaining = [
            example
            for example in examples
            if _factorized_probe_example_key(example) not in selected_keys
        ]
        selected.extend(_even_sample(remaining, min(int(requested) - len(selected), len(remaining))))
    return sorted(selected[: int(requested)], key=_factorized_probe_example_sort_key)


def _dedupe_probe_examples(examples: list[FrameProbeExample]) -> list[FrameProbeExample]:
    deduped: dict[tuple, FrameProbeExample] = {}
    for example in examples:
        deduped.setdefault(_factorized_probe_example_key(example), example)
    return sorted(deduped.values(), key=_factorized_probe_example_sort_key)


def _factorized_probe_example_key(example: FrameProbeExample) -> tuple:
    return (
        example.role,
        int(example.number),
        tuple(example.prefix),
        example.target,
        example.instance_id,
    )


def _factorized_probe_example_sort_key(example: FrameProbeExample) -> tuple:
    return (
        example.role,
        len(str(int(example.number))),
        int(example.number),
        tuple(example.prefix),
        example.target,
        example.instance_id or "",
    )


def _example_from_number(number: int):
    class _Example:
        def __init__(self, value: int):
            self.number = int(value)
            self.text = english_number_name(int(value))

    return _Example(number)


def _factorized_operational_probes_for_example(example) -> list[FrameProbeExample]:
    if int(example.number) < 1:
        return []
    return _factorized_probe_examples_from_labels(int(example.number))


def _functional_operational_probes_for_example(example) -> list[FrameProbeExample]:
    if int(example.number) < 1:
        return []
    return _functional_probe_examples_from_labels(int(example.number))


def _factorized_schema_probes_for_tree(tree: FactorizedExampleTree) -> list[FrameProbeExample]:
    return _factorized_owner_probe_examples_from_labels(int(tree.number), include_emit=True)


def _factorized_probe_examples_from_labels(number: int) -> list[FrameProbeExample]:
    return _factorized_owner_probe_examples_from_labels(int(number), include_emit=False)


def _functional_probe_examples_from_labels(number: int) -> list[FrameProbeExample]:
    examples: list[FrameProbeExample] = []
    for label in build_functional_token_labels(int(number)):
        for node_id in _functional_probe_nodes(label.active):
            examples.append(
                FrameProbeExample(
                    role=node_id,
                    number=int(label.number),
                    prefix=label.prefix,
                    target=label.target,
                    instance_id=f"{node_id}@{label.context}",
                    context=label.context,
                    subtype=label.subtype,
                )
            )
    return _dedupe_probe_examples(examples)


def _functional_probe_nodes(active: tuple[str, ...]) -> tuple[str, ...]:
    active_set = set(active)
    return tuple(node for node in FUNCTIONAL_PROBE_NODES if node in active_set)


def _factorized_owner_probe_examples_from_labels(number: int, *, include_emit: bool) -> list[FrameProbeExample]:
    examples: list[FrameProbeExample] = []
    for label in build_token_labels(int(number)):
        examples.append(_factorized_probe_example(label, label.owner))
        if include_emit:
            emit = next(
                instance
                for instance in label.active_closure
                if instance.family.value == "EMIT"
            )
            examples.append(_factorized_probe_example(label, emit))
    return _dedupe_probe_examples(examples)


def _factorized_probe_example(label, instance: QuantumInstance) -> FrameProbeExample:
    return FrameProbeExample(
        role=instance.family.value,
        number=int(label.number),
        prefix=label.prefix,
        target=label.target,
        instance_id=_factorized_instance_id(instance),
        context="/".join(instance.path),
        subtype=_factorized_instance_subtype(instance),
    )


def _factorized_instance_id(instance: QuantumInstance) -> str:
    args = ",".join(f"{key}={value}" for key, value in instance.args)
    path = "/".join(instance.path) or "root"
    return f"{instance.family.value}({args})@{path}"


def _factorized_instance_subtype(instance: QuantumInstance) -> str:
    return instance.family.value.lower()


def _factorized_tree_for_example(example) -> FactorizedExampleTree:
    number = int(example.number)
    if not 0 < number <= 999_999:
        raise ValueError(f"eng_factorized supports numbers 1..999999, got {number}.")
    expected_words = english_number_name(number).split()
    actual_words = str(example.text).split()
    if actual_words != expected_words:
        raise ValueError(f"eng_factorized text mismatch for {number}: {actual_words} vs {expected_words}")
    plan = build_factorized_plan(number)
    nodes = build_factorized_nodes(number)
    occurrences = [
        FactorizedNodeOccurrence(
            number=number,
            text=str(example.text),
            node_id=_factorized_instance_id(node.q),
            node_type=node.q.family.value,
            value=None,
            role="/".join(node.q.path),
            token_positions=tuple(range(node.span[0], node.span[1])),
            children=tuple(_factorized_instance_id(child.q) for child in node.children),
            quantum_id=node.q.family.value,
            instance_id=_factorized_instance_id(node.q),
            args=tuple(f"{key}={value}" for key, value in node.q.args),
            context="/".join(node.q.path),
            schema_children=_factorized_schema_children(node.q.family.value),
        )
        for node in nodes
    ]
    return FactorizedExampleTree(
        number=number,
        text=str(example.text),
        root_node_id=_factorized_instance_id(nodes[0].q),
        occurrences=tuple(occurrences),
    )


def _factorized_schema_children(quantum_id: str | None) -> tuple[str, ...]:
    return tuple(parent for parent, child in FACTORIZED_SCHEMA_EDGES if child == quantum_id)


def _occurrence_quantum_id(occurrence: FactorizedNodeOccurrence) -> str:
    return occurrence.quantum_id or occurrence.node_type


def _occurrence_instance_id(occurrence: FactorizedNodeOccurrence) -> str:
    return occurrence.instance_id or occurrence.node_id


def _factorized_subtype(occurrence: FactorizedNodeOccurrence) -> str:
    return (occurrence.quantum_id or occurrence.node_type).lower()


def _mermaid_node_id(node_id: str) -> str:
    return "f_" + "".join(char if char.isalnum() else "_" for char in node_id)


def _value_id(word: str) -> str:
    return f"VALUE({word.replace(' ', '_')})"


def _value_label(value_id: str) -> str:
    token = value_id.removeprefix("VALUE(").removesuffix(")").replace("_", " ")
    return f"Value: {token}"


def _surface_marker_id(word: str) -> str:
    return "MARKER(structure)" if _is_structural_word(word) else "MARKER(content)"


def _is_structural_word(word: str) -> bool:
    return word in {"hundred", "thousand", "million", "billion"}


def _marker_label(marker_id: str) -> str:
    marker = marker_id.removeprefix("MARKER(").removesuffix(")").replace("_", " ")
    return f"Marker: {marker}"


def _surface_category_id(word: str) -> str:
    if word in {"hundred", "thousand", "million", "billion"}:
        return "EMIT_STRUCTURAL_MARKER"
    try:
        value = _word_value(word)
    except ValueError:
        return "EMIT_VALUE_WORD"
    if 10 <= value <= 19:
        return "EMIT_TEEN_WORD"
    if value >= 20 and value % 10 == 0:
        return "EMIT_TENS_WORD"
    return "EMIT_VALUE_WORD"


def _word_value(word: str) -> int:
    values = {
        english_number_name(value): value
        for value in range(1, 100)
    }
    if word not in values:
        raise ValueError(f"Unknown English number word: {word!r}")
    return values[word]


def _scale_id_for_role(role: str) -> str | None:
    scale_roles = {
        "thousands": "thousand",
        "ten_thousands": "ten_thousand",
        "hundred_thousands": "hundred_thousand",
    }
    scale = scale_roles.get(role)
    if scale is None:
        return None
    return f"SCALE({scale})"


def _scale_label(scale_id: str) -> str:
    scale = scale_id.removeprefix("SCALE(").removesuffix(")")
    return f"Scale: {scale.replace('_', ' ')}"


def _contextual_token_label(observed_id: str) -> str:
    word, _, role = observed_id.partition("__")
    return f"{word.replace('_', ' ')} @ {role.replace('_', ' ')}"


def _contextual_role_id(word: str, role: str) -> str:
    if _is_structural_word(word):
        return f"ROLE({word.replace(' ', '_')})"
    return f"ROLE({word.replace(' ', '_')}__{role})"


def _contextual_role_label(role_id: str) -> str:
    inner = role_id.removeprefix("ROLE(").removesuffix(")")
    word, _, role = inner.partition("__")
    if not role:
        return f"Role: {word.replace('_', ' ')}"
    return f"Role: {word.replace('_', ' ')} @ {role.replace('_', ' ')}"


def _contextual_probe_ids_for_occurrence(word: str, role: str) -> tuple[str, ...]:
    marker_id = _surface_marker_id(word)
    role_id = _contextual_role_id(word, role)
    if _is_structural_word(word):
        return (marker_id, role_id)
    return (marker_id, _value_id(word), role_id)


def _probe_target_tokens(occurrences: list[tuple[int, int, FrameProbeExample]]) -> tuple[str, ...]:
    return tuple(sorted({example.target for _, _, example in occurrences if example.target != EOS_TARGET}))


def _contextual_tokens_mermaid(
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    scale_nodes: set[str],
    value_nodes: set[str],
    role_nodes: set[str],
) -> str:
    marker_nodes = {node for node in nodes if node.startswith("MARKER(")}
    lines = ["graph TD\n"]
    for subgraph_name, ids in [
        ("Markers", sorted(marker_nodes)),
        ("Scales", sorted(scale_nodes)),
        ("Values", sorted(value_nodes)),
        ("Roles", sorted(role_nodes)),
    ]:
        lines.append(f"    subgraph {_mermaid_node_id(subgraph_name)}[\"{_mermaid_label(subgraph_name)}\"]\n")
        for node_id in ids:
            if node_id in nodes:
                lines.append(f"        {_mermaid_node_id(node_id)}[\"{_mermaid_label(nodes[node_id])}\"]\n")
        lines.append("    end\n")
    for parent, child in edges:
        lines.append(
            f"    {_mermaid_node_id(parent)} --> {_mermaid_node_id(child)}\n"
        )
    return "".join(lines)


def _select_tagged_probe_examples(
    eval_examples: list,
    token_counts_by_example: list[int],
    occurrences_by_role: dict[str, list[tuple[int, int, FrameProbeExample]]],
    probe_max_size: int | None,
) -> tuple[dict[str, list[FrameProbeExample]], list[TaggedProbeEvalExample]]:
    tags_by_example = [
        [set() for _ in range(token_count)]
        for token_count in token_counts_by_example
    ]
    examples_by_role: dict[str, list[FrameProbeExample]] = {}
    for role, occurrences in occurrences_by_role.items():
        selected = occurrences
        if probe_max_size is not None and int(probe_max_size) < len(occurrences):
            selected = _even_sample(occurrences, int(probe_max_size))
        examples_by_role[role] = [example for _, _, example in selected]
        for example_index, token_index, _ in selected:
            tags_by_example[example_index][token_index].add(role)

    tagged_examples = []
    for example, tags in zip(eval_examples, tags_by_example):
        normalized_tags = tuple(tuple(sorted(tag_set)) for tag_set in tags)
        if any(normalized_tags):
            tagged_examples.append(
                TaggedProbeEvalExample(
                    number=int(example.number),
                    text=str(example.text),
                    tags=normalized_tags,
                )
            )
    return examples_by_role, tagged_examples


def _virtual_token_id(word: str, role: str) -> str:
    return f"{word}__{role}".replace(" ", "_")


def _virtual_token_label(role: str) -> str:
    word, _, semantic_role = role.partition("__")
    return f"{word} ({semantic_role.replace('_', ' ')})"


def _mermaid_label(label: str) -> str:
    return label.replace('"', '\\"')


def _virtual_token_roles(number: int, words: list[str]) -> list[str]:
    value_roles = _value_position_roles(number, words)
    return [
        _surface_token_role(word, value_role)
        for word, value_role in zip(words, value_roles)
    ]


def _surface_token_role(word: str, value_role: str) -> str:
    if word == "hundred":
        return "hundred_marker"
    if word in {"thousand", "million", "billion"}:
        return "scale"
    return value_role


def _value_position_roles(number: int, words: list[str]) -> list[str]:
    if not words:
        return []
    if number < 0:
        raise ValueError("number must be non-negative.")
    if number < 20:
        return ["units"]
    if number < 100:
        tens, ones = divmod(number, 10)
        del tens
        roles = ["tens"]
        if ones:
            roles.append("units")
        return roles
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        del hundreds
        roles = ["hundreds", "hundreds"]
        if remainder:
            roles.extend(_value_position_roles(remainder, words[2:]))
        return roles
    for scale, role in [
        (1_000_000_000, "billions"),
        (1_000_000, "millions"),
        (1_000, "thousands"),
    ]:
        block, remainder = divmod(number, scale)
        if block:
            block_words = english_number_name(block).split()
            roles = [
                _scale_block_role(block_role, scale)
                for block_role in _value_position_roles(block, block_words)
            ] + [role]
            if remainder:
                roles.extend(_value_position_roles(remainder, english_number_name(remainder).split()))
            return roles
    raise ValueError(f"Cannot derive value positions for number: {number}")


def _scale_block_role(role: str, scale: int) -> str:
    scale_roles = {
        1_000: {
            "units": "thousands",
            "tens": "ten_thousands",
            "hundreds": "hundred_thousands",
        },
        1_000_000: {
            "units": "millions",
            "tens": "ten_millions",
            "hundreds": "hundred_millions",
        },
        1_000_000_000: {
            "units": "billions",
            "tens": "ten_billions",
            "hundreds": "hundred_billions",
        },
    }
    return scale_roles.get(int(scale), {}).get(role, role)


def _probe_cache_path(config, suite_id: str, eval_examples: list, probe_max_size: int | None) -> Path | None:
    if not getattr(config, "save_dir", None):
        return None
    save_dir = Path(config.save_dir)
    cache_dir = save_dir.parent / "cache" / "probes"
    eval_digest = hashlib.sha256()
    for example in eval_examples:
        eval_digest.update(str(int(example.number)).encode("utf-8"))
        eval_digest.update(b"\0")
        eval_digest.update(str(example.text).encode("utf-8"))
        eval_digest.update(b"\0")
    key = {
        "version": PROBE_CACHE_VERSION,
        "suite_id": suite_id,
        "language": config.language,
        "eval_digest": eval_digest.hexdigest(),
        "probe_max_size": None if probe_max_size is None else int(probe_max_size),
    }
    if suite_id in {"eng_factorized", "eng_functional"}:
        key.update(
            {
                "max_number": int(getattr(config, "max_number", 999_999)),
                "split_seed": int(getattr(config, "split_seed", 0)),
                "eval_size": getattr(config, "eval_size", None),
            }
        )
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{suite_id}-{digest}.json"


def _write_probe_cache(path: Path, suite: ProbeSuite) -> None:
    os.makedirs(path.parent, exist_ok=True)
    payload = {
        "version": PROBE_CACHE_VERSION,
        "id": suite.id,
        "nodes": suite.nodes,
        "edges": suite.edges,
        "mermaid": suite.mermaid,
        "tagged_examples": [
            {
                "number": example.number,
                "text": example.text,
                "tags": [list(tags or ()) for tags in example.tags],
            }
            for example in suite.tagged_examples
        ],
        "probes": [
            {
                "id": probe.id,
                "label": probe.label,
                "target_tokens": list(probe.target_tokens),
                "examples": [
                    {
                        "role": example.role,
                        "number": example.number,
                        "prefix": list(example.prefix),
                        "target": example.target,
                        "instance_id": example.instance_id,
                        "context": example.context,
                        "subtype": example.subtype,
                    }
                    for example in probe.examples
                ],
            }
            for probe in suite.probes
        ],
        "factorized_occurrences": [
            _factorized_occurrence_to_dict(occurrence)
            for occurrence in suite.factorized_occurrences
        ],
        "factorized_trees": [
            {
                "number": tree.number,
                "text": tree.text,
                "root_node_id": tree.root_node_id,
                "occurrences": [
                    _factorized_occurrence_to_dict(occurrence)
                    for occurrence in tree.occurrences
                ],
            }
            for tree in suite.factorized_trees
        ],
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as handle:
        json.dump(payload, handle)
    os.replace(temp_path, path)


def _read_probe_cache(path: Path) -> ProbeSuite | None:
    with open(path, "r") as handle:
        payload = json.load(handle)
    if payload.get("version") != PROBE_CACHE_VERSION:
        return None
    required_keys = {"id", "nodes", "edges", "mermaid", "probes", "tagged_examples"}
    if not required_keys.issubset(payload):
        return None
    return ProbeSuite(
        id=str(payload["id"]),
        nodes={str(key): str(value) for key, value in payload["nodes"].items()},
        edges=[(str(parent), str(child)) for parent, child in payload["edges"]],
        mermaid=str(payload["mermaid"]),
        tagged_examples=[
            TaggedProbeEvalExample(
                number=int(item["number"]),
                text=str(item["text"]),
                tags=tuple(
                    tuple(str(tag) for tag in tags)
                    if isinstance(tags, list)
                    else (() if tags is None else (str(tags),))
                    for tags in item["tags"]
                ),
            )
            for item in payload["tagged_examples"]
        ],
        probes=[
            ProbeSpec(
                id=str(item["id"]),
                label=str(item["label"]),
                target_tokens=tuple(str(token) for token in item.get("target_tokens") or ()),
                examples=[
                    FrameProbeExample(
                        role=str(example["role"]),
                        number=int(example["number"]),
                        prefix=tuple(str(word) for word in example["prefix"]),
                        target=str(example["target"]),
                        instance_id=None if example.get("instance_id") is None else str(example["instance_id"]),
                        context=None if example.get("context") is None else str(example["context"]),
                        subtype=None if example.get("subtype") is None else str(example["subtype"]),
                    )
                    for example in item["examples"]
                ],
            )
            for item in payload["probes"]
        ],
        factorized_occurrences=[
            _factorized_occurrence_from_dict(item)
            for item in payload.get("factorized_occurrences") or []
        ],
        factorized_trees=[
            FactorizedExampleTree(
                number=int(item["number"]),
                text=str(item["text"]),
                root_node_id=str(item["root_node_id"]),
                occurrences=tuple(
                    _factorized_occurrence_from_dict(occurrence)
                    for occurrence in item.get("occurrences") or []
                ),
            )
            for item in payload.get("factorized_trees") or []
        ],
    )


def _factorized_occurrence_to_dict(occurrence: FactorizedNodeOccurrence) -> dict:
    return {
        "number": occurrence.number,
        "text": occurrence.text,
        "node_id": occurrence.node_id,
        "node_type": occurrence.node_type,
        "value": occurrence.value,
        "role": occurrence.role,
        "token_positions": list(occurrence.token_positions),
        "children": list(occurrence.children),
        "quantum_id": occurrence.quantum_id,
        "instance_id": occurrence.instance_id,
        "args": list(occurrence.args),
        "context": occurrence.context,
        "schema_children": list(occurrence.schema_children),
    }


def _factorized_occurrence_from_dict(item: dict) -> FactorizedNodeOccurrence:
    return FactorizedNodeOccurrence(
        number=int(item["number"]),
        text=str(item["text"]),
        node_id=str(item["node_id"]),
        node_type=str(item["node_type"]),
        value=None if item.get("value") is None else int(item["value"]),
        role=str(item["role"]),
        token_positions=tuple(int(position) for position in item.get("token_positions") or []),
        children=tuple(str(child) for child in item.get("children") or []),
        quantum_id=None if item.get("quantum_id") is None else str(item["quantum_id"]),
        instance_id=None if item.get("instance_id") is None else str(item["instance_id"]),
        args=tuple(str(arg) for arg in item.get("args") or ()),
        context=None if item.get("context") is None else str(item["context"]),
        schema_children=tuple(str(child) for child in item.get("schema_children") or ()),
    )
