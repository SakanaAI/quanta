from __future__ import annotations

from collections import Counter
import json
import math
from typing import Iterable, Mapping

import numpy as np

from .types import AlignedEventTrace, CoverageComparison, CoverageReport, EventTrace, PosetStructure


def coverage_report(
    traces: Iterable[EventTrace | AlignedEventTrace],
    structure: PosetStructure,
) -> CoverageReport:
    rows = tuple(traces)
    nodes = structure.nodes
    aligned_rows = tuple(_aligned_supervision(trace, nodes) for trace in rows)
    activity = Counter({node: 0 for node in nodes})
    eligible_negative = Counter({node: 0 for node in nodes})
    outcomes: dict[str, Counter[str]] = {node: Counter() for node in nodes}
    semantic_pairs: Counter[str] = Counter()
    parent_contexts: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    pairwise: Counter[str] = Counter()
    higher_order: Counter[str] = Counter()
    parent_only = Counter()
    parent_child = Counter()

    for trace, (activity_row, semantic_row) in zip(rows, aligned_rows):
        active_indices = tuple(index for index, value in enumerate(activity_row) if value)
        active = {nodes[index] for index in active_indices}
        signatures[_signature_key(nodes[index] for index in active_indices)] += 1
        for index, node in enumerate(nodes):
            if activity_row[index]:
                activity[node] += 1
                outcomes[node][_value_key(semantic_row[index])] += 1
            elif all(parent in active for parent in structure.parents[node]):
                eligible_negative[node] += 1
        for parent, child in structure.edges:
            key = f"{parent}->{child}"
            if parent in active and child in active:
                parent_child[key] += 1
            elif parent in active:
                parent_only[key] += 1
        active_order = tuple(nodes[index] for index in active_indices)
        for left_index, left in enumerate(active_order):
            for right in active_order[left_index + 1 :]:
                pairwise[f"{left}|{right}"] += 1
        if len(active_order) >= 3:
            higher_order[_signature_key(active_order)] += 1

    compact_nodes = {
        node
        for node in nodes
        if len(outcomes[node]) <= 64
    }
    node_to_index = {node: index for index, node in enumerate(nodes)}
    for activity_row, semantic_row in aligned_rows:
        active = {nodes[index] for index, value in enumerate(activity_row) if value}
        active_semantics = [
            (node, _value_key(semantic_row[index]))
            for index, node in enumerate(nodes)
            if activity_row[index] and node in compact_nodes
        ]
        for left_index, (left_node, left_value) in enumerate(active_semantics):
            for right_node, right_value in active_semantics[left_index + 1 :]:
                semantic_pairs[f"{left_node}={left_value}|{right_node}={right_value}"] += 1
        for node in active:
            parents = structure.parents[node]
            if not parents or any(parent not in compact_nodes for parent in parents):
                continue
            parent_values = ",".join(
                f"{parent}={_value_key(semantic_row[node_to_index[parent]])}"
                for parent in parents
            )
            parent_contexts[f"{node}<-{parent_values}"] += 1

    if rows and nodes:
        matrix = np.asarray(
            [activity_row for activity_row, _ in aligned_rows],
            dtype=np.float64,
        )
        rank = int(np.linalg.matrix_rank(matrix))
        singular = np.linalg.svd(matrix, compute_uv=False)
        positive = singular[singular > np.finfo(np.float64).eps * max(matrix.shape) * singular[0]] if singular.size else []
        condition = float(positive[0] / positive[-1]) if len(positive) else None
        if condition is not None and not math.isfinite(condition):
            condition = None
    else:
        matrix = np.empty((0, len(nodes)), dtype=np.float64)
        rank = 0
        condition = None
    one_class = tuple(
        node
        for node in nodes
        if activity[node] + eligible_negative[node] > 0
        and (activity[node] == 0 or eligible_negative[node] == 0)
    )
    columns: dict[bytes, list[str]] = {}
    for index, node in enumerate(nodes):
        columns.setdefault(matrix[:, index].tobytes(), []).append(node)
    indistinguishable = tuple(
        tuple(group)
        for group in columns.values()
        if len(group) > 1
    )
    return CoverageReport(
        event_count=len(rows),
        activity_counts=dict(sorted(activity.items())),
        negative_eligible_counts=dict(sorted(eligible_negative.items())),
        semantic_outcome_counts={node: dict(sorted(counts.items())) for node, counts in outcomes.items()},
        semantic_pair_counts=dict(sorted(semantic_pairs.items())),
        parent_semantic_context_counts=dict(sorted(parent_contexts.items())),
        trace_signature_counts=dict(sorted(signatures.items())),
        parent_only_counts=dict(sorted(parent_only.items())),
        parent_plus_child_counts=dict(sorted(parent_child.items())),
        pairwise_coactivation=dict(sorted(pairwise.items())),
        higher_order_coactivation=dict(sorted(higher_order.items())),
        incidence_rank=rank,
        incidence_condition=condition,
        one_class_activity_nodes=one_class,
        indistinguishable_activity_groups=indistinguishable,
    )


def compare_coverage(
    training_traces: Iterable[EventTrace | AlignedEventTrace],
    audit_traces: Iterable[EventTrace | AlignedEventTrace],
    structure: PosetStructure,
    *,
    validation_semantic_outcomes: Mapping[str, Iterable[object]] | None = None,
    rare_semantic_threshold: int = 0,
) -> CoverageComparison:
    if int(rare_semantic_threshold) < 0:
        raise ValueError("rare_semantic_threshold must be non-negative")
    train_rows = tuple(training_traces)
    audit_rows = tuple(audit_traces)
    train = coverage_report(train_rows, structure)
    audit = coverage_report(audit_rows, structure)
    train_signatures = set(train.trace_signature_counts)
    audit_signatures = set(audit.trace_signature_counts)
    train_states = {_value_key(trace.state) for trace in train_rows}
    audit_states = {_value_key(trace.state) for trace in audit_rows}
    audit_only_quanta = tuple(
        node
        for node in structure.nodes
        if audit.activity_counts[node] > 0 and train.activity_counts[node] == 0
    )
    audit_only_outcomes = {}
    for node in structure.nodes:
        missing = set(audit.semantic_outcome_counts[node]) - set(train.semantic_outcome_counts[node])
        if missing:
            audit_only_outcomes[node] = tuple(sorted(missing))
    audit_only_pairs = tuple(
        sorted(set(audit.semantic_pair_counts) - set(train.semantic_pair_counts))
    )
    audit_only_parent_contexts: dict[str, tuple[str, ...]] = {}
    missing_parent_contexts = set(audit.parent_semantic_context_counts) - set(
        train.parent_semantic_context_counts
    )
    for node in structure.nodes:
        prefix = f"{node}<-"
        rows = tuple(sorted(item for item in missing_parent_contexts if item.startswith(prefix)))
        if rows:
            audit_only_parent_contexts[node] = rows
    absent_training_outcomes: dict[str, tuple[str, ...]] = {}
    rare_training_outcomes: dict[str, tuple[tuple[str, int], ...]] = {}
    if validation_semantic_outcomes is not None:
        for node in structure.nodes:
            full_outcomes = tuple(
                sorted(_value_key(value) for value in validation_semantic_outcomes[node])
            )
            training_counts = train.semantic_outcome_counts[node]
            absent = tuple(value for value in full_outcomes if training_counts.get(value, 0) == 0)
            if absent:
                absent_training_outcomes[node] = absent
            if rare_semantic_threshold:
                rare = tuple(
                    (value, training_counts[value])
                    for value in full_outcomes
                    if 0 < training_counts.get(value, 0) <= int(rare_semantic_threshold)
                )
                if rare:
                    rare_training_outcomes[node] = rare
    return CoverageComparison(
        train=train,
        audit=audit,
        trace_overlap=len(train_signatures & audit_signatures),
        predictive_state_overlap=len(train_states & audit_states),
        audit_only_quanta=audit_only_quanta,
        audit_only_semantic_outcomes=audit_only_outcomes,
        audit_only_semantic_pairs=audit_only_pairs,
        audit_only_parent_semantic_contexts=audit_only_parent_contexts,
        training_absent_semantic_outcomes=absent_training_outcomes,
        rare_training_semantic_outcomes=rare_training_outcomes,
    )


def _signature_key(nodes: Iterable[str]) -> str:
    return "|".join(nodes)


def _value_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _aligned_supervision(
    trace: EventTrace | AlignedEventTrace,
    nodes: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[object | None, ...]]:
    """Return node-aligned activity and semantics without expanding invocations."""
    activity_targets = trace.activity_targets
    if activity_targets and len(activity_targets) == len(nodes):
        return activity_targets, trace.semantic_targets
    active = set(trace.active_nodes)
    invocation_by_node = {item.quantum_id: item for item in trace.invocations}
    return (
        tuple(int(node in active) for node in nodes),
        tuple(invocation_by_node[node].output if node in active else None for node in nodes),
    )
