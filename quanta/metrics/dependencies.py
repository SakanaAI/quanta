from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from quanta.metrics.utils import (
    is_finite_learning_step,
    learned_threshold_bits,
    loss_nats_to_bits,
    sliding_min,
)


DEPENDENCY_SATISFIED = "dependency_satisfied"
DEPENDENCY_VIOLATED = "dependency_violated"
DEPENDENCY_CENSORED = "dependency_censored"


def dependency_pair_violated(dependent_step, prerequisite_step) -> bool:
    return is_finite_learning_step(dependent_step) and not (
        is_finite_learning_step(prerequisite_step) and prerequisite_step < dependent_step
    )


def dependency_timing_status(dependent_step, prerequisite_steps: Iterable) -> str:
    prerequisite_steps = list(prerequisite_steps)
    if any(dependency_pair_violated(dependent_step, step) for step in prerequisite_steps):
        return DEPENDENCY_VIOLATED
    if prerequisite_steps and all(is_finite_learning_step(step) for step in prerequisite_steps):
        return DEPENDENCY_SATISFIED
    return DEPENDENCY_CENSORED


def get_ancestors(node, dependencies) -> set[int]:
    graph = _normalize_graph(dependencies)
    return _ancestors(int(node), graph, visiting=set())


def compute_poset_dependencies_error(
    subtask_losses,
    codes,
    task_type,
    graph_dependencies=None,
    threshold=None,
    window=100,
):
    """Measure violations of explicitly declared direct dependency edges."""
    return _compute_order_error(
        subtask_losses,
        codes,
        task_type,
        graph_dependencies,
        threshold,
        window,
        transitive=False,
        metric_function=compute_poset_dependencies_error,
    )


def _compute_order_error(
    losses,
    codes,
    task_type,
    graph_dependencies,
    threshold,
    window,
    *,
    transitive,
    metric_function,
):
    threshold = learned_threshold_bits() if threshold is None else float(threshold)
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    if len(losses) != len(codes):
        raise ValueError("subtask_losses and codes must have the same length.")

    learning_steps = [_learning_step(curve, threshold, window) for curve in losses]
    edges = _dependency_edges(codes, task_type, graph_dependencies, transitive)
    candidates = [(child, parent) for child, parent in edges if is_finite_learning_step(learning_steps[child])]
    violations = sum(
        dependency_pair_violated(learning_steps[child], learning_steps[parent])
        for child, parent in candidates
    )
    _store_diagnostics(metric_function, threshold, len(candidates), len(edges))
    return (
        violations / len(candidates) if candidates else 0.0,
        int(violations),
        len(candidates),
    )


def _learning_step(curve, threshold: float, window: int):
    learned = np.flatnonzero(sliding_min(loss_nats_to_bits(curve), window) < threshold)
    return int(learned[0]) if len(learned) else np.inf


def _dependency_edges(codes, task_type, dependencies, transitive: bool):
    code_to_index = {_code_key(code): index for index, code in enumerate(codes)}
    if len(code_to_index) != len(codes):
        raise ValueError("codes must be unique.")

    if str(task_type).lower() in {"cxor", "cnand"}:
        graph = _normalize_graph(dependencies)
        edges = []
        for child_index, code in enumerate(codes):
            parents = _ancestors(int(code), graph, set()) if transitive else set(graph.get(int(code), []))
            edges.extend(
                (child_index, code_to_index[parent])
                for parent in parents
                if parent in code_to_index
            )
        return edges

    sets = [set(code) if isinstance(code, (list, tuple, set)) else {code} for code in codes]
    return [
        (child, parent)
        for child, child_set in enumerate(sets)
        for parent, parent_set in enumerate(sets)
        if parent_set < child_set and (transitive or len(child_set) - len(parent_set) == 1)
    ]


def _normalize_graph(dependencies) -> dict[int, list[int]]:
    return {
        int(node): [int(parent) for parent in parents]
        for node, parents in (dependencies or {}).items()
    }


def _ancestors(node: int, graph: dict[int, list[int]], visiting: set[int]) -> set[int]:
    if node in visiting:
        raise ValueError("graph_dependencies must be acyclic.")
    ancestors = set()
    for parent in graph.get(node, []):
        ancestors.add(parent)
        ancestors.update(_ancestors(parent, graph, visiting | {node}))
    return ancestors


def _code_key(code):
    return tuple(code) if isinstance(code, (list, tuple, set)) else int(code)


def _store_diagnostics(function, threshold: float, candidates: int, total: int) -> None:
    function.last_threshold_bits = threshold
    function.last_candidate_dependencies = candidates
    function.last_total_dependencies = total
    function.last_candidate_dependency_fraction = candidates / total if total else 0.0
