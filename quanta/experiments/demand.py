from __future__ import annotations

import math
from typing import Any

import numpy as np

from quanta.experiments.ideal_sampling import threshold_ideal_mixture
from quanta.utils import theoretical_alpha

from .demand_solver import (
    closure_product,
    compute_actual_tail_alpha,
    fit_target_distribution,
)


DEMAND_MODES = {"shortcut", "composition", "uniform"}


def resolve_task_demand(
    *,
    graph_dependencies: dict[int, list[int]],
    node_depths: dict[int, int],
    beta: float,
    base_freq: float,
    mode: str,
    fit_tolerance: float = 0.02,
    trace_sampling: str = "principal",
) -> dict[str, Any]:
    """Resolve target frequencies and report their closure-induced demand."""
    if mode not in DEMAND_MODES:
        raise ValueError(f"quanta_demand must be one of {sorted(DEMAND_MODES)}.")
    nodes = sorted(node_depths)
    desired = np.asarray(
        [float(base_freq) * float(beta) ** (-int(node_depths[node])) for node in nodes],
        dtype=float,
    )
    rows, columns = _closure_entries(nodes, graph_dependencies)
    if trace_sampling == "ideal_threshold":
        if mode != "composition":
            raise ValueError("ideal_threshold trace sampling requires composition demand.")
        mixture = threshold_ideal_mixture(
            marginals={node: float(desired[index]) for index, node in enumerate(nodes)},
            graph_dependencies=graph_dependencies,
        )
        induced = np.asarray([mixture["marginals"][node] for node in nodes], dtype=float)
        induced_normalized = induced / induced.sum()
        actual_beta = fit_depth_beta(induced, node_depths, nodes)
        return {
            "mode": mode,
            "trace_sampling": trace_sampling,
            "target_frequencies": dict(mixture["marginals"]),
            "desired_quanta_demand": {
                node: float(induced_normalized[index]) for index, node in enumerate(nodes)
            },
            "induced_quanta_demand": {
                node: float(induced_normalized[index]) for index, node in enumerate(nodes)
            },
            "induced_quanta_demand_raw": dict(mixture["marginals"]),
            "ideal_mixture": mixture,
            "desired_beta": float(beta),
            "actual_induced_beta": float(actual_beta),
            "actual_tail_alpha": float(compute_actual_tail_alpha(induced)),
            "comparison_beta": float(beta),
            "relative_rmse": 0.0,
            "fit_tolerance": float(fit_tolerance),
            "close_fit": True,
        }
    if trace_sampling != "principal":
        raise ValueError("trace_sampling must be 'principal' or 'ideal_threshold'.")
    if mode == "shortcut":
        target_weights = desired
        target = desired / desired.sum()
    elif mode == "uniform":
        target_weights = np.full(len(nodes), float(base_freq), dtype=float)
        target = target_weights / target_weights.sum()
    else:
        order = sorted(
            range(len(nodes)),
            key=lambda index: node_depths[nodes[index]],
            reverse=True,
        )
        target = fit_target_distribution(rows, columns, desired, order)
        target_weights = target

    induced = closure_product(rows, columns, target, len(nodes))
    desired_normalized = desired / desired.sum()
    induced_normalized = induced / induced.sum()
    relative_rmse = _relative_rmse(induced_normalized, desired_normalized)
    actual_beta = fit_depth_beta(induced, node_depths, nodes)
    actual_tail_alpha = compute_actual_tail_alpha(induced)
    close_fit = relative_rmse <= float(fit_tolerance)
    comparison_beta = (
        float(beta)
        if mode == "shortcut" or close_fit
        else actual_beta
    )
    return {
        "mode": mode,
        "target_frequencies": {
            node: float(target_weights[index]) for index, node in enumerate(nodes)
        },
        "desired_quanta_demand": {
            node: float(desired_normalized[index]) for index, node in enumerate(nodes)
        },
        "induced_quanta_demand": {
            node: float(induced_normalized[index]) for index, node in enumerate(nodes)
        },
        "induced_quanta_demand_raw": {
            node: float(induced[index]) for index, node in enumerate(nodes)
        },
        "desired_beta": float(beta),
        "actual_induced_beta": float(actual_beta),
        "actual_tail_alpha": float(actual_tail_alpha),
        "comparison_beta": float(comparison_beta),
        "relative_rmse": float(relative_rmse),
        "fit_tolerance": float(fit_tolerance),
        "close_fit": bool(close_fit),
    }


def select_theoretical_alpha(
    *,
    demand: dict[str, Any],
    rho: float,
    beta: float,
    delta: float,
) -> tuple[float, str]:
    if demand.get("theory_alpha_source") == "flat_frequency":
        return float(demand["actual_tail_alpha"]), "flat_frequency"
    if demand.get("theory_alpha_source") == "finite_rank_demand":
        source = str(demand["theory_alpha_source"])
        return float(demand["theoretical_alpha"]), source
    if (
        demand.get("mode") == "composition"
        and not bool(demand.get("close_fit", False))
        and math.isclose(float(delta), 0.0, abs_tol=1e-12)
        and demand.get("actual_tail_alpha") is not None
    ):
        return float(demand["actual_tail_alpha"]), "actual_induced_tail"
    comparison_beta = float(demand.get("comparison_beta", beta))
    return theoretical_alpha(rho, comparison_beta, delta), "depth_beta"


def closure_matrix(
    nodes: list[int],
    graph_dependencies: dict[int, list[int]],
) -> np.ndarray:
    rows, columns = _closure_entries(nodes, graph_dependencies)
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    matrix[rows, columns] = 1.0
    return matrix


def induced_quanta_demand(
    target_frequencies: dict[int, float],
    graph_dependencies: dict[int, list[int]],
) -> dict[int, float]:
    nodes = sorted(target_frequencies)
    weights = np.asarray([target_frequencies[node] for node in nodes], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("target frequencies must be non-negative with positive total.")
    rows, columns = _closure_entries(nodes, graph_dependencies)
    demand = closure_product(rows, columns, weights / weights.sum(), len(nodes))
    return {node: float(demand[index]) for index, node in enumerate(nodes)}


def fit_depth_beta(
    demand: np.ndarray,
    node_depths: dict[int, int],
    nodes: list[int],
) -> float:
    means = []
    for depth in sorted(set(node_depths.values())):
        values = [
            demand[index]
            for index, node in enumerate(nodes)
            if node_depths[node] == depth and demand[index] > 0
        ]
        if values:
            means.append((depth, float(np.mean(values))))
    if len(means) < 2:
        return 1.0
    slope, _ = np.polyfit(
        np.asarray([depth for depth, _ in means], dtype=float),
        np.log([value for _, value in means]),
        1,
    )
    return float(math.exp(-slope))


def _closure_entries(
    nodes: list[int],
    graph_dependencies: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    index = {node: position for position, node in enumerate(nodes)}
    rows = []
    columns = []
    for target_position, target in enumerate(nodes):
        for quantum in _ancestral_closure(target, graph_dependencies, set()):
            if quantum in index:
                rows.append(index[quantum])
                columns.append(target_position)
    return np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)


def _ancestral_closure(
    node: int,
    graph_dependencies: dict[int, list[int]],
    visiting: set[int],
) -> set[int]:
    if node in visiting:
        raise ValueError("graph_dependencies must be acyclic.")
    closure = {int(node)}
    for parent in graph_dependencies.get(int(node), []):
        closure.update(_ancestral_closure(int(parent), graph_dependencies, visiting | {node}))
    return closure


def _relative_rmse(actual: np.ndarray, desired: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(desired**2)))
    return float(np.sqrt(np.mean((actual - desired) ** 2)) / max(denominator, 1e-12))
