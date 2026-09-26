from __future__ import annotations

import numpy as np


def compute_actual_tail_alpha(induced_quanta_demand: np.ndarray) -> float:
    """Fit the unresolved-mass tail after learning quanta in demand order."""
    demand = np.asarray(induced_quanta_demand, dtype=float).reshape(-1)
    if demand.size == 0:
        raise ValueError("induced_quanta_demand must be non-empty.")
    if not np.all(np.isfinite(demand)):
        raise ValueError("induced_quanta_demand must contain only finite values.")
    if np.any(demand < 0):
        raise ValueError("induced_quanta_demand must be non-negative.")
    total = float(demand.sum())
    if total <= 0:
        raise ValueError("induced_quanta_demand must have positive total mass.")

    weights = np.sort(demand / total)[::-1]
    capacities = np.arange(1, weights.size, dtype=float)
    tails = 1.0 - np.cumsum(weights)[:-1]
    valid = np.isfinite(tails) & (tails > 0)
    if np.count_nonzero(valid) < 2:
        return 0.0

    slope, _ = np.polyfit(
        np.log(capacities[valid]),
        np.log(tails[valid]),
        1,
    )
    return float(-slope)


def fit_target_distribution(
    rows: np.ndarray,
    columns: np.ndarray,
    desired: np.ndarray,
    reverse_topological_order: list[int],
) -> np.ndarray:
    desired = desired / desired.sum()
    n_nodes = len(desired)
    exact = _mobius_inverse(
        rows,
        columns,
        desired,
        n_nodes,
        reverse_topological_order,
    )
    if np.min(exact) >= -1e-12 and exact.sum() > 0:
        exact = np.maximum(exact, 0.0)
        return exact / exact.sum()

    target = _project_simplex(np.maximum(exact, 0.0))
    induced = closure_product(rows, columns, target, n_nodes)
    scale = float(np.dot(desired, induced) / np.dot(desired, desired))
    max_row = int(np.bincount(rows, minlength=n_nodes).max(initial=1))
    max_column = int(np.bincount(columns, minlength=n_nodes).max(initial=1))
    lipschitz = 2.0 * (max_row * max_column + np.dot(desired, desired))
    step_size = 1.0 / max(lipschitz, 1e-12)

    for _ in range(2000):
        residual = closure_product(rows, columns, target, n_nodes) - scale * desired
        gradient = np.bincount(
            columns,
            weights=residual[rows],
            minlength=n_nodes,
        )
        next_target = _project_simplex(target - step_size * 2.0 * gradient)
        next_scale = max(0.0, scale + step_size * 2.0 * np.dot(desired, residual))
        change = max(
            np.max(np.abs(next_target - target)),
            abs(next_scale - scale),
        )
        target, scale = next_target, next_scale
        if change < 1e-11:
            break
    return target


def closure_product(
    rows: np.ndarray,
    columns: np.ndarray,
    target: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    return np.bincount(rows, weights=target[columns], minlength=n_nodes)


def _mobius_inverse(
    rows: np.ndarray,
    columns: np.ndarray,
    desired: np.ndarray,
    n_nodes: int,
    reverse_topological_order: list[int],
) -> np.ndarray:
    target = np.zeros(n_nodes, dtype=float)
    descendants = [
        columns[(rows == node) & (columns != node)]
        for node in range(n_nodes)
    ]
    for node in reverse_topological_order:
        target[node] = desired[node] - target[descendants[node]].sum()
    return target


def _project_simplex(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    valid = ordered - cumulative / np.arange(1, len(values) + 1) > 0
    threshold = cumulative[np.flatnonzero(valid)[-1]] / np.count_nonzero(valid)
    return np.maximum(values - threshold, 0.0)
