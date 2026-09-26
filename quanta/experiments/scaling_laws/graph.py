from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from quanta.experiments.demand import (
    fit_depth_beta,
    resolve_task_demand,
    select_theoretical_alpha,
)
from quanta.experiments.demand_solver import compute_actual_tail_alpha


def generate_layered_poset(
    *,
    rho: float,
    beta: float,
    base_tasks: int,
    base_freq: float,
    max_depth: int,
    m: int,
    seed: int,
    delta: float = 0.0,
    quanta_demand: str = "shortcut",
    quanta_demand_fit_tolerance: float = 0.02,
    trace_sampling: str = "principal",
    graph_family: str = "exponential_random",
    flat_frequency_exponent: float | None = None,
    target_alpha: float | None = None,
) -> dict[str, Any]:
    if graph_family == "flat_roots":
        return _generate_flat_roots(
            base_tasks=base_tasks,
            base_freq=base_freq,
            max_depth=max_depth,
            seed=seed,
            frequency_exponent=flat_frequency_exponent,
        )
    if graph_family in {"exponential_paired", "exponential_depth_paired"}:
        return _generate_exponential_paired_poset(
            rho=rho,
            beta=beta,
            base_tasks=base_tasks,
            base_freq=base_freq,
            max_depth=max_depth,
            m=m,
            seed=seed,
            delta=delta,
            quanta_demand=quanta_demand,
            trace_sampling=trace_sampling,
            target_alpha=target_alpha,
            quanta_demand_fit_tolerance=quanta_demand_fit_tolerance,
            graph_family=graph_family,
        )
    if graph_family != "exponential_random":
        raise ValueError(f"Unknown graph_family: {graph_family!r}.")
    rng = random.Random(seed)
    depth_nodes: dict[int, list[int]] = {}
    node_depths: dict[int, int] = {}
    graph_dependencies: dict[int, list[int]] = {}
    next_node = 0

    for depth in range(max_depth + 1):
        count = max(1, int(round(base_tasks * (rho ** depth))))
        nodes = list(range(next_node, next_node + count))
        depth_nodes[depth] = nodes
        next_node += count
        for node in nodes:
            node_depths[node] = depth
        if depth == 0:
            continue

        candidates = depth_nodes[depth - 1]
        parent_count = min(m, len(candidates))
        for node in nodes:
            graph_dependencies[node] = sorted(rng.sample(candidates, parent_count))

    demand = resolve_task_demand(
        graph_dependencies=graph_dependencies,
        node_depths=node_depths,
        beta=beta,
        base_freq=base_freq,
        mode=quanta_demand,
        fit_tolerance=quanta_demand_fit_tolerance,
        trace_sampling=trace_sampling,
    )
    selected_alpha, theory_alpha_source = select_theoretical_alpha(
        demand=demand,
        rho=rho,
        beta=beta,
        delta=delta,
    )
    demand.update(
        {
            "theoretical_alpha": float(selected_alpha),
            "theory_alpha_source": theory_alpha_source,
        }
    )
    return {
        "rho": float(rho),
        "beta": float(beta),
        "delta": float(delta),
        "base_tasks": int(base_tasks),
        "base_freq": float(base_freq),
        "max_depth": int(max_depth),
        "m": int(m),
        "seed": int(seed),
        "graph_family": "exponential_random",
        "depth_nodes": depth_nodes,
        "node_depths": node_depths,
        "graph_dependencies": graph_dependencies,
        "task_frequencies": demand["target_frequencies"],
        "quanta_demand": demand,
    }


def _generate_flat_roots(
    *,
    base_tasks: int,
    base_freq: float,
    max_depth: int,
    seed: int,
    frequency_exponent: float | None,
) -> dict[str, Any]:
    """Construct matched depth-zero cNAND tasks with rank-varying frequency."""
    if int(max_depth) != 0:
        raise ValueError("flat_roots requires max_depth: 0.")
    if int(base_tasks) <= 1:
        raise ValueError("flat_roots requires base_tasks > 1.")
    if frequency_exponent is None or float(frequency_exponent) <= 0.0:
        raise ValueError("flat_roots requires a positive flat_frequency_exponent.")

    nodes = list(range(int(base_tasks)))
    ranks = nodes.copy()
    random.Random(int(seed)).shuffle(ranks)
    weights_by_rank = np.asarray(
        [float(base_freq) * float(rank + 1) ** (-float(frequency_exponent)) for rank in nodes],
        dtype=float,
    )
    target_frequencies = {
        int(node): float(weights_by_rank[int(rank)])
        for node, rank in zip(nodes, ranks)
    }
    weights = np.asarray([target_frequencies[node] for node in nodes], dtype=float)
    normalized = weights / weights.sum()
    actual_tail_alpha = float(compute_actual_tail_alpha(weights))
    rank_alpha = float(frequency_exponent) - 1.0
    demand = {
        "mode": "flat_frequency",
        "trace_sampling": "principal",
        "target_frequencies": target_frequencies,
        "desired_quanta_demand": {
            int(node): float(normalized[index]) for index, node in enumerate(nodes)
        },
        "induced_quanta_demand": {
            int(node): float(normalized[index]) for index, node in enumerate(nodes)
        },
        "induced_quanta_demand_raw": target_frequencies,
        "frequency_ranks": {int(node): int(rank + 1) for node, rank in zip(nodes, ranks)},
        "frequency_exponent": float(frequency_exponent),
        "actual_rank_alpha": rank_alpha,
        "rank_demand_fit_min": 1,
        "rank_demand_fit_max": int(base_tasks),
        "desired_beta": 1.0,
        "actual_induced_beta": 1.0,
        "actual_tail_alpha": actual_tail_alpha,
        "comparison_beta": 1.0,
        "relative_rmse": 0.0,
        "fit_tolerance": 0.0,
        "close_fit": True,
        "theory_alpha_source": "flat_frequency",
        "theoretical_alpha": rank_alpha,
        "demand_construction": "rank_power",
    }
    return {
        "rho": 1.0,
        "beta": 1.0,
        "delta": 0.0,
        "base_tasks": int(base_tasks),
        "base_freq": float(base_freq),
        "max_depth": 0,
        "m": 0,
        "seed": int(seed),
        "graph_family": "flat_roots",
        "depth_nodes": {0: nodes},
        "node_depths": {int(node): 0 for node in nodes},
        "graph_dependencies": {},
        "task_frequencies": target_frequencies,
        "quanta_demand": demand,
    }


def _generate_exponential_paired_poset(
    *,
    rho: float,
    beta: float,
    base_tasks: int,
    base_freq: float,
    max_depth: int,
    m: int,
    seed: int,
    delta: float,
    quanta_demand: str,
    trace_sampling: str,
    target_alpha: float | None,
    quanta_demand_fit_tolerance: float,
    graph_family: str,
) -> dict[str, Any]:
    """Build a regular module tree with an exact compact-ideal mixture.

    A sample chooses one terminal module and activates its unique ancestor path.
    Terminal probabilities are the Mobius inverse of the requested module
    marginals.  The marginals may be specified either directly as a finite
    rank power law or by the depth law ``p_d proportional to beta**(-d)``.
    When the Mobius inverse is non-negative, the desired distribution is
    feasible by construction and every sampled trace contains only
    ``m * (depth + 1)`` quanta.
    """
    if trace_sampling != "ideal_path":
        raise ValueError(f"{graph_family} graphs require ideal_path sampling.")
    if quanta_demand != "composition":
        raise ValueError(f"{graph_family} graphs require composition demand.")
    if int(m) != 2:
        raise ValueError(f"{graph_family} graphs currently require m=2.")
    if int(base_tasks) % int(m) != 0:
        raise ValueError("base_tasks must be divisible by m for paired modules.")
    if not math.isclose(float(base_freq), 1.0):
        raise ValueError(f"{graph_family} graphs currently require base_freq=1.")
    branching_factor = int(round(float(rho)))
    if branching_factor < 1 or not math.isclose(float(rho), float(branching_factor)):
        raise ValueError(f"{graph_family} graphs require an integer rho >= 1.")

    depth_first = graph_family == "exponential_depth_paired"
    if depth_first:
        if target_alpha is not None:
            raise ValueError(
                "exponential_depth_paired derives alpha from rho and beta; "
                "target_alpha must be null."
            )
        if float(beta) <= float(rho):
            raise ValueError(
                "exponential_depth_paired requires beta > rho so compact-path "
                "terminal probabilities are positive."
            )
        alpha = float(math.log(float(beta), float(rho)) - 1.0)
    else:
        alpha = (
            float(target_alpha)
            if target_alpha is not None
            else float(math.log(float(beta), float(rho)) - 1.0)
        )
    if alpha <= 0:
        raise ValueError(f"{graph_family} graphs require a positive alpha.")

    root_module_count = int(base_tasks) // int(m)
    module_counts = [
        root_module_count * branching_factor**depth
        for depth in range(int(max_depth) + 1)
    ]
    level_counts = [int(m) * count for count in module_counts]
    cumulative_counts = np.cumsum(level_counts).astype(int).tolist()

    depth_nodes: dict[int, list[int]] = {}
    node_depths: dict[int, int] = {}
    graph_dependencies: dict[int, list[int]] = {}
    modules_by_depth: list[list[list[int]]] = []
    parent_modules_by_depth: list[list[int]] = []
    next_node = 0
    for depth, module_count in enumerate(module_counts):
        count = int(m) * module_count
        nodes = list(range(next_node, next_node + count))
        next_node += count
        depth_nodes[depth] = nodes
        for node in nodes:
            node_depths[node] = depth
        modules = [nodes[start : start + int(m)] for start in range(0, count, int(m))]
        modules_by_depth.append(modules)
        if depth == 0:
            parent_modules_by_depth.append([-1] * len(modules))
            continue
        parents = [index // branching_factor for index in range(len(modules))]
        parent_modules_by_depth.append(parents)
        previous_modules = modules_by_depth[depth - 1]
        for module, parent_index in zip(modules, parents):
            parent_nodes = list(previous_modules[parent_index])
            for node in module:
                graph_dependencies[node] = parent_nodes

    module_marginals_by_depth: list[np.ndarray] = []
    if depth_first:
        # Every module at depth d has exactly the requested beta^{-d}
        # marginal.  With beta > rho, Mobius inversion on the regular tree
        # gives positive terminal mass p_d * (1 - rho / beta).
        for depth, module_count in enumerate(module_counts):
            module_marginals_by_depth.append(
                np.full(
                    module_count,
                    float(base_freq) * float(beta) ** (-depth),
                    dtype=float,
                )
            )
    else:
        # Give each paired module the mean probability of its two consecutive
        # microscopic ranks.  This retains paired execution while making the
        # finite rank-demand law itself (not merely its depth anchors) match alpha.
        rank_power = 1.0 + alpha
        rank_offset = 0
        for module_count in module_counts:
            values = []
            for module_index in range(module_count):
                first_rank = rank_offset + int(m) * module_index + 1
                ranks = np.arange(first_rank, first_rank + int(m), dtype=float)
                values.append(float(np.mean(ranks ** (-rank_power))))
            module_marginals_by_depth.append(np.asarray(values, dtype=float))
            rank_offset += int(m) * module_count

    # Exactly one root module is selected per sample.  This fixes the otherwise
    # irrelevant global scale of the marginals and makes terminal weights sum 1.
    normalization = float(module_marginals_by_depth[0].sum())
    module_marginals_by_depth = [
        values / normalization for values in module_marginals_by_depth
    ]
    terminal_probabilities_by_depth: list[np.ndarray] = []
    for depth, marginals_at_depth in enumerate(module_marginals_by_depth):
        if depth == int(max_depth):
            terminal = marginals_at_depth.copy()
        else:
            child_marginals = module_marginals_by_depth[depth + 1].reshape(
                -1, branching_factor
            )
            terminal = marginals_at_depth - child_marginals.sum(axis=1)
        if float(terminal.min(initial=0.0)) < -1e-12:
            raise ValueError(
                "Requested rank demand is infeasible for compact path ideals: "
                f"minimum terminal probability at depth {depth} is {terminal.min():.6g}."
            )
        terminal_probabilities_by_depth.append(np.maximum(terminal, 0.0))

    terminal_probability_sum = float(
        sum(values.sum() for values in terminal_probabilities_by_depth)
    )
    if not math.isclose(terminal_probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(
            "Compact path terminal probabilities must sum to one; "
            f"got {terminal_probability_sum:.12g}."
        )

    marginals = {
        node: float(module_marginals_by_depth[depth][module_index])
        for depth, modules in enumerate(modules_by_depth)
        for module_index, module in enumerate(modules)
        for node in module
    }
    raw_demand = np.asarray(
        [marginals[node] for node in sorted(node_depths)], dtype=float
    )
    normalized = raw_demand / raw_demand.sum()
    rank_fit_min = max(2, int(base_tasks))
    rank_fit_max = max(rank_fit_min + 1, int(cumulative_counts[-1]) // 2)
    actual_rank_alpha = _fit_rank_demand_alpha(
        raw_demand,
        rank_min=rank_fit_min,
        rank_max=rank_fit_max,
    )
    rank_alpha_relative_error = abs(actual_rank_alpha - alpha) / alpha
    actual_beta = fit_depth_beta(raw_demand, node_depths, sorted(node_depths))
    depth_beta_relative_error = abs(actual_beta - float(beta)) / float(beta)
    relative_rmse = (
        depth_beta_relative_error if depth_first else rank_alpha_relative_error
    )
    theory_alpha_source = "depth_beta" if depth_first else "finite_rank_demand"
    ideal_sampler = {
        "type": "paired_module_path",
        "module_size": int(m),
        "branching_factor": int(branching_factor),
        "modules_by_depth": modules_by_depth,
        "parent_modules_by_depth": parent_modules_by_depth,
        "terminal_probabilities_by_depth": [
            [float(value) for value in values]
            for values in terminal_probabilities_by_depth
        ],
        "marginals": marginals,
        "expected_active_nodes": float(raw_demand.sum()),
        "max_active_nodes": int(m) * (int(max_depth) + 1),
    }
    demand = {
        "mode": "composition",
        "trace_sampling": "ideal_path",
        "target_frequencies": marginals,
        "desired_quanta_demand": {
            node: float(normalized[index])
            for index, node in enumerate(sorted(node_depths))
        },
        "induced_quanta_demand": {
            node: float(normalized[index])
            for index, node in enumerate(sorted(node_depths))
        },
        "induced_quanta_demand_raw": marginals,
        "ideal_sampler": ideal_sampler,
        "desired_beta": float(beta),
        "actual_induced_beta": float(actual_beta),
        "actual_tail_alpha": float(compute_actual_tail_alpha(raw_demand)),
        "actual_rank_alpha": float(actual_rank_alpha),
        "rank_demand_fit_min": int(rank_fit_min),
        "rank_demand_fit_max": int(rank_fit_max),
        "rank_demand_alpha_relative_error": float(rank_alpha_relative_error),
        "depth_beta_relative_error": float(depth_beta_relative_error),
        "comparison_beta": float(beta),
        "relative_rmse": float(relative_rmse),
        "fit_tolerance": float(quanta_demand_fit_tolerance),
        "close_fit": bool(relative_rmse <= quanta_demand_fit_tolerance),
        "theoretical_alpha": float(alpha),
        "theory_alpha_source": theory_alpha_source,
        "demand_construction": "depth_beta" if depth_first else "rank_power",
        "cumulative_counts": cumulative_counts,
        "level_counts": level_counts,
    }
    return {
        "rho": float(rho),
        "beta": float(beta),
        "delta": float(delta),
        "base_tasks": int(base_tasks),
        "base_freq": float(base_freq),
        "max_depth": int(max_depth),
        "m": int(m),
        "seed": int(seed),
        "graph_family": graph_family,
        "target_alpha": float(alpha),
        "depth_nodes": depth_nodes,
        "node_depths": node_depths,
        "graph_dependencies": graph_dependencies,
        "task_frequencies": marginals,
        "quanta_demand": demand,
    }


def _fit_rank_demand_alpha(
    demand: np.ndarray,
    *,
    rank_min: int,
    rank_max: int,
) -> float:
    """Fit p_k ~ k^{-(1+alpha)} on a preregistered finite-rank interval."""
    ordered = np.sort(np.asarray(demand, dtype=float).reshape(-1))[::-1]
    ranks = np.arange(1, ordered.size + 1, dtype=float)
    selected = (
        (ranks >= int(rank_min))
        & (ranks <= int(rank_max))
        & np.isfinite(ordered)
        & (ordered > 0)
    )
    if int(np.count_nonzero(selected)) < 2:
        raise ValueError(
            f"Rank-demand fit [{rank_min}, {rank_max}] contains fewer than two points."
        )
    slope, _ = np.polyfit(np.log(ranks[selected]), np.log(ordered[selected]), 1)
    return float(-slope - 1.0)
