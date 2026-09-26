from __future__ import annotations

from typing import Any

import numpy as np


def threshold_ideal_mixture(
    *,
    marginals: dict[int, float],
    graph_dependencies: dict[int, list[int]],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Represent monotone node marginals as a finite mixture of order ideals.

    The construction samples a shared threshold U and activates every node whose
    marginal demand is at least U.  Scaling all marginals to have maximum one
    removes the empty trace while preserving their relative law.
    """
    nodes = sorted(int(node) for node in marginals)
    if not nodes:
        raise ValueError("ideal marginals must be non-empty.")
    values = np.asarray([float(marginals[node]) for node in nodes], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("ideal marginals must be finite.")
    if np.any(values < 0):
        raise ValueError("ideal marginals must be non-negative.")
    maximum = float(values.max(initial=0.0))
    if maximum <= 0:
        raise ValueError("ideal marginals must have positive maximum demand.")

    normalized = values / maximum
    node_index = {node: index for index, node in enumerate(nodes)}
    for child, parents in graph_dependencies.items():
        child = int(child)
        if child not in node_index:
            continue
        for parent in parents:
            parent = int(parent)
            if parent not in node_index:
                continue
            if normalized[node_index[parent]] + float(tolerance) < normalized[node_index[child]]:
                raise ValueError(
                    "ideal marginals must be order-reversing: "
                    f"parent {parent} has demand {normalized[node_index[parent]]:.6g} "
                    f"below child {child} demand {normalized[node_index[child]]:.6g}."
                )

    levels = np.asarray(sorted({float(value) for value in normalized if value > 0}, reverse=True))
    next_levels = np.concatenate([levels[1:], np.zeros(1, dtype=float)])
    probabilities = levels - next_levels
    masks = normalized[None, :] >= levels[:, None] - float(tolerance)
    if np.any(probabilities < -float(tolerance)):
        raise RuntimeError("threshold ideal mixture produced negative probabilities.")
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()

    reconstructed = probabilities @ masks.astype(float)
    if not np.allclose(reconstructed, normalized, atol=max(float(tolerance), 1e-9), rtol=0.0):
        raise RuntimeError("threshold ideal mixture did not reconstruct its marginals.")

    return {
        "nodes": nodes,
        "marginals": {node: float(normalized[index]) for index, node in enumerate(nodes)},
        "marginal_scale": maximum,
        "thresholds": [float(value) for value in levels],
        "probabilities": [float(value) for value in probabilities],
        "ideal_masks": masks.tolist(),
    }
