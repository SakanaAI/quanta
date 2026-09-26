"""Canonical composition pass for an exact-GD discovery artifact.

The priority stage decides which candidates exist.  This module only adds the
minimal support and temporal closure implied by the simple, one-parent rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PriorityEdge = tuple[int, int, int, int]
EPSILON = 1.0e-12


def _select_edges(
    supports: Sequence[np.ndarray], *, maximum_closure_cost: float
) -> tuple[tuple[PriorityEdge, ...], list[dict[str, Any]]]:
    """Choose each child's cheapest earlier-layer parent, when admissible."""

    candidates: list[dict[str, Any]] = []
    selected: list[PriorityEdge] = []
    for child_layer in range(1, len(supports)):
        child_supports = np.asarray(supports[child_layer], dtype=bool)
        for child in range(child_supports.shape[1]):
            child_mask = child_supports[:, child]
            child_count = int(child_mask.sum())
            possible: list[dict[str, Any]] = []
            for parent_layer in range(child_layer):
                parent_supports = np.asarray(supports[parent_layer], dtype=bool)
                for parent in range(parent_supports.shape[1]):
                    missing = int(
                        (child_mask & ~parent_supports[:, parent]).sum()
                    )
                    record = {
                        "edge": [parent_layer, parent, child_layer, child],
                        "closure_additions": missing,
                        "child_active_events": child_count,
                        "closure_cost": missing / max(child_count, 1),
                        "parent_demand": float(parent_supports[:, parent].mean()),
                    }
                    candidates.append(record)
                    possible.append(record)
            best = min(
                possible,
                key=lambda value: (
                    value["closure_cost"],
                    -value["parent_demand"],
                    value["edge"],
                ),
            )
            if float(best["closure_cost"]) <= maximum_closure_cost:
                selected.append(tuple(int(value) for value in best["edge"]))
    return tuple(sorted(selected)), candidates


def _close_supports(
    supports: Sequence[np.ndarray], edges: Sequence[PriorityEdge]
) -> tuple[np.ndarray, ...]:
    """Apply the canonical implication ``child active => parent active``."""

    closed = [np.asarray(value, dtype=bool).copy() for value in supports]
    for parent_layer, parent, child_layer, child in reversed(tuple(edges)):
        closed[parent_layer][:, parent] |= closed[child_layer][:, child]
    return tuple(closed)


def _close_temporal_priority(
    temporal_priority: Sequence[np.ndarray], edges: Sequence[PriorityEdge]
) -> tuple[np.ndarray, ...]:
    """Make each selected parent available no later than its child."""

    curves = [
        np.maximum(np.asarray(value, dtype=np.float64), 0.0)
        for value in temporal_priority
    ]
    totals = [value.sum(axis=1) for value in curves]
    if any(np.any(value <= EPSILON) for value in totals):
        raise ValueError("every temporal priority curve must have positive mass")
    cumulative = [
        np.cumsum(value, axis=1) / total[:, None]
        for value, total in zip(curves, totals)
    ]
    for parent_layer, parent, child_layer, child in reversed(tuple(edges)):
        cumulative[parent_layer][parent] = np.maximum(
            cumulative[parent_layer][parent], cumulative[child_layer][child]
        )
    return tuple(
        np.maximum(
            np.diff(
                np.concatenate((np.zeros((len(values), 1)), values), axis=1
            ),
            axis=1,
        ),
        0.0,
        )
        * total[:, None]
        for values, total in zip(cumulative, totals)
    )


def _fit_metrics(
    supports: Sequence[np.ndarray], curves: Sequence[np.ndarray], fields: np.ndarray
) -> dict[str, Any]:
    residual_sse = []
    explained = []
    for layer, (support, curve) in enumerate(zip(supports, curves)):
        observed = np.asarray(fields[layer], dtype=np.float64)
        prediction = np.asarray(support, dtype=np.float64) @ np.asarray(curve)
        residual = float(np.square(observed - prediction).sum())
        total = float(np.square(observed).sum())
        residual_sse.append(residual)
        explained.append(1.0 - residual / max(total, EPSILON))
    return {
        "residual_sse_by_layer": residual_sse,
        "field_explained_fraction_by_layer": explained,
        "demand_by_layer": [
            np.asarray(value, dtype=np.float64).mean(axis=0).tolist()
            for value in supports
        ],
        "mean_active_candidates": float(
            sum(np.asarray(value).sum(axis=1).mean() for value in supports)
        ),
    }


def build_simple_closure_graph(
    run_dir: Path, *, maximum_closure_cost: float
) -> Path:
    """Write the single supported composition artifact for a discovery run."""

    if not 0.0 <= maximum_closure_cost <= 1.0:
        raise ValueError("maximum_edge_closure_cost must lie in [0, 1]")
    run_dir = Path(run_dir)
    priority_summary = json.loads((run_dir / "priority_summary.json").read_text())
    with np.load(run_dir / "priority_analysis.npz", allow_pickle=True) as payload:
        supports = tuple(
            np.asarray(value, dtype=bool) for value in payload["train_supports"]
        )
        curves = tuple(
            np.asarray(value, dtype=np.float64)
            for value in payload["temporal_priority"]
        )
        fields = np.asarray(payload["train_credit"], dtype=np.float64)
    edges, candidates = _select_edges(
        supports, maximum_closure_cost=maximum_closure_cost
    )
    closed_supports = _close_supports(supports, edges)
    closed_curves = _close_temporal_priority(curves, edges)
    edge_records = []
    for edge in edges:
        parent_layer, parent, child_layer, child = edge
        child_mask = closed_supports[child_layer][:, child]
        parent_cumulative = np.cumsum(closed_curves[parent_layer][parent])
        child_cumulative = np.cumsum(closed_curves[child_layer][child])
        parent_cumulative /= max(parent_cumulative[-1], EPSILON)
        child_cumulative /= max(child_cumulative[-1], EPSILON)
        source_record = next(
            value for value in candidates if tuple(value["edge"]) == edge
        )
        edge_records.append(
            {
                **source_record,
                "closed_support_violations": int(
                    (child_mask & ~closed_supports[parent_layer][:, parent]).sum()
                ),
                "minimum_temporal_margin": float(
                    np.min(parent_cumulative - child_cumulative)
                ),
            }
        )
    output_dir = run_dir / "factorization"
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, support in enumerate(closed_supports):
        np.save(output_dir / f"support_layer_{layer}.npy", support)
    np.savez(
        output_dir / "temporal_priority.npz",
        **{f"layer_{layer}": value for layer, value in enumerate(closed_curves)},
    )
    summary = {
        "method": "quanta_factorization_v1_simple_closure",
        "scientific_status": (
            "Fixed dynamics candidates receive only the minimal one-parent "
            "support and temporal closure selected by the configured cost ceiling."
        ),
        "source_run": str(run_dir),
        "candidate_counts": [int(value.shape[1]) for value in closed_supports],
        "minimum_candidate_gain_fraction": float(
            priority_summary["minimum_component_gain_fraction_of_original_sse"]
        ),
        "edge_selection": {
            "rule": "minimum child-to-parent support closure cost",
            "maximum_closure_cost": maximum_closure_cost,
            "one_parent_per_child": True,
            "candidate_edges": candidates,
        },
        "edges": [list(edge) for edge in edges],
        "edge_count": len(edges),
        "edge_records": edge_records,
        "support_decoder": "minimal_downward_completion",
        "temporal_projection": "minimal_parent_acceleration",
        **_fit_metrics(closed_supports, closed_curves, fields),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path
