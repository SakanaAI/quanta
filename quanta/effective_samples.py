from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EffectiveSampleData:
    codes: list[int]
    eval_steps: np.ndarray
    probabilities: dict[int, float]
    batch_size: float
    node_depths: dict[int, int]
    axes: dict[int, np.ndarray]


def effective_sample_data_from_results(
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    batch_size: float | None = None,
) -> EffectiveSampleData:
    config = config or {}
    codes = [int(code) for code in results["codes"]]
    eval_steps = np.asarray(results["eval_steps"], dtype=float)
    probabilities = {
        int(code): float(probability)
        for code, probability in results.get("task_probabilities", {}).items()
    }
    missing_probabilities = [code for code in codes if code not in probabilities]
    if missing_probabilities:
        raise ValueError(
            "Effective-sample plotting requires task_probabilities for every code; "
            f"missing: {', '.join(map(str, missing_probabilities))}."
        )

    resolved_batch_size = (
        float(batch_size)
        if batch_size is not None
        else _batch_size(results, config)
    )
    node_depths = {
        int(code): int(depth)
        for code, depth in results.get("node_depths", {}).items()
    }
    missing_depths = [code for code in codes if code not in node_depths]
    if missing_depths:
        raise ValueError(
            "Effective-sample plotting requires node_depths for every code; "
            f"missing: {', '.join(map(str, missing_depths))}."
        )

    axes = _exact_effective_sample_axes(results, codes, eval_steps)
    if axes is None:
        axes = {
            code: eval_steps * resolved_batch_size * probabilities[code]
            for code in codes
        }
    return EffectiveSampleData(
        codes=codes,
        eval_steps=eval_steps,
        probabilities=probabilities,
        batch_size=resolved_batch_size,
        node_depths=node_depths,
        axes=axes,
    )


def _batch_size(results: dict[str, Any], config: dict[str, Any]) -> float:
    if config.get("batch_size") is not None:
        return float(config["batch_size"])
    training_samples = results.get("training_samples")
    if training_samples:
        values = {float(value) for value in training_samples}
        if len(values) == 1:
            return values.pop()
    raise ValueError(
        "Effective-sample plotting requires batch_size in config metadata or "
        "a constant training_samples history."
    )


def _exact_effective_sample_axes(
    results: dict[str, Any],
    codes: list[int],
    eval_steps: np.ndarray,
) -> dict[int, np.ndarray] | None:
    stored = results.get("effective_samples")
    if not isinstance(stored, dict):
        return None
    axes: dict[int, np.ndarray] = {}
    for code in codes:
        values = stored.get(code, stored.get(str(code)))
        if values is None:
            return None
        axis = np.asarray(values, dtype=float)
        if len(axis) != len(eval_steps):
            return None
        axes[code] = axis
    return axes
