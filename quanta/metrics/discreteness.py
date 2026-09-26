from __future__ import annotations

import os

import numpy as np

from quanta.metrics.utils import (
    UNLEARNED_THRESHOLD_BITS_ENV,
    learned_threshold_bits,
    loss_nats_to_bits,
    random_prediction_loss_bits,
)


def compute_discreteness_transition_error(
    subtask_losses,
    codes,
    unlearned_quanta_threshold=None,
    learned_quanta_threshold=None,
) -> dict[str, float | int]:
    if len(subtask_losses) != len(codes):
        raise ValueError("subtask_losses and codes must have the same length.")
    if not codes:
        return _empty_metrics()

    unlearned = (
        float(os.environ.get(UNLEARNED_THRESHOLD_BITS_ENV, random_prediction_loss_bits()))
        if unlearned_quanta_threshold is None
        else float(unlearned_quanta_threshold)
    )
    learned = (
        learned_threshold_bits()
        if learned_quanta_threshold is None
        else float(learned_quanta_threshold)
    )
    if learned < 0 or unlearned <= learned:
        raise ValueError("thresholds must satisfy 0 <= learned < unlearned.")

    transitions = [
        _transition_metrics(loss_nats_to_bits(curve), unlearned, learned)
        for curve in subtask_losses
    ]
    transitions = [transition for transition in transitions if transition is not None]
    if not transitions:
        metrics = _empty_metrics()
        metrics["total_candidates"] = len(codes)
        return metrics

    steps, step_errors, area_errors = map(np.asarray, zip(*transitions))
    return {
        "dte_steps": float(step_errors.mean()),
        "dte_area": float(area_errors.mean()),
        "total_quanta": len(transitions),
        "candidate_quanta": len(transitions),
        "total_candidates": len(codes),
        "candidate_quanta_fraction": len(transitions) / len(codes),
        "avg_transition_steps": float(steps.mean()),
        "min_transition_steps": int(steps.min()),
        "max_transition_steps": int(steps.max()),
    }


def _transition_metrics(curve, unlearned: float, learned: float):
    learned_indices = np.flatnonzero(curve < learned)
    if not len(learned_indices):
        return None
    end = int(learned_indices[0])
    unlearned_indices = np.flatnonzero(curve[:end] > unlearned)
    if not len(unlearned_indices):
        return None
    start = int(unlearned_indices[-1])
    steps = end - start
    return steps, steps / end, _distribution_weighted_error(curve, start, end)


def _distribution_weighted_error(curve, start: int, end: int) -> float:
    random_loss = random_prediction_loss_bits()
    bounded = np.clip(curve, 0.0, random_loss)
    target = np.where(np.arange(len(curve)) < (start + end) / 2, random_loss, 0.0)
    width = len(curve) - 1
    if width <= 0:
        return 0.0
    return float(np.trapezoid(np.abs(bounded - target) / random_loss) / width)


def _empty_metrics() -> dict[str, float | int]:
    return {
        "dte_steps": 0.0,
        "dte_area": 0.0,
        "total_quanta": 0,
        "candidate_quanta": 0,
        "total_candidates": 0,
        "candidate_quanta_fraction": 0.0,
        "avg_transition_steps": 0.0,
        "min_transition_steps": 0,
        "max_transition_steps": 0,
    }
