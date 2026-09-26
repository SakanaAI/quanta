from __future__ import annotations

import os

import numpy as np


NATS_TO_BITS = np.log2(np.e)
LEARNED_THRESHOLD_BITS_ENV = "LEARNED_QUANTA_THRESHOLD"
UNLEARNED_THRESHOLD_BITS_ENV = "UNLEARNED_QUANTA_THRESHOLD"


def learned_threshold_bits(default: float = 0.05) -> float:
    value = float(os.environ.get(LEARNED_THRESHOLD_BITS_ENV, default))
    if value <= 0:
        raise ValueError(f"{LEARNED_THRESHOLD_BITS_ENV} must be positive.")
    return value


def loss_nats_to_bits(loss):
    return np.asarray(loss, dtype=float) * NATS_TO_BITS


def random_prediction_loss_bits(num_classes: int = 2) -> float:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")
    return float(np.log2(num_classes))


def sliding_min(values, window: int):
    if window <= 0:
        raise ValueError("window must be positive.")
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = np.min(values[start : index + 1])
    return result


def is_finite_learning_step(step) -> bool:
    return step is not None and bool(np.isfinite(step))
