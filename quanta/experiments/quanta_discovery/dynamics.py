"""Small shared helpers for the exact-GD discovery and Q-model replay paths."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from quanta.experiments.number_naming.model import DecoderTransformerLM

from .qgraph import QuantumGraphEdge, validate_quantum_graph


EPSILON = 1.0e-12


def prediction_event_offsets(texts: Sequence[str]) -> np.ndarray:
    """Return stable offsets for each example's target-word and EOS events."""

    counts = np.fromiter(
        (len(str(text).split()) + 1 for text in texts),
        dtype=np.int64,
        count=len(texts),
    )
    if len(counts) == 0 or np.any(counts <= 0):
        raise ValueError("training texts must contain prediction events")
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))


def batch_prediction_event_ids(
    example_indices: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Expand example IDs into row-major global prediction-event IDs."""

    indices = np.asarray(example_indices, dtype=np.int64)
    boundaries = np.asarray(offsets, dtype=np.int64)
    if indices.ndim != 1 or boundaries.ndim != 1 or len(boundaries) < 2:
        raise ValueError("indices must be 1D and offsets must contain boundaries")
    if len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    if int(indices.min()) < 0 or int(indices.max()) >= len(boundaries) - 1:
        raise ValueError("example index is outside the event-offset table")
    starts = boundaries[indices]
    counts = boundaries[indices + 1] - starts
    repeated_starts = np.repeat(starts, counts)
    row_starts = np.repeat(np.cumsum(counts) - counts, counts)
    return repeated_starts + np.arange(int(counts.sum()), dtype=np.int64) - row_starts


def block_parameters_by_layer(
    model: DecoderTransformerLM,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Return all parameters owned by each complete transformer residual block."""

    return tuple(tuple(layer.parameters()) for layer in model.layers)


def flattened_block_parameters(model: DecoderTransformerLM) -> torch.Tensor:
    return torch.stack(
        [
            torch.cat([parameter.flatten() for parameter in parameters])
            for parameters in block_parameters_by_layer(model)
        ]
    )


def flattened_mlp_parameters(model: DecoderTransformerLM) -> torch.Tensor:
    """Return the MLP-only vectors used for writer-complexity diagnostics."""

    return torch.stack(
        [
            torch.cat(
                [
                    layer.mlp_input.weight.flatten(),
                    layer.mlp_input.bias.flatten(),
                    layer.mlp_output.weight.flatten(),
                    layer.mlp_output.bias.flatten(),
                ]
            )
            for layer in model.layers
        ]
    )


def cumulative_existence_at_step(
    temporal_priority: np.ndarray, *, step: int, total_steps: int
) -> np.ndarray:
    """Interpolate bin priority into an existence value at one optimizer step."""

    curves = np.maximum(np.asarray(temporal_priority, dtype=np.float64), 0.0)
    if curves.ndim != 2 or curves.shape[1] <= 0:
        raise ValueError("temporal_priority must have shape [quantum, interval]")
    if not 0 <= int(step) <= int(total_steps) or int(total_steps) <= 0:
        raise ValueError("step must lie in [0, total_steps]")
    boundaries = np.linspace(0.0, float(total_steps), curves.shape[1] + 1)
    completed = min(
        int(np.searchsorted(boundaries[1:], float(step), side="right")),
        curves.shape[1],
    )
    acquired = curves[:, :completed].sum(axis=1)
    if completed < curves.shape[1] and step > boundaries[completed]:
        fraction = (step - boundaries[completed]) / (
            boundaries[completed + 1] - boundaries[completed]
        )
        acquired = acquired + float(fraction) * curves[:, completed]
    total = curves.sum(axis=1)
    return np.divide(
        acquired, total, out=np.zeros_like(acquired), where=total > EPSILON
    ).astype(np.float32)


def close_existence(
    existences: Sequence[np.ndarray], edges: Sequence[QuantumGraphEdge]
) -> tuple[np.ndarray, ...]:
    """Apply exact temporal downward closure, ``existence_child <= existence_parent``."""

    values = [np.asarray(value, dtype=np.float32).copy() for value in existences]
    canonical = validate_quantum_graph(tuple(len(value) for value in values), edges)
    for parent_layer, parent, child_layer, child in reversed(canonical):
        values[parent_layer][parent] = max(
            float(values[parent_layer][parent]),
            float(values[child_layer][child]),
        )
    return tuple(values)
