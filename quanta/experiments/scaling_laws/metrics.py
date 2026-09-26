from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch.nn as nn


TAIL_MEDIAN_EVAL_POINTS = 6


def tail_median(
    values: Sequence[float] | None,
    *,
    n_points: int = TAIL_MEDIAN_EVAL_POINTS,
) -> float:
    """Return the temporal median over the final evaluation points."""
    if not values:
        return float("nan")
    tail = np.asarray(values[-max(1, int(n_points)) :], dtype=float)
    finite_tail = tail[np.isfinite(tail)]
    if finite_tail.size == 0:
        return float("nan")
    return float(np.median(finite_tail))


def trainable_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count all trainable parameters and the subset owned by embeddings."""
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
        if parameter.requires_grad
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    n_parameters = sum(parameter.numel() for parameter in parameters)
    n_embedding_parameters = sum(
        parameter.numel()
        for parameter in parameters
        if id(parameter) in embedding_parameter_ids
    )
    return {
        "n_parameters": int(n_parameters),
        "n_embedding_parameters": int(n_embedding_parameters),
        "n_non_embedding_parameters": int(n_parameters - n_embedding_parameters),
    }
