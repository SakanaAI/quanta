from __future__ import annotations

import torch

from quanta.config import ScalingLawsConfig
from quanta.experiments.common import set_optimizer_learning_rate
from quanta.experiments.scaling_laws.cnand_values import (
    masked_token_cross_entropy,
    uses_masked_token_supervision,
)


def _batch_size(batch) -> int:
    if isinstance(batch, dict):
        if "node_raw" in batch:
            return int(batch["node_raw"].shape[0])
        if "features" in batch:
            return int(batch["features"].shape[0])
        return int(batch["input_ids"].shape[0])
    return int(batch.size(0))


def _permute_batch(batch, permutation: torch.Tensor):
    if not isinstance(batch, dict):
        return batch[permutation]
    output = {}
    batch_size = int(permutation.shape[0])
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            output[key] = value[permutation]
        else:
            output[key] = value
    return output


def _local_loss_sum(model, x, y, loss_sum_fn, config: ScalingLawsConfig) -> torch.Tensor:
    logits = model(x)
    if uses_masked_token_supervision(config):
        return masked_token_cross_entropy(logits, x, reduction="sum")
    return loss_sum_fn(logits, y)
