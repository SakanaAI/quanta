from __future__ import annotations

import torch
import torch.nn.functional as F

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.batch_common import (
    CNAND_TOKEN_OUT,
    IGNORE_INDEX,
)


def evaluate_cnand_out_values(
    *,
    batch_cache: dict[str, torch.Tensor],
    local_bits: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(local_bits.shape[0])
    n_nodes = int(local_bits.shape[1])
    values = torch.zeros((batch_size, n_nodes), dtype=torch.long, device=local_bits.device)
    parent_indices = batch_cache["parent_indices"]
    parent_masks = batch_cache["parent_masks"]
    node_depths = batch_cache["node_depths"]
    if "node_functions" in batch_cache and batch_cache["node_functions"].numel() > 0:
        max_local_bits = local_bits.shape[2]
        powers = 2 ** torch.arange(max_local_bits, device=local_bits.device)
        bit_indices = (local_bits * powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        expanded_functions = batch_cache["node_functions"].unsqueeze(0).expand(batch_size, -1, -1)
        local_signal = expanded_functions.gather(2, bit_indices.unsqueeze(2)).squeeze(2)
    else:
        local_signal = (~torch.where(
            batch_cache["local_bit_masks"].unsqueeze(0),
            local_bits,
            torch.ones_like(local_bits),
        ).bool().all(dim=2)).to(dtype=torch.long)
    for depth in sorted({int(value) for value in node_depths.detach().cpu().tolist()}):
        nodes = (node_depths == depth).nonzero(as_tuple=False).flatten()
        if not bool(nodes.numel()):
            continue
        node_parent_indices = parent_indices[nodes].clamp_min(0)
        node_parent_masks = parent_masks[nodes]
        has_parents = node_parent_masks.any(dim=1)
        if bool((~has_parents).any().item()):
            base_nodes = nodes[~has_parents]
            values[:, base_nodes] = local_signal[:, base_nodes]
        if not bool(has_parents.any().item()):
            continue
        composite_nodes = nodes[has_parents]
        composite_parent_indices = node_parent_indices[has_parents]
        composite_parent_masks = node_parent_masks[has_parents]
        parent_values = values[:, composite_parent_indices]
        parent_values = torch.where(
            composite_parent_masks.unsqueeze(0),
            parent_values,
            torch.ones_like(parent_values),
        )
        parent_nand = (~parent_values.bool().all(dim=2)).to(dtype=torch.long)
        values[:, composite_nodes] = (~(parent_nand.bool() & local_signal[:, composite_nodes].bool())).to(dtype=torch.long)
    return values







def uses_cnand_tokens(config: ScalingLawsConfig) -> bool:
    return config.task == "cnand"


def uses_masked_token_supervision(config: ScalingLawsConfig) -> bool:
    return uses_cnand_tokens(config)


def masked_token_cross_entropy(logits: torch.Tensor, batch: dict[str, torch.Tensor], *, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["loss_targets"].reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )


def masked_token_loss_statistics(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | int]:
    targets = batch["loss_targets"].reshape(-1)
    token_count = int((targets != IGNORE_INDEX).sum().item())
    if token_count <= 0:
        raise ValueError("Masked-token evaluation requires at least one supervised token.")
    loss_sum_nats = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return {
        "loss_sum_nats": loss_sum_nats,
        "token_count": token_count,
        "mean_loss_nats": loss_sum_nats / token_count,
    }


def masked_token_accuracy(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> float:
    pred = logits.argmax(dim=-1)
    mask = batch["loss_mask"]
    if not bool(mask.any().item()):
        return float("nan")
    return float((pred[mask] == batch["loss_targets"][mask]).float().mean().item())


def active_quantum_out_loss_statistics(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[int, dict[str, float | int]]:
    required = {"input_ids", "active_mask", "quantum_ids", "true_values"}
    if not required.issubset(batch):
        return {}

    output_mask = batch["active_mask"] & (batch["input_ids"] == CNAND_TOKEN_OUT)
    rows, slots = output_mask.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return {}

    quantum_ids = batch["quantum_ids"][rows, slots].to(dtype=torch.long)
    targets = batch["true_values"][rows, quantum_ids].to(dtype=torch.long)
    losses = F.cross_entropy(logits[rows, slots], targets, reduction="none")
    statistics: dict[int, dict[str, float | int]] = {}
    for quantum_id in torch.unique(quantum_ids).detach().cpu().tolist():
        quantum_mask = quantum_ids == int(quantum_id)
        loss_sum = losses[quantum_mask].sum()
        token_count = int(quantum_mask.sum().item())
        statistics[int(quantum_id)] = {
            "loss_sum_nats": float(loss_sum.item()),
            "token_count": token_count,
            "mean_loss_nats": float((loss_sum / token_count).item()),
        }
    return statistics
