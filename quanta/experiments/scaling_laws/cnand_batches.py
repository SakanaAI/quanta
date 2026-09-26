from __future__ import annotations

import torch
import torch.nn.functional as F

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.batch_common import (
    CNAND_TOKEN_OUT,
    CNAND_TOKEN_PAD,
    IGNORE_INDEX,
    CNAND_TOKEN_SIGN_POS,
    CNAND_TOKEN_SIGN_NEG,
)
from quanta.experiments.scaling_laws.cnand_cache import build_cnand_batch_cache
from quanta.experiments.scaling_laws.cnand_values import (
    evaluate_cnand_out_values,
    masked_token_accuracy,
    masked_token_cross_entropy,
    uses_cnand_tokens,
    uses_masked_token_supervision,
)


def _build_sampled_cnand_batch(
    *,
    config: ScalingLawsConfig,
    probabilities: torch.Tensor,
    batch_cache: dict[str, torch.Tensor],
    batch_size: int,
    device,
    ideal_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if config.task == "multitask_sparse_parity":
        target_indices = torch.multinomial(
            probabilities.unsqueeze(0).expand(int(batch_size), -1),
            1,
            replacement=False,
        ).reshape(-1)
        local_bits = torch.randint(
            0,
            2,
            (int(batch_size), int(config.n_bits)),
            dtype=torch.long,
            device=device,
        ).to(dtype=torch.float32).mul(2).sub(1)
        labels = (local_bits.masked_select(
            batch_cache["parity_masks"][target_indices]
        ).reshape(int(batch_size), int(config.parity_subset_size)).prod(dim=1) < 0).to(dtype=torch.long)
        selector = F.one_hot(target_indices, num_classes=int(probabilities.numel())).to(dtype=local_bits.dtype)
        return {
            "features": torch.cat([selector, local_bits], dim=1),
            "target_indices": target_indices.unsqueeze(1),
        }, labels
    if config.trace_sampling == "ideal_path":
        path_probabilities = batch_cache["path_ideal_probabilities"]
        path_masks = batch_cache["path_ideal_masks"]
        if int(batch_size) == 0:
            ideal_indices = torch.empty((0,), dtype=torch.long, device=device)
            active_node_mask = torch.empty(
                (0, path_masks.shape[1]), dtype=torch.bool, device=device
            )
            target_indices = torch.empty((0, 1), dtype=torch.long, device=device)
        else:
            ideal_indices = torch.multinomial(
                path_probabilities,
                int(batch_size),
                replacement=True,
            )
            active_node_mask = path_masks[ideal_indices]
            target_indices = batch_cache["path_terminal_quantum_ids"][
                ideal_indices
            ].unsqueeze(1)
        batch = build_cnand_batch_from_active_nodes(
            config=config,
            active_node_mask=active_node_mask,
            batch_cache=batch_cache,
            device=device,
            target_indices=target_indices,
        )
        batch["ideal_indices"] = ideal_indices
        return batch
    if config.trace_sampling == "ideal_threshold":
        ideal_probabilities = batch_cache["ideal_probabilities"]
        ideal_masks = batch_cache["ideal_masks"]
        if int(batch_size) == 0:
            active_node_mask = torch.empty(
                (0, ideal_masks.shape[1]), dtype=torch.bool, device=device
            )
            ideal_indices = torch.empty((0,), dtype=torch.long, device=device)
        else:
            if ideal_index is None:
                ideal_index = torch.multinomial(
                    ideal_probabilities,
                    1,
                    replacement=True,
                )
            else:
                ideal_index = ideal_index.reshape(1).to(
                    dtype=torch.long,
                    device=device,
                )
            ideal_indices = ideal_index.expand(int(batch_size))
            active_node_mask = ideal_masks[ideal_index].expand(int(batch_size), -1)
        batch = build_cnand_batch_from_active_nodes(
            config=config,
            active_node_mask=active_node_mask,
            batch_cache=batch_cache,
            device=device,
        )
        batch["ideal_indices"] = ideal_indices
        return batch
    if config.trace_sampling != "principal":
        raise ValueError(
            "trace_sampling must be 'principal', 'ideal_threshold', "
            "or 'ideal_path'."
        )
    if int(batch_size) == 0:
        target_indices = torch.empty((0, 1), dtype=torch.long, device=device)
    else:
        target_indices = torch.multinomial(
            probabilities.unsqueeze(0).expand(int(batch_size), -1),
            1,
            replacement=False,
        )
    return build_cnand_batch_from_targets(
        config=config,
        target_indices=target_indices,
        batch_cache=batch_cache,
        device=device,
    )


def build_cnand_batch_from_targets(
    *,
    config: ScalingLawsConfig,
    target_indices: torch.Tensor,
    batch_cache: dict[str, torch.Tensor],
    device,
    local_bits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if target_indices.ndim == 1:
        target_indices = target_indices.reshape(-1, 1)
    target_indices = target_indices.to(dtype=torch.long, device=device)
    batch_size = int(target_indices.shape[0])
    seq_len = int(batch_cache["slot_ids"].shape[0])
    node_depths = batch_cache["node_depths"]
    if batch_size == 0:
        empty_shape = (0, seq_len)
        return {
            "input_ids": torch.empty(empty_shape, dtype=torch.long, device=device),
            "loss_targets": torch.empty(empty_shape, dtype=torch.long, device=device),
            "active_mask": torch.empty(empty_shape, dtype=torch.bool, device=device),
            "loss_mask": torch.empty(empty_shape, dtype=torch.bool, device=device),
            "slot_ids": torch.empty(empty_shape, dtype=torch.long, device=device),
            "depth_ids": torch.empty(empty_shape, dtype=torch.long, device=device),
            "quantum_ids": torch.empty(empty_shape, dtype=torch.long, device=device),
            "kappa_values": torch.empty(empty_shape, dtype=torch.long, device=device),
            "theoretical_p_q": torch.empty(empty_shape, dtype=torch.float32, device=device),
            "graph_depths": torch.empty((0,), dtype=torch.long, device=device),
            "target_indices": target_indices,
            "target_quantum_ids": target_indices,
            "true_values": torch.empty(empty_shape, dtype=torch.long, device=device),
        }
    closure_masks = batch_cache["closure_masks"]
    target_closure = closure_masks[target_indices].any(dim=1)
    return build_cnand_batch_from_active_nodes(
        config=config,
        active_node_mask=target_closure,
        batch_cache=batch_cache,
        device=device,
        target_indices=target_indices,
        local_bits=local_bits,
    )


def build_cnand_batch_from_active_nodes(
    *,
    config: ScalingLawsConfig,
    active_node_mask: torch.Tensor,
    batch_cache: dict[str, torch.Tensor],
    device,
    target_indices: torch.Tensor | None = None,
    local_bits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if active_node_mask.ndim != 2:
        raise ValueError("active_node_mask must have shape [batch, nodes].")
    active_node_mask = active_node_mask.to(dtype=torch.bool, device=device)
    batch_size, n_nodes = active_node_mask.shape
    if int(n_nodes) != int(batch_cache["node_depths"].numel()):
        raise ValueError("active_node_mask node dimension must match the graph.")
    if batch_size and not bool(active_node_mask.any(dim=1).all().item()):
        raise ValueError("Each cNAND ideal must contain at least one active node.")

    node_depths = batch_cache["node_depths"]
    if target_indices is None:
        depth_scores = torch.where(
            active_node_mask,
            node_depths.unsqueeze(0).expand(batch_size, -1),
            torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=device),
        )
        target_indices = depth_scores.argmax(dim=1, keepdim=True)
    elif target_indices.ndim == 1:
        target_indices = target_indices.reshape(-1, 1)
    target_indices = target_indices.to(dtype=torch.long, device=device)
    graph_depths = torch.where(
        active_node_mask,
        node_depths.unsqueeze(0).expand(batch_size, -1),
        torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=device),
    ).max(dim=1).values

    batch = build_cnand_out_token_batch(
        config=config,
        target_indices=target_indices,
        batch_cache=batch_cache,
        active_node_mask=active_node_mask,
        graph_depths=graph_depths,
        batch_size=batch_size,
        device=device,
        local_bits=local_bits,
    )
    return compact_cnand_active_tokens(batch)


def compact_cnand_active_tokens(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    active_mask = batch["active_mask"]
    if active_mask.ndim != 2 or active_mask.shape[0] == 0:
        return batch
    active_counts = active_mask.to(dtype=torch.long).sum(dim=1)
    max_active = int(active_counts.max().item())
    if max_active == int(active_mask.shape[1]):
        return batch
    if max_active == 0:
        raise ValueError("Cannot compact cNAND batch with no active tokens.")

    batch_size = int(active_mask.shape[0])
    slot_order = torch.arange(active_mask.shape[1], device=active_mask.device).unsqueeze(0).expand_as(active_mask)
    inactive_order = slot_order + int(active_mask.shape[1])
    compact_indices = torch.where(active_mask, slot_order, inactive_order).argsort(dim=1)[:, :max_active]
    row_indices = torch.arange(batch_size, device=active_mask.device).unsqueeze(1)
    compact_active = torch.arange(max_active, device=active_mask.device).unsqueeze(0) < active_counts.unsqueeze(1)
    compacted: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:2] == active_mask.shape:
            gathered = value[row_indices, compact_indices]
            if key == "active_mask":
                gathered = compact_active
            compacted[key] = gathered
        else:
            compacted[key] = value
    return compacted


def build_cnand_out_token_batch(
    *,
    config: ScalingLawsConfig,
    target_indices: torch.Tensor,
    batch_cache: dict[str, torch.Tensor],
    active_node_mask: torch.Tensor,
    graph_depths: torch.Tensor,
    batch_size: int,
    device,
    local_bits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    n_nodes = int(batch_cache["quantum_ids"].shape[0])
    tokens_per_node = int(batch_cache["tokens_per_node"].item())
    seq_len = n_nodes * tokens_per_node
    uses_lut_functions = getattr(config, "n_lut_functions", None) is not None
    local_bit_width = tokens_per_node - 2 if uses_lut_functions else tokens_per_node - 1
    if local_bits is None:
        local_bits = torch.randint(
            0,
            2,
            (int(batch_size), n_nodes, local_bit_width),
            dtype=torch.long,
            device=device,
        )
    else:
        expected_shape = (int(batch_size), n_nodes, local_bit_width)
        if tuple(local_bits.shape) != expected_shape:
            raise ValueError(f"local_bits must have shape {expected_shape}.")
        local_bits = local_bits.to(dtype=torch.long, device=device)
    local_bits = torch.where(
        batch_cache["local_bit_masks"].unsqueeze(0),
        local_bits,
        torch.zeros_like(local_bits),
    )
    values = evaluate_cnand_out_values(batch_cache=batch_cache, local_bits=local_bits)

    input_ids = torch.full((int(batch_size), seq_len), CNAND_TOKEN_PAD, dtype=torch.long, device=device)
    active_mask = active_node_mask.repeat_interleave(tokens_per_node, dim=1)
    local_token_offsets = torch.arange(local_bit_width, dtype=torch.long, device=device)
    local_token_slots = torch.arange(n_nodes, dtype=torch.long, device=device).unsqueeze(1) * tokens_per_node + local_token_offsets
    input_ids[:, local_token_slots.reshape(-1)] = local_bits.reshape(int(batch_size), -1)

    if uses_lut_functions:
        sign_slots = torch.arange(tokens_per_node - 2, n_nodes * tokens_per_node, tokens_per_node, dtype=torch.long, device=device)
        if "node_signs" in batch_cache:
            node_signs = batch_cache["node_signs"].to(device=device)
        else:
            node_funcs = batch_cache["node_functions"]
            _, node_fn_inverse = torch.unique(node_funcs, dim=0, return_inverse=True)
            node_signs = 4 + node_fn_inverse
        input_ids[:, sign_slots] = node_signs.unsqueeze(0).expand(int(batch_size), -1)

    output_slots = batch_cache["output_token_slots"]
    input_ids[:, output_slots] = CNAND_TOKEN_OUT
    input_ids = torch.where(active_mask, input_ids, torch.full_like(input_ids, CNAND_TOKEN_PAD))

    target_slots = output_slots[target_indices]
    loss_mask = torch.zeros((int(batch_size), seq_len), dtype=torch.bool, device=device)
    loss_targets = torch.full((int(batch_size), seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    if config.loss_supervision == "all":
        loss_mask[:, output_slots] = active_node_mask
        loss_targets[:, output_slots] = torch.where(
            active_node_mask,
            values,
            torch.full_like(values, IGNORE_INDEX),
        )
    elif config.loss_supervision == "target_only":
        loss_mask.scatter_(1, target_slots, True)
        loss_targets.scatter_(1, target_slots, values.gather(1, target_indices))
    else:
        raise ValueError("loss_supervision must be 'all' or 'target_only'.")

    token_node_ids = batch_cache["token_node_ids"]
    slot_ids = batch_cache["slot_ids"].unsqueeze(0).expand(int(batch_size), -1)
    return {
        "input_ids": input_ids,
        "loss_targets": loss_targets,
        "active_mask": active_mask,
        "active_node_mask": active_node_mask,
        "loss_mask": loss_mask,
        "slot_ids": slot_ids,
        "depth_ids": batch_cache["node_depths"][token_node_ids].unsqueeze(0).expand(int(batch_size), -1),
        "quantum_ids": batch_cache["quantum_ids"][token_node_ids].unsqueeze(0).expand(int(batch_size), -1),
        "kappa_values": batch_cache["kappa_values"][token_node_ids].unsqueeze(0).expand(int(batch_size), -1),
        "theoretical_p_q": batch_cache["theoretical_p_q"][token_node_ids].unsqueeze(0).expand(int(batch_size), -1),
        "graph_depths": graph_depths,
        "target_indices": target_indices,
        "target_quantum_ids": batch_cache["quantum_ids"][target_indices],
        "target_slots": target_slots,
        "local_bits": local_bits,
        "true_values": values,
    }


def build_sampled_cnand_batch(
    *,
    config,
    task_spec,
    probabilities,
    batch_cache,
    device,
    batch_size = None,
    ideal_index: torch.Tensor | None = None,
):
    batch_size = int(config.batch_size if batch_size is None else batch_size)
    x = _build_sampled_cnand_batch(
        config=config,
        probabilities=probabilities,
        batch_cache=batch_cache,
        batch_size=batch_size,
        device=device,
        ideal_index=ideal_index,
    )
    if config.task == "multitask_sparse_parity":
        return x
    return x, x["loss_targets"]
