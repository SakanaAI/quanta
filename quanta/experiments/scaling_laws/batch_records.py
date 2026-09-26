from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.cnand_batches import (
    build_cnand_batch_from_targets,
    build_sampled_cnand_batch,
)
from quanta.experiments.scaling_laws.cnand_values import uses_cnand_tokens
from quanta.utils import bipolar_parity_labels, bipolar_random_bits


def prediction_sample_records(
    *,
    model: nn.Module,
    config: ScalingLawsConfig,
    task_spec,
    probabilities: torch.Tensor,
    batch_cache: dict[str, torch.Tensor],
    node_depths: dict[int, int],
    device,
    sample_count: int = 2,
) -> list[dict[str, Any]]:
    model.eval()
    sampled_indices = torch.multinomial(probabilities, int(sample_count), replacement=True)
    if config.task == "cnand":
        if config.trace_sampling in {"ideal_threshold", "ideal_path"}:
            x, y = build_sampled_cnand_batch(
                config=config,
                task_spec=task_spec,
                probabilities=probabilities,
                batch_cache=batch_cache,
                device=device,
                batch_size=int(sample_count),
            )
            sampled_indices = x["target_indices"].reshape(-1)
        else:
            x = build_cnand_batch_from_targets(
                config=config,
                target_indices=sampled_indices,
                batch_cache=batch_cache,
                device=device,
            )
            y = x["loss_targets"]
    else:
        bits = bipolar_random_bits((int(sample_count), config.n_bits), dtype=torch.float32, device=device)
        parity_masks = batch_cache["parity_masks"][sampled_indices]
        y = bipolar_parity_labels(bits, parity_masks)
        selector = torch.nn.functional.one_hot(
            sampled_indices,
            num_classes=len(task_spec.codes),
        ).to(dtype=bits.dtype)
        x = {
            "features": torch.cat([selector, bits], dim=1),
            "target_indices": sampled_indices.unsqueeze(1),
        }

    with torch.no_grad():
        logits = model(x)
        if uses_cnand_tokens(config):
            probabilities_pred = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            masked_slots_by_row = [x["loss_mask"][row].nonzero(as_tuple=False).flatten() for row in range(logits.shape[0])]
            product_pred = []
            product_target = []
            true_probabilities = []
            sample_loss_bits = []
            for row, masked_slots in enumerate(masked_slots_by_row):
                row_pred = pred[row, masked_slots]
                row_target = y[row, masked_slots]
                row_prob = probabilities_pred[row, masked_slots, row_target]
                product_pred.append(row_pred)
                product_target.append(row_target)
                true_probabilities.append(row_prob)
                sample_loss_bits.append(
                    -torch.log2(row_prob.clamp_min(torch.finfo(row_prob.dtype).tiny)).mean()
                )
            sample_loss_bits = torch.stack(sample_loss_bits)
        else:
            probabilities_pred = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            product_pred = pred
            product_target = y
            true_probabilities = probabilities_pred.gather(1, y.to(torch.long).reshape(-1, 1)).squeeze(1)
            sample_loss_bits = -torch.log2(true_probabilities.clamp_min(torch.finfo(true_probabilities.dtype).tiny))

    records = []
    for row, task_index in enumerate(sampled_indices.detach().cpu().tolist()):
        code = int(task_spec.codes[int(task_index)])
        if uses_cnand_tokens(config):
            predicted_label = _tensor_to_python_values(product_pred[row])
            true_label = _tensor_to_python_values(product_target[row])
            correct = bool((product_pred[row] == product_target[row]).all().item())
        else:
            predicted_label = int(product_pred[row].detach().cpu().item())
            true_label = _prediction_value(product_target[row])
            correct = bool(predicted_label == true_label)
        record = {
            "task": code,
            "depth": int(node_depths[code]),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": correct,
            "loss_bits": _rounded_scalar(sample_loss_bits[row]),
        }
        if uses_cnand_tokens(config):
            masked_slots = x["loss_mask"][row].nonzero(as_tuple=False).flatten().detach().cpu()
            row_pred = product_pred[row]
            row_target = product_target[row]
            record["masked_slots"] = [int(slot) for slot in masked_slots.tolist()]
            record["masked_quantum_ids"] = _tensor_to_python_values(x["quantum_ids"][row, masked_slots])
            record["masked_depths"] = _tensor_to_python_values(x["depth_ids"][row, masked_slots])
            record["true_label_probability"] = _tensor_to_python_values(true_probabilities[row])
            record["active_nodes"] = int(x["active_mask"][row].sum().detach().cpu().item())
            record["graph_depth"] = int(x["graph_depths"][row].detach().cpu().item())
        else:
            record["true_label_probability"] = _rounded_scalar(true_probabilities[row])
        records.append(record)
    return records




def _prediction_value(tensor: torch.Tensor) -> int | list[int | float]:
    if tensor.ndim == 0:
        return int(tensor.detach().cpu().item())
    return _tensor_to_python_values(tensor)


def _tensor_to_python_values(tensor: torch.Tensor) -> list[int | float]:
    values = tensor.detach().cpu().tolist()
    output = []
    for value in values:
        scalar = float(value)
        if scalar.is_integer():
            output.append(int(scalar))
        else:
            output.append(round(scalar, 6))
    return output


def _rounded_scalar(tensor: torch.Tensor) -> int | float:
    scalar = float(tensor.detach().cpu().item())
    if scalar.is_integer():
        return int(scalar)
    return round(scalar, 6)
