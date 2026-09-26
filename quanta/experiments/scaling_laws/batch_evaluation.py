from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather_object

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.cnand_batches import (
    build_cnand_batch_from_active_nodes,
    build_cnand_batch_from_targets,
)
from quanta.experiments.scaling_laws.cnand_values import (
    active_quantum_out_loss_statistics,
    masked_token_accuracy,
    masked_token_loss_statistics,
    uses_masked_token_supervision,
)
from quanta.experiments.scaling_laws.cnand_cache import build_seeded_cnand_batch_cache
from quanta.metrics import loss_nats_to_bits
from quanta.utils import bipolar_parity_labels, bipolar_random_bits


def evaluate_weighted_task_loss(
    *,
    model: nn.Module,
    loss_fn,
    config: ScalingLawsConfig,
    task_spec,
    probabilities: torch.Tensor,
    node_depths: dict[int, int],
    batch_cache: dict[str, torch.Tensor] | None = None,
    device,
    accelerator: Any | None = None,
) -> dict[str, Any]:
    model.eval()
    if batch_cache is None:
        batch_cache = build_seeded_cnand_batch_cache(config, task_spec, device)
    if config.trace_sampling == "ideal_path":
        return _evaluate_weighted_path_loss(
            model=model,
            config=config,
            task_spec=task_spec,
            probabilities=probabilities,
            node_depths=node_depths,
            batch_cache=batch_cache,
            device=device,
            accelerator=accelerator,
        )
    if config.trace_sampling == "ideal_threshold":
        return _evaluate_weighted_ideal_loss(
            model=model,
            config=config,
            task_spec=task_spec,
            node_depths=node_depths,
            batch_cache=batch_cache,
            device=device,
            accelerator=accelerator,
        )

    local_results = []
    evaluation_config = config
    process_index = int(accelerator.process_index) if accelerator is not None else 0
    num_processes = int(accelerator.num_processes) if accelerator is not None else 1
    with torch.no_grad():
        for index in range(process_index, len(task_spec.codes), num_processes):
            code = task_spec.codes[index]
            x, y = cached_cnand_eval_batch(
                config=evaluation_config,
                task_index=index,
                batch_cache=batch_cache,
                device=device,
            )
            with accelerator.autocast() if accelerator is not None else nullcontext():
                pred = model(x)
            quantum_context_losses = active_quantum_out_loss_statistics(pred, x)
            if uses_masked_token_supervision(evaluation_config):
                loss_statistics = masked_token_loss_statistics(pred, x)
                result = {
                    "loss_sum_nats": float(loss_statistics["loss_sum_nats"].item()),
                    "token_count": int(loss_statistics["token_count"]),
                    "mean_loss_nats": float(loss_statistics["mean_loss_nats"].item()),
                    "accuracy": masked_token_accuracy(pred, x),
                }
            else:
                mean_loss_nats = float(loss_fn(pred, y).item())
                token_count = int(y.numel())
                result = {
                    "loss_sum_nats": mean_loss_nats * token_count,
                    "token_count": token_count,
                    "mean_loss_nats": mean_loss_nats,
                    "accuracy": float((pred.argmax(dim=-1).squeeze() == y.squeeze()).float().mean().item()),
                }
            local_results.append(
                {
                    "index": index,
                    "code": int(code),
                    "quantum_context_losses": quantum_context_losses,
                    **result,
                }
            )

    gathered_results = (
        gather_object(local_results)
        if accelerator is not None and num_processes > 1
        else local_results
    )
    ordered_results = sorted(gathered_results, key=lambda result: result["index"])
    if len(ordered_results) != len(task_spec.codes):
        raise RuntimeError(
            f"Expected evaluation results for {len(task_spec.codes)} tasks, got {len(ordered_results)}."
        )
    task_results = ordered_results
    accuracies = [result["accuracy"] for result in task_results]

    loss_sums = torch.tensor(
        [result["loss_sum_nats"] for result in task_results],
        dtype=torch.float64,
    )
    token_counts = torch.tensor(
        [result["token_count"] for result in task_results],
        dtype=torch.float64,
    )
    mean_losses = torch.tensor(
        [result["mean_loss_nats"] for result in task_results],
        dtype=torch.float64,
    )
    accuracy_tensor = torch.tensor(accuracies, dtype=torch.float64)
    probability_cpu = probabilities.detach().cpu().to(dtype=torch.float64)
    weighted_token_count = float((probability_cpu * token_counts).sum().item())
    if weighted_token_count <= 0:
        raise ValueError("Weighted evaluation requires a positive supervised-token count.")
    weighted_token_loss_nats = float(
        (probability_cpu * loss_sums).sum().item() / weighted_token_count
    )
    weighted_token_loss_bits = float(loss_nats_to_bits(weighted_token_loss_nats))
    task_weighted_loss_nats = float((probability_cpu * mean_losses).sum().item())
    task_weighted_loss_bits = float(loss_nats_to_bits(task_weighted_loss_nats))
    if config.eval_loss_formula == "quanta_weighted":
        eval_loss_nats = weighted_token_loss_nats
        eval_loss_bits = weighted_token_loss_bits
    elif config.eval_loss_formula == "task_weighted":
        eval_loss_nats = task_weighted_loss_nats
        eval_loss_bits = task_weighted_loss_bits
    else:
        raise ValueError(
            "eval_loss_formula must be 'quanta_weighted' or 'task_weighted'."
        )
    mean_task_loss_nats = float(mean_losses.mean().item())

    depth_values = sorted({int(node_depths[int(code)]) for code in task_spec.codes})
    mean_depth_loss = {}
    mean_depth_accuracy = {}
    weighted_depth_loss = {}
    weighted_depth_accuracy = {}
    weighted_depth_probability = {}
    task_losses = {}
    quantum_loss_numerators = {int(code): 0.0 for code in task_spec.codes}
    quantum_context_probabilities = {int(code): 0.0 for code in task_spec.codes}
    quantum_mean_numerators = {int(code): 0.0 for code in task_spec.codes}
    quantum_context_counts = {int(code): 0 for code in task_spec.codes}
    for index, code in enumerate(task_spec.codes):
        result = task_results[index]
        task_probability = float(probability_cpu[index].item())
        task_losses[int(code)] = {
            "loss_sum_nats": float(result["loss_sum_nats"]),
            "token_count": int(result["token_count"]),
            "mean_loss_nats": float(result["mean_loss_nats"]),
            "mean_loss_bits": float(loss_nats_to_bits(result["mean_loss_nats"])),
            "loss_nats": float(result["mean_loss_nats"]),
            "loss_bits": float(loss_nats_to_bits(result["mean_loss_nats"])),
            "accuracy": float(accuracies[index]),
            "probability": task_probability,
            "depth": int(node_depths[int(code)]),
        }
        for quantum_id, context_loss in result.get("quantum_context_losses", {}).items():
            quantum_id = int(quantum_id)
            quantum_loss_numerators[quantum_id] += (
                task_probability * float(context_loss["mean_loss_nats"])
            )
            quantum_context_probabilities[quantum_id] += task_probability
            quantum_mean_numerators[quantum_id] += float(
                context_loss["mean_loss_nats"]
            )
            quantum_context_counts[quantum_id] += 1
    for depth in depth_values:
        indices = [index for index, code in enumerate(task_spec.codes) if int(node_depths[int(code)]) == depth]
        depth_losses = mean_losses[indices]
        depth_accuracies = accuracy_tensor[indices]
        depth_probs = probability_cpu[indices]
        depth_probability = float(depth_probs.sum().item())
        mean_depth_loss[depth] = {
            "loss_nats": float(depth_losses.mean().item()),
            "loss_bits": float(loss_nats_to_bits(depth_losses.mean().item())),
        }
        mean_depth_accuracy[depth] = float(depth_accuracies.mean().item())
        weighted_depth_loss[depth] = {
            "loss_nats": float((depth_probs * depth_losses).sum().item()),
            "loss_bits": float(loss_nats_to_bits((depth_probs * depth_losses).sum().item())),
        }
        weighted_depth_accuracy[depth] = (
            float((depth_probs * depth_accuracies).sum().item() / depth_probability)
            if depth_probability > 0
            else float("nan")
        )
        weighted_depth_probability[depth] = depth_probability

    quantum_losses = {}
    quantum_mean_losses = {}
    for code in task_spec.codes:
        quantum_id = int(code)
        context_probability = quantum_context_probabilities[quantum_id]
        if context_probability > 0:
            loss_nats = quantum_loss_numerators[quantum_id] / context_probability
            quantum_losses[quantum_id] = {
                "loss_nats": float(loss_nats),
                "loss_bits": float(loss_nats_to_bits(loss_nats)),
                "context_probability": float(context_probability),
            }
        context_count = quantum_context_counts[quantum_id]
        if context_count <= 0:
            continue
        quantum_mean_loss_nats = (
            quantum_mean_numerators[quantum_id] / context_count
        )
        quantum_mean_losses[quantum_id] = {
            "loss_nats": float(quantum_mean_loss_nats),
            "loss_bits": float(loss_nats_to_bits(quantum_mean_loss_nats)),
            "context_count": int(context_count),
        }

    diagnostics = {
        "weighted_token_loss_nats": weighted_token_loss_nats,
        "weighted_token_loss_bits": weighted_token_loss_bits,
        "weighted_token_count": weighted_token_count,
        "quanta_weighted_loss_nats": weighted_token_loss_nats,
        "quanta_weighted_loss_bits": weighted_token_loss_bits,
        "task_weighted_loss_nats": task_weighted_loss_nats,
        "task_weighted_loss_bits": task_weighted_loss_bits,
        "eval_loss_formula": config.eval_loss_formula,
        "loss_supervision": config.loss_supervision,
        "eval_loss_nats": eval_loss_nats,
        "eval_loss_bits": eval_loss_bits,
        "total_loss_nats": eval_loss_nats,
        "total_loss_bits": eval_loss_bits,
        "weighted_accuracy": float((probability_cpu * accuracy_tensor).sum().item()),
        "mean_task_loss_nats": mean_task_loss_nats,
        "mean_task_loss_bits": float(loss_nats_to_bits(mean_task_loss_nats)),
        "mean_task_accuracy": float(accuracy_tensor.mean().item()),
        "mean_depth_loss": mean_depth_loss,
        "mean_depth_accuracy": mean_depth_accuracy,
        "weighted_depth_loss": weighted_depth_loss,
        "weighted_depth_accuracy": weighted_depth_accuracy,
        "weighted_depth_probability": weighted_depth_probability,
        "task_losses": task_losses,
        "quantum_losses": quantum_losses,
        "quantum_mean_losses": quantum_mean_losses,
    }
    return diagnostics


def _evaluate_weighted_path_loss(
    *,
    model: nn.Module,
    config: ScalingLawsConfig,
    task_spec,
    probabilities: torch.Tensor,
    node_depths: dict[int, int],
    batch_cache: dict[str, torch.Tensor],
    device,
    accelerator: Any | None,
) -> dict[str, Any]:
    """Evaluate every path quantum on its causal closure."""
    evaluation_config = copy.copy(config)
    evaluation_config.loss_supervision = "target_only"
    process_index = int(accelerator.process_index) if accelerator is not None else 0
    num_processes = int(accelerator.num_processes) if accelerator is not None else 1
    sample_count = int(config.eval_samples_per_task)
    local_node_indices = list(range(process_index, len(task_spec.codes), num_processes))
    targets = torch.tensor(
        [index for index in local_node_indices for _ in range(sample_count)],
        dtype=torch.long,
        device=device,
    )
    evaluation_batch_size = min(max(1, int(config.batch_size)), 256)
    loss_sums = {index: 0.0 for index in local_node_indices}
    correct_counts = {index: 0 for index in local_node_indices}
    counts = {index: 0 for index in local_node_indices}

    with torch.no_grad():
        for start in range(0, int(targets.numel()), evaluation_batch_size):
            chunk_targets = targets[start : start + evaluation_batch_size]
            batch = build_cnand_batch_from_targets(
                config=evaluation_config,
                target_indices=chunk_targets,
                batch_cache=batch_cache,
                device=device,
            )
            with accelerator.autocast() if accelerator is not None else nullcontext():
                logits = model(batch)
            rows, slots = batch["loss_mask"].nonzero(as_tuple=True)
            row_targets = batch["loss_targets"][rows, slots]
            row_losses = F.cross_entropy(
                logits[rows, slots], row_targets, reduction="none"
            )
            row_correct = logits[rows, slots].argmax(dim=-1).eq(row_targets)
            for node_index in torch.unique(chunk_targets).detach().cpu().tolist():
                node_mask = chunk_targets == int(node_index)
                loss_sums[int(node_index)] += float(row_losses[node_mask].sum().item())
                correct_counts[int(node_index)] += int(row_correct[node_mask].sum().item())
                counts[int(node_index)] += int(node_mask.sum().item())

    local_results = [
        {
            "index": int(index),
            "loss_sum_nats": float(loss_sums[index]),
            "correct_count": int(correct_counts[index]),
            "count": int(counts[index]),
        }
        for index in local_node_indices
    ]
    gathered = (
        gather_object(local_results)
        if accelerator is not None and num_processes > 1
        else local_results
    )
    results = sorted(gathered, key=lambda result: result["index"])
    if len(results) != len(task_spec.codes):
        raise RuntimeError(
            f"Expected ideal-path evaluation for {len(task_spec.codes)} quanta, got {len(results)}."
        )

    probability_cpu = probabilities.detach().cpu().to(dtype=torch.float64)
    mean_losses = torch.tensor(
        [result["loss_sum_nats"] / result["count"] for result in results],
        dtype=torch.float64,
    )
    accuracies = torch.tensor(
        [result["correct_count"] / result["count"] for result in results],
        dtype=torch.float64,
    )
    weighted_loss_nats = float((probability_cpu * mean_losses).sum().item())
    weighted_accuracy = float((probability_cpu * accuracies).sum().item())
    mean_task_loss_nats = float(mean_losses.mean().item())
    task_losses: dict[int, dict[str, float | int]] = {}
    quantum_losses: dict[int, dict[str, float]] = {}
    quantum_mean_losses: dict[int, dict[str, float | int]] = {}
    for index, code in enumerate(task_spec.codes):
        loss_nats = float(mean_losses[index].item())
        loss_bits = float(loss_nats_to_bits(loss_nats))
        task_losses[int(code)] = {
            "loss_sum_nats": float(results[index]["loss_sum_nats"]),
            "token_count": int(results[index]["count"]),
            "mean_loss_nats": loss_nats,
            "mean_loss_bits": loss_bits,
            "loss_nats": loss_nats,
            "loss_bits": loss_bits,
            "accuracy": float(accuracies[index].item()),
            "probability": float(probability_cpu[index].item()),
            "depth": int(node_depths[int(code)]),
        }
        quantum_losses[int(code)] = {
            "loss_nats": loss_nats,
            "loss_bits": loss_bits,
            "context_probability": float(probability_cpu[index].item()),
        }
        quantum_mean_losses[int(code)] = {
            "loss_nats": loss_nats,
            "loss_bits": loss_bits,
            "context_count": int(results[index]["count"]),
        }

    mean_depth_loss: dict[int, dict[str, float]] = {}
    mean_depth_accuracy: dict[int, float] = {}
    weighted_depth_loss: dict[int, dict[str, float]] = {}
    weighted_depth_accuracy: dict[int, float] = {}
    weighted_depth_probability: dict[int, float] = {}
    for depth in sorted(set(node_depths.values())):
        indices = [
            index
            for index, code in enumerate(task_spec.codes)
            if int(node_depths[int(code)]) == int(depth)
        ]
        depth_losses = mean_losses[indices]
        depth_accuracies = accuracies[indices]
        depth_probabilities = probability_cpu[indices]
        depth_probability = float(depth_probabilities.sum().item())
        mean_depth_loss[int(depth)] = {
            "loss_nats": float(depth_losses.mean().item()),
            "loss_bits": float(loss_nats_to_bits(depth_losses.mean().item())),
        }
        mean_depth_accuracy[int(depth)] = float(depth_accuracies.mean().item())
        weighted_loss = float((depth_probabilities * depth_losses).sum().item())
        weighted_depth_loss[int(depth)] = {
            "loss_nats": weighted_loss,
            "loss_bits": float(loss_nats_to_bits(weighted_loss)),
        }
        weighted_depth_accuracy[int(depth)] = float(
            (depth_probabilities * depth_accuracies).sum().item() / depth_probability
        )
        weighted_depth_probability[int(depth)] = depth_probability

    weighted_loss_bits = float(loss_nats_to_bits(weighted_loss_nats))
    return {
        "weighted_token_loss_nats": weighted_loss_nats,
        "weighted_token_loss_bits": weighted_loss_bits,
        "weighted_token_count": 1.0,
        "quanta_weighted_loss_nats": weighted_loss_nats,
        "quanta_weighted_loss_bits": weighted_loss_bits,
        "task_weighted_loss_nats": weighted_loss_nats,
        "task_weighted_loss_bits": weighted_loss_bits,
        "eval_loss_formula": config.eval_loss_formula,
        "loss_supervision": config.loss_supervision,
        "eval_loss_nats": weighted_loss_nats,
        "eval_loss_bits": weighted_loss_bits,
        "total_loss_nats": weighted_loss_nats,
        "total_loss_bits": weighted_loss_bits,
        "weighted_accuracy": weighted_accuracy,
        "mean_task_loss_nats": mean_task_loss_nats,
        "mean_task_loss_bits": float(loss_nats_to_bits(mean_task_loss_nats)),
        "mean_task_accuracy": float(accuracies.mean().item()),
        "mean_depth_loss": mean_depth_loss,
        "mean_depth_accuracy": mean_depth_accuracy,
        "weighted_depth_loss": weighted_depth_loss,
        "weighted_depth_accuracy": weighted_depth_accuracy,
        "weighted_depth_probability": weighted_depth_probability,
        "task_losses": task_losses,
        "quantum_losses": quantum_losses,
        "quantum_mean_losses": quantum_mean_losses,
    }


def _evaluate_weighted_ideal_loss(
    *,
    model: nn.Module,
    config: ScalingLawsConfig,
    task_spec,
    node_depths: dict[int, int],
    batch_cache: dict[str, torch.Tensor],
    device,
    accelerator: Any | None,
) -> dict[str, Any]:
    ideal_masks = batch_cache["ideal_masks"]
    ideal_probabilities = batch_cache["ideal_probabilities"]
    if ideal_masks.numel() == 0 or ideal_probabilities.numel() == 0:
        raise ValueError("ideal_threshold evaluation requires a non-empty ideal mixture.")
    process_index = int(accelerator.process_index) if accelerator is not None else 0
    num_processes = int(accelerator.num_processes) if accelerator is not None else 1
    sample_count = int(config.eval_samples_per_task)
    # Evaluation follows the same physical-batch limit as training.  In
    # particular, an otherwise memory-safe accumulated training run must not
    # OOM before its first update by evaluating all samples for an ideal at
    # once.
    evaluation_batch_size = min(
        sample_count,
        int(config.batch_size) // int(config.gradient_accumulation_steps),
    )
    local_results = []
    with torch.no_grad():
        for ideal_index in range(process_index, int(ideal_masks.shape[0]), num_processes):
            loss_sum_nats = 0.0
            token_count = 0
            correct_count = 0.0
            quantum_loss_sums: dict[int, float] = {}
            quantum_token_counts: dict[int, int] = {}
            quantum_correct_counts: dict[int, float] = {}
            for start in range(0, sample_count, evaluation_batch_size):
                chunk_size = min(evaluation_batch_size, sample_count - start)
                active_node_mask = ideal_masks[ideal_index].unsqueeze(0).expand(chunk_size, -1)
                batch = build_cnand_batch_from_active_nodes(
                    config=config,
                    active_node_mask=active_node_mask,
                    batch_cache=batch_cache,
                    device=device,
                )
                with accelerator.autocast() if accelerator is not None else nullcontext():
                    logits = model(batch)
                statistics = masked_token_loss_statistics(logits, batch)
                chunk_token_count = int(statistics["token_count"])
                loss_sum_nats += float(statistics["loss_sum_nats"].item())
                token_count += chunk_token_count
                correct_count += float(masked_token_accuracy(logits, batch)) * chunk_token_count
                chunk_quantum_losses = active_quantum_out_loss_statistics(logits, batch)
                for quantum_id, context_loss in chunk_quantum_losses.items():
                    quantum_id = int(quantum_id)
                    quantum_loss_sums[quantum_id] = quantum_loss_sums.get(quantum_id, 0.0) + float(
                        context_loss["loss_sum_nats"]
                    )
                    quantum_token_counts[quantum_id] = quantum_token_counts.get(quantum_id, 0) + int(
                        context_loss["token_count"]
                    )
                for quantum_id, accuracy in _active_quantum_accuracies(logits, batch).items():
                    quantum_id = int(quantum_id)
                    quantum_correct_counts[quantum_id] = quantum_correct_counts.get(quantum_id, 0.0) + float(
                        accuracy
                    ) * int(chunk_quantum_losses[quantum_id]["token_count"])

            quantum_context_losses = {
                quantum_id: {
                    "loss_sum_nats": loss_sum,
                    "token_count": quantum_token_counts[quantum_id],
                    "mean_loss_nats": loss_sum / quantum_token_counts[quantum_id],
                }
                for quantum_id, loss_sum in quantum_loss_sums.items()
            }
            quantum_context_accuracies = {
                quantum_id: quantum_correct_counts[quantum_id] / quantum_token_counts[quantum_id]
                for quantum_id in quantum_token_counts
            }
            local_results.append(
                {
                    "index": ideal_index,
                    "loss_sum_nats": loss_sum_nats,
                    "token_count": token_count,
                    "mean_loss_nats": loss_sum_nats / token_count,
                    "correct_count": correct_count,
                    "quantum_context_losses": quantum_context_losses,
                    "quantum_context_accuracies": quantum_context_accuracies,
                }
            )

    gathered = (
        gather_object(local_results)
        if accelerator is not None and num_processes > 1
        else local_results
    )
    results = sorted(gathered, key=lambda result: result["index"])
    if len(results) != int(ideal_masks.shape[0]):
        raise RuntimeError(
            f"Expected {int(ideal_masks.shape[0])} ideal evaluation results, got {len(results)}."
        )
    weights = ideal_probabilities.detach().cpu().to(dtype=torch.float64)
    per_sample_loss_sums = torch.tensor(
        [result["loss_sum_nats"] / sample_count for result in results], dtype=torch.float64
    )
    per_sample_token_counts = torch.tensor(
        [result["token_count"] / sample_count for result in results], dtype=torch.float64
    )
    per_sample_correct_counts = torch.tensor(
        [result["correct_count"] / sample_count for result in results], dtype=torch.float64
    )
    mean_context_losses = torch.tensor(
        [result["mean_loss_nats"] for result in results], dtype=torch.float64
    )
    expected_tokens = float((weights * per_sample_token_counts).sum().item())
    if expected_tokens <= 0:
        raise ValueError("ideal evaluation requires positive expected supervised-token count.")
    quanta_weighted_loss_nats = float(
        (weights * per_sample_loss_sums).sum().item() / expected_tokens
    )
    task_weighted_loss_nats = float((weights * mean_context_losses).sum().item())
    weighted_accuracy = float(
        (weights * per_sample_correct_counts).sum().item() / expected_tokens
    )

    normalized_marginals = batch_cache["theoretical_p_q"].detach().cpu().to(dtype=torch.float64)
    normalized_marginals /= normalized_marginals.sum()
    task_losses: dict[int, dict[str, float | int]] = {}
    quantum_losses: dict[int, dict[str, float]] = {}
    quantum_mean_losses: dict[int, dict[str, float | int]] = {}
    for node_index, code in enumerate(task_spec.codes):
        quantum_id = int(code)
        weighted_loss_numerator = 0.0
        weighted_accuracy_numerator = 0.0
        context_probability = 0.0
        context_losses = []
        for ideal_index, result in enumerate(results):
            context_loss = result["quantum_context_losses"].get(quantum_id)
            if context_loss is None:
                continue
            context_weight = float(weights[ideal_index].item())
            context_probability += context_weight
            mean_loss = float(context_loss["mean_loss_nats"])
            weighted_loss_numerator += context_weight * mean_loss
            weighted_accuracy_numerator += context_weight * float(
                result["quantum_context_accuracies"][quantum_id]
            )
            context_losses.append(mean_loss)
        if context_probability <= 0:
            raise RuntimeError(f"Quantum {quantum_id} is absent from every sampled ideal.")
        loss_nats = weighted_loss_numerator / context_probability
        accuracy = weighted_accuracy_numerator / context_probability
        loss_bits = float(loss_nats_to_bits(loss_nats))
        task_losses[quantum_id] = {
            "loss_sum_nats": float(loss_nats * sample_count),
            "token_count": int(sample_count),
            "mean_loss_nats": float(loss_nats),
            "mean_loss_bits": loss_bits,
            "loss_nats": float(loss_nats),
            "loss_bits": loss_bits,
            "accuracy": float(accuracy),
            "probability": float(normalized_marginals[node_index].item()),
            "depth": int(node_depths[quantum_id]),
        }
        quantum_losses[quantum_id] = {
            "loss_nats": float(loss_nats),
            "loss_bits": loss_bits,
            "context_probability": float(context_probability),
        }
        mean_context_loss = float(sum(context_losses) / len(context_losses))
        quantum_mean_losses[quantum_id] = {
            "loss_nats": mean_context_loss,
            "loss_bits": float(loss_nats_to_bits(mean_context_loss)),
            "context_count": int(len(context_losses)),
        }

    mean_task_loss_nats = float(
        sum(values["loss_nats"] for values in task_losses.values()) / len(task_losses)
    )
    mean_task_accuracy = float(
        sum(values["accuracy"] for values in task_losses.values()) / len(task_losses)
    )
    mean_depth_loss: dict[int, dict[str, float]] = {}
    mean_depth_accuracy: dict[int, float] = {}
    weighted_depth_loss: dict[int, dict[str, float]] = {}
    weighted_depth_accuracy: dict[int, float] = {}
    weighted_depth_probability: dict[int, float] = {}
    for depth in sorted(set(node_depths.values())):
        depth_codes = [int(code) for code in task_spec.codes if node_depths[int(code)] == depth]
        depth_probability = sum(task_losses[code]["probability"] for code in depth_codes)
        mean_loss = sum(task_losses[code]["loss_nats"] for code in depth_codes) / len(depth_codes)
        mean_accuracy = sum(task_losses[code]["accuracy"] for code in depth_codes) / len(depth_codes)
        weighted_loss = sum(
            task_losses[code]["probability"] * task_losses[code]["loss_nats"]
            for code in depth_codes
        )
        weighted_accuracy_at_depth = sum(
            task_losses[code]["probability"] * task_losses[code]["accuracy"]
            for code in depth_codes
        ) / depth_probability
        mean_depth_loss[int(depth)] = {
            "loss_nats": float(mean_loss),
            "loss_bits": float(loss_nats_to_bits(mean_loss)),
        }
        mean_depth_accuracy[int(depth)] = float(mean_accuracy)
        weighted_depth_loss[int(depth)] = {
            "loss_nats": float(weighted_loss),
            "loss_bits": float(loss_nats_to_bits(weighted_loss)),
        }
        weighted_depth_accuracy[int(depth)] = float(weighted_accuracy_at_depth)
        weighted_depth_probability[int(depth)] = float(depth_probability)

    return {
        "weighted_token_loss_nats": quanta_weighted_loss_nats,
        "weighted_token_loss_bits": float(loss_nats_to_bits(quanta_weighted_loss_nats)),
        "weighted_token_count": expected_tokens,
        "quanta_weighted_loss_nats": quanta_weighted_loss_nats,
        "quanta_weighted_loss_bits": float(loss_nats_to_bits(quanta_weighted_loss_nats)),
        "task_weighted_loss_nats": task_weighted_loss_nats,
        "task_weighted_loss_bits": float(loss_nats_to_bits(task_weighted_loss_nats)),
        "eval_loss_nats": quanta_weighted_loss_nats,
        "eval_loss_bits": float(loss_nats_to_bits(quanta_weighted_loss_nats)),
        "total_loss_nats": quanta_weighted_loss_nats,
        "total_loss_bits": float(loss_nats_to_bits(quanta_weighted_loss_nats)),
        "weighted_accuracy": weighted_accuracy,
        "mean_task_loss_nats": mean_task_loss_nats,
        "mean_task_loss_bits": float(loss_nats_to_bits(mean_task_loss_nats)),
        "mean_task_accuracy": mean_task_accuracy,
        "mean_depth_loss": mean_depth_loss,
        "mean_depth_accuracy": mean_depth_accuracy,
        "weighted_depth_loss": weighted_depth_loss,
        "weighted_depth_accuracy": weighted_depth_accuracy,
        "weighted_depth_probability": weighted_depth_probability,
        "task_losses": task_losses,
        "quantum_losses": quantum_losses,
        "quantum_mean_losses": quantum_mean_losses,
        "eval_loss_formula": config.eval_loss_formula,
        "loss_supervision": config.loss_supervision,
        "ideal_context_losses": [float(value) for value in mean_context_losses.tolist()],
        "ideal_context_probabilities": [float(value) for value in weights.tolist()],
    }


def _active_quantum_accuracies(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[int, float]:
    rows, slots = batch["loss_mask"].nonzero(as_tuple=True)
    quantum_ids = batch["quantum_ids"][rows, slots].to(dtype=torch.long)
    predictions = logits[rows, slots].argmax(dim=-1)
    targets = batch["loss_targets"][rows, slots]
    return {
        int(quantum_id): float(
            (predictions[quantum_ids == int(quantum_id)] == targets[quantum_ids == int(quantum_id)])
            .to(dtype=torch.float32)
            .mean()
            .item()
        )
        for quantum_id in torch.unique(quantum_ids).detach().cpu().tolist()
    }


def cached_cnand_eval_batch(
    *,
    config: ScalingLawsConfig,
    task_index: int,
    batch_cache: dict[str, torch.Tensor],
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.task == "cnand":
        if bool(getattr(config, "exhaustive_local_eval", False)):
            local_bit_width = int(batch_cache["local_bit_masks"].shape[1])
            size = 1 << local_bit_width
            assignments = torch.arange(size, device=device, dtype=torch.long)
            bits = (
                (assignments.unsqueeze(1) >> torch.arange(local_bit_width, device=device))
                & 1
            ).to(dtype=torch.long)
            local_bits = torch.zeros(
                (size, int(batch_cache["quantum_ids"].numel()), local_bit_width),
                dtype=torch.long,
                device=device,
            )
            local_bits[:, int(task_index), :] = bits
        else:
            size = int(config.eval_samples_per_task)
            local_bits = None
        task_indices = torch.full((size,), int(task_index), dtype=torch.long, device=device)
        batch = build_cnand_batch_from_targets(
            config=config,
            target_indices=task_indices,
            batch_cache=batch_cache,
            device=device,
            local_bits=local_bits,
        )
        return batch, batch["loss_targets"]

    size = int(config.eval_samples_per_task)
    bits = bipolar_random_bits((size, config.n_bits), dtype=torch.float32, device=device)

    parity_mask = batch_cache["parity_masks"][task_index].expand(size, -1)
    y = bipolar_parity_labels(bits, parity_mask)
    task_indices = torch.full((size,), int(task_index), dtype=torch.long, device=device)
    selector = F.one_hot(task_indices, num_classes=len(batch_cache["parity_masks"])).to(dtype=bits.dtype)
    return {
        "features": torch.cat([selector, bits], dim=1),
        "target_indices": task_indices.unsqueeze(1),
    }, y
