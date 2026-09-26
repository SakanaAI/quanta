from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from quanta.config import load_experiment_config
from quanta.experiments.number_naming.task import NumberNamingBatch
from quanta.experiments.quanta_discovery.qmodel import (
    AttentionQuantumCache,
    CheapCausalAttentionQuantumLayer,
    QuantumReadout,
)
from quanta.experiments.quanta_discovery.qgraph import (
    QuantumGraphEdge,
    forward_quanta,
    forward_quanta_step,
)
from quanta.experiments.quanta_discovery.dynamics import (
    batch_prediction_event_ids,
    close_existence,
    cumulative_existence_at_step,
    prediction_event_offsets,
)
from quanta.experiments.quanta_discovery.trajectory import build_model_and_task
from quanta.experiments.quanta_discovery.capacity import (
    allocate_writer_counts,
    allocate_overcomplete_writer_counts,
    functional_writer_metrics,
    fixed_qmodel_parameter_count,
    parameter_count,
)
from quanta.experiments.scaling_laws.training.trainer_curriculum import (
    scheduled_learning_rate,
)
from quanta.utils import set_seeds


EVENT_RECONSTRUCTION_WEIGHT = 1.0
GATE_SUPERVISION_WEIGHT = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the exact source Adam stream and fit one Q-model on every "
            "matching post-update batch/checkpoint pair."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--priority-dir",
        type=Path,
        help=(
            "Priority analysis to execute. Defaults to run_dir/factorization; "
            "joint analyses expose their selected graph through summary.selected."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/quanta_discovery/number_naming/small.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--greedy-max-examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _slice_batch(batch: NumberNamingBatch, start: int, end: int) -> NumberNamingBatch:
    return NumberNamingBatch(
        input_ids=batch.input_ids[start:end],
        attention_mask=batch.attention_mask[start:end],
        labels=batch.labels[start:end],
        numbers=batch.numbers[start:end],
        texts=batch.texts[start:end],
    )


def _set_source_trainable(source: torch.nn.Module, trainable: bool) -> None:
    for parameter in source.parameters():
        parameter.requires_grad_(bool(trainable))
    source.train(bool(trainable))


def _uniform_replay_steps(total_steps: int, count: int) -> np.ndarray:
    """Choose distinct post-update source steps, including the final state."""

    if not 1 <= int(count) <= int(total_steps):
        raise ValueError("replay checkpoint count must lie in [1, total_steps]")
    steps = np.rint(np.linspace(1, int(total_steps), int(count))).astype(np.int64)
    if len(np.unique(steps)) != int(count):
        raise RuntimeError("uniform replay checkpoint grid contains duplicates")
    return steps


def _save_q_checkpoint(
    output_dir: Path,
    *,
    q_optimizer_step: int,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    optimizer: torch.optim.Optimizer,
    record: dict[str, Any],
) -> None:
    """Persist a resumable trainable-Q snapshot at one optimizer step.

    The frozen readout is deliberately omitted: it is a copy of the sampled
    source checkpoint during IID replay, while evaluation restores the desired
    source checkpoint and copies its readout.
    """

    checkpoint_dir = output_dir / "q_checkpoints" / f"step_{q_optimizer_step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for layer, module in enumerate(modules):
        torch.save(module.state_dict(), checkpoint_dir / f"layer_{layer}.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    metadata = {
        "q_optimizer_step": int(q_optimizer_step),
        "checkpoint_record": record,
        "readout": (
            "not saved; restore the desired source checkpoint and copy its "
            "frozen final norm/token head before evaluation"
        ),
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _replay_iid_source_checkpoints(
    source: torch.nn.Module,
    task: Any,
    *,
    run_dir: Path,
    checkpoint_files: Sequence[str],
    checkpoint_steps: Sequence[int],
    initial_state: dict[str, torch.Tensor],
    batch_indices: np.ndarray | None,
    total_steps: int,
    selected_steps: np.ndarray,
    learning_rate: float,
    replay_tolerance: float,
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, float], float]:
    """Exactly replay the source once and retain a uniformly selected state grid."""

    source.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.SGD(source.parameters(), lr=float(learning_rate))
    selected = {int(step) for step in selected_steps}
    cached: dict[int, dict[str, torch.Tensor]] = {}
    losses: dict[int, float] = {}
    reference_lookup = {int(step): index for index, step in enumerate(checkpoint_steps)}
    replay_max_error = 0.0
    for source_step in range(1, int(total_steps) + 1):
        ids = (
            np.arange(len(task.train), dtype=np.int64)
            if batch_indices is None
            else np.asarray(batch_indices[source_step - 1], dtype=np.int64)
        )
        batch = task.encode_examples(
            [task.train[int(index)] for index in ids], device=device
        )
        _set_source_trainable(source, True)
        optimizer.zero_grad(set_to_none=True)
        logits = source(**batch.model_inputs)
        loss = task.compute_loss(logits, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if source_step in reference_lookup:
            reference = torch.load(
                run_dir / checkpoint_files[reference_lookup[source_step]],
                map_location=device,
                weights_only=True,
            )
            replay_error = max(
                float((source.state_dict()[name] - value).abs().max())
                for name, value in reference.items()
            )
            replay_max_error = max(replay_max_error, replay_error)
            if replay_error > float(replay_tolerance):
                raise RuntimeError(
                    f"source replay diverged at step {source_step}: max error "
                    f"{replay_error:g}, tolerance={float(replay_tolerance):g}"
                )
        if source_step in selected:
            cached[source_step] = {
                name: value.detach().cpu().clone()
                for name, value in source.state_dict().items()
            }
            losses[source_step] = float(loss.detach())
        del batch, logits, loss
    if set(cached) != selected:
        raise RuntimeError("failed to retain every requested IID replay checkpoint")
    return cached, losses, replay_max_error


def _linear_ramp(step: int, ramp_steps: int) -> float:
    if int(ramp_steps) <= 0:
        return 1.0
    return min(1.0, float(step) / float(ramp_steps))


def _delayed_ramp(step: int, start_step: int, ramp_steps: int) -> float:
    if int(step) <= int(start_step):
        return 0.0
    return _linear_ramp(int(step) - int(start_step), int(ramp_steps))


@torch.no_grad()
def _greedy_tokens(
    number: int,
    *,
    tokenizer: Any,
    max_seq_len: int,
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    device: torch.device,
    routing_mode: str = "layer_residual_roots",
    enforce_parent_gate_closure: bool = True,
) -> tuple[int, ...]:
    digits = tuple(int(character) for character in str(int(number)))
    prompt = [tokenizer.bos_id]
    prompt.extend(tokenizer.token_to_id[f"<D{digit}>"] for digit in digits)
    prompt.append(tokenizer.sep_id)
    caches: tuple[AttentionQuantumCache, ...] | None = None
    step = None
    for position, token in enumerate(prompt):
        step = forward_quanta_step(
            source,
            readout,
            modules,
            torch.as_tensor([token], device=device),
            position,
            existences,
            edges,
            caches,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        caches = step.caches
    assert step is not None
    generated: list[int] = []
    for position in range(len(prompt), int(max_seq_len)):
        token = int(step.logits[0].argmax())
        generated.append(token)
        if token == tokenizer.eos_id:
            break
        step = forward_quanta_step(
            source,
            readout,
            modules,
            torch.as_tensor([token], device=device),
            position,
            existences,
            edges,
            caches,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        caches = step.caches
    return tuple(generated)


def _load_priority(
    run_dir: Path,
    priority_dir: Path | None = None,
    *,
    raw_stage_b: bool = False,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[QuantumGraphEdge, ...], dict[str, Any]]:
    if raw_stage_b and priority_dir is None:
        priority_summary = json.loads((run_dir / "priority_summary.json").read_text())
        with np.load(run_dir / "priority_analysis.npz", allow_pickle=True) as payload:
            supports = tuple(
                np.asarray(value, dtype=bool) for value in payload["train_supports"]
            )
            temporal = tuple(
                np.asarray(value, dtype=np.float64)
                for value in payload["temporal_priority"]
            )
        return supports, temporal, (), {
            "method": "quanta_factorization_v1_raw_stage_b",
            "source_priority_method": priority_summary["method"],
            "edges": [],
        }
    priority_dir = run_dir / "factorization" if priority_dir is None else priority_dir
    summary = json.loads((priority_dir / "summary.json").read_text())
    with np.load(priority_dir / "temporal_priority.npz") as payload:
        layers = sorted(payload.files, key=lambda value: int(value.split("_")[-1]))
        temporal = tuple(np.asarray(payload[name], dtype=np.float64) for name in layers)
    supports = tuple(
        np.load(priority_dir / f"support_layer_{layer}.npy", mmap_mode="r")
        for layer in range(len(temporal))
    )
    edge_values = [] if raw_stage_b else summary["edges"]
    edges = tuple(tuple(int(item) for item in edge) for edge in edge_values)
    if raw_stage_b:
        summary = {
            **summary,
            "method": f"{summary['method']}_raw_stage_b",
            "edges": [],
        }
    return supports, temporal, edges, summary


def _existence_table(
    temporal_priority: Sequence[np.ndarray],
    edges: Sequence[QuantumGraphEdge],
    total_steps: int,
    *,
    apply_closure: bool = True,
) -> tuple[np.ndarray, ...]:
    tables = [
        np.empty((int(total_steps) + 1, len(curves)), dtype=np.float32)
        for curves in temporal_priority
    ]
    for step in range(int(total_steps) + 1):
        raw = tuple(
            cumulative_existence_at_step(
                curves, step=step, total_steps=int(total_steps)
            )
            for curves in temporal_priority
        )
        values = close_existence(raw, edges) if apply_closure else raw
        for layer, value in enumerate(values):
            tables[layer][step] = value
    return tuple(tables)


def _all_layer_ordered_edges(
    quantum_counts: Sequence[int],
) -> tuple[QuantumGraphEdge, ...]:
    """Return every globally admissible earlier-layer message channel."""

    return tuple(
        (parent_layer, parent, child_layer, child)
        for child_layer, child_count in enumerate(quantum_counts)
        for parent_layer in range(child_layer)
        for parent in range(int(quantum_counts[parent_layer]))
        for child in range(int(child_count))
    )


def _straight_through_global_edge_scales(
    edge_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return global hard-forward edge scales and their sigmoid relaxation."""

    probabilities = torch.sigmoid(edge_logits)
    hard = (probabilities >= 0.5).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach(), probabilities, hard


def _event_coordinates(batch: NumberNamingBatch) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.nonzero(batch.labels[:, 1:] != -100, as_tuple=True)


def _reconstruction_coordinates(
    batch: NumberNamingBatch, positions: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions == "prediction_positions":
        return _event_coordinates(batch)
    if positions == "all_valid_positions":
        return torch.nonzero(batch.attention_mask.to(torch.bool), as_tuple=True)
    raise ValueError(f"unsupported reconstruction positions: {positions!r}")


def _binary_gate_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    effective_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure a local gate against a fixed binary dynamics-support target."""

    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=bool).reshape(-1)
    if probability.shape != target.shape:
        raise ValueError("gate probabilities and labels must have matching shapes")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("gate probabilities must lie in [0, 1]")

    def report(prediction: np.ndarray) -> dict[str, float | int]:
        prediction = np.asarray(prediction, dtype=bool).reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError("gate predictions and labels must have matching shapes")
        tp = int(np.logical_and(prediction, target).sum())
        fp = int(np.logical_and(prediction, ~target).sum())
        tn = int(np.logical_and(~prediction, ~target).sum())
        fn = int(np.logical_and(~prediction, target).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        return {
            "count": int(target.size),
            "support_prevalence": float(target.mean()),
            "prediction_prevalence": float(prediction.mean()),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (recall + specificity),
        }

    local = report(probability >= 0.5)
    local["brier"] = float(np.square(probability - target).mean())
    local["log_loss_nats"] = float(
        -(
            target * np.log(np.clip(probability, 1.0e-7, 1.0))
            + (~target) * np.log(np.clip(1.0 - probability, 1.0e-7, 1.0))
        ).mean()
    )
    result: dict[str, Any] = {"local_gate": local}
    if effective_predictions is not None:
        result["effective_gate"] = report(effective_predictions)
    return result


def _source_update_targets(trace: Any) -> tuple[torch.Tensor, ...]:
    boundaries = trace.block_boundaries
    return tuple(after - before for before, after in zip(boundaries[:-1], boundaries[1:]))


def _global_denominators(
    targets: Sequence[torch.Tensor], rows: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, int]:
    event_energy = torch.stack(
        [target[rows, positions].square().sum() for target in targets]
    ).clamp_min(1.0e-12)
    return event_energy, int(len(rows))


def _gate_positive_weights(
    supports: Sequence[np.ndarray], event_ids: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    result = []
    for support in supports:
        labels = np.asarray(support[event_ids], dtype=np.float32)
        positives = labels.sum(axis=0)
        negatives = len(labels) - positives
        weights = np.divide(
            negatives,
            np.maximum(positives, 1.0),
            out=np.ones_like(positives),
            where=(positives > 0.0) & (negatives > 0.0),
        )
        result.append(torch.as_tensor(weights, device=device))
    return tuple(result)


def _writer_metric_record(value: Any) -> dict[str, Any]:
    return {
        "demand": value.demand.tolist(),
        "tau_50_steps": value.tau_50_steps.tolist(),
        "packet_effective_groups": value.packet_effective_groups.tolist(),
        "packet_groups_for_90_percent_energy": value.packet_groups_for_90_percent_energy.tolist(),
        "priority_weighted_instantaneous_groups": value.priority_weighted_instantaneous_groups.tolist(),
    }


def _writer_l0_target_penalty(
    quantum: Any,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    rows: torch.Tensor,
    positions: torch.Tensor,
    *,
    scope: str,
    target: float,
    microbatch_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match writer activity at the configured aggregation scope."""

    if scope == "global_event":
        writers_per_event = torch.stack(
            [
                quantum.writer_gates[layer][rows, positions].sum(dim=-1)
                for layer in range(len(modules))
            ]
        ).sum(dim=0)
        observed = writers_per_event.mean()
        penalty = (
            (observed / float(target) - 1.0).square()
            * float(microbatch_fraction)
        )
        return penalty, observed
    if scope != "per_active_quantum":
        raise ValueError(f"unsupported writer L0 scope: {scope}")

    penalties = []
    observed_values = []
    for layer, module in enumerate(modules):
        writer_gates = quantum.writer_gates[layer][rows, positions]
        local_gates = quantum.local_gates[layer][rows, positions]
        for q in range(module.quantum_count):
            active = local_gates[:, q].detach() > 0.5
            active_count = int(active.sum())
            if active_count == 0:
                continue
            indices = torch.nonzero(
                module.writer_owner == q, as_tuple=False
            ).flatten()
            observed = writer_gates[:, indices][active].sum() / active_count
            observed_values.append(observed)
            penalties.append((observed / float(target) - 1.0).square())
    if not penalties:
        zero = quantum.logits.sum() * 0.0
        return zero, zero.detach()
    penalty = torch.stack(penalties).mean() * float(microbatch_fraction)
    observed = torch.stack(observed_values).mean()
    return penalty, observed


@torch.no_grad()
def _evaluate_teacher_forced(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    batch_size: int,
    edge_scales: dict[QuantumGraphEdge, float] | None = None,
    routing_mode: str = "layer_residual_roots",
    enforce_parent_gate_closure: bool = True,
) -> dict[str, Any]:
    source.eval()
    for module in (*modules, readout):
        module.eval()
    total = source_correct = quantum_correct = agreement = 0
    source_nll = quantum_nll = kl = 0.0
    digit_counts: dict[int, list[int]] = {}
    quantum_endpoint = [0 for _ in modules]
    quantum_entropy = [0.0 for _ in modules]
    quantum_active = [0.0 for _ in modules]
    quantum_values = [0 for _ in modules]
    writer_active = [0.0 for _ in modules]
    writer_values = [0 for _ in modules]
    active_quanta_per_event: list[torch.Tensor] = []
    active_writers_per_event: list[torch.Tensor] = []
    for start in range(0, len(examples), int(batch_size)):
        selected = list(examples[start : start + int(batch_size)])
        batch = task.encode_examples(selected, device=device)
        rows, positions = _event_coordinates(batch)
        source_trace = source.residual_trace(**batch.model_inputs)
        e = tuple(value.expand(len(selected), -1) for value in existences)
        quantum = forward_quanta(
            source,
            readout,
            modules,
            batch.input_ids,
            batch.attention_mask,
            e,
            edges,
            hard_gates=True,
            edge_scales=edge_scales,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        event_quanta = torch.zeros(len(rows), device=device)
        event_writers = torch.zeros(len(rows), device=device)
        for layer, module in enumerate(modules):
            gate_probabilities = torch.sigmoid(
                quantum.gate_logits[layer][rows, positions]
            )
            gate_active = quantum.effective_gates[layer][rows, positions]
            gate_entropy = -(
                gate_probabilities.clamp_min(1.0e-7)
                * gate_probabilities.clamp_min(1.0e-7).log()
                + (1.0 - gate_probabilities).clamp_min(1.0e-7)
                * (1.0 - gate_probabilities).clamp_min(1.0e-7).log()
            )
            quantum_endpoint[layer] += int(
                ((gate_probabilities <= 0.1) | (gate_probabilities >= 0.9)).sum()
            )
            quantum_entropy[layer] += float(gate_entropy.sum())
            quantum_active[layer] += float(gate_active.sum())
            quantum_values[layer] += gate_probabilities.numel()
            owner_active = gate_active.index_select(1, module.writer_owner)
            effective_writer_active = (
                quantum.writer_gates[layer][rows, positions] * owner_active
            )
            writer_active[layer] += float(effective_writer_active.sum())
            writer_values[layer] += effective_writer_active.numel()
            event_quanta += gate_active.sum(dim=1)
            event_writers += effective_writer_active.sum(dim=1)
        active_quanta_per_event.append(event_quanta.cpu())
        active_writers_per_event.append(event_writers.cpu())
        labels = batch.labels[:, 1:][rows, positions]
        source_logits = source_trace.logits[rows, positions]
        quantum_logits = quantum.logits[rows, positions]
        source_predictions = source_logits.argmax(dim=-1)
        quantum_predictions = quantum_logits.argmax(dim=-1)
        source_correct += int((source_predictions == labels).sum())
        quantum_correct += int((quantum_predictions == labels).sum())
        agreement += int((quantum_predictions == source_predictions).sum())
        source_nll += float(F.cross_entropy(source_logits, labels, reduction="sum"))
        quantum_nll += float(F.cross_entropy(quantum_logits, labels, reduction="sum"))
        kl += float(
            F.kl_div(
                F.log_softmax(quantum_logits, dim=-1),
                F.softmax(source_logits, dim=-1),
                reduction="sum",
            )
        )
        total += len(labels)
        valid = batch.labels[:, 1:] != -100
        predictions = quantum.logits[:, :-1].argmax(dim=-1)
        for row, example in enumerate(selected):
            digit = len(str(int(example.number)))
            row_valid = valid[row]
            record = digit_counts.setdefault(digit, [0, 0])
            record[0] += int((predictions[row][row_valid] == batch.labels[row, 1:][row_valid]).sum())
            record[1] += int(row_valid.sum())
    active_quanta = torch.cat(active_quanta_per_event)
    active_writers = torch.cat(active_writers_per_event)
    return {
        "examples": len(examples),
        "prediction_events": total,
        "source_accuracy": source_correct / max(total, 1),
        "quantum_accuracy": quantum_correct / max(total, 1),
        "quantum_source_argmax_agreement": agreement / max(total, 1),
        "source_nll_bits": source_nll / max(total, 1) / math.log(2.0),
        "quantum_nll_bits": quantum_nll / max(total, 1) / math.log(2.0),
        "source_to_quantum_kl": kl / max(total, 1),
        "quantum_accuracy_by_digit": {
            str(digit): values[0] / max(values[1], 1)
            for digit, values in sorted(digit_counts.items())
        },
        "execution": {
            "quantum_gate_endpoint_fraction_by_layer": [
                value / max(count, 1)
                for value, count in zip(quantum_endpoint, quantum_values)
            ],
            "quantum_gate_entropy_nats_by_layer": [
                value / max(count, 1)
                for value, count in zip(quantum_entropy, quantum_values)
            ],
            "effective_quantum_activation_fraction_by_layer": [
                value / max(count, 1)
                for value, count in zip(quantum_active, quantum_values)
            ],
            "effective_writer_activation_fraction_by_layer": [
                value / max(count, 1)
                for value, count in zip(writer_active, writer_values)
            ],
            "active_quanta_per_event": {
                "mean": float(active_quanta.mean()),
                "median": float(active_quanta.median()),
                "p90": float(torch.quantile(active_quanta, 0.9)),
            },
            "active_writers_per_event": {
                "mean": float(active_writers.mean()),
                "median": float(active_writers.median()),
                "p90": float(torch.quantile(active_writers, 0.9)),
            },
        },
    }


@torch.no_grad()
def _evaluate_greedy(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    routing_mode: str = "layer_residual_roots",
    enforce_parent_gate_closure: bool = True,
) -> dict[str, Any]:
    exact = terminated = aligned_correct = aligned_total = 0
    errors = []
    for index, example in enumerate(examples):
        prediction = _greedy_tokens(
            int(example.number),
            tokenizer=task.tokenizer,
            max_seq_len=int(task.config.max_seq_len),
            source=source,
            readout=readout,
            modules=modules,
            existences=existences,
            edges=edges,
            device=device,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        target = tuple(
            task.tokenizer.token_to_id[token] for token in example.text.split()
        ) + (task.tokenizer.eos_id,)
        exact += int(prediction == target)
        terminated += int(bool(prediction) and prediction[-1] == task.tokenizer.eos_id)
        aligned_correct += sum(int(left == right) for left, right in zip(prediction, target))
        aligned_total += max(len(prediction), len(target))
        if prediction != target and len(errors) < 32:
            errors.append({"number": int(example.number), "target_ids": list(target), "prediction_ids": list(prediction)})
        if (index + 1) % 250 == 0:
            print(json.dumps({"phase": "greedy_evaluation", "completed": index + 1, "total": len(examples)}), flush=True)
    return {
        "examples": len(examples),
        "exact_sequence_accuracy": exact / max(len(examples), 1),
        "termination_rate": terminated / max(len(examples), 1),
        "position_aligned_token_accuracy": aligned_correct / max(aligned_total, 1),
        "first_errors": errors,
    }


def main() -> None:
    args = parse_args()
    if int(args.evaluation_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    if not math.isfinite(float(args.replay_tolerance)) or float(args.replay_tolerance) <= 0.0:
        raise ValueError("replay-tolerance must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config("quanta_discovery", args.config)
    if int(config.q_microbatch_size) <= 0:
        raise ValueError("q_microbatch_size must be positive")
    source_steps = int(config.steps or 0)
    q_optimizer_steps = int(config.q_optimizer_steps or source_steps)
    set_seeds(int(args.seed))
    task, source = build_model_and_task(config, device=device)
    batch_indices_path = run_dir / "batch_indices.npy"
    batch_indices = (
        np.load(batch_indices_path, mmap_mode="r")
        if batch_indices_path.exists()
        else None
    )
    offsets_path = run_dir / "event_offsets.npy"
    offsets = (
        np.load(offsets_path, mmap_mode="r")
        if offsets_path.exists()
        else prediction_event_offsets([example.text for example in task.train])
    )
    priority_dir = (
        None if args.priority_dir is None else args.priority_dir.resolve()
    )
    global_edge_discovery = config.q_edge_discovery == "global_static"
    supports, temporal_priority, fixed_edges, priority_summary = _load_priority(
        run_dir,
        priority_dir,
        raw_stage_b=global_edge_discovery,
    )
    discovery_priority_path = run_dir / "priority_summary.json"
    discovery_priority_summary = (
        json.loads(discovery_priority_path.read_text())
        if discovery_priority_path.exists()
        else {}
    )
    existence_tables = _existence_table(
        temporal_priority,
        fixed_edges,
        source_steps,
        apply_closure=not global_edge_discovery,
    )
    np.savez(
        output_dir / "existence_by_step.npz",
        **{f"layer_{layer}": values for layer, values in enumerate(existence_tables)},
    )

    checkpoint_metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text())
    checkpoint_files = checkpoint_metadata["checkpoint_files"]
    checkpoint_steps = checkpoint_metadata["checkpoint_steps"]
    initial_state = torch.load(run_dir / checkpoint_files[0], map_location=device, weights_only=True)
    source.load_state_dict(initial_state, strict=True)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=float(config.lr))

    quantum_counts = tuple(value.shape[1] for value in supports)
    candidate_edges = (
        _all_layer_ordered_edges(quantum_counts)
        if global_edge_discovery
        else fixed_edges
    )
    embedding_parameter_count = int(
        initial_state["token_embedding.weight"].numel()
        + initial_state["position_embedding.weight"].numel()
    )
    fixed_parameters, parameters_per_writer = fixed_qmodel_parameter_count(
        quantum_counts,
        d_model=int(config.d_model),
        vocabulary_size=int(initial_state["token_embedding.weight"].shape[0]),
        attention_rank=int(config.q_attention_rank),
        attention_value_dim=int(config.q_attention_value_dim),
        embedding_parameter_count=embedding_parameter_count,
        attention_direct_residual=bool(config.q_attention_direct_residual),
        share_attention_projections=bool(config.q_share_attention_projections),
        attention_query_offsets=bool(config.q_attention_query_offsets),
    )
    source_parameter_count = sum(value.numel() for value in initial_state.values())
    functional = functional_writer_metrics(
        np.load(run_dir / "functional_residual_writes.npy", mmap_mode="r"),
        temporal_priority, supports, energy=float(config.q_functional_rank_energy),
    )
    if config.q_writer_total_budget is None:
        writer_counts = tuple(allocate_overcomplete_writer_counts(
            metric.complexity,
            minimum=int(config.q_writer_minimum),
            factor=float(config.q_writer_overcomplete_factor),
            maximum=int(config.q_writer_maximum),
        ) for metric in functional)
        writer_allocation = "independent_overcomplete"
    else:
        flat_complexity = np.concatenate(
            [metric.complexity for metric in functional]
        )
        flat_counts = allocate_writer_counts(
            flat_complexity, int(config.q_writer_total_budget)
        )
        writer_counts_list = []
        offset = 0
        for metric in functional:
            count = len(metric.complexity)
            writer_counts_list.append(flat_counts[offset : offset + count])
            offset += count
        writer_counts = tuple(writer_counts_list)
        writer_allocation = "complexity_fixed_total"
    modules = tuple(
        CheapCausalAttentionQuantumLayer(
            int(config.d_model),
            count,
            attention_rank=int(config.q_attention_rank),
            attention_value_dim=int(config.q_attention_value_dim),
            attention_direct_residual=bool(config.q_attention_direct_residual),
            share_attention_projections=bool(config.q_share_attention_projections),
            attention_query_offsets=bool(config.q_attention_query_offsets),
            writer_counts=writer_counts,
            writer_average_k=float(config.q_writer_average_k),
            writer_activation=str(config.q_writer_activation),
            writer_sparsity=str(config.q_writer_sparsity),
            writer_jump_threshold=float(config.q_writer_jump_threshold),
            writer_jump_bandwidth=float(config.q_writer_jump_bandwidth),
        ).to(device)
        for count, writer_counts in zip(quantum_counts, writer_counts)
    )
    readout = QuantumReadout(
        int(config.d_model),
        int(initial_state["token_embedding.weight"].shape[0]),
    ).to(device)
    readout.copy_source_readout(source)
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    q_parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
    ]
    edge_logits = (
        torch.nn.Parameter(torch.zeros(len(candidate_edges), device=device))
        if global_edge_discovery
        else None
    )
    if edge_logits is not None:
        q_parameters.append(edge_logits)
    q_optimizer = torch.optim.Adam(q_parameters, lr=float(config.q_learning_rate))
    report_every = max(1, q_optimizer_steps // 20)
    q_checkpoint_steps = set(int(value) for value in config.q_checkpoint_steps)
    history = []
    replay_max_error = 0.0
    checkpoint_lookup = {int(step): index for index, step in enumerate(checkpoint_steps)}
    iid_replay_steps: np.ndarray | None = None
    iid_sampled_steps: np.ndarray | None = None
    iid_source_states: dict[int, dict[str, torch.Tensor]] | None = None
    iid_source_losses: dict[int, float] | None = None
    frozen_edge_scales: torch.Tensor | None = None
    edge_learning_steps = q_optimizer_steps - int(config.q_edge_freeze_steps)
    if config.q_training_mode == "iid_checkpoint_replay":
        iid_replay_steps = _uniform_replay_steps(
            source_steps, int(config.q_iid_checkpoint_count)
        )
        (
            iid_source_states,
            iid_source_losses,
            replay_max_error,
        ) = _replay_iid_source_checkpoints(
            source,
            task,
            run_dir=run_dir,
            checkpoint_files=checkpoint_files,
            checkpoint_steps=checkpoint_steps,
            initial_state=initial_state,
            batch_indices=batch_indices,
            total_steps=source_steps,
            selected_steps=iid_replay_steps,
            learning_rate=float(config.lr),
            replay_tolerance=float(args.replay_tolerance),
            device=device,
        )
        iid_sampled_steps = np.random.default_rng(int(args.seed)).choice(
            iid_replay_steps, size=q_optimizer_steps, replace=True
        )
        np.save(output_dir / "iid_replay_checkpoint_steps.npy", iid_replay_steps)
        np.save(output_dir / "iid_sampled_checkpoint_steps.npy", iid_sampled_steps)

    for step in range(1, q_optimizer_steps + 1):
        if iid_sampled_steps is None:
            source_step = step
            ids = (
                np.arange(len(task.train), dtype=np.int64)
                if batch_indices is None
                else np.asarray(batch_indices[step - 1], dtype=np.int64)
            )
        else:
            source_step = int(iid_sampled_steps[step - 1])
            ids = np.arange(len(task.train), dtype=np.int64)
        examples = [task.train[int(index)] for index in ids]
        batch = task.encode_examples(examples, device=device)
        if iid_source_states is None:
            learning_rate = scheduled_learning_rate(
                float(config.lr),
                step,
                source_steps,
                str(config.scheduler),
                warmup_phase=float(config.warmup_phase),
                plateau_phase=float(config.plateau_phase),
            )
            for group in source_optimizer.param_groups:
                group["lr"] = float(learning_rate)
            _set_source_trainable(source, True)
            source_optimizer.zero_grad(set_to_none=True)
            source_logits_before = source(**batch.model_inputs)
            source_loss = task.compute_loss(source_logits_before, batch)
            source_loss.backward()
            source_optimizer.step()
            source_optimizer.zero_grad(set_to_none=True)
            del source_logits_before

            if step in checkpoint_lookup:
                reference = torch.load(
                    run_dir / checkpoint_files[checkpoint_lookup[step]],
                    map_location=device,
                    weights_only=True,
                )
                replay_error = max(
                    float((source.state_dict()[name] - value).abs().max())
                    for name, value in reference.items()
                )
                replay_max_error = max(replay_max_error, replay_error)
                if replay_error > float(args.replay_tolerance):
                    raise RuntimeError(
                        f"source replay diverged at step {step}: max error "
                        f"{replay_error:g}, tolerance={float(args.replay_tolerance):g}"
                    )
        else:
            source.load_state_dict(iid_source_states[source_step], strict=True)
            source_loss = torch.tensor(
                iid_source_losses[source_step], device=device
            )

        _set_source_trainable(source, False)
        readout.copy_source_readout(source)
        with torch.no_grad():
            source_trace = source.residual_trace(**batch.model_inputs)
            source_targets = _source_update_targets(source_trace)
            event_rows, event_positions = _event_coordinates(batch)
            event_energy, event_count = _global_denominators(
                source_targets, event_rows, event_positions
            )
            reconstruction_rows, reconstruction_positions = _reconstruction_coordinates(
                batch, str(config.q_reconstruction_positions)
            )
            reconstruction_energy, reconstruction_count = _global_denominators(
                source_targets, reconstruction_rows, reconstruction_positions
            )
        event_ids = batch_prediction_event_ids(ids, offsets)
        if len(event_ids) != event_count:
            raise RuntimeError("Q stream event IDs do not align with the source batch")
        positive_weights = (
            _gate_positive_weights(supports, event_ids, device)
            if config.q_balance_gate_bce
            else tuple(None for _ in supports)
        )
        existence = tuple(
            torch.as_tensor(table[source_step], device=device)
            for table in existence_tables
        )
        q_optimizer.zero_grad(set_to_none=True)
        totals = {
            "event_reconstruction": 0.0,
            "output_kl": 0.0,
            "gate": 0.0,
            "gate_binary_entropy": 0.0,
            "writer_l0_target_penalty": 0.0,
            "writer_l0_observed": 0.0,
            "edge_l0_expected": 0.0,
            "edge_l0_penalty": 0.0,
            "edge_binary_entropy": 0.0,
            "edge_binary_entropy_penalty": 0.0,
        }
        binary_ramp = _delayed_ramp(
            step,
            int(config.q_gate_entropy_start_step),
            int(config.q_gate_entropy_ramp_steps),
        )
        writer_l0_ramp = _linear_ramp(
            step, int(config.q_writer_l0_ramp_steps)
        )
        event_cursor = 0
        for start in range(0, len(ids), int(config.q_microbatch_size)):
            end = min(start + int(config.q_microbatch_size), len(ids))
            micro = _slice_batch(batch, start, end)
            edge_scales: dict[QuantumGraphEdge, torch.Tensor] | None = None
            edge_probabilities: torch.Tensor | None = None
            if edge_logits is not None:
                if frozen_edge_scales is None:
                    scales, edge_probabilities, _ = (
                        _straight_through_global_edge_scales(edge_logits)
                    )
                else:
                    scales = frozen_edge_scales
                    edge_probabilities = torch.sigmoid(edge_logits.detach())
                edge_scales = {
                    edge: scales[index]
                    for index, edge in enumerate(candidate_edges)
                }
            rows, positions = _event_coordinates(micro)
            reconstruction_rows, reconstruction_positions = _reconstruction_coordinates(
                micro, str(config.q_reconstruction_positions)
            )
            micro_event_count = len(rows)
            micro_event_ids = event_ids[event_cursor : event_cursor + micro_event_count]
            event_cursor += micro_event_count
            e = tuple(value[None].expand(end - start, -1) for value in existence)
            quantum = forward_quanta(
                source,
                readout,
                modules,
                micro.input_ids,
                micro.attention_mask,
                e,
                candidate_edges,
                straight_through_gates=True,
                detach_gates_from_writes=bool(
                    config.detach_gate_gradients
                ),
                edge_scales=edge_scales,
                routing_mode=str(config.q_routing_mode),
                enforce_parent_gate_closure=not global_edge_discovery,
            )
            event_reconstruction = torch.stack(
                [
                    (
                        prediction[reconstruction_rows, reconstruction_positions]
                        - target[start:end][reconstruction_rows, reconstruction_positions]
                    ).square().sum()
                    / reconstruction_energy[layer]
                    for layer, (prediction, target) in enumerate(zip(quantum.updates, source_targets))
                ]
            ).mean()
            zero = event_reconstruction * 0.0
            output_kl = zero
            if float(config.q_output_kl_weight) > 0.0:
                source_event_logits = source_trace.logits[start:end][rows, positions]
                quantum_event_logits = quantum.logits[rows, positions]
                output_kl = F.kl_div(
                    F.log_softmax(quantum_event_logits, dim=-1),
                    F.softmax(source_event_logits, dim=-1),
                    reduction="sum",
                ) / max(event_count, 1)
            gate_loss = zero
            binary_entropy = zero
            total_gate_values = max(
                event_count * sum(quantum_counts), 1
            )
            for layer, module in enumerate(modules):
                gate_logits = quantum.gate_logits[layer][rows, positions]
                gate_labels = torch.as_tensor(
                    np.asarray(supports[layer][micro_event_ids], dtype=np.float32),
                    device=device,
                )
                gate_loss = gate_loss + F.binary_cross_entropy_with_logits(
                    gate_logits,
                    gate_labels,
                    pos_weight=positive_weights[layer],
                    reduction="sum",
                ) / total_gate_values
                probabilities = torch.sigmoid(gate_logits).clamp(1.0e-7, 1.0 - 1.0e-7)
                binary_entropy = binary_entropy - (
                    probabilities * probabilities.log()
                    + (1.0 - probabilities) * torch.log1p(-probabilities)
                ).sum() / total_gate_values
            writer_l0_penalty = zero
            writer_l0_observed = zero.detach()
            if config.q_writer_sparsity == "l0_target":
                writer_l0_penalty, writer_l0_observed = _writer_l0_target_penalty(
                    quantum,
                    modules,
                    rows,
                    positions,
                    scope=str(config.q_writer_l0_scope),
                    target=float(config.q_writer_l0_target),
                    microbatch_fraction=micro_event_count / max(event_count, 1),
                )
            edge_l0_penalty = zero
            edge_binary_entropy = zero
            edge_binary_entropy_penalty = zero
            if edge_probabilities is not None and frozen_edge_scales is None:
                edge_l0_penalty = (
                    float(config.q_edge_l0_weight)
                    * _linear_ramp(step, int(config.q_edge_l0_ramp_steps))
                    * edge_probabilities.sum()
                    * (micro_event_count / max(event_count, 1))
                )
                probabilities = edge_probabilities.clamp(1.0e-7, 1.0 - 1.0e-7)
                edge_binary_entropy = -(
                    probabilities * probabilities.log()
                    + (1.0 - probabilities) * torch.log1p(-probabilities)
                ).sum() * (micro_event_count / max(event_count, 1))
                edge_binary_entropy_penalty = (
                    float(config.q_edge_entropy_weight)
                    * _delayed_ramp(
                        step,
                        int(config.q_edge_entropy_start_step),
                        int(config.q_edge_entropy_ramp_steps),
                    )
                    * edge_binary_entropy
                )
            functional_loss = (
                EVENT_RECONSTRUCTION_WEIGHT * event_reconstruction
                + float(config.q_output_kl_weight) * output_kl
                + edge_l0_penalty
                + edge_binary_entropy_penalty
            )
            auxiliary_loss = (
                GATE_SUPERVISION_WEIGHT * gate_loss
                + float(config.q_gate_entropy_weight)
                * float(binary_ramp)
                * binary_entropy
                + float(config.q_writer_l0_weight)
                * float(writer_l0_ramp)
                * writer_l0_penalty
            )
            functional_loss.backward(retain_graph=True)
            functional_edge_gradient = (
                None
                if edge_logits is None or edge_logits.grad is None
                else edge_logits.grad.detach().clone()
            )
            auxiliary_loss.backward()
            if functional_edge_gradient is not None:
                edge_logits.grad.copy_(functional_edge_gradient)
            for name, value in (
                ("event_reconstruction", event_reconstruction),
                ("output_kl", output_kl),
                ("gate", gate_loss),
                ("gate_binary_entropy", binary_entropy),
                ("writer_l0_target_penalty", writer_l0_penalty),
                ("writer_l0_observed", writer_l0_observed),
                (
                    "edge_l0_expected",
                    zero if edge_probabilities is None else edge_probabilities.sum(),
                ),
                ("edge_l0_penalty", edge_l0_penalty),
                ("edge_binary_entropy", edge_binary_entropy),
                ("edge_binary_entropy_penalty", edge_binary_entropy_penalty),
            ):
                totals[name] += float(value.detach())
        if event_cursor != event_count:
            raise RuntimeError("Q microbatch accumulation lost prediction events")
        torch.nn.utils.clip_grad_norm_(q_parameters, 5.0)
        q_optimizer.step()
        if (
            edge_logits is not None
            and frozen_edge_scales is None
            and step == edge_learning_steps
        ):
            frozen_edge_scales = (
                (torch.sigmoid(edge_logits.detach()) >= 0.5)
                .to(dtype=edge_logits.dtype)
                .detach()
            )
        should_report = step == 1 or step % report_every == 0 or step == q_optimizer_steps
        if should_report or step in q_checkpoint_steps:
            record = {
                "step": step,
                "source_checkpoint_step": source_step,
                "source_loss_nats": float(source_loss.detach()),
                "existence_mean": float(
                    np.mean([table[source_step].mean() for table in existence_tables])
                ),
                "binary_ramp": binary_ramp,
                "writer_l0_ramp": writer_l0_ramp,
                "edge_l0_ramp": _linear_ramp(
                    step, int(config.q_edge_l0_ramp_steps)
                ),
                "edge_entropy_ramp": _delayed_ramp(
                    step,
                    int(config.q_edge_entropy_start_step),
                    int(config.q_edge_entropy_ramp_steps),
                ),
                "edge_graph_frozen": frozen_edge_scales is not None,
                "reconstruction_position_count": reconstruction_count,
                **totals,
            }
            if should_report:
                history.append(record)
                print(json.dumps(record), flush=True)
            if step in q_checkpoint_steps:
                _save_q_checkpoint(
                    output_dir,
                    q_optimizer_step=step,
                    modules=modules,
                    optimizer=q_optimizer,
                    record=record,
                )
        del batch, source_trace, source_targets, source_loss

    if iid_source_states is not None:
        source.load_state_dict(iid_source_states[source_steps], strict=True)
    _set_source_trainable(source, False)
    readout.copy_source_readout(source)
    if edge_logits is None:
        executed_edges = fixed_edges
        edge_probability_records: list[dict[str, Any]] = []
    else:
        final_probabilities = torch.sigmoid(edge_logits.detach())
        final_hard = (final_probabilities >= 0.5).to(torch.bool)
        executed_edges = tuple(
            edge
            for edge, selected in zip(candidate_edges, final_hard.tolist())
            if selected
        )
        edge_probability_records = [
            {
                "edge": list(edge),
                "probability": float(probability),
                "selected": bool(selected),
            }
            for edge, probability, selected in zip(
                candidate_edges,
                final_probabilities.cpu().tolist(),
                final_hard.cpu().tolist(),
            )
        ]
        np.savez(
            output_dir / "global_edge_state.npz",
            candidate_edges=np.asarray(candidate_edges, dtype=np.int64),
            logits=edge_logits.detach().cpu().numpy(),
            probabilities=final_probabilities.cpu().numpy(),
            selected=final_hard.cpu().numpy(),
        )
    final_existence = tuple(
        torch.as_tensor(table[-1], device=device)[None]
        for table in existence_tables
    )
    # Persist the completed fit before behavioral evaluation so a diagnostic
    # failure cannot discard a full trajectory replay.
    for layer, module in enumerate(modules):
        torch.save(module.state_dict(), output_dir / f"layer_{layer}.pt")
    torch.save(readout.state_dict(), output_dir / "readout.pt")
    train_metrics = _evaluate_teacher_forced(
        source,
        readout,
        modules,
        task,
        list(task.train),
        final_existence,
        executed_edges,
        device=device,
        batch_size=int(args.evaluation_batch_size),
        routing_mode=str(config.q_routing_mode),
        enforce_parent_gate_closure=not global_edge_discovery,
    )
    eval_metrics = _evaluate_teacher_forced(
        source,
        readout,
        modules,
        task,
        list(task.eval_examples),
        final_existence,
        executed_edges,
        device=device,
        batch_size=int(args.evaluation_batch_size),
        routing_mode=str(config.q_routing_mode),
        enforce_parent_gate_closure=not global_edge_discovery,
    )
    edge_interventions = []
    for disabled_edges in ((edge,) for edge in executed_edges):
        intervention = _evaluate_teacher_forced(
            source,
            readout,
            modules,
            task,
            list(task.eval_examples),
            final_existence,
            executed_edges,
            device=device,
            batch_size=int(args.evaluation_batch_size),
            edge_scales={edge: 0.0 for edge in disabled_edges},
            routing_mode=str(config.q_routing_mode),
            enforce_parent_gate_closure=not global_edge_discovery,
        )
        edge_interventions.append(
            {
                "disabled_edges": [list(edge) for edge in disabled_edges],
                "quantum_accuracy": intervention["quantum_accuracy"],
                "accuracy_change": (
                    intervention["quantum_accuracy"]
                    - eval_metrics["quantum_accuracy"]
                ),
                "quantum_nll_bits": intervention["quantum_nll_bits"],
                "nll_bits_change": (
                    intervention["quantum_nll_bits"]
                    - eval_metrics["quantum_nll_bits"]
                ),
            }
        )
    greedy_count = min(int(args.greedy_max_examples), len(task.eval_examples))
    greedy_metrics = _evaluate_greedy(
        source,
        readout,
        modules,
        task,
        list(task.eval_examples[:greedy_count]),
        final_existence,
        executed_edges,
        device=device,
        routing_mode=str(config.q_routing_mode),
        enforce_parent_gate_closure=not global_edge_discovery,
    ) if greedy_count > 0 else None

    measured_executable = embedding_parameter_count + parameter_count(modules) + parameter_count((readout,))
    summary = {
        "method": (
            "quanta_model_v2_global_message_edges"
            if global_edge_discovery
            else "quanta_model_v1_event_only"
        ),
        "source_run": str(run_dir),
        "priority_source": {
            "directory": str(
                run_dir
                if global_edge_discovery and priority_dir is None
                else (
                    run_dir / "factorization"
                    if priority_dir is None
                    else priority_dir
                )
            ),
            "raw_stage_b": bool(global_edge_discovery),
        },
        "priority_method": priority_summary["method"],
        "priority_identity": discovery_priority_summary.get(
            "priority_identity", "unspecified"
        ),
        "source_data_equivalence": {
            "source_batch_identity": (
                "full population" if batch_indices is None else "saved example indices"
            ),
            "same_order": True,
            "source_and_q_batch_size": int(config.batch_size),
            "q_microbatch_size": int(config.q_microbatch_size),
            "q_update_frequency": (
                "one Q-model Adam update after every source update"
                if config.q_training_mode == "online_trajectory"
                else "one Q-model Adam update per IID sampled source checkpoint"
            ),
            "source_replay_max_parameter_error": replay_max_error,
            "source_replay_absolute_tolerance": float(args.replay_tolerance),
        },
        "trajectory_alignment": (
            "At source step s, the configured source optimizer first produces "
            "theta_s from batch B_s; the Q objective then uses B_s, source "
            "checkpoint theta_s, and e_q(s)."
            if config.q_training_mode == "online_trajectory"
            else (
                "A single fixed Q-model is optimized on IID draws of "
                f"{int(config.q_iid_checkpoint_count)} uniformly spaced post-update "
                "source checkpoints; each draw uses theta_s, the full training "
                "panel, and e_q(s)."
            )
        ),
        "q_training": {
            "mode": str(config.q_training_mode),
            "q_optimizer_steps": q_optimizer_steps,
            "q_checkpoint_steps": sorted(q_checkpoint_steps),
            "iid_checkpoint_count": (
                None
                if iid_replay_steps is None
                else int(len(iid_replay_steps))
            ),
            "iid_checkpoint_steps": (
                None
                if iid_replay_steps is None
                else iid_replay_steps.tolist()
            ),
        },
        "quantum_counts_by_layer": list(quantum_counts),
        "attention_rank_per_quantum": int(config.q_attention_rank),
        "attention_value_dim_per_quantum": int(config.q_attention_value_dim),
        "attention_direct_residual": bool(config.q_attention_direct_residual),
        "share_attention_projections": bool(config.q_share_attention_projections),
        "attention_query_offsets": bool(config.q_attention_query_offsets),
        "routing_mode": str(config.q_routing_mode),
        "reconstruction_positions": str(config.q_reconstruction_positions),
        "writer_mode": (
            f"{config.q_writer_sparsity}_{config.q_writer_activation}_"
            "unit_normalized_writers"
            + (
                "_with_controlled_attention_residual"
                if config.q_attention_direct_residual
                else ""
            )
        ),
        "normalization": "fixed_layer_norm",
        "writer_counts_by_layer": [value.tolist() for value in writer_counts],
        "writer_jump_thresholds_by_layer": [
            module.writer_jump_threshold.detach().abs().cpu().tolist()
            for module in modules
        ],
        "writer_batch_topk_cutoffs_by_layer": [
            module.writer_cutoffs.detach().cpu().tolist() for module in modules
        ],
        "writer_allocation": {
            "mode": writer_allocation,
            "total_budget": config.q_writer_total_budget,
            "functional_rank_energy": float(config.q_functional_rank_energy),
            "functional_complexity_by_layer": [value.complexity.tolist() for value in functional],
            "functional_eigenvalues_by_layer": [[spectrum.tolist() for spectrum in value.eigenvalues] for value in functional],
            "zero_energy_by_layer": [value.zero_energy.tolist() for value in functional],
            "total_writers": int(sum(value.sum() for value in writer_counts)),
        },
        "edge_discovery": {
            "mode": str(config.q_edge_discovery),
            "candidate_edges": [list(edge) for edge in candidate_edges],
            "candidate_count": len(candidate_edges),
            "selected_edges": [list(edge) for edge in executed_edges],
            "selected_count": len(executed_edges),
            "global_edge_probability_records": edge_probability_records,
            "raw_stage_b_supports_and_temporal_priority": global_edge_discovery,
            "existence_closure_applied": not global_edge_discovery,
            "parent_gate_closure_applied": not global_edge_discovery,
            "router_gradients": (
                "reconstruction_plus_output_kl_plus_global_edge_l0"
                + (
                    "_plus_global_edge_binary_entropy"
                    if float(config.q_edge_entropy_weight) > 0.0
                    else ""
                )
                if global_edge_discovery
                else "not_applicable"
            ),
        },
        "discovered_edges": [list(edge) for edge in executed_edges],
        "executed_edges": [list(edge) for edge in executed_edges],
        "message_interface": (
            "initial_residual_plus_declared_parent_messages"
            if config.q_routing_mode == "initial_residual_plus_parents"
            else "edge_children_receive_only_parent_messages"
        ),
        "readout": "frozen copy of current source final norm and token head",
        "parameter_count": {
            "shared_source_embeddings": embedding_parameter_count,
            "quantum_layers": parameter_count(modules),
            "quantum_readout": parameter_count((readout,)),
            "executable_total": measured_executable,
            "source_total": source_parameter_count,
            "fraction_of_source": measured_executable / source_parameter_count,
        },
        "training": {
            "optimizer": "Adam",
            "steps": q_optimizer_steps,
            "full_batch_size": int(config.batch_size),
            "microbatch_size": int(config.q_microbatch_size),
            "learning_rate": float(config.q_learning_rate),
            "gate_gradient_routing": {
                "functional_objective_to_execution_gates": not config.detach_gate_gradients,
                "gate_bce_to_gate_logits": True,
                "writer_l0_to_gate_logits": False,
                "gate_entropy_to_gate_logits": float(config.q_gate_entropy_weight) > 0.0,
            },
            "objective": {
                (
                    "all_valid_position_block_update_relative_sse"
                    if config.q_reconstruction_positions == "all_valid_positions"
                    else "event_position_block_update_relative_sse"
                ): EVENT_RECONSTRUCTION_WEIGHT,
                "source_output_kl": float(config.q_output_kl_weight),
                "support_gate_bce": GATE_SUPERVISION_WEIGHT,
                "batch_topk_average_writers": float(config.q_writer_average_k),
                "writer_activation": str(config.q_writer_activation),
                "writer_sparsity": str(config.q_writer_sparsity),
                "jumprelu_fixed_threshold": float(config.q_writer_jump_threshold),
                "jumprelu_surrogate_bandwidth": float(config.q_writer_jump_bandwidth),
                "writer_l0_scope": str(config.q_writer_l0_scope),
                "writer_l0_target": float(config.q_writer_l0_target),
                "writer_l0_weight": float(config.q_writer_l0_weight),
                "writer_l0_ramp_steps": int(config.q_writer_l0_ramp_steps),
                "gate_binary_entropy": float(config.q_gate_entropy_weight),
                "global_edge_l0_weight": float(config.q_edge_l0_weight),
                "global_edge_l0_ramp_steps": int(config.q_edge_l0_ramp_steps),
                "global_edge_binary_entropy_weight": float(
                    config.q_edge_entropy_weight
                ),
                "global_edge_binary_entropy_start_step": int(
                    config.q_edge_entropy_start_step
                ),
                "global_edge_binary_entropy_ramp_steps": int(
                    config.q_edge_entropy_ramp_steps
                ),
                "global_edge_freeze_steps": int(config.q_edge_freeze_steps),
            },
            "balanced_gate_bce": bool(config.q_balance_gate_bce),
            "gate_binary_start_step": int(config.q_gate_entropy_start_step),
            "gate_binary_ramp_steps": int(config.q_gate_entropy_ramp_steps),
            "history": history,
        },
        "metrics": {
            "train": train_metrics,
            "eval": eval_metrics,
            "greedy_eval": greedy_metrics,
            "held_out_edge_zero_interventions": edge_interventions,
        },
        "scientific_status": (
            f"Priority identity: {discovery_priority_summary.get('priority_identity', 'unspecified')}. "
            + (
                "Global edge selection used raw Stage-B supports and temporal "
                "priority. A selected edge globally adds the parent's executed "
                "message to the child's initial-residual input without imposing "
                "activation or acquisition closure. Functional message use still "
                "requires intervention evidence."
                if global_edge_discovery
                else "Graph edges impose availability closure and replace an edged "
                "child's ordinary residual input with its parent message. Functional "
                "message use still requires intervention evidence."
            )
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    print(f"saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()
