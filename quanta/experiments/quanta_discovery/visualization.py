from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .qmodel import CheapCausalAttentionQuantumLayer


EPSILON = 1.0e-12


@dataclass(frozen=True)
class DynamicsDiagnostics:
    checkpoint_steps: np.ndarray
    temporal_priority: tuple[np.ndarray, ...]
    cumulative_priority: tuple[np.ndarray, ...]
    model_improvement: np.ndarray
    improvement_label: str


def normalized_cumulative_priority(curves: np.ndarray) -> np.ndarray:
    """Turn nonnegative priority mass into one 0-to-1 acquisition curve."""

    values = np.maximum(np.asarray(curves, dtype=np.float64), 0.0)
    if values.ndim != 2:
        raise ValueError("priority curves must have shape [quantum, interval]")
    cumulative = np.concatenate(
        (np.zeros((len(values), 1)), np.cumsum(values, axis=1)), axis=1
    )
    totals = cumulative[:, -1:]
    return np.divide(
        cumulative,
        totals,
        out=np.zeros_like(cumulative),
        where=totals > EPSILON,
    )


def _factorization_dir(run_dir: Path) -> Path:
    path = run_dir / "factorization"
    if not path.exists():
        raise FileNotFoundError(f"missing composition artifact: {path}")
    return path


def _load_curves(run_dir: Path) -> tuple[np.ndarray, ...]:
    with np.load(_factorization_dir(run_dir) / "temporal_priority.npz") as payload:
        names = sorted(payload.files, key=lambda value: int(value.rsplit("_", 1)[1]))
        return tuple(np.asarray(payload[name], dtype=np.float64) for name in names)


def _checkpoint_steps(run_dir: Path, interval_count: int) -> np.ndarray:
    metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text())
    steps = np.asarray(metadata["checkpoint_steps"], dtype=np.int64)
    if len(steps) != int(interval_count) + 1:
        raise ValueError("checkpoint count does not align with priority intervals")
    return steps


def _source_improvement(
    run_dir: Path, checkpoint_steps: np.ndarray
) -> tuple[np.ndarray, str]:
    evaluations = run_dir / "checkpoint_evaluation.json"
    if evaluations.exists():
        records = json.loads(evaluations.read_text())
        steps = np.asarray([record["step"] for record in records], dtype=np.int64)
        if not np.array_equal(steps, checkpoint_steps):
            raise ValueError("checkpoint evaluation steps do not align with dynamics")
        nll = np.asarray([record["nll_bits"] for record in records], dtype=np.float64)
        return nll[0] - nll, "held-out NLL improvement (bits/event)"

    losses = np.load(run_dir / "losses.npy", mmap_mode="r")
    metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text())
    train_count = int(metadata["num_train_prediction_events"])
    if losses.shape[1] != len(checkpoint_steps):
        raise ValueError("event-loss checkpoints do not align with dynamics")
    eval_losses = np.asarray(losses[train_count:], dtype=np.float64)
    if len(eval_losses) == 0:
        eval_losses = np.asarray(losses[:train_count], dtype=np.float64)
        label = "training-panel NLL improvement (nats/event)"
    else:
        label = "held-out NLL improvement (nats/event)"
    mean_loss = eval_losses.mean(axis=0)
    return mean_loss[0] - mean_loss, label


def load_dynamics_diagnostics(run_dir: Path) -> DynamicsDiagnostics:
    run_dir = run_dir.resolve()
    temporal = _load_curves(run_dir)
    if not temporal:
        raise ValueError("factorization contains no layers")
    interval_count = temporal[0].shape[1]
    if any(value.shape[1] != interval_count for value in temporal):
        raise ValueError("all layers must share priority intervals")
    steps = _checkpoint_steps(run_dir, interval_count)
    improvement, label = _source_improvement(run_dir, steps)
    return DynamicsDiagnostics(
        checkpoint_steps=steps,
        temporal_priority=temporal,
        cumulative_priority=tuple(
            normalized_cumulative_priority(value) for value in temporal
        ),
        model_improvement=improvement,
        improvement_label=label,
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > EPSILON else 0.0


def priority_improvement_correlations(
    diagnostics: DynamicsDiagnostics,
) -> dict[str, object]:
    curves = [
        (layer, quantum, values)
        for layer, matrix in enumerate(diagnostics.cumulative_priority)
        for quantum, values in enumerate(matrix)
    ]
    aggregate = np.mean([values for _, _, values in curves], axis=0)
    return {
        "aggregate_pearson": _pearson(aggregate, diagnostics.model_improvement),
        "by_quantum": [
            {
                "layer": layer,
                "quantum": quantum,
                "pearson": _pearson(values, diagnostics.model_improvement),
            }
            for layer, quantum, values in curves
        ],
    }


def plot_cumulative_priority(
    diagnostics: DynamicsDiagnostics, output: Path
) -> Path:
    layers = len(diagnostics.cumulative_priority)
    figure, axes = plt.subplots(layers, 1, figsize=(10, 2.8 * layers), squeeze=False)
    for layer, values in enumerate(diagnostics.cumulative_priority):
        axis = axes[layer, 0]
        for quantum, curve in enumerate(values):
            axis.plot(diagnostics.checkpoint_steps, curve, label=f"Q{quantum}")
        axis.set_title(f"Layer {layer}: cumulative priority / existence schedule")
        axis.set_ylabel("normalized cumulative mass")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
        if len(values) <= 16:
            axis.legend(ncol=min(8, len(values)), fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel("source optimizer step")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_priority_improvement_correlation(
    diagnostics: DynamicsDiagnostics, output: Path
) -> tuple[Path, dict[str, object]]:
    records = priority_improvement_correlations(diagnostics)
    flat_curves = [
        curve for layer in diagnostics.cumulative_priority for curve in layer
    ]
    aggregate = np.mean(flat_curves, axis=0)
    quantum_records = records["by_quantum"]
    labels = [f"L{item['layer']}Q{item['quantum']}" for item in quantum_records]
    correlations = [float(item["pearson"]) for item in quantum_records]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(
        diagnostics.checkpoint_steps,
        aggregate,
        color="#e5821f",
        linewidth=2.0,
        label="mean quantum acquisition",
    )
    scaled_improvement = diagnostics.model_improvement
    maximum = float(np.max(np.abs(scaled_improvement)))
    if maximum > EPSILON:
        scaled_improvement = scaled_improvement / maximum
    axes[0].plot(
        diagnostics.checkpoint_steps,
        scaled_improvement,
        color="#2f6f9f",
        linewidth=2.0,
        label="normalized model improvement",
    )
    axes[0].set_title(f"Trajectory correlation r={records['aggregate_pearson']:.3f}")
    axes[0].set_xlabel("source optimizer step")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    colors = ["#2f8f5b" if value >= 0.0 else "#c94c4c" for value in correlations]
    axes[1].bar(np.arange(len(labels)), correlations, color=colors)
    axes[1].axhline(0.0, color="#7a746d", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=75, fontsize=7)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_title(f"Each acquisition curve vs {diagnostics.improvement_label}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output, records


def write_dynamics_figures(run_dir: Path, output_dir: Path) -> dict[str, object]:
    diagnostics = load_dynamics_diagnostics(run_dir)
    cumulative = plot_cumulative_priority(
        diagnostics, output_dir / "cumulative_priority.png"
    )
    correlation, records = plot_priority_improvement_correlation(
        diagnostics, output_dir / "priority_model_correlation.png"
    )
    manifest = {
        "cumulative_priority": str(cumulative),
        "priority_model_correlation": str(correlation),
        "improvement_label": diagnostics.improvement_label,
        "correlations": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dynamics_visualization.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def writer_activity(
    module: CheapCausalAttentionQuantumLayer,
    routed_inputs: torch.Tensor,
    attention_context: torch.Tensor,
    effective_gates: torch.Tensor,
    writer_gates: torch.Tensor,
    *,
    batch: int,
    position: int,
) -> np.ndarray:
    """Return one nonnegative contribution proxy per scalar writer."""

    normalized = module.norm(routed_inputs[int(batch), int(position)])
    writer_inputs = normalized.index_select(0, module.writer_owner)
    local = (writer_inputs * module.local_weight).sum(dim=-1) + module.local_bias
    context = attention_context[int(batch), int(position)].index_select(
        0, module.writer_owner
    )
    features = F.gelu(local + (context * module.context_scale).sum(dim=-1))
    direction_norm = module.output_weight.norm(dim=-1)
    owner_gates = effective_gates[int(batch), int(position)].index_select(
        0, module.writer_owner
    )
    execution = writer_gates[int(batch), int(position)]
    return (
        features.abs() * direction_norm * owner_gates * execution
    ).detach().cpu().numpy()


def writer_box_layout(
    count: int,
    *,
    center: tuple[float, float],
    max_width: float = 0.14,
    max_height: float = 0.045,
) -> tuple[tuple[float, float, float], ...]:
    """Pack writer squares beneath a quantum within a fixed visual footprint."""

    count = int(count)
    if count <= 0:
        return ()
    columns = max(1, int(np.ceil(np.sqrt(count * max_width / max_height))))
    rows = int(np.ceil(count / columns))
    size = min(max_width / columns, max_height / rows)
    used_columns = min(count, columns)
    left = float(center[0]) - 0.5 * used_columns * size
    top = float(center[1]) - 0.042
    return tuple(
        (
            left + (index % columns) * size,
            top - (index // columns + 1) * size,
            size,
        )
        for index in range(count)
    )
