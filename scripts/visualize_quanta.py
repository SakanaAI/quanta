from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image
import torch

from quanta.config import load_experiment_config
from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.number_naming.names import ONES, TENS, english_number_name
from quanta.experiments.number_naming.tokenizer import EOS, NumberNamingTokenizer
from quanta.experiments.quanta_discovery.qgraph import (
    QuantumGraphEdge,
    forward_quanta,
)
from quanta.experiments.quanta_discovery.qmodel import (
    CheapCausalAttentionQuantumLayer,
    QuantumReadout,
    freeze,
)
from quanta.experiments.quanta_discovery.visualization import (
    writer_activity,
    writer_box_layout,
    write_dynamics_figures,
)


BLUE = "#2f6f9f"
GREEN = "#2f8f5b"
ORANGE = "#e5821f"
RED = "#c94c4c"
TEXT = "#171717"
LIGHT = "#ffffff"
PANEL = "#fffdf9"
PANEL_EDGE = "#d8d1c8"
MUTED = "#7a746d"


@dataclass(frozen=True)
class ProbeFrame:
    number: int
    text: str
    mode: str
    step: int
    target: str
    prediction: str
    transformer_prediction: str
    qmodel_prefix: tuple[str, ...]
    transformer_prefix: tuple[str, ...]
    confidence: float
    top_tokens: tuple[tuple[str, float], ...]
    probabilities: tuple[np.ndarray, ...]
    active: tuple[np.ndarray, ...]
    contribution_norms: tuple[np.ndarray, ...]
    writer_activities: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Quanta-discovery v0")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostics = commands.add_parser(
        "dynamics", help="Plot acquisition curves and source-model improvement"
    )
    diagnostics.add_argument("run_dir", type=Path)
    diagnostics.add_argument("--output-dir", type=Path)

    probes = commands.add_parser(
        "probes", help="Animate hard quantum and scalar-writer activity"
    )
    probes.add_argument("run_dir", type=Path)
    probes.add_argument("--qmodel-dir", type=Path)
    probes.add_argument("--config", type=Path)
    probes.add_argument(
        "--probe", type=int, nargs="+", default=[7, 42, 132, 1005, 530802]
    )
    probes.add_argument("--output-dir", type=Path)
    probes.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    probes.add_argument(
        "--mode", choices=("teacher_forced", "greedy"), default="teacher_forced"
    )
    probes.add_argument("--frame-ms", type=int, default=950)
    probes.add_argument(
        "--hide-title",
        action="store_true",
        help="Omit the internal visualization title for paper-facing panels.",
    )
    probes.add_argument(
        "--compact-layout",
        action="store_true",
        help="Reduce the gap between token rows and the quantum network.",
    )
    return parser.parse_args()


def _canonical_tokenizer() -> NumberNamingTokenizer:
    return NumberNamingTokenizer([*ONES[1:], *TENS.values(), "hundred", "thousand"])


def _blend(left: str, right: str, amount: float) -> str:
    amount = 0.0 if not np.isfinite(amount) else float(np.clip(amount, 0.0, 1.0))
    a = np.asarray([int(left[index : index + 2], 16) for index in (1, 3, 5)])
    b = np.asarray([int(right[index : index + 2], 16) for index in (1, 3, 5)])
    rgb = np.rint((1.0 - amount) * a + amount * b).astype(int)
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _safe_token(tokenizer: NumberNamingTokenizer, token_id: int) -> str:
    token = tokenizer.id_to_token[int(token_id)]
    return "EOS" if token == EOS else token


def _prompt_ids(tokenizer: NumberNamingTokenizer, number: int) -> list[int]:
    return [
        tokenizer.bos_id,
        *(tokenizer.token_to_id[f"<D{digit}>"] for digit in str(int(number))),
        tokenizer.sep_id,
    ]


def _top_tokens(
    tokenizer: NumberNamingTokenizer, distribution: torch.Tensor
) -> tuple[tuple[str, float], ...]:
    probabilities, ids = torch.topk(distribution, k=min(4, distribution.numel()))
    return tuple(
        (_safe_token(tokenizer, int(token_id)), float(probability))
        for probability, token_id in zip(probabilities, ids)
    )


def _config_path(run_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    path = run_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError("pass --config because run_dir/config.json is missing")
    return path


def _load_visual_model(
    run_dir: Path,
    qmodel_dir: Path,
    config: Any,
    device: torch.device,
) -> tuple[
    DecoderTransformerLM,
    QuantumReadout,
    tuple[CheapCausalAttentionQuantumLayer, ...],
    dict[str, Any],
]:
    summary = json.loads((qmodel_dir / "summary.json").read_text())
    if summary.get("method") not in {
        "quanta_model_v0",
        "quanta_model_event_v1",
        "quanta_model_v1_event_only",
        "quanta_model_v2_global_message_edges",
    }:
        raise ValueError("visualization accepts only supported Q-model artifacts")
    metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text())
    state = torch.load(
        run_dir / metadata["checkpoint_files"][-1],
        map_location=device,
        weights_only=True,
    )
    source = DecoderTransformerLM(
        vocab_size=int(state["token_embedding.weight"].shape[0]),
        max_seq_len=int(config.max_seq_len),
        d_model=int(config.d_model),
        n_layers=int(config.n_layers),
        n_heads=int(config.n_heads),
        dropout=float(config.dropout),
        pad_id=0,
        mlp_ratio=float(config.mlp_ratio),
    ).to(device)
    source.load_state_dict(state, strict=True)
    freeze(source)
    modules = tuple(
        CheapCausalAttentionQuantumLayer(
            int(config.d_model),
            int(count),
            attention_rank=int(summary["attention_rank_per_quantum"]),
            attention_value_dim=int(summary["attention_value_dim_per_quantum"]),
            attention_direct_residual=bool(
                summary.get("attention_direct_residual", False)
            ),
            share_attention_projections=bool(
                summary.get(
                    "share_attention_projections",
                    config.q_share_attention_projections,
                )
            ),
            writer_counts=writer_counts,
            writer_average_k=float(summary.get("training", {}).get("objective", {}).get("batch_topk_average_writers", 1.0)),
            writer_activation=str(summary.get("training", {}).get("objective", {}).get("writer_activation", "jumprelu")),
            writer_sparsity=str(summary.get("training", {}).get("objective", {}).get("writer_sparsity", "batch_topk")),
            writer_jump_threshold=float(
                summary.get("training", {}).get("objective", {}).get(
                    "jumprelu_fixed_threshold",
                    summary.get("training", {}).get("objective", {}).get(
                        "jumprelu_initial_threshold", 0.0
                    ),
                )
            ),
            writer_jump_bandwidth=float(summary.get("training", {}).get("objective", {}).get("jumprelu_surrogate_bandwidth", 0.1)),
        ).to(device)
        for count, writer_counts in zip(
            summary["quantum_counts_by_layer"], summary["writer_counts_by_layer"]
        )
    )
    for layer, module in enumerate(modules):
        module.load_state_dict(
            torch.load(
                qmodel_dir / f"layer_{layer}.pt",
                map_location=device,
                weights_only=True,
            ),
            strict=True,
        )
        module.eval()
    readout = QuantumReadout(
        int(config.d_model), int(state["token_embedding.weight"].shape[0])
    ).to(device)
    readout_state = torch.load(
        qmodel_dir / "readout.pt", map_location=device, weights_only=True
    )
    if set(readout_state) == {"head.weight", "head.bias"}:
        # Legacy v0 checkpoints trained only a copied output head.  The current
        # readout additionally stores the source final normalization, which was
        # implicit in those artifacts.
        readout.copy_source_readout(source)
        readout.load_state_dict(readout_state, strict=False)
    else:
        readout.load_state_dict(readout_state, strict=True)
    readout.eval()
    return source, readout, modules, summary


def _load_existences(qmodel_dir: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    with np.load(qmodel_dir / "existence_by_step.npz") as payload:
        names = sorted(payload.files, key=lambda value: int(value.rsplit("_", 1)[1]))
        return tuple(
            torch.as_tensor(payload[name][-1], device=device)[None] for name in names
        )


def _edges(summary: dict[str, Any]) -> tuple[QuantumGraphEdge, ...]:
    return tuple(
        tuple(int(value) for value in edge) for edge in summary["executed_edges"]
    )


def _enforce_parent_gate_closure(summary: dict[str, Any]) -> bool:
    """Match the activation semantics used to fit the saved Q-model."""

    return summary.get("edge_discovery", {}).get("mode") != "global_static"


def _frame_activity(
    trace: Any,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    position: int,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    probabilities = tuple(
        torch.sigmoid(value[0, position]).cpu().numpy() for value in trace.gate_logits
    )
    active = tuple(
        value[0, position].cpu().numpy() >= 0.5 for value in trace.effective_gates
    )
    effective_probabilities = tuple(
        np.where(enabled, probability, 0.0)
        for probability, enabled in zip(probabilities, active)
    )
    norms = tuple(
        value[0, position].norm(dim=-1).cpu().numpy()
        for value in trace.contributions
    )
    writers = tuple(
        writer_activity(
            module,
            trace.routed_inputs[layer],
            trace.attention_contexts[layer],
            trace.effective_gates[layer],
            trace.writer_gates[layer],
            batch=0,
            position=position,
        )
        for layer, module in enumerate(modules)
    )
    return effective_probabilities, active, norms, writers


@torch.no_grad()
def _teacher_forced_trace(
    number: int,
    *,
    tokenizer: NumberNamingTokenizer,
    max_seq_len: int,
    source: DecoderTransformerLM,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    device: torch.device,
    routing_mode: str,
    enforce_parent_gate_closure: bool = True,
) -> tuple[ProbeFrame, ...]:
    text = english_number_name(number)
    encoded = tokenizer.encode(number, text, max_seq_len=max_seq_len)
    inputs = torch.as_tensor(encoded.input_ids, device=device)[None]
    mask = torch.ones_like(inputs)
    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        edges,
        hard_gates=True,
        routing_mode=routing_mode,
        enforce_parent_gate_closure=enforce_parent_gate_closure,
    )
    source_trace = source.residual_trace(inputs, mask)
    labels = np.asarray(encoded.labels[1:], dtype=int)
    positions = np.flatnonzero(labels != -100)
    frames = []
    for step, position in enumerate(positions):
        probability, active, norms, writers = _frame_activity(
            trace, modules, int(position)
        )
        distribution = torch.softmax(trace.logits[0, position], dim=-1)
        predicted = int(distribution.argmax())
        source_predicted = int(source_trace.logits[0, position].argmax())
        frames.append(
            ProbeFrame(
                number=number,
                text=text,
                mode="teacher_forced",
                step=step,
                target=_safe_token(tokenizer, int(labels[position])),
                prediction=_safe_token(tokenizer, predicted),
                transformer_prediction=_safe_token(tokenizer, source_predicted),
                qmodel_prefix=tuple(text.split()[:step]),
                transformer_prefix=tuple(text.split()[:step]),
                confidence=float(distribution[predicted]),
                top_tokens=_top_tokens(tokenizer, distribution),
                probabilities=probability,
                active=active,
                contribution_norms=norms,
                writer_activities=writers,
            )
        )
    return tuple(frames)


@torch.no_grad()
def _greedy_trace(
    number: int,
    *,
    tokenizer: NumberNamingTokenizer,
    max_seq_len: int,
    source: DecoderTransformerLM,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    device: torch.device,
    routing_mode: str,
    enforce_parent_gate_closure: bool = True,
) -> tuple[ProbeFrame, ...]:
    text = english_number_name(number)
    target_tokens = (*text.split(), "EOS")
    prompt = _prompt_ids(tokenizer, number)
    q_ids: list[int] = []
    source_ids: list[int] = []
    source_prefix: list[str] = []
    source_finished = False
    frames = []
    for step in range(min(len(target_tokens) + 3, max_seq_len - len(prompt))):
        q_sequence = torch.as_tensor([*prompt, *q_ids], device=device)[None]
        q_trace = forward_quanta(
            source,
            readout,
            modules,
            q_sequence,
            torch.ones_like(q_sequence),
            existences,
            edges,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        position = q_sequence.shape[1] - 1
        distribution = torch.softmax(q_trace.logits[0, position], dim=-1)
        predicted = int(distribution.argmax())
        if source_finished:
            source_prediction = "<finished>"
            source_predicted = tokenizer.eos_id
        else:
            source_sequence = torch.as_tensor([*prompt, *source_ids], device=device)[None]
            source_logits = source.residual_trace(
                source_sequence, torch.ones_like(source_sequence)
            ).logits
            source_predicted = int(source_logits[0, -1].argmax())
            source_prediction = _safe_token(tokenizer, source_predicted)
        probability, active, norms, writers = _frame_activity(q_trace, modules, position)
        frames.append(
            ProbeFrame(
                number=number,
                text=text,
                mode="greedy",
                step=step,
                target=target_tokens[step] if step < len(target_tokens) else "<past target>",
                prediction=_safe_token(tokenizer, predicted),
                transformer_prediction=source_prediction,
                qmodel_prefix=tuple(_safe_token(tokenizer, value) for value in q_ids),
                transformer_prefix=tuple(source_prefix),
                confidence=float(distribution[predicted]),
                top_tokens=_top_tokens(tokenizer, distribution),
                probabilities=probability,
                active=active,
                contribution_norms=norms,
                writer_activities=writers,
            )
        )
        q_ids.append(predicted)
        if not source_finished:
            source_ids.append(source_predicted)
            source_prefix.append(source_prediction)
            source_finished = source_predicted == tokenizer.eos_id
        if predicted == tokenizer.eos_id:
            break
    return tuple(frames)


def _positions(counts: Sequence[int]) -> dict[tuple[int, int], tuple[float, float]]:
    return {
        (layer, quantum): (
            0.36 + 1.16 * layer,
            0.88 - quantum * (0.76 / max(int(count) - 1, 1)),
        )
        for layer, count in enumerate(counts)
        for quantum in range(int(count))
    }


def _draw_elbow_arrow(
    ax: Any,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
    color: str,
    alpha: float,
    zorder: int = 2,
    arrowstyle: str = "-|>",
) -> None:
    """Draw one solid elbow route with rounded, monotonic corners."""

    x0, y0 = start
    x1, y1 = end
    mid_x = x0 + 0.58 * (x1 - x0)
    if abs(y1 - y0) < 1.0e-8:
        vertices = [(x0, y0), (x1, y1)]
        codes = [MatplotlibPath.MOVETO, MatplotlibPath.LINETO]
    else:
        direction = 1.0 if y1 > y0 else -1.0
        radius = min(0.035, 0.22 * abs(y1 - y0), 0.16 * abs(x1 - x0))
        vertices = [
            (x0, y0),
            (mid_x - radius, y0),
            (mid_x, y0),
            (mid_x, y0 + direction * radius),
            (mid_x, y1 - direction * radius),
            (mid_x, y1),
            (mid_x + radius, y1),
            (x1, y1),
        ]
        codes = [
            MatplotlibPath.MOVETO,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE3,
            MatplotlibPath.CURVE3,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE3,
            MatplotlibPath.CURVE3,
            MatplotlibPath.LINETO,
        ]
    ax.add_patch(
        FancyArrowPatch(
            path=MatplotlibPath(vertices, codes),
            arrowstyle=arrowstyle,
            mutation_scale=9,
            linewidth=width,
            color=color,
            alpha=alpha,
            fill=False,
            zorder=zorder,
        )
    )


def _draw_aligned_token_rows(
    axis: Any, frame: ProbeFrame, *, right: float, y: float = 1.34
) -> None:
    gold_tokens = (*frame.text.split(), "EOS")
    gold_visible = gold_tokens[: frame.step + 1]
    transformer_tokens = (*frame.transformer_prefix, frame.transformer_prediction)
    qmodel_tokens = (*frame.qmodel_prefix, frame.prediction)
    visible_count = max(len(gold_visible), len(transformer_tokens), len(qmodel_tokens))
    labels = ("Ground truth:", "Transformer:", "Q-model:")
    rows = (gold_visible, transformer_tokens, qmodel_tokens)
    gap = 0.055
    for row, label in enumerate(labels):
        axis.text(
            0.0,
            y - row * gap,
            label,
            fontsize=9.2,
            color=TEXT,
            fontweight="bold",
            va="center",
        )
    cursor = 0.55
    for index in range(visible_count):
        if cursor > right - 0.75:
            axis.text(cursor, y - gap, "...", fontsize=9, color=MUTED)
            break
        values = tuple(tokens[index] if index < len(tokens) else "" for tokens in rows)
        current = index == frame.step
        for row, value in enumerate(values):
            color = TEXT
            if current and row > 0 and values[0]:
                color = GREEN if value == values[0] else RED
            axis.text(
                cursor,
                y - row * gap,
                value,
                fontsize=8.8,
                color=color,
                fontweight="bold" if current else "normal",
                va="center",
            )
        cursor += max(0.22, 0.05 * max(map(len, values), default=1) + 0.11)


def _draw_top_tokens(
    axis: Any, frame: ProbeFrame, *, right: float, y: float = 1.43
) -> None:
    token_x = right - 0.44
    bar_x = right - 0.40
    bar_width = 0.33
    axis.text(
        bar_x + 0.5 * bar_width,
        y + 0.05,
        "Top tokens",
        fontsize=9.5,
        color=TEXT,
        fontweight="bold",
        ha="center",
    )
    for row, (token, probability) in enumerate(frame.top_tokens):
        yy = y - row * 0.052
        axis.text(token_x, yy, token, fontsize=8.0, color=TEXT, ha="right", va="center")
        axis.add_patch(
            plt.Rectangle(
                (bar_x, yy - 0.017),
                bar_width,
                0.034,
                facecolor="#fff9f1",
                edgecolor=PANEL_EDGE,
                linewidth=0.6,
                zorder=1,
            )
        )
        axis.add_patch(
            plt.Rectangle(
                (bar_x, yy - 0.017),
                bar_width * probability,
                0.034,
                facecolor=ORANGE if token == frame.target else _blend(LIGHT, ORANGE, 0.45),
                edgecolor="none",
                zorder=2,
            )
        )
        axis.text(
            bar_x + bar_width - 0.01,
            yy,
            f"{probability:.2f}",
            fontsize=6.7,
            color=TEXT,
            ha="right",
            va="center",
            zorder=3,
        )


def _render_frame(
    frame: ProbeFrame,
    *,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    edges: Sequence[QuantumGraphEdge],
    output: Path,
    show_title: bool = True,
    compact_layout: bool = False,
) -> None:
    counts = [module.quantum_count for module in modules]
    positions = _positions(counts)
    if compact_layout:
        positions = {key: (x, y + 0.10) for key, (x, y) in positions.items()}
    width = 1.16 * max(len(counts) - 1, 1) + 1.35
    figure, axis = plt.subplots(
        figsize=(12.2, 7.7) if compact_layout else (12.2, 9.4), dpi=170
    )
    figure.patch.set_facecolor(LIGHT)
    axis.set_facecolor(LIGHT)
    if show_title:
        axis.set_title(
            f"Quanta v0 · {frame.mode.replace('_', ' ')} · hard gates and scalar writers",
            fontsize=13,
            fontweight="bold",
            pad=7,
        )
    axis.text(
        0.0,
        1.44 if compact_layout else 1.49,
        rf"${frame.number:,}\;\longrightarrow$ {frame.text}",
        fontsize=9.5,
        fontweight="bold", color=TEXT,
    )
    _draw_aligned_token_rows(
        axis, frame, right=width, y=1.27 if compact_layout else 1.34
    )
    _draw_top_tokens(
        axis, frame, right=width, y=1.33 if compact_layout else 1.43
    )

    output_x = positions[(len(counts) - 1, 0)][0] + 0.74
    axis.plot(
        [output_x, output_x],
        [0.06, 1.03 if compact_layout else 0.96],
        color=ORANGE,
        linewidth=1.2,
    )
    axis.annotate(
        "",
        xy=(output_x, 0.015),
        xytext=(output_x, 0.095),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.2},
    )
    axis.text(
        output_x + 0.06,
        0.97 if compact_layout else 0.90,
        "Q-residual",
        ha="left",
        va="center",
        fontsize=8,
        color=ORANGE,
    )

    # Draw direct quantum-to-residual routes first. Each quantum has its own
    # orthogonal lane so these routes do not merge with graph messages.
    residual_y_top = 0.84 if compact_layout else 0.76
    for layer, module in enumerate(modules):
        layer_max = max(float(frame.writer_activities[layer].max()), 1.0e-12)
        owners = module.writer_owner.cpu().numpy()
        for quantum in range(module.quantum_count):
            x, y = positions[(layer, quantum)]
            probability = float(frame.probabilities[layer][quantum])
            active = bool(frame.active[layer][quantum])
            axis.scatter(
                [x], [y], s=500,
                color=_blend(LIGHT, ORANGE, 0.08 + 0.84 * probability),
                edgecolor=ORANGE if active else PANEL_EDGE,
                linewidth=1.0,
                zorder=4,
            )
            axis.text(
                x, y, rf"$q_{{{quantum}}}$", ha="center", va="center", fontsize=9,
                fontweight="bold" if active else "normal", zorder=5,
            )
            writer_indices = np.flatnonzero(owners == quantum)
            boxes = writer_box_layout(len(writer_indices), center=(x, y))
            for box, writer in zip(boxes, writer_indices):
                bx, by, size = box
                intensity = float(frame.writer_activities[layer][writer]) / layer_max
                axis.add_patch(
                    plt.Rectangle(
                        (bx, by), size * 0.88, size * 0.88,
                        facecolor=_blend(LIGHT, ORANGE, intensity),
                        edgecolor=PANEL_EDGE,
                        linewidth=0.2,
                        zorder=4,
                    )
                )
            # A hard-active quantum always writes to the residual.  Omit inactive
            # routes rather than drawing an ambiguous, low-opacity background line.
            if active:
                residual_y = residual_y_top - 0.18 * layer
                residual_y += 0.018 * (
                    quantum - 0.5 * (module.quantum_count - 1)
                )
                _draw_elbow_arrow(
                    axis,
                    start=(x + 0.09, y),
                    end=(output_x - 0.04, residual_y),
                    width=0.7 + min(
                        0.7, 0.5 * float(frame.contribution_norms[layer][quantum])
                    ),
                    color=ORANGE,
                    alpha=0.9,
                    zorder=1,
                )
        axis.text(
            positions[(layer, 0)][0], 1.08 if compact_layout else 0.96,
            f"Layer {layer}: {int(frame.active[layer].sum())}/{module.quantum_count} active",
            ha="center", fontsize=9, fontweight="bold", color=TEXT,
        )

    # Draw graph messages after residual routes so causal edges remain primary.
    for parent_layer, parent, child_layer, child in edges:
        parent_position = positions[(parent_layer, parent)]
        child_position = positions[(child_layer, child)]
        used = bool(frame.active[parent_layer][parent] and frame.active[child_layer][child])
        _draw_elbow_arrow(
            axis,
            start=(parent_position[0] + 0.07, parent_position[1]),
            end=(child_position[0] - 0.07, child_position[1]),
            width=1.7 if used else 0.8,
            color=ORANGE if used else PANEL_EDGE,
            alpha=0.9 if used else 0.45,
            zorder=3,
        )
    axis.text(
        width, -0.04,
        "Small boxes below each quantum are its writers; orange intensity is the "
        "writer's gated contribution magnitude.",
        fontsize=8, color=MUTED, ha="right",
    )
    axis.set_xlim(-0.25, width + 0.25)
    axis.set_ylim(-0.08, 1.49 if compact_layout else 1.56)
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor=LIGHT)
    plt.close(figure)


def _write_gif(frame_paths: Sequence[Path], output: Path, frame_ms: int) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    if not images:
        raise ValueError("a probe produced no frames")
    try:
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=int(frame_ms),
            loop=0,
            optimize=True,
        )
    finally:
        for image in images:
            image.close()


def _frame_record(frame: ProbeFrame) -> dict[str, Any]:
    return {
        "number": frame.number,
        "step": frame.step,
        "target": frame.target,
        "prediction": frame.prediction,
        "transformer_prediction": frame.transformer_prediction,
        "gate_probabilities": [value.tolist() for value in frame.probabilities],
        "hard_active": [value.astype(int).tolist() for value in frame.active],
        "writer_activities": [value.tolist() for value in frame.writer_activities],
    }


def _run_probes(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    qmodel_dir = args.qmodel_dir.resolve() if args.qmodel_dir else run_dir / "qmodel"
    output_dir = args.output_dir.resolve() if args.output_dir else qmodel_dir / "visualizations"
    device = torch.device(args.device)
    config = load_experiment_config(
        "quanta_discovery", _config_path(run_dir, args.config)
    )
    source, readout, modules, summary = _load_visual_model(
        run_dir, qmodel_dir, config, device
    )
    tokenizer = _canonical_tokenizer()
    if tokenizer.vocab_size != source.token_embedding.num_embeddings:
        raise ValueError("canonical tokenizer does not match the source checkpoint")
    existences = _load_existences(qmodel_dir, device)
    edges = _edges(summary)
    routing_mode = str(summary.get("routing_mode", config.q_routing_mode))
    enforce_parent_gate_closure = _enforce_parent_gate_closure(summary)
    trace_function = _teacher_forced_trace if args.mode == "teacher_forced" else _greedy_trace
    manifest: dict[str, Any] = {
        "method": "quanta_v0_probe_activity",
        "mode": args.mode,
        "enforce_parent_gate_closure": enforce_parent_gate_closure,
        "writer_box_semantics": "gated scalar-writer contribution magnitude",
        "probes": {},
    }
    for number in args.probe:
        frames = trace_function(
            int(number),
            tokenizer=tokenizer,
            max_seq_len=int(config.max_seq_len),
            source=source,
            readout=readout,
            modules=modules,
            existences=existences,
            edges=edges,
            device=device,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        suffix = "" if args.mode == "teacher_forced" else "_greedy"
        frame_dir = output_dir / f"probe_{number}{suffix}_frames"
        paths = []
        for index, frame in enumerate(frames):
            path = frame_dir / f"frame_{index:02d}.png"
            _render_frame(
                frame,
                modules=modules,
                edges=edges,
                output=path,
                show_title=not args.hide_title,
                compact_layout=args.compact_layout,
            )
            paths.append(path)
        gif = output_dir / f"probe_{number}{suffix}_activity.gif"
        _write_gif(paths, gif, args.frame_ms)
        manifest["probes"][str(number)] = {
            "gif": str(gif),
            "frames": [_frame_record(frame) for frame in frames],
        }
        print(gif)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"probe_activity_{args.mode}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


def main() -> None:
    args = parse_args()
    if args.command == "dynamics":
        run_dir = args.run_dir.resolve()
        output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "visualizations"
        manifest = write_dynamics_figures(run_dir, output_dir)
        print(json.dumps(manifest, indent=2))
    else:
        _run_probes(args)


if __name__ == "__main__":
    main()
