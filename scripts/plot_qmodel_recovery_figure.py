#!/usr/bin/env python3
"""Build paper-facing Q-discovery dynamics and probe activation panels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from quanta.experiments.quanta_discovery.visualization import (
    DynamicsDiagnostics,
    load_dynamics_diagnostics,
    normalized_cumulative_priority,
)


ORANGE = "#e5821f"
LIGHT_ORANGE = "#fff8ef"
TEXT = "#222222"
MUTED = "#706b65"
GRID = "#ded8d0"
LAYER_COLORS = ("#3f78a8", "#3d8b63", "#d97706")


@dataclass(frozen=True)
class Description:
    concept: str
    f1: float


@dataclass(frozen=True)
class SemanticAudit:
    quanta: dict[str, Description]
    writers: dict[str, Description]


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text())


def _description(record: dict[str, Any], field: str) -> Description:
    hypothesis = record[field]["all"]
    return Description(
        concept=humanize_concept(str(hypothesis["concept"])),
        f1=float(hypothesis["test"]["f1"]),
    )


def load_semantic_audit(audit_dir: Path) -> SemanticAudit:
    audit = load_json(audit_dir / "audit.json")
    dashboards = load_json(audit_dir / "writer_dashboards.json")
    if not isinstance(audit, dict) or not isinstance(dashboards, list):
        raise ValueError("unexpected semantic-audit format")
    quantum_records = audit["semantics"]["quantum"]["features"]
    return SemanticAudit(
        quanta={
            str(record["name"]): _description(record, "compound_hypotheses")
            for record in quantum_records
        },
        writers={
            str(record["name"]): _description(
                record, "conditional_compound_hypotheses"
            )
            for record in dashboards
        },
    )


def humanize_concept(concept: str) -> str:
    """Render the audit's exact concept syntax as a compact display label."""

    value_names = {
        "CHUNK_HAS_TAIL": "chunk has tail",
        "CHUNK_HAS_HUNDREDS": "chunk has hundreds",
        "EMIT_VALUE": "emit value",
        "EOS_DECISION": "EOS decision",
        "FORM_UNIT": "unit form",
        "GROUP_LOW": "low chunk",
        "ONE_HUNDRED": "one-hundred rule",
        "UNIT_LEX": "unit lexeme",
        "START": "sequence start",
    }

    def replace(match: re.Match[str]) -> str:
        field, raw_value = match.groups()
        value = value_names.get(raw_value, raw_value.replace("_", " ").lower())
        if field == "previous_target":
            return f"after {value}"
        if field == "target_position":
            return f"output position {value}"
        if field == "digit_length":
            return f"{value}-digit input"
        if field == "value_role":
            return f"{value} value"
        return value

    readable = re.sub(r"([a-z_]+)=([A-Za-z0-9_]+)", replace, concept)
    readable = readable.replace(" OR ", " or ").replace(" AND ", " and ")
    readable = readable.replace("(", "").replace(")", "")
    return readable


def _blend(left: str, right: str, amount: float) -> tuple[float, float, float]:
    start = np.asarray(to_rgb(left))
    end = np.asarray(to_rgb(right))
    return tuple(start + float(np.clip(amount, 0.0, 1.0)) * (end - start))


def _normalized_improvement(diagnostics: DynamicsDiagnostics) -> np.ndarray:
    improvement = np.asarray(diagnostics.model_improvement, dtype=np.float64)
    scale = float(np.max(np.abs(improvement)))
    return improvement / scale if scale > 0.0 else improvement


def load_factorization_dynamics(
    source_run: Path, factorization_dir: Path | None = None
) -> DynamicsDiagnostics:
    """Load source improvement with curves from a selected candidate basis."""

    diagnostics = load_dynamics_diagnostics(source_run)
    if factorization_dir is None:
        return diagnostics
    with np.load(factorization_dir / "temporal_priority.npz") as payload:
        names = sorted(
            payload.files, key=lambda value: int(value.rsplit("_", 1)[1])
        )
        temporal = tuple(
            np.asarray(payload[name], dtype=np.float64) for name in names
        )
    if not temporal:
        raise ValueError("selected factorization contains no layers")
    interval_count = len(diagnostics.checkpoint_steps) - 1
    if any(curves.shape[1] != interval_count for curves in temporal):
        raise ValueError(
            "selected factorization does not align with source checkpoints"
        )
    return DynamicsDiagnostics(
        checkpoint_steps=diagnostics.checkpoint_steps,
        temporal_priority=temporal,
        cumulative_priority=tuple(
            normalized_cumulative_priority(curves) for curves in temporal
        ),
        model_improvement=diagnostics.model_improvement,
        improvement_label=diagnostics.improvement_label,
    )


def plot_discovery_dynamics(
    diagnostics: DynamicsDiagnostics, output_stem: Path
) -> dict[str, Any]:
    quantum_count = sum(map(len, diagnostics.cumulative_priority))
    figure_height = 4.1 if quantum_count > 9 else 3.25
    figure, axis = plt.subplots(figsize=(4.5, figure_height))
    styles = ("-", "--", "-.", ":")
    acquisition_steps: dict[str, int | None] = {}
    for layer, curves in enumerate(diagnostics.cumulative_priority):
        for quantum, curve in enumerate(curves):
            name = f"L{layer}Q{quantum}"
            color = _blend(
                LAYER_COLORS[layer],
                "#ffffff",
                0.12 * quantum / max(len(curves) - 1, 1),
            )
            axis.plot(
                diagnostics.checkpoint_steps,
                curve,
                color=color,
                linestyle=styles[quantum % len(styles)],
                linewidth=1.65,
                label=name,
                zorder=2,
            )
            crossings = np.flatnonzero(curve >= 0.5)
            step = (
                int(diagnostics.checkpoint_steps[crossings[0]])
                if len(crossings)
                else None
            )
            acquisition_steps[name] = step
            if step is not None:
                axis.scatter(step, 0.5, s=13, color=color, zorder=3)
    axis.plot(
        diagnostics.checkpoint_steps,
        _normalized_improvement(diagnostics),
        color=TEXT,
        linestyle=(0, (4, 2)),
        linewidth=2.0,
        label="held-out NLL improvement",
        zorder=1,
    )
    axis.set(
        xlim=(0, min(3_500, int(diagnostics.checkpoint_steps[-1]))),
        ylim=(-0.03, 1.03),
        xlabel="source optimizer step",
        ylabel="normalized cumulative mass",
    )
    axis.grid(alpha=0.22, linewidth=0.55)
    legend_columns = 4 if quantum_count > 9 else 3
    axis.legend(
        ncol=legend_columns,
        fontsize=6.2 if legend_columns == 4 else 7.0,
        frameon=False,
        loc="lower right",
        columnspacing=0.9,
        handlelength=2.0,
    )
    figure.tight_layout(pad=0.45)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    return {"acquisition_50pct_steps": acquisition_steps}


def _quantum_rows(writer_counts: list[list[int]]) -> list[tuple[int, int, int, int]]:
    rows = []
    for layer, counts in enumerate(writer_counts):
        offset = 0
        for quantum, count in enumerate(counts):
            rows.append((layer, quantum, offset, int(count)))
            offset += int(count)
    return rows


def _active_writers(
    frame: dict[str, Any], rows: list[tuple[int, int, int, int]]
) -> list[tuple[str, float]]:
    active = []
    activities = frame["writer_activities"]
    for layer, quantum, offset, count in rows:
        for local_writer in range(count):
            writer = offset + local_writer
            magnitude = float(activities[layer][writer])
            if magnitude > 0.0:
                active.append((f"L{layer}.Q{quantum}.W{writer}", magnitude))
    return active


def plot_probe_activation_summary(
    frames: list[dict[str, Any]],
    writer_counts: list[list[int]],
    semantics: SemanticAudit,
    output_stem: Path,
) -> dict[str, Any]:
    rows = _quantum_rows(writer_counts)
    n_rows = len(rows)
    n_steps = len(frames)
    figure, axis = plt.subplots(figsize=(6.6, 4.3))
    axis.set_xlim(-0.05, n_steps + 0.05)
    axis.set_ylim(-0.65, n_rows + 0.25)
    axis.invert_yaxis()

    for row, (layer, quantum, offset, count) in enumerate(rows):
        name = f"L{layer}.Q{quantum}"
        description = semantics.quanta[name]
        axis.text(
            -0.12,
            row + 0.43,
            f"{name}  {description.concept}  ({description.f1:.2f})",
            ha="right",
            va="center",
            fontsize=8.1,
            color=TEXT,
        )
        for step, frame in enumerate(frames):
            gate = float(frame["gate_probabilities"][layer][quantum])
            hard_active = bool(frame["hard_active"][layer][quantum])
            left = step + 0.08
            bottom = row + 0.08
            width = 0.84
            height = 0.70
            axis.add_patch(
                FancyBboxPatch(
                    (left, bottom),
                    width,
                    height,
                    boxstyle="round,pad=0.015,rounding_size=0.04",
                    facecolor=_blend("#ffffff", LIGHT_ORANGE, 0.85),
                    edgecolor=ORANGE if hard_active else GRID,
                    linewidth=1.15 if hard_active else 0.5,
                )
            )
            axis.add_patch(
                plt.Circle(
                    (left + 0.11, bottom + 0.20),
                    0.065,
                    facecolor=_blend("#ffffff", ORANGE, gate),
                    edgecolor=ORANGE if hard_active else GRID,
                    linewidth=0.7,
                )
            )
            values = np.asarray(
                frame["writer_activities"][layer][offset : offset + count],
                dtype=np.float64,
            )
            frame_max = max(
                max(
                    np.asarray(layer_values, dtype=np.float64).max(initial=0.0)
                    for layer_values in frame["writer_activities"]
                ),
                1.0e-12,
            )
            size = min(0.105, 0.57 / max(count, 1))
            gap = 0.018
            start = left + 0.11
            for local_writer, magnitude in enumerate(values):
                intensity = np.sqrt(float(magnitude) / frame_max)
                x = start + local_writer * (size + gap)
                axis.add_patch(
                    Rectangle(
                        (x, bottom + 0.40),
                        size,
                        size,
                        facecolor=_blend("#ffffff", ORANGE, intensity),
                        edgecolor=GRID,
                        linewidth=0.35,
                    )
                )
            axis.text(
                left + 0.11,
                bottom + 0.61,
                f"{int(np.count_nonzero(values))}/{count} writers",
                ha="left",
                va="center",
                fontsize=6.4,
                color=MUTED,
            )

    for step, frame in enumerate(frames):
        axis.text(
            step + 0.5,
            -0.34,
            str(frame["prediction"]),
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=TEXT,
        )
    for boundary in (2, 4):
        axis.axhline(boundary - 0.10, color=GRID, linewidth=0.75)
    axis.text(
        n_steps,
        n_rows + 0.02,
        "circle = quantum gate; boxes = scalar writers; orange = activity",
        ha="right",
        va="center",
        fontsize=7.0,
        color=MUTED,
    )
    axis.axis("off")
    figure.tight_layout(pad=0.4)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    return {
        "active_quantum_counts": [
            int(sum(sum(layer) for layer in frame["hard_active"]))
            for frame in frames
        ],
        "active_writer_counts": [len(_active_writers(frame, rows)) for frame in frames],
    }


def write_interpretation_report(
    frames: list[dict[str, Any]],
    writer_counts: list[list[int]],
    semantics: SemanticAudit,
    output: Path,
    *,
    probe: int,
) -> None:
    rows = _quantum_rows(writer_counts)
    lines = [
        f"# Artifact-backed interpretation of the {probe} trace",
        "",
        "Quantum descriptions are calibration-selected and scored "
        "globally on held-out events. Writer descriptions are selected and scored "
        "conditional on the owner quantum being active. They describe activation "
        "support; they are not causal role names or evidence of source-mechanism "
        "identity.",
        "",
    ]
    for frame in frames:
        lines.append(f"## {frame['prediction']}")
        active_quanta = []
        for layer, quantum, _offset, _count in rows:
            if frame["hard_active"][layer][quantum]:
                name = f"L{layer}.Q{quantum}"
                description = semantics.quanta[name]
                active_quanta.append(
                    f"`{name}` -- {description.concept} (held-out F1 "
                    f"{description.f1:.3f})"
                )
        lines.extend(("", "Active quanta:", ""))
        lines.extend(f"- {item}" for item in active_quanta)
        lines.extend(("", "Active writers:", ""))
        for name, magnitude in _active_writers(frame, rows):
            description = semantics.writers[name]
            lines.append(
                f"- `{name}` -- {description.concept} (owner-conditional held-out "
                f"F1 {description.f1:.3f}; activity {magnitude:.3f})"
            )
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--factorization-dir", type=Path)
    parser.add_argument("--qmodel-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe", type=int, default=346)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qmodel = load_json(args.qmodel_dir / "summary.json")
    trace = load_json(args.trace_json)
    if not isinstance(qmodel, dict) or not isinstance(trace, dict):
        raise ValueError("unexpected Q-model or trace manifest format")
    frames = trace["probes"][str(args.probe)]["frames"]
    semantics = load_semantic_audit(args.audit_dir)
    dynamics = load_factorization_dynamics(
        args.source_run, args.factorization_dir
    )
    dynamics_metrics = plot_discovery_dynamics(
        dynamics, args.output_dir / "qmodel_discovery_dynamics"
    )
    activation_metrics = plot_probe_activation_summary(
        frames,
        qmodel["writer_counts_by_layer"],
        semantics,
        args.output_dir / f"qmodel_{args.probe}_activation_summary",
    )
    write_interpretation_report(
        frames,
        qmodel["writer_counts_by_layer"],
        semantics,
        args.output_dir / f"qmodel_{args.probe}_interpretation.md",
        probe=args.probe,
    )
    metrics = {"dynamics": dynamics_metrics, "activation": activation_metrics}
    (args.output_dir / "qmodel_recovery_figure.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
