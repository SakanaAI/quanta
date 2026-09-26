#!/usr/bin/env python3
"""Build the three-law HSP resource-scaling figure for Section 4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np


EXPECTED_WIDTHS = (8, 16, 24, 32, 48, 64, 96, 128, 256, 512)
TAIL_WINDOW_STEPS = 25_000
DATA_SMOOTHING_RADIUS = 1
DATA_ACQUISITION_THRESHOLD = 0.8
DATA_PLATEAU_IMPROVEMENT = 0.0005
DATA_PLATEAU_PATIENCE = 10
DATA_VISUALIZATION_POINTS = 10
COMMON_EXPECTED = {
    "base_tasks": 8,
    "branching_factor": 2,
    "max_depth": 5,
    "demand_law": "rho_beta",
    "model_seed": 0,
    "data_seed": 7000,
    "eval_seed": 10000,
    "hidden_layers": 5,
    "task_conditioning": "early",
    "parameterization": "mup",
    "mup_base_width": 8,
    "mup_delta_width": 16,
    "optimizer": "sgd",
    "learning_rate": 0.0075,
    "batch_size": 16_384,
    "microbatch_size": 16_384,
    "eval_every": 5_000,
}


@dataclass(frozen=True)
class ScalingSpec:
    key: str
    target_alpha: float
    display_alpha: str
    beta: float
    default_checkpoint: int
    color: str
    checkpoint_overrides: dict[int, int] = field(default_factory=dict)


SPECS = (
    ScalingSpec(
        key="alpha020",
        target_alpha=0.2,
        display_alpha="0.20",
        beta=2.29739671,
        default_checkpoint=15_000_000,
        color="#4477AA",
    ),
    ScalingSpec(
        key="alpha025",
        target_alpha=0.25,
        display_alpha="0.25",
        beta=2.37841423,
        default_checkpoint=10_000_000,
        color="#EE7733",
    ),
    ScalingSpec(
        key="alpha0339",
        target_alpha=math.log2(2.53) - 1.0,
        display_alpha="0.339",
        beta=2.53,
        default_checkpoint=20_000_000,
        color="#228833",
        checkpoint_overrides={24: 18_000_000, 128: 19_000_000, 256: 19_000_000},
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def validate_config(path: Path, config: dict[str, Any], spec: ScalingSpec) -> None:
    expected = {**COMMON_EXPECTED, "beta": spec.beta}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}")


def tail_loss(
    path: Path,
    spec: ScalingSpec,
    total_parameters_by_width: dict[int, int] | None = None,
) -> dict[str, int | float | str]:
    status = load_json(path / "status.json")
    config = status["config"]
    validate_config(path, config, spec)
    width = int(config["width"])
    checkpoint = spec.checkpoint_overrides.get(width, spec.default_checkpoint)

    run_status = str(status.get("status"))
    allowed_failed_run = spec.key == "alpha0339" and width == 256
    if run_status != "complete" and not (allowed_failed_run and run_status == "failed"):
        raise ValueError(f"unexpected run status {run_status!r}: {path}")

    try:
        summary = load_json(path / "summary.json")
        total_parameters = int(summary["parameter_counts"]["total"])
    except (json.JSONDecodeError, KeyError):
        if not allowed_failed_run or total_parameters_by_width is None:
            raise
        total_parameters = total_parameters_by_width[width]

    if total_parameters_by_width is not None:
        expected_total = total_parameters_by_width[width]
        if total_parameters != expected_total:
            raise ValueError(
                f"{path}: total parameters {total_parameters} != {expected_total}"
            )

    with np.load(path / "trajectory.npz", allow_pickle=False) as trajectory:
        steps = trajectory["steps"]
        losses = trajectory["weighted_losses_bits"]
        window = (steps >= checkpoint - TAIL_WINDOW_STEPS) & (steps <= checkpoint)
        if not np.any(window) or int(steps[window][-1]) != checkpoint:
            raise ValueError(f"{path}: missing checkpoint {checkpoint}")
        selected_losses = losses[window]
        if not np.all(np.isfinite(selected_losses)):
            raise ValueError(f"{path}: non-finite loss in selected tail")
        loss = float(np.median(selected_losses))

    return {
        "width": width,
        "total_parameters": total_parameters,
        "loss_bits": loss,
        "checkpoint": checkpoint,
        "run_status": run_status,
    }


def load_sweep(
    root: Path,
    spec: ScalingSpec,
    total_parameters_by_width: dict[int, int] | None = None,
) -> list[dict[str, int | float | str]]:
    records = sorted(
        (
            tail_loss(path.parent, spec, total_parameters_by_width)
            for path in root.glob("*/status.json")
        ),
        key=lambda record: int(record["width"]),
    )
    widths = tuple(int(record["width"]) for record in records)
    if widths != EXPECTED_WIDTHS:
        raise ValueError(f"expected widths {EXPECTED_WIDTHS}, found {widths}")
    return records


def log_log_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    prediction = intercept + slope * np.log(x)
    residual = np.log(y) - prediction
    centered = np.log(y) - np.log(y).mean()
    return {
        "alpha_hat": float(-slope),
        "intercept": float(intercept),
        "r_squared": float(
            1.0 - np.dot(residual, residual) / np.dot(centered, centered)
        ),
    }


def three_checkpoint_median(values: np.ndarray) -> np.ndarray:
    """Suppress isolated evaluation spikes without changing the time grid."""
    return np.asarray(
        [
            np.median(
                values[
                    max(0, index - DATA_SMOOTHING_RADIUS) : min(
                        len(values), index + DATA_SMOOTHING_RADIUS + 1
                    )
                ]
            )
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def data_scaling_record(path: Path, spec: ScalingSpec) -> dict[str, Any]:
    """Load one fixed-width trajectory under the declared post-hoc rule."""
    status = load_json(path / "status.json")
    config = status["config"]
    if status.get("status") != "complete":
        raise ValueError(f"expected a complete run: {path}")
    if int(config["width"]) != 2048 or int(config["batch_size"]) != 4096:
        raise ValueError(f"expected width=2048 and batch=4096: {path}")
    if not math.isclose(float(config["beta"]), spec.beta, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"unexpected beta for {spec.key}: {path}")

    with np.load(path / "trajectory.npz", allow_pickle=False) as trajectory:
        steps = trajectory["steps"].astype(np.int64)
        losses = trajectory["weighted_losses_bits"].astype(np.float64)
        coefficients = trajectory["functional_coefficients"].astype(np.float64)

    smoothed_losses = three_checkpoint_median(losses)
    acquired = (coefficients >= DATA_ACQUISITION_THRESHOLD).sum(axis=1)
    event_indices = np.flatnonzero(np.diff(acquired) > 0) + 1
    if len(event_indices) < 3:
        raise ValueError(f"fewer than three distinct acquisition events: {path}")
    start_index = int(event_indices[2])

    best_loss = float(smoothed_losses[start_index])
    best_index = start_index
    end_index: int | None = None
    for index in range(start_index + 1, len(steps)):
        if smoothed_losses[index] < best_loss * (1.0 - DATA_PLATEAU_IMPROVEMENT):
            best_loss = float(smoothed_losses[index])
            best_index = index
        if index - best_index >= DATA_PLATEAU_PATIENCE:
            end_index = best_index
            break
    if end_index is None:
        raise ValueError(f"no plateau reached under the declared rule: {path}")

    selected_steps = steps[start_index : end_index + 1]
    selected_losses = smoothed_losses[start_index : end_index + 1]
    examples = selected_steps.astype(np.float64) * float(config["batch_size"])
    fit = log_log_fit(examples, selected_losses)
    target_exponent = spec.target_alpha / (1.0 + spec.target_alpha)
    target_intercept = float(
        np.mean(np.log(selected_losses) + target_exponent * np.log(examples))
    )
    return {
        "path": str(path),
        "batch_size": int(config["batch_size"]),
        "width": int(config["width"]),
        "smoothing": "three_checkpoint_median",
        "acquisition_threshold": DATA_ACQUISITION_THRESHOLD,
        "plateau_relative_improvement": DATA_PLATEAU_IMPROVEMENT,
        "plateau_patience_evaluations": DATA_PLATEAU_PATIENCE,
        "start_step": int(selected_steps[0]),
        "end_step": int(selected_steps[-1]),
        "trigger_step": int(steps[end_index + DATA_PLATEAU_PATIENCE]),
        "start_acquired_tasks": int(acquired[start_index]),
        "fit": {
            **fit,
            "target_exponent": target_exponent,
            "relative_error": abs(fit["alpha_hat"] - target_exponent)
            / target_exponent,
            "target_intercept": target_intercept,
        },
        "examples": examples,
        "losses": selected_losses,
    }


def save_figure(
    sweeps: list[tuple[ScalingSpec, list[dict[str, int | float | str]]]],
    data_records: list[tuple[ScalingSpec, dict[str, Any]]],
    output: Path,
    manuscript_figure: Path | None,
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)
    axis, data_axis = axes
    reports: list[dict[str, Any]] = []
    parameter_legend_handles: list[Line2D] = []
    data_legend_handles: list[Line2D] = []

    for spec, records in sweeps:
        parameters = np.asarray(
            [record["total_parameters"] for record in records], dtype=np.float64
        )
        losses = np.asarray([record["loss_bits"] for record in records])
        fit = log_log_fit(parameters, losses)
        grid = np.geomspace(parameters.min(), parameters.max(), 300)
        fitted = np.exp(fit["intercept"]) * grid ** (-fit["alpha_hat"])
        target_intercept = float(
            np.mean(np.log(losses) + spec.target_alpha * np.log(parameters))
        )
        target = np.exp(target_intercept) * grid ** (-spec.target_alpha)

        axis.scatter(
            parameters,
            losses,
            s=29,
            color=spec.color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axis.plot(grid, fitted, color=spec.color, linewidth=2.0, zorder=2)
        axis.plot(
            grid,
            target,
            color=spec.color,
            linestyle="--",
            linewidth=1.25,
            alpha=0.8,
            zorder=1,
        )
        parameter_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=spec.color,
                linewidth=2.0,
                marker="o",
                markersize=4.5,
                label=(
                    rf"$\alpha={spec.display_alpha}$ "
                    rf"($\widehat{{\alpha}}={fit['alpha_hat']:.3f}$)"
                ),
            )
        )
        reports.append(
            {
                "key": spec.key,
                "target_alpha": spec.target_alpha,
                "tail_window_steps": TAIL_WINDOW_STEPS,
                "fit": {
                    **fit,
                    "relative_error": abs(fit["alpha_hat"] - spec.target_alpha)
                    / spec.target_alpha,
                },
                "points": records,
            }
        )

    axis.set(
        xscale="log",
        yscale="log",
        xticks=(1e4, 1e5, 1e6),
        xlabel=r"Parameters $P$",
        ylabel="held-out loss (bits)",
        title="Parameter scaling",
    )
    parameter_legend_handles.extend(
        (
            Line2D([0], [0], color="0.2", linewidth=2.0, label="empirical fit"),
            Line2D(
                [0],
                [0],
                color="0.2",
                linewidth=1.25,
                linestyle="--",
                label="theoretical slope",
            ),
        )
    )
    axis.legend(
        handles=parameter_legend_handles,
        frameon=False,
        fontsize=6.8,
        loc="lower left",
        handlelength=2.0,
    )
    axis.grid(which="major", color="0.91", linewidth=0.7)

    for spec, record in data_records:
        examples = np.asarray(record["examples"], dtype=np.float64)
        losses = np.asarray(record["losses"], dtype=np.float64)
        display_indices = np.linspace(
            0, len(examples) - 1, DATA_VISUALIZATION_POINTS, dtype=int
        )
        fit = record["fit"]
        grid = np.geomspace(examples.min(), examples.max(), 300)
        fitted = np.exp(fit["intercept"]) * grid ** (-fit["alpha_hat"])
        target = np.exp(fit["target_intercept"]) * grid ** (-fit["target_exponent"])
        data_axis.scatter(
            examples[display_indices],
            losses[display_indices],
            s=13,
            color=spec.color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        data_axis.plot(grid, fitted, color=spec.color, linewidth=2.0, zorder=2)
        data_axis.plot(
            grid,
            target,
            color=spec.color,
            linestyle="--",
            linewidth=1.25,
            alpha=0.8,
            zorder=1,
        )
        data_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=spec.color,
                linewidth=2.0,
                marker="o",
                markersize=4.0,
                label=(
                    rf"$\alpha={spec.display_alpha}$ "
                    rf"($\widehat{{b}}={fit['alpha_hat']:.3f}$)"
                ),
            )
        )

    data_axis.set(
        xscale="log",
        yscale="log",
        xlim=(6e9, 3.6e10),
        xticks=(6e9, 1.5e10, 3e10),
        xlabel=r"Data $D = B \times S$",
        title="Data scaling",
    )
    data_axis.set_xticklabels(
        (r"$6\times10^9$", r"$1.5\times10^{10}$", r"$3\times10^{10}$")
    )
    data_axis.xaxis.set_minor_locator(NullLocator())
    data_axis.legend(
        handles=data_legend_handles,
        frameon=False,
        fontsize=6.8,
        loc="lower left",
        handlelength=2.0,
    )
    data_axis.grid(which="major", color="0.91", linewidth=0.7)

    axis.set(ylim=(0.1, 1.1), yticks=(1.0, 0.3, 0.1))
    axis.set_yticklabels((r"$10^0$", r"$3\times10^{-1}$", r"$10^{-1}$"))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_locator(NullLocator())
    data_axis.yaxis.set_minor_locator(NullLocator())

    figure.tight_layout(w_pad=1.6, pad=0.5)
    figure.savefig(output / "hsp_resource_scaling.pdf", bbox_inches="tight")
    figure.savefig(
        output / "hsp_resource_scaling.png", dpi=300, bbox_inches="tight"
    )
    if manuscript_figure is not None:
        manuscript_figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(manuscript_figure, bbox_inches="tight")
    plt.close(figure)
    return {
        "status": "one_seed_windowed_endpoint",
        "widths": list(EXPECTED_WIDTHS),
        "parameter_coordinate": "total_trainable_parameters",
        "fit_space": "ordinary_least_squares_in_log_log_space",
        "sweeps": reports,
        "data_scaling": {
            "status": "post_hoc_fixed_protocol_diagnostic",
            "visualization_points_per_trajectory": DATA_VISUALIZATION_POINTS,
            "records": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"examples", "losses"}
                }
                for _, record in data_records
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha020-root", type=Path, required=True)
    parser.add_argument("--alpha025-root", type=Path, required=True)
    parser.add_argument("--alpha0339-root", type=Path, required=True)
    parser.add_argument("--data-alpha020-run", type=Path, required=True)
    parser.add_argument("--data-alpha025-run", type=Path, required=True)
    parser.add_argument("--data-alpha0339-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manuscript-figure", type=Path)
    args = parser.parse_args()

    roots = {
        "alpha020": args.alpha020_root,
        "alpha025": args.alpha025_root,
        "alpha0339": args.alpha0339_root,
    }
    alpha020_records = load_sweep(roots["alpha020"], SPECS[0])
    total_parameters_by_width = {
        int(record["width"]): int(record["total_parameters"])
        for record in alpha020_records
    }
    sweeps = [(SPECS[0], alpha020_records)]
    for spec in SPECS[1:]:
        sweeps.append(
            (
                spec,
                load_sweep(roots[spec.key], spec, total_parameters_by_width),
            )
        )

    data_paths = {
        "alpha020": args.data_alpha020_run,
        "alpha025": args.data_alpha025_run,
        "alpha0339": args.data_alpha0339_run,
    }
    data_records = [
        (spec, data_scaling_record(data_paths[spec.key], spec)) for spec in SPECS
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = save_figure(sweeps, data_records, args.output_dir, args.manuscript_figure)
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
