#!/usr/bin/env python3
"""Plot one composed-task staircase against a multitask HSP aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def event_for(
    summary: dict[str, Any], *, mode: str, group: int
) -> dict[str, Any]:
    matches = [
        event
        for event in summary["event_records"]
        if event["mode"] == mode and int(event["group"]) == group
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {mode} event for group {group}, found {len(matches)}"
        )
    return matches[0]


def plot_staircase(
    run_dir: Path,
    output: Path,
    tracked_task: int,
    x_limit: float,
    preview_output: Path | None = None,
) -> dict[str, Any]:
    status = load_json(run_dir / "status.json")
    if status.get("status") != "complete":
        raise ValueError(f"incomplete run: {run_dir}")
    summary = load_json(run_dir / "summary.json")
    with np.load(run_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        steps = trajectory["steps"].copy()
        losses = trajectory["task_losses_bits"].copy()
        aggregate = trajectory["weighted_losses_bits"].copy()
        probabilities = trajectory["probabilities"].copy()

    if not 0 <= tracked_task < losses.shape[1]:
        raise ValueError(
            f"tracked task {tracked_task} is outside [0, {losses.shape[1]})"
        )
    factor_specs = {spec["name"]: spec for spec in summary["factor_specs"]}
    if set(factor_specs) != {"A", "B"}:
        raise ValueError("the staircase panel requires exactly A and B factors")
    if int(factor_specs["B"]["sharing"]) != 1:
        raise ValueError("the B factor must be task-private")

    shared_group = tracked_task // int(factor_specs["A"]["sharing"])
    shared_event = event_for(summary, mode="A", group=shared_group)
    private_event = event_for(summary, mode="B", group=tracked_task)
    private_time = private_event["time_80pct"]
    shared_time = shared_event["time_80pct"]
    if private_time is None or shared_time is None:
        raise ValueError("both tracked constituents must be acquired")
    if float(private_time) >= float(shared_time):
        raise ValueError(
            "the private constituent must precede the shared constituent "
            "to form a staircase"
        )

    figure, axis = plt.subplots(figsize=(4.8, 3.25))
    for task in range(losses.shape[1]):
        if task == tracked_task:
            continue
        axis.plot(
            steps,
            losses[:, task],
            color="0.45",
            alpha=0.10,
            linewidth=0.55,
            zorder=1,
        )
    axis.plot(
        steps,
        aggregate,
        color="crimson",
        linewidth=2.1,
        label="weighted loss",
        zorder=3,
    )
    axis.plot(
        steps,
        losses[:, tracked_task],
        color="#2166ac",
        linewidth=2.2,
        label="one composed task",
        zorder=4,
    )
    for acquisition_time in (float(private_time), float(shared_time)):
        axis.axvline(
            acquisition_time,
            color="#2166ac",
            alpha=0.35,
            linestyle="--",
            linewidth=0.75,
            zorder=2,
        )
    axis.set(
        xlim=(0, x_limit),
        ylim=(-0.03, 0.86),
        xlabel="Optimizer step",
    )
    axis.set_xticks(np.linspace(0, x_limit, 5, dtype=int))
    axis.legend(
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="upper right",
        columnspacing=1.2,
        handlelength=2.4,
    )
    figure.tight_layout(pad=0.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    if preview_output is not None:
        preview_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_output, dpi=240, bbox_inches="tight")
    plt.close(figure)

    exact_time = summary["crossing_exact"][tracked_task]
    return {
        "run_dir": str(run_dir),
        "tracked_task": tracked_task,
        "tracked_task_probability": float(probabilities[tracked_task]),
        "shared_group": shared_group,
        "private_80pct_step": float(private_time),
        "shared_80pct_step": float(shared_time),
        "tracked_task_exact_step": (
            float(exact_time) if exact_time is not None else None
        ),
        "n_tasks": int(losses.shape[1]),
        "completed_events": int(summary["completed_events"]),
        "total_events": int(summary["total_events"]),
        "best_exact_tasks": int(summary["best_exact_tasks"]),
        "final_exact_tasks": int(summary["final_exact_tasks"]),
        "final_weighted_loss_bits": float(summary["final_weighted_loss_bits"]),
        "conditional_plateau_bits": float(
            summary["conditional_bayes_losses_bits"]["1"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracked-task", type=int, required=True)
    parser.add_argument("--x-limit", type=float, default=15_000)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    metrics = plot_staircase(
        args.run_dir,
        args.output,
        args.tracked_task,
        args.x_limit,
        args.preview_output,
    )
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w") as handle:
            json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
