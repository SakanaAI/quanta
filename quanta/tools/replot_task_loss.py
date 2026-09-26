#!/usr/bin/env python3
"""Replot task loss decomposition from results.json with custom step ranges and depth aggregation."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from quanta.figures.common import should_save_pdf

def main():
    parser = argparse.ArgumentParser(
        description="Replot task loss decomposition from results.json with custom step ranges and optional depth decomposition."
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to results.json file."
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=0,
        help="Minimum optimization step to plot (inclusive)."
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=5000,
        help="Maximum optimization step to plot (inclusive)."
    )
    parser.add_argument(
        "--by-depth",
        action="store_true",
        help="Decompose and aggregate curves by depth instead of listing all tasks."
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Path to save the output figure. Defaults to <tasks/depth>_loss_<min>_<max>.png in the same directory."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="DPI of the output figure (default: 320)."
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="9.5,5.4",
        help="Figure size as 'width,height' (default: '9.5,5.4')."
    )
    args = parser.parse_args()

    # Parse figsize
    try:
        figsize = tuple(map(float, args.figsize.split(",")))
    except Exception:
        raise ValueError("figsize must be in the format 'width,height' (e.g. '9.5,5.4')")

    # Load results.json
    print(f"Loading results from {args.results_json}...")
    with open(args.results_json, "r") as f:
        results = json.load(f)

    eval_records = results["eval_records"]
    node_depths = results["node_depths"]
    mask_mode = results.get("attention", "unknown")

    # Filter records based on step range
    filtered_records = [
        record for record in eval_records
        if args.min_step <= int(record["step"]) <= args.max_step
    ]

    if not filtered_records:
        raise ValueError(f"No records found in step range [{args.min_step}, {args.max_step}]")

    steps = [int(record["step"]) for record in filtered_records]

    # In JSON, keys are always strings
    codes_str = list(node_depths.keys())

    curves = {
        code_str: [
            float(record["task_losses"][code_str]["loss_bits"])
            for record in filtered_records
        ]
        for code_str in codes_str
    }

    mean_curve = np.nanmean(
        np.asarray([curves[code_str] for code_str in codes_str], dtype=float),
        axis=0,
    )

    node_depths_int = {code_str: int(depth) for code_str, depth in node_depths.items()}
    maximum_depth = max(node_depths_int.values(), default=0)

    depth_colors = LinearSegmentedColormap.from_list(
        "depth_blue_yellow",
        ["#08306b", "#ffd84d"],
    )

    fig, axis = plt.subplots(figsize=figsize, dpi=220)

    if args.by_depth:
        # Group codes by depth
        depth_to_codes = {}
        for code_str, d in node_depths_int.items():
            depth_to_codes.setdefault(d, []).append(code_str)

        # Compute curves per depth
        depth_curves = {}
        for d, codes_at_depth in depth_to_codes.items():
            depth_curves[d] = np.nanmean(
                np.asarray([curves[c] for c in codes_at_depth], dtype=float),
                axis=0
            )

        # Plot curves for each depth in order
        for d in sorted(depth_curves.keys()):
            color = depth_colors(
                float(d) / float(maximum_depth)
                if maximum_depth > 0
                else 0.0
            )
            axis.plot(
                steps,
                depth_curves[d],
                color=color,
                linewidth=1.8,
                alpha=0.9,
                label=f"Depth {d}",
                zorder=2,
            )
        title_prefix = "Depth decomposition"
    else:
        # Plot curves for each individual task
        for code_str in codes_str:
            depth = node_depths_int[code_str]
            color = depth_colors(
                float(depth) / float(maximum_depth)
                if maximum_depth > 0
                else 0.0
            )
            axis.plot(
                steps,
                curves[code_str],
                color=color,
                linewidth=0.55,
                alpha=0.32,
                zorder=1,
            )
        title_prefix = "Task loss decomposition"

    # Plot overall mean curve
    axis.plot(
        steps,
        mean_curve,
        color="#d62728",
        linewidth=2.2,
        alpha=0.98,
        label="Mean task loss",
        zorder=10,
    )

    axis.set_title(
        f"{title_prefix} | {mask_mode} | Steps {args.min_step}-{args.max_step}",
        fontsize=11,
        fontweight="bold",
    )
    axis.set_xlabel("Optimization steps", fontweight="bold")
    axis.set_ylabel("Loss (bits)", fontweight="bold")
    axis.set_xlim(left=min(steps), right=max(steps))
    axis.grid(True, alpha=0.22, linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(fontsize=8, frameon=False)

    if args.output_path:
        output_path = args.output_path
    else:
        name_prefix = "depth_loss" if args.by_depth else "tasks_loss"
        output_path = (
            ROOT
            / "scripts/data/replot_task_loss"
            / f"{name_prefix}_{args.min_step}_{args.max_step}.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    if should_save_pdf():
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved decomposed task loss figure to {output_path}")

if __name__ == "__main__":
    main()
