from __future__ import annotations

import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from quanta.effective_samples import EffectiveSampleData
from .common import should_save_pdf


def plot_effective_sample_decomposition(
    *,
    output_image_path: str | None,
    data: EffectiveSampleData,
    curves: dict[int, np.ndarray],
    threshold_bits: float,
    title: str,
    x_start: float | None,
    x_lim: float | None,
    ylim: float | None,
    x_scale: str,
    overall_curve: np.ndarray | None = None,
    overall_curve_label: str | None = None,
    save_pdf: bool = False,
    show: bool = False,
):
    losses_by_code = {
        code: np.asarray(curves[index], dtype=float)
        for index, code in enumerate(data.codes)
        if index in curves
    }
    first_crossing_indices = {
        code: _first_crossing_index(losses_by_code[code], threshold_bits)
        for code in losses_by_code
    }
    codes_by_depth: dict[int, list[int]] = defaultdict(list)
    for code in data.codes:
        codes_by_depth[data.node_depths[code]].append(code)

    depth_values = sorted(codes_by_depth)
    columns = min(2, max(1, len(depth_values)))
    depth_rows = max(1, math.ceil(len(depth_values) / columns))
    fig = plt.figure(
        figsize=(12, 3.8 * depth_rows + 5.0),
        dpi=300,
    )
    grid = fig.add_gridspec(
        depth_rows + 1,
        columns,
        height_ratios=[1.0] * depth_rows + [1.35],
        hspace=0.68,
        wspace=0.25,
    )
    depth_axes = [
        fig.add_subplot(grid[row, column])
        for row in range(depth_rows)
        for column in range(columns)
    ]
    combined_axis = fig.add_subplot(grid[depth_rows, :])

    for axis, depth in zip(depth_axes, depth_values):
        depth_codes = codes_by_depth[depth]
        depth_probability = sum(data.probabilities[code] for code in depth_codes)
        expected_depth_per_batch = data.batch_size * depth_probability
        per_task_per_batch = [
            data.batch_size * data.probabilities[code]
            for code in depth_codes
        ]
        samples_to_threshold = []
        samples_after_previous_depths = []
        previous_depth_ready_index = _previous_depths_ready_index(
            depth=depth,
            codes=data.codes,
            depths=data.node_depths,
            first_crossing_indices=first_crossing_indices,
        )

        for code in depth_codes:
            effective_samples = data.axes[code]
            losses = losses_by_code[code]
            n_points = min(len(effective_samples), len(losses))
            crossing_index = first_crossing_indices[code]
            if crossing_index is not None and crossing_index < n_points:
                samples_to_threshold.append(float(effective_samples[crossing_index]))
            if previous_depth_ready_index is not None:
                subsequent_crossing = _first_crossing_index(
                    losses[:n_points],
                    threshold_bits,
                    after_index=previous_depth_ready_index,
                )
                if subsequent_crossing is not None:
                    samples_after_previous_depths.append(
                        float(
                            effective_samples[subsequent_crossing]
                            - effective_samples[previous_depth_ready_index]
                        )
                    )
            axis.plot(
                effective_samples[:n_points],
                losses[:n_points],
                linewidth=0.8,
                alpha=0.65,
                color="#315f85",
            )

        axis.set_title(
            f"Depth {depth}: {len(depth_codes)} tasks\n"
            f"{expected_depth_per_batch:.1f} depth samples/batch, "
            f"{min(per_task_per_batch):.2f}-{max(per_task_per_batch):.2f} per task/batch\n"
            f"Mean total samples to loss < {threshold_bits:g}: "
            f"{_threshold_summary(samples_to_threshold, len(depth_codes))}\n"
            f"Mean samples after shallower depths ready: "
            f"{_post_readiness_summary(samples_after_previous_depths, len(depth_codes), depth)}",
            fontsize=9,
            fontweight="bold",
        )
        _style_axis(
            axis,
            x_start=x_start,
            x_lim=x_lim,
            ylim=ylim,
            x_scale=x_scale,
        )

    for axis in depth_axes[len(depth_values):]:
        axis.set_visible(False)

    _plot_combined_panel(
        axis=combined_axis,
        data=data,
        losses_by_code=losses_by_code,
        overall_curve=overall_curve,
        overall_curve_label=overall_curve_label,
    )
    _style_axis(
        combined_axis,
        x_start=x_start,
        x_lim=x_lim,
        ylim=ylim,
        x_scale=x_scale,
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.08, right=0.98)
    if output_image_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        fig.savefig(output_image_path, bbox_inches="tight", dpi=400)
        if save_pdf or should_save_pdf():
            fig.savefig(
                os.path.splitext(output_image_path)[0] + ".pdf",
                bbox_inches="tight",
            )
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _plot_combined_panel(
    *,
    axis,
    data: EffectiveSampleData,
    losses_by_code: dict[int, np.ndarray],
    overall_curve: np.ndarray | None,
    overall_curve_label: str | None,
) -> None:
    color_map = LinearSegmentedColormap.from_list(
        "task_blue_yellow",
        ["#08306b", "#ffd84d"],
    )
    ordered_codes = sorted(
        data.codes,
        key=lambda code: (data.node_depths[code], code),
    )
    max_rank = max(len(ordered_codes) - 1, 1)
    for rank, code in enumerate(ordered_codes):
        effective_samples = data.axes[code]
        losses = losses_by_code[code]
        n_points = min(len(effective_samples), len(losses))
        axis.plot(
            effective_samples[:n_points],
            losses[:n_points],
            linewidth=0.8,
            alpha=0.65,
            color=color_map(rank / max_rank),
        )

    if overall_curve is not None:
        representative_axis = np.mean(
            np.stack([data.axes[code] for code in ordered_codes]),
            axis=0,
        )
        n_points = min(len(representative_axis), len(overall_curve))
        axis.plot(
            representative_axis[:n_points],
            overall_curve[:n_points],
            color="#d62728",
            linewidth=1.6,
            label=overall_curve_label,
        )
    axis.set_title(
        f"All Depths: {len(data.codes)} tasks",
        fontsize=10,
        fontweight="bold",
    )


def _style_axis(axis, *, x_start, x_lim, ylim, x_scale) -> None:
    axis.set_xlabel("Effective cumulative target samples per task")
    axis.set_ylabel("Task loss (bits)")
    axis.set_xscale(x_scale)
    if x_start is not None or x_lim is not None:
        axis.set_xlim(left=x_start, right=x_lim)
    axis.set_ylim(bottom=0.0, top=ylim)
    axis.grid(True, linestyle="--", alpha=0.35)


def _threshold_summary(samples: list[float], total_tasks: int) -> str:
    if not samples:
        return f"not reached (0/{total_tasks} tasks)"
    mean_samples = float(np.mean(samples))
    return f"{mean_samples:,.0f} ({len(samples)}/{total_tasks} tasks)"


def _post_readiness_summary(
    samples: list[float],
    total_tasks: int,
    depth: int,
) -> str:
    if depth == 0:
        return "n/a (no shallower depths)"
    return _threshold_summary(samples, total_tasks)


def _first_crossing_index(
    losses: np.ndarray,
    threshold: float,
    *,
    after_index: int | None = None,
) -> int | None:
    start = 0 if after_index is None else int(after_index) + 1
    crossings = np.flatnonzero(losses[start:] < threshold)
    if len(crossings) == 0:
        return None
    return start + int(crossings[0])


def _previous_depths_ready_index(
    *,
    depth: int,
    codes: list[int],
    depths: dict[int, int],
    first_crossing_indices: dict[int, int | None],
) -> int | None:
    if depth == 0:
        return None
    shallower_codes = [code for code in codes if depths[code] < depth]
    crossing_indices = [first_crossing_indices[code] for code in shallower_codes]
    if any(index is None for index in crossing_indices):
        return None
    return max(int(index) for index in crossing_indices)
