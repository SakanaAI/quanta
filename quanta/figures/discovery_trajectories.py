import logging
import os

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

from quanta.metrics import (
    learned_threshold_bits,
)
from quanta.effective_samples import EffectiveSampleData
from .effective_samples import plot_effective_sample_decomposition
from .common import (
    node_depth,
    should_save_pdf,
)


def plot_discovery_trajectories(
    output_image_path: str | None,
    codes: list,
    subtask_losses: list[list[float]],
    samples: list[int],
    effective_samples: dict[int, list[float]] | None = None,
    effective_sample_data: EffectiveSampleData | None = None,
    overall_loss_bits: list[float] | None = None,
    quantum_subtask_losses: list[list[float]] | None = None,
    mean_quantum_subtask_losses: list[list[float]] | None = None,
    subtask_train_losses: list[list[float]] | None = None,
    graph_dependencies: dict[int, list[int]] | None = None,
    task_frequencies: dict[int, float] | None = None,
    model_name: str = "MLP",
    depth: int | str = "unknown",
    width: int | str = "unknown",
    task_name: str | None = None,
    window: int = 100,
    smoothing: float = 0.0,
    x_start: float = 1e5,
    x_lim: float | None = None,
    ylim: float | None = None,
    x_axis: str = "steps",
    x_scale: str = "log",
    loss_decomposition: str = "tasks",
    weighted_loss: bool = True,
    record_train_loss: bool = False,
    plot_total_loss: bool = False,
    plot_relative_loss: bool = False,
    discreteness_transition_error: bool = False,
    discreteness_value: float | None = None,
    discreteness_area: float | None = None,
    discreteness_candidate_fraction: float | None = None,
    save_pdf: bool = False,
    legend: bool = False,
    animate: bool = False,
    show: bool = False,
):
    del task_frequencies, record_train_loss
    threshold = learned_threshold_bits()

    if not codes or not subtask_losses or not samples:
        logging.warning("discovery_trajectories figure=skipped reason=missing_results")
        return None
    if animate and not output_image_path:
        raise ValueError("output_image_path is required when animate=True.")
    if x_axis not in {"steps", "samples", "effective_samples"}:
        raise ValueError("x_axis must be 'steps', 'samples', or 'effective_samples'.")
    if (
        x_axis == "effective_samples"
        and effective_samples is None
        and effective_sample_data is None
    ):
        raise ValueError(
            "x_axis='effective_samples' requires expected samples for each task."
        )
    if x_scale not in {"log", "linear"}:
        raise ValueError("x_scale must be 'log' or 'linear'.")
    if loss_decomposition not in {"tasks", "tasks_dependencies_ready", "quanta"}:
        raise ValueError(
            "loss_decomposition must be 'tasks', 'tasks_dependencies_ready', or 'quanta'."
        )
    if not 0.0 <= float(smoothing) <= 1.0:
        raise ValueError("smoothing must be between 0 and 1.")
    if not isinstance(weighted_loss, bool):
        raise ValueError("weighted_loss must be true or false.")
    if plot_total_loss and overall_loss_bits is None:
        raise ValueError(
            "plot_total_loss=True requires the saved overall weighted eval loss."
        )

    is_cxor = all(isinstance(code, (int, np.integer)) for code in codes)
    task_name = task_name or "CXOR"
    shared_x_values = _aligned_cumulative_samples(samples, subtask_losses)
    curve_x_values = _curve_x_values(
        codes,
        shared_x_values,
        (
            effective_sample_data.axes
            if x_axis == "effective_samples" and effective_sample_data is not None
            else effective_samples if x_axis == "effective_samples" else None
        ),
    )
    overall_x_values = _representative_x_values(curve_x_values)
    raw_curves = _loss_curves(subtask_losses, smoothing)
    raw_readiness_curves = _loss_curves(subtask_losses, smoothing=0.0)
    if loss_decomposition == "tasks":
        displayed_curves = raw_curves
        metric_losses = _curves_bits_to_nats(raw_readiness_curves, len(codes))
        panel_label = "Task Loss"
    elif loss_decomposition == "tasks_dependencies_ready":
        displayed_curves = _dependencies_ready_loss_curves(
            raw_curves=raw_curves,
            readiness_curves=raw_readiness_curves,
            smoothing=smoothing,
            codes=codes,
            graph_dependencies=graph_dependencies if is_cxor else None,
            threshold=threshold,
        )
        metric_losses = _dependencies_ready_metric_losses(
            raw_readiness_curves=raw_readiness_curves,
            codes=codes,
            graph_dependencies=graph_dependencies if is_cxor else None,
            threshold=threshold,
        )
        panel_label = "Task Loss, Dependencies Ready"
    else:
        selected_quantum_losses = (
            quantum_subtask_losses
            if weighted_loss
            else mean_quantum_subtask_losses
        )
        if selected_quantum_losses is None:
            raise ValueError(
                "loss_decomposition='quanta' requires saved quantum losses "
                "for the selected weighting mode."
            )
        if len(selected_quantum_losses) != len(codes):
            raise ValueError("Quantum losses must have one curve per code.")
        displayed_curves = _loss_curves(selected_quantum_losses, smoothing)
        metric_losses = _curves_bits_to_nats(
            _loss_curves(selected_quantum_losses, smoothing=0.0),
            len(codes),
        )
        panel_label = (
            "Quantum OUT Loss Across Weighted Target Contexts"
            if weighted_loss
            else "Quantum OUT Loss, Mean Across Target Contexts"
        )
    if plot_relative_loss:
        displayed_curves = _relative_loss_curves(displayed_curves)
    overall_curve = None
    if overall_loss_bits is not None:
        overall_curve = _ema_smooth(np.asarray(overall_loss_bits, dtype=float), smoothing)
        if plot_relative_loss:
            overall_curve = _relative_loss_curve(overall_curve)
    if x_lim is None:
        x_lim = float(max(values[-1] for values in curve_x_values.values()))

    del subtask_train_losses, discreteness_transition_error, discreteness_value, discreteness_area, discreteness_candidate_fraction
    panel_metrics = _panel_metrics(
        losses_nats=metric_losses,
        codes=codes,
        graph_dependencies=graph_dependencies,
        steps={
            code: curve_x_values[index]
            for index, code in enumerate(codes)
            if index in curve_x_values
        },
        overall_steps=overall_x_values,
        x_axis=x_axis,
    )

    title = _figure_title(
        task_name=task_name,
        model_name=model_name,
        depth=depth,
        width=width,
    )

    if x_axis == "effective_samples" and effective_sample_data is not None and not animate:
        return plot_effective_sample_decomposition(
            output_image_path=output_image_path,
            data=effective_sample_data,
            curves=displayed_curves,
            threshold_bits=threshold,
            title=f"{task_name} {panel_label} by Effective Samples Received\n{title}",
            x_start=x_start,
            x_lim=x_lim,
            ylim=ylim,
            x_scale=x_scale,
            overall_curve=overall_curve if plot_total_loss else None,
            overall_curve_label=(
                "Overall weighted loss" if weighted_loss else "Overall mean task loss"
            ),
            save_pdf=save_pdf,
            show=show,
        )

    fig, axes = _create_axes()
    _draw_panels(
        fig=fig,
        axes=axes,
        curve_x_values=curve_x_values,
        overall_x_values=overall_x_values,
        curves=displayed_curves,
        codes=codes,
        is_cxor=is_cxor,
        graph_dependencies=graph_dependencies,
        x_start=x_start,
        x_lim=x_lim,
        y_limit=_panel_y_limit(
            displayed_curves,
            plot_relative_loss,
            overall_curve=overall_curve if plot_total_loss else None,
            ylim=ylim,
        ),
        x_axis=x_axis,
        x_scale=x_scale,
        title=title,
        panel_metrics=panel_metrics,
        panel_label=panel_label,
        plot_total_loss=plot_total_loss,
        overall_curve=overall_curve,
        overall_curve_label=(
            "Overall weighted loss" if weighted_loss else "Overall mean task loss"
        ),
        plot_relative_loss=plot_relative_loss,
        legend=legend,
    )
    if animate:
        _save_animation_to_path(
            fig=fig,
            axes=axes,
            output_path=output_image_path,
            codes=codes,
            is_cxor=is_cxor,
            graph_dependencies=graph_dependencies,
            curve_x_values=curve_x_values,
            overall_x_values=overall_x_values,
            animation_time_values=shared_x_values,
            curves=displayed_curves,
            x_start=x_start,
            x_lim=x_lim,
            ylim=ylim,
            x_axis=x_axis,
            x_scale=x_scale,
            title=title,
            panel_metrics=panel_metrics,
            panel_label=panel_label,
            plot_total_loss=plot_total_loss,
            overall_curve=overall_curve,
            overall_curve_label=(
                "Overall weighted loss" if weighted_loss else "Overall mean task loss"
            ),
            plot_relative_loss=plot_relative_loss,
            legend=legend,
        )
        return None

    _finalize_static_figure(fig, axes, output_image_path, legend=legend, save_pdf=save_pdf, show=show)
    return fig


def _figure_title(
    *,
    task_name,
    model_name,
    depth,
    width,
):
    return f"{task_name} with {model_name} (d={depth}, w={width})"


def _candidate_suffix(fraction: float | None, label: str) -> str:
    if fraction is None:
        return ""
    return f" ({100.0 * float(fraction):.0f}% {label})"


def _panel_metrics(
    losses_nats,
    codes,
    graph_dependencies,
    steps,
    overall_steps,
    x_axis,
):
    del losses_nats, codes, graph_dependencies, steps, overall_steps
    return {"unit": _metric_unit_label(x_axis)}


def _panel_title(label: str, metrics: dict) -> str:
    return f"{label} ({metrics.get('unit', 'steps')})"


def _metric_unit_label(x_axis: str) -> str:
    if x_axis == "effective_samples":
        return "effective samples"
    if x_axis == "samples":
        return "training samples"
    return "steps"


def _aligned_cumulative_samples(samples, subtask_losses):
    samples = np.array(samples)
    if len(samples) > 1 and samples[-1] > samples[0] and np.all(np.diff(samples) >= 0):
        cum_samples = samples
    else:
        cum_samples = np.cumsum(samples)

    if len(subtask_losses) > 0 and len(cum_samples) != len(subtask_losses[0]):
        ratio = len(cum_samples) // len(subtask_losses[0])
        if ratio > 1:
            cum_samples = cum_samples[::ratio]
    return cum_samples


def _curve_x_values(codes, shared_x_values, effective_samples):
    if effective_samples is None:
        return {
            index: np.asarray(shared_x_values, dtype=float)
            for index in range(len(codes))
        }
    normalized = {int(code): values for code, values in effective_samples.items()}
    missing = [int(code) for code in codes if int(code) not in normalized]
    if missing:
        raise ValueError(
            "Missing effective-sample axes for task codes: "
            + ", ".join(map(str, missing))
        )
    return {
        index: np.asarray(normalized[int(code)], dtype=float)
        for index, code in enumerate(codes)
    }


def _representative_x_values(curve_x_values):
    arrays = list(curve_x_values.values())
    min_len = min((len(values) for values in arrays), default=0)
    if min_len <= 0:
        return np.asarray([], dtype=float)
    return np.mean(
        np.stack([np.asarray(values[:min_len], dtype=float) for values in arrays]),
        axis=0,
    )


def _loss_curves(losses_by_task, smoothing):
    curves = {}
    for i, losses in enumerate(losses_by_task):
        losses_bits = np.array(losses, dtype=float) * np.log2(np.e)
        curves[i] = _ema_smooth(losses_bits, smoothing)
    return curves


def _overall_discreteness_area(losses_by_task) -> float | None:
    if not losses_by_task:
        return None
    min_len = min((len(curve) for curve in losses_by_task if len(curve) > 0), default=0)
    if min_len <= 0:
        return None
    overall = np.nanmean(np.stack([np.asarray(curve[:min_len], dtype=float) for curve in losses_by_task]), axis=0)
    metrics = compute_discreteness_transition_error([overall], codes=[0])
    if int(metrics.get("total_quanta", 0)) <= 0:
        return None
    return float(metrics["dte_area"])


def _curves_bits_to_nats(curves, n_codes):
    return [
        (np.asarray(curves[index], dtype=float) / np.log2(np.e)).tolist()
        for index in range(n_codes)
        if index in curves
    ]


def _ema_smooth(values, smoothing):
    smoothing = float(smoothing)
    values = np.asarray(values, dtype=float)
    if smoothing <= 0.0 or len(values) == 0:
        return values
    smoothed = np.array(values, dtype=float, copy=True)
    previous = np.nan
    for index, value in enumerate(values):
        if not np.isfinite(value):
            smoothed[index] = previous
            continue
        if not np.isfinite(previous):
            previous = value
        else:
            previous = smoothing * previous + (1.0 - smoothing) * value
        smoothed[index] = previous
    return smoothed


def _relative_loss_curves(curves):
    return {
        index: _relative_loss_curve(curve)
        for index, curve in curves.items()
    }


def _relative_loss_curve(curve):
    values = np.asarray(curve, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values
    min_value = float(np.nanmin(values[finite]))
    max_value = float(np.nanmax(values[finite]))
    scale = max_value - min_value
    if scale <= 1e-12:
        normalized = np.ones_like(values, dtype=float)
        normalized[~finite] = np.nan
        return normalized
    return (values - min_value) / scale


def _dependencies_ready_loss_curves(
    *,
    raw_curves,
    readiness_curves,
    smoothing,
    codes,
    graph_dependencies,
    threshold,
):
    if not graph_dependencies:
        return {index: np.array(curve, dtype=float, copy=True) for index, curve in raw_curves.items()}

    index_by_code = {int(code): index for index, code in enumerate(codes)}
    learned_index = {
        index: _first_learned_index(curve, threshold)
        for index, curve in readiness_curves.items()
    }
    proper_curves = {}
    for index, code in enumerate(codes):
        if index not in raw_curves:
            continue
        curve = np.asarray(raw_curves[index], dtype=float)
        proper_curve = np.array(curve, dtype=float, copy=True)
        ancestor_indices = [
            index_by_code[int(ancestor)]
            for ancestor in _closure_dependencies(int(code), graph_dependencies)
            if int(ancestor) in index_by_code
        ]
        if not ancestor_indices:
            proper_curves[index] = np.array(raw_curves[index], dtype=float, copy=True)
            continue
        dependency_ready_steps = [learned_index.get(ancestor) for ancestor in ancestor_indices]
        if any(step is None for step in dependency_ready_steps):
            ready_step = len(curve)
        else:
            ready_step = max(int(step) for step in dependency_ready_steps)
        if len(curve) > 0 and ready_step > 0:
            proper_curve[: min(ready_step, len(curve))] = curve[0]
        proper_curves[index] = proper_curve
    return proper_curves


def _dependencies_ready_metric_losses(
    *,
    raw_readiness_curves,
    codes,
    graph_dependencies,
    threshold,
):
    proper_bits = _dependencies_ready_loss_curves(
        raw_curves=raw_readiness_curves,
        readiness_curves=raw_readiness_curves,
        smoothing=0.0,
        codes=codes,
        graph_dependencies=graph_dependencies,
        threshold=threshold,
    )
    return _curves_bits_to_nats(proper_bits, len(codes))


def _closure_dependencies(node, graph_dependencies):
    deps = set()
    stack = list(graph_dependencies.get(int(node), []))
    while stack:
        parent = int(stack.pop())
        if parent in deps:
            continue
        deps.add(parent)
        stack.extend(graph_dependencies.get(parent, []))
    return deps


def _first_learned_index(curve, threshold):
    values = np.asarray(curve, dtype=float)
    learned = np.where(values <= float(threshold))[0]
    if len(learned) == 0:
        return None
    return int(learned[0])


def _create_axes():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axis = plt.subplots(1, 1, figsize=(8.2, 4.8), dpi=300)
    return fig, [axis]


def _draw_panels(
    *,
    fig,
    axes,
    curve_x_values,
    overall_x_values,
    curves,
    codes,
    is_cxor,
    graph_dependencies,
    x_start,
    x_lim,
    y_limit,
    x_axis,
    x_scale,
    title,
    panel_metrics,
    panel_label,
    plot_total_loss,
    overall_curve,
    overall_curve_label,
    plot_relative_loss,
    legend,
):
    for axis in axes:
        axis.set_xscale("linear")
        axis.clear()

    axis = axes[0]
    _plot_curves(axis, curve_x_values, curves, codes, is_cxor, graph_dependencies, label_curves=legend)
    if plot_total_loss:
        _plot_overall_curve(axis, overall_x_values, overall_curve, label=overall_curve_label)
    axis.set_title(_panel_title(panel_label, panel_metrics), fontsize=8, fontweight="bold", pad=6)
    if y_limit is not None:
        axis.set_ylim(*y_limit)

    fig.suptitle(title, x=0.5, y=0.985, ha="center", fontsize=10, fontweight="bold")
    _apply_axis_labels(
        axes,
        x_start,
        x_lim=x_lim,
        plot_relative_loss=plot_relative_loss,
        x_axis=x_axis,
        x_scale=x_scale,
    )
    _style_axes(axes)
    if legend:
        axes[0].legend(
            title="Subtask",
            title_fontsize=8,
            prop={"size": 6, "weight": "normal"},
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            facecolor="#ffffff",
            edgecolor="#cccccc",
        )


def _plot_curves(ax, x_values, curves, codes, is_cxor, graph_deps, label_curves):
    if x_values is None:
        return
    rank_by_index = _rank_by_index(codes, is_cxor, graph_deps)
    max_rank = max(rank_by_index.values(), default=0)
    for i, code in enumerate(codes):
        if i not in curves:
            continue
        curve_x = x_values[i]
        n_points = min(len(curve_x), len(curves[i]))
        if n_points <= 0:
            continue
        label, task_depth = _curve_label_and_depth(code, is_cxor, graph_deps)
        ax.plot(
            curve_x[:n_points],
            curves[i][:n_points],
            label=label if label_curves else None,
            color=_rank_color(rank_by_index.get(i, task_depth), max_rank),
            alpha=0.75,
            linewidth=0.5,
            zorder=2 + task_depth,
        )


def _panel_y_limit(curves, plot_relative_loss, *, overall_curve=None, ylim=None):
    curves_for_limits = dict(curves)
    if overall_curve is not None:
        curves_for_limits["overall"] = overall_curve
    limit = _curves_y_limit(
        curves_for_limits,
        lower_bound=-0.03 if plot_relative_loss else 0.0,
    )
    if ylim is None:
        return limit
    upper = float(ylim)
    return (limit[0], upper) if limit is not None else None


def _curves_y_limit(curves, *, lower_bound=0.0):
    values = [
        value
        for curve in curves.values()
        for value in np.asarray(curve, dtype=float)
        if np.isfinite(value)
    ]
    if not values:
        return None
    low = float(min(values))
    high = float(max(values))
    if high <= low:
        pad = max(abs(high) * 0.03, 1e-6)
    else:
        pad = (high - low) * 0.03
    lower = max(float(lower_bound), low - pad)
    return lower, high + pad


def _depth_by_index(codes, is_cxor, graph_deps):
    return {
        index: _curve_label_and_depth(code, is_cxor, graph_deps)[1]
        for index, code in enumerate(codes)
    }


def _rank_by_index(codes, is_cxor, graph_deps):
    sortable = []
    for index, code in enumerate(codes):
        label, task_depth = _curve_label_and_depth(code, is_cxor, graph_deps)
        sortable.append((task_depth, _sort_value(code), label, index))
    return {
        index: rank
        for rank, (*_unused, index) in enumerate(sorted(sortable))
    }


def _sort_value(code):
    if isinstance(code, str):
        return code
    if isinstance(code, (int, np.integer)):
        return int(code)
    return tuple(int(item) for item in code)


def _rank_color(rank: int, max_rank: int):
    cmap = LinearSegmentedColormap.from_list("depth_blue_yellow", ["#08306b", "#ffd84d"])
    if max_rank <= 0:
        return cmap(0.0)
    return cmap(min(max(float(rank) / float(max_rank), 0.0), 1.0))


def _curve_label_and_depth(code, is_cxor, graph_deps):
    if isinstance(code, str):
        return code, 0

    if is_cxor:
        node = int(code)
        task_depth = node_depth(node, graph_deps)
        if graph_deps and node in graph_deps and len(graph_deps[node]) > 0:
            parents_str = ",".join(map(str, sorted(graph_deps[node])))
            return "$T_{\\{" + str(node) + " \\leftarrow \\{" + parents_str + "\\}\\}}$", task_depth
        return "$T_{\\{" + str(node) + "\\}}$", task_depth

    code_list = list(code)
    if len(code_list) == 1:
        return "$T_{\\{" + str(code_list[0]) + "\\}}$", 0
    code_str = ",".join(map(str, sorted(code_list)))
    return "$T_{\\{" + code_str + "\\}}$", 1


def _plot_overall_curve(ax, cum_samples, overall_curve, label=None):
    if cum_samples is None or overall_curve is None:
        return
    min_len = min(len(cum_samples), len(overall_curve))
    if min_len <= 0:
        return
    ax.plot(
        cum_samples[:min_len],
        overall_curve[:min_len],
        color="#d62728",
        linestyle="-",
        linewidth=1.6,
        alpha=0.95,
        zorder=10,
        label=label,
    )


def _apply_axis_labels(
    axes,
    x_start,
    x_lim=None,
    plot_relative_loss=False,
    x_axis="steps",
    x_scale="log",
):
    for index, axis in enumerate(axes):
        axis.set_xscale(x_scale)
        axis.set_xlim(left=x_start, right=x_lim)
        ylabel = "Relative Loss" if plot_relative_loss else "Loss (bits)"
        axis.set_ylabel(ylabel, fontsize=9, fontweight="bold", labelpad=6)
        if plot_relative_loss:
            axis.set_ylim(-0.03, 1.5)
        if _should_label_x_axis(index, len(axes)):
            xlabel = (
                "Optimization steps"
                if x_axis == "steps"
                else "Training samples"
                if x_axis == "samples"
                else "Effective cumulative target samples per task"
            )
            axis.set_xlabel(xlabel, fontsize=9, fontweight="bold", labelpad=8)


def _should_label_x_axis(index: int, n_axes: int) -> bool:
    if n_axes == 1:
        return True
    if n_axes == 2:
        return index == 1
    return index >= n_axes - 2


def _style_axes(axes):
    for axis in axes:
        axis.grid(True, which="major", ls="--", color="#e0e0e0", alpha=0.6, zorder=0)
        for spine in ["top", "right", "left", "bottom"]:
            axis.spines[spine].set_visible(True)
            axis.spines[spine].set_color("#888888")
            axis.spines[spine].set_linewidth(0.8)
        axis.tick_params(axis="both", which="major", labelsize=8)
        axis.tick_params(axis="both", which="minor", labelsize=8)


def _finalize_static_figure(fig, axes, output_image_path, *, legend=True, save_pdf=False, show=False):
    del legend
    fig.tight_layout(pad=1.3, rect=(0.0, 0.0, 1.0, 0.94))
    if len(axes) == 1:
        fig.subplots_adjust(bottom=0.18, top=0.82)
    else:
        fig.subplots_adjust(bottom=0.11, top=0.86, hspace=0.28, wspace=0.16)
    if output_image_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        fig.savefig(output_image_path, bbox_inches="tight", dpi=500)
        if save_pdf or should_save_pdf():
            pdf_path = os.path.splitext(output_image_path)[0] + ".pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        logging.info("discovery_trajectories figure=%s", output_image_path)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def _save_animation_to_path(
    *,
    fig,
    axes,
    output_path,
    codes,
    is_cxor,
    graph_dependencies,
    curve_x_values,
    overall_x_values,
    animation_time_values,
    curves,
    x_start,
    x_lim,
    ylim,
    x_axis,
    x_scale,
    title,
    panel_metrics,
    panel_label,
    plot_total_loss,
    overall_curve,
    overall_curve_label,
    plot_relative_loss,
    legend,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    max_points = max((len(curve) for curve in curves.values()), default=0)
    if max_points <= 1:
        logging.warning("discovery_trajectories gif=skipped reason=insufficient_points")
        plt.close(fig)
        return
    frame_count = min(80, max(12, max_points))
    frame_ends = _linear_time_frame_ends(
        animation_time_values,
        max_points=max_points,
        frame_count=frame_count,
    )
    y_limit = _panel_y_limit(
        curves,
        plot_relative_loss,
        overall_curve=overall_curve if plot_total_loss else None,
        ylim=ylim,
    )

    def draw(frame_end):
        curve_slice = {index: curve[:frame_end] for index, curve in curves.items()}
        _draw_panels(
            fig=fig,
            axes=axes,
            curve_x_values={
                index: values[:frame_end]
                for index, values in curve_x_values.items()
            },
            overall_x_values=overall_x_values[:frame_end],
            curves=curve_slice,
            codes=codes,
            is_cxor=is_cxor,
            graph_dependencies=graph_dependencies,
            x_start=x_start,
            x_lim=x_lim,
            y_limit=y_limit,
            x_axis=x_axis,
            x_scale=x_scale,
            title=title,
            panel_metrics=panel_metrics,
            panel_label=panel_label,
            plot_total_loss=plot_total_loss,
            overall_curve=overall_curve[:frame_end] if overall_curve is not None else None,
            overall_curve_label=overall_curve_label,
            plot_relative_loss=plot_relative_loss,
            legend=legend,
        )
        fig.tight_layout(pad=1.3, rect=(0.0, 0.0, 1.0, 0.94))
        if len(axes) == 1:
            fig.subplots_adjust(bottom=0.18, top=0.82)
        else:
            fig.subplots_adjust(bottom=0.11, top=0.86, hspace=0.28, wspace=0.16)
        return axes

    animation = FuncAnimation(fig, draw, frames=frame_ends, interval=120, blit=False, repeat_delay=800)
    fig.tight_layout(pad=1.3, rect=(0.0, 0.0, 1.0, 0.94))
    animation.save(output_path, writer=PillowWriter(fps=8), dpi=160)
    plt.close(fig)
    logging.info("discovery_trajectories gif=%s", output_path)


def _linear_time_frame_ends(
    time_values,
    *,
    max_points: int,
    frame_count: int,
) -> np.ndarray:
    """Reveal observations at a constant training-time rate.

    The displayed x-axis may be steps, samples, or per-task effective samples.
    Animation timing always follows the shared training timeline supplied by
    ``samples``. Repeated frame ends deliberately preserve pauses across long
    intervals with no evaluation.
    """
    values = np.asarray(time_values, dtype=float)[: int(max_points)]
    if (
        len(values) != int(max_points)
        or len(values) <= 1
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) < 0)
        or values[-1] <= values[0]
    ):
        return np.linspace(1, int(max_points), int(frame_count), dtype=int)
    frame_times = np.linspace(values[0], values[-1], int(frame_count))
    return np.clip(
        np.searchsorted(values, frame_times, side="right"),
        1,
        int(max_points),
    )
