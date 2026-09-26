from __future__ import annotations

import logging

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .common import save_figure


POINT_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
]


def plot_rank_demand(
    output_image_path: str,
    demand_diagnostics: dict,
    *,
    save_pdf: bool = False,
) -> None:
    """Plot the finite microscopic rank law and its truncated unmet-demand tail."""
    raw_mapping = demand_diagnostics.get("induced_quanta_demand_raw") or {}
    if not raw_mapping:
        logging.warning("rank_demand figure=skipped reason=missing_demand")
        return
    demand = np.sort(np.asarray(list(raw_mapping.values()), dtype=float))[::-1]
    ranks = np.arange(1, demand.size + 1, dtype=float)
    alpha = float(demand_diagnostics["theoretical_alpha"])
    fit_min = int(demand_diagnostics.get("rank_demand_fit_min", 1))
    fit_max = int(demand_diagnostics.get("rank_demand_fit_max", demand.size))
    fit_alpha = demand_diagnostics.get("actual_rank_alpha")
    theory_source = demand_diagnostics.get("theory_alpha_source")

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=300)
    ax = axes[0]
    ax.loglog(ranks, demand, color=POINT_COLORS[0], linewidth=1.5, label="Induced demand")
    anchor_rank = min(max(fit_min, 1), demand.size)
    anchor = demand[anchor_rank - 1]
    reference = anchor * (ranks / anchor_rank) ** (-(1.0 + alpha))
    reference_label = (
        rf"Depth-derived $\alpha={alpha:.3f}$"
        if theory_source == "depth_beta"
        else rf"Target $\alpha={alpha:.3f}$"
    )
    ax.loglog(
        ranks,
        reference,
        "--",
        color="#222222",
        linewidth=1.0,
        label=reference_label,
    )
    ax.axvspan(fit_min, fit_max, color="#009E73", alpha=0.10, label="Preregistered fit")
    measured = "n/a" if fit_alpha is None else f"{float(fit_alpha):.3f}"
    rank_title = (
        rf"Stepwise rank diagnostic ($\hat{{\alpha}}={measured}$)"
        if theory_source == "depth_beta"
        else rf"Microscopic rank demand ($\hat{{\alpha}}={measured}$)"
    )
    ax.set_title(rank_title)
    ax.set_xlabel(r"Demand rank $k$")
    ax.set_ylabel(r"Activation probability $p_k$")
    ax.legend(fontsize=7)

    tail_ax = axes[1]
    capacities = np.arange(1, demand.size, dtype=float)
    finite_tail = np.cumsum(demand[::-1])[::-1][1:]
    tail_ax.loglog(capacities, finite_tail, color=POINT_COLORS[1], linewidth=1.5, label="Finite unmet demand")
    tail_anchor_rank = min(max(fit_min, 1), finite_tail.size)
    tail_anchor = finite_tail[tail_anchor_rank - 1]
    tail_reference = tail_anchor * (capacities / tail_anchor_rank) ** (-alpha)
    tail_ax.loglog(capacities, tail_reference, "--", color="#222222", linewidth=1.0, label=rf"Infinite-tail target $K^{{-{alpha:.3f}}}$")
    tail_ax.axvspan(fit_min, min(fit_max, finite_tail.size), color="#009E73", alpha=0.10)
    finite_alpha = demand_diagnostics.get("actual_tail_alpha")
    finite_label = "n/a" if finite_alpha is None else f"{float(finite_alpha):.3f}"
    tail_ax.set_title(rf"Finite truncated tail (global slope={finite_label})")
    tail_ax.set_xlabel(r"Resolved frontier $K$")
    tail_ax.set_ylabel(r"$\sum_{{k>K}} p_k$")
    tail_ax.legend(fontsize=7)

    for current_ax in axes:
        current_ax.grid(True, which="major", ls="--", color="#dddddd", alpha=0.7)
    if theory_source == "depth_beta":
        target_beta = demand_diagnostics.get("desired_beta")
        measured_beta = demand_diagnostics.get("actual_induced_beta")
        if target_beta is not None and measured_beta is not None:
            fig.suptitle(
                rf"Depth law: $\beta={float(target_beta):.3f}$, "
                rf"measured $\hat{{\beta}}={float(measured_beta):.3f}$; "
                rf"$\alpha=\log_\rho\beta-1={alpha:.3f}$",
                fontsize=9,
            )
    fig.tight_layout()
    save_figure(fig, output_image_path, dpi=500, save_pdf=save_pdf)
    logging.info("rank_demand figure=%s", output_image_path)


def plot_scaling_laws(
    output_image_path: str,
    pair_summaries: list[dict],
    *,
    save_pdf: bool = False,
    task_name: str | None = None,
    model_name: str | None = None,
    depth: int | str | None = None,
    width: int | str | None = None,
    ylim: float | None = None,
    theory_line_anchor: str = "first",
) -> None:
    if not pair_summaries:
        logging.warning("scaling_laws figure=skipped reason=missing_results")
        return
    if theory_line_anchor not in {"first", "log_mean", "median"}:
        raise ValueError(
            "theory_line_anchor must be 'first', 'log_mean', or 'median'."
        )

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.2), dpi=300)
    panel_specs = (
        {
            "title": "Endpoint / all parameters",
            "x_key": "n_parameters_mean",
            "x_fallback": "n_parameters_mean",
            "y_key": "final_eval_loss_bits_mean",
            "y_fallback": "final_eval_loss_bits_mean",
            "error_prefix": "final_eval_loss_bits",
            "fit_name": "endpoint_all_parameters",
            "alpha_key": "empirical_alpha",
            "intercept_key": "fit_intercept",
            "r_squared_key": "fit_r_squared",
            "xlabel": r"Trainable parameters $P$",
        },
        {
            "title": "Endpoint / non-embedding parameters",
            "x_key": "n_non_embedding_parameters_mean",
            "x_fallback": "n_parameters_mean",
            "y_key": "final_eval_loss_bits_mean",
            "y_fallback": "final_eval_loss_bits_mean",
            "error_prefix": "final_eval_loss_bits",
            "fit_name": "endpoint_non_embedding_parameters",
            "alpha_key": "non_embedding_empirical_alpha",
            "intercept_key": "non_embedding_fit_intercept",
            "r_squared_key": "non_embedding_fit_r_squared",
            "xlabel": r"Non-embedding parameters $P_{\rm nonemb}$",
        },
        {
            "title": "Tail median / all parameters",
            "x_key": "n_parameters_mean",
            "x_fallback": "n_parameters_mean",
            "y_key": "tail_median_eval_loss_bits_mean",
            "y_fallback": "final_eval_loss_bits_mean",
            "error_prefix": "tail_median_eval_loss_bits",
            "fit_name": "tail_median_all_parameters",
            "alpha_key": "tail_median_empirical_alpha",
            "intercept_key": "tail_median_fit_intercept",
            "r_squared_key": "tail_median_fit_r_squared",
            "xlabel": r"Trainable parameters $P$",
        },
    )
    weighted_loss = all(
        bool(pair.get("weighted_loss", True))
        for pair in pair_summaries
    )
    loss_label = "Weighted eval loss (bits)" if weighted_loss else "Mean task loss (bits)"
    fig.suptitle(
        _plot_title(
            task_name,
            model_name=_display_model_name(model_name),
            depth=depth,
            width=width if width is not None else _width_range_label(pair_summaries),
        ),
        fontsize=10,
        fontweight="bold",
        y=1.01,
    )

    for ax, panel in zip(axes, panel_specs):
        plotted_y_maxima: list[float] = []
        legend_values: tuple[float | None, float | None, str | None, float | None] | None = None
        for index, pair in enumerate(pair_summaries):
            color = POINT_COLORS[index % len(POINT_COLORS)]
            points = list(pair.get("width_summaries", []))
            points.sort(
                key=lambda point: float(
                    point.get(panel["x_key"], point[panel["x_fallback"]])
                )
            )
            if not points:
                continue
            x = np.asarray(
                [point.get(panel["x_key"], point[panel["x_fallback"]]) for point in points],
                dtype=float,
            )
            y = np.asarray(
                [point.get(panel["y_key"], point[panel["y_fallback"]]) for point in points],
                dtype=float,
            )
            error_prefix = panel["error_prefix"]
            if error_prefix.startswith("tail_median") and not any(
                f"{error_prefix}_ci95" in point for point in points
            ):
                error_prefix = "final_eval_loss_bits"
            yerr = np.asarray(
                [_error_bar(point, str(error_prefix)) for point in points],
                dtype=float,
            )
            fit = pair.get("fits", {}).get(panel["fit_name"], {})
            empirical_alpha = fit.get(
                "empirical_alpha",
                pair.get(panel["alpha_key"], pair.get("empirical_alpha")),
            )
            fit_intercept = fit.get(
                "fit_intercept",
                pair.get(panel["intercept_key"], pair.get("fit_intercept")),
            )
            r_squared = fit.get(
                "r_squared",
                pair.get(panel["r_squared_key"], pair.get("fit_r_squared")),
            )
            theoretical = pair.get("theoretical_alpha")
            if legend_values is None:
                legend_values = (
                    empirical_alpha,
                    theoretical,
                    pair.get("theory_alpha_source"),
                    r_squared,
                )
            ax.errorbar(
                x,
                y,
                yerr=yerr if np.any(yerr > 0) else None,
                marker="o",
                linestyle="none",
                elinewidth=1.0,
                capsize=3,
                color=color,
            )
            plotted_y_maxima.extend((y + np.maximum(yerr, 0.0)).tolist())
            if empirical_alpha is not None and fit_intercept is not None and len(x) > 1:
                fit_y = np.exp(float(fit_intercept)) * x ** (-float(empirical_alpha))
                ax.plot(x, fit_y, linestyle="-", linewidth=1.0, color=color, alpha=0.8)
                plotted_y_maxima.extend(fit_y.tolist())
            if theoretical is not None and len(x) > 1 and np.all(y > 0):
                if theory_line_anchor == "log_mean":
                    intercept = np.mean(np.log(y) + float(theoretical) * np.log(x))
                    anchor = np.exp(intercept) * x ** (-float(theoretical))
                elif theory_line_anchor == "median":
                    median_index = len(x) // 2
                    anchor = y[median_index] * (
                        x / x[median_index]
                    ) ** (-float(theoretical))
                else:
                    anchor = y[0] * (x / x[0]) ** (-float(theoretical))
                ax.plot(x, anchor, linestyle="--", linewidth=1.0, color="#000000", alpha=0.8)
                plotted_y_maxima.extend(anchor.tolist())
            for point, x_value, y_value in zip(points, x, y):
                if point.get("width") is not None:
                    ax.annotate(
                        f"w={int(point['width'])}",
                        (x_value, y_value),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=6,
                        color=color,
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(panel["xlabel"], fontsize=9, fontweight="bold", labelpad=6)
        ax.set_ylabel(loss_label, fontsize=8, fontweight="bold", labelpad=5)
        ax.set_title(panel["title"], fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        finite_maxima = [value for value in plotted_y_maxima if np.isfinite(value) and value > 0]
        if ylim is not None:
            lower, _ = ax.get_ylim()
            upper = float(ylim)
            ax.set_ylim(bottom=min(lower, upper / 10.0), top=upper)
        elif finite_maxima:
            lower, _ = ax.get_ylim()
            ax.set_ylim(bottom=lower, top=max(finite_maxima) * 1.03)
        ax.grid(True, which="major", ls="--", color="#e0e0e0", alpha=0.6, zorder=0)
        empirical, theoretical, source, r_squared = legend_values or (None, None, None, None)
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="-", color=POINT_COLORS[0], label=_alpha_label("Empirical alpha", empirical)),
                Line2D([0], [0], linestyle="--", linewidth=1.0, color="#000000", label=_alpha_label(_theoretical_alpha_prefix(source), theoretical)),
                Line2D([0], [0], linestyle="none", color="#000000", label=_relative_error_label(_relative_error(empirical, theoretical))),
                Line2D([0], [0], linestyle="none", color="#000000", label=_r_squared_label(r_squared)),
            ],
            prop={"size": 6.5},
            frameon=True,
            framealpha=0.9,
            facecolor="#ffffff",
            edgecolor="#cccccc",
        )
    fig.tight_layout()
    save_figure(fig, output_image_path, dpi=500, save_pdf=save_pdf)
    logging.info("scaling_laws figure=%s", output_image_path)


def _error_bar(point: dict, prefix: str) -> float:
    ci95 = point.get(f"{prefix}_ci95")
    if ci95 is not None:
        return float(ci95)
    se = point.get(f"{prefix}_se")
    if se is not None:
        return float(se)
    n_runs = max(1, int(point.get("n_runs", 1)))
    return float(point.get(f"{prefix}_std", 0.0)) / np.sqrt(n_runs)


def _alpha_label(prefix: str, alpha: float | None) -> str:
    value = r"\mathrm{n/a}" if alpha is None else f"{float(alpha):.3f}"
    return rf"{prefix} $\alpha={value}$"


def _theoretical_alpha_prefix(theory_alpha_source: str | None) -> str:
    if theory_alpha_source == "actual_induced_tail":
        return "Theoretical alpha (actual induced tail)"
    return "Theoretical alpha"


def _relative_error(empirical_alpha: float | None, theoretical_alpha: float | None) -> float | None:
    if empirical_alpha is None or theoretical_alpha is None:
        return None
    theoretical_alpha = float(theoretical_alpha)
    if theoretical_alpha == 0:
        return None
    return abs(float(empirical_alpha) - theoretical_alpha) / abs(theoretical_alpha)


def _relative_error_label(relative_error: float | None) -> str:
    value = r"\mathrm{n/a}" if relative_error is None else rf"{100.0 * float(relative_error):.1f}\%"
    return rf"Relative Error $={value}$"


def _r_squared_label(r_squared: float | None) -> str:
    value = r"\mathrm{n/a}" if r_squared is None else f"{float(r_squared):.4f}"
    return rf"$R^2={value}$"


def _plot_title(
    task_name: str | None,
    *,
    model_name: str | None = None,
    depth: int | str | None = None,
    width: int | str | None = None,
) -> str:
    title = "Parameters Power Scaling"
    if task_name:
        title += f" on {_format_task_name(task_name)}"
    if model_name:
        title += f" with {model_name}"
    if depth is not None:
        title += f" (depth={_display_value(depth)})"
    return title


def _display_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    if "transformer" in str(model_name).lower():
        return "Transformer"
    return str(model_name)


def _display_value(value) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _width_range_label(pair_summaries: list[dict]) -> str | None:
    width = sorted(
        {
            int(point["width"])
            for pair in pair_summaries
            for point in pair.get("width_summaries", [])
            if point.get("width") is not None
        }
    )
    if not width:
        return None
    if len(width) == 1:
        return str(width[0])
    return f"{width[0]}-{width[-1]}"


def _format_task_name(task_name: str) -> str:
    if task_name.lower() == "cnand":
        return "cNAND"
    return task_name.upper()
