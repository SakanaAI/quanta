from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from quanta.experiments.scaling_laws.aggregation import (
    aggregate_scaling_results,
    select_best_runs,
)
from quanta.figures import plot_scaling_laws, plot_discovery_trajectories
from quanta.figures.common import load_results, load_config_metadata
from quanta.effective_samples import effective_sample_data_from_results


def parse_args():
    parser = argparse.ArgumentParser(description="Plot figures from a saved quanta results.pkl file.")
    parser.add_argument("results", help="Path to an experiment directory or a results.pkl file.")
    parser.add_argument("--output", help="Output image path. Defaults to the experiment folder convention.")
    parser.add_argument("--window", type=int, default=100, help="Legacy metric window for implicit curriculum plots.")
    parser.add_argument("--smoothing", type=float, default=0.0, help="EMA smoothing weight in [0, 1].")
    parser.add_argument("--x-start", type=float, default=10.0, help="Left x-axis limit for implicit curriculum plots.")
    parser.add_argument("--x-lim", type=float, default=None, help="Upper x-axis limit for implicit curriculum plots.")
    parser.add_argument("--ylim", type=float, default=None, help="Upper y-axis limit in plot units.")
    parser.add_argument(
        "--x-axis",
        choices=["steps", "samples", "effective_samples"],
        default="steps",
        help="Metric and curve x-axis.",
    )
    parser.add_argument("--x-scale", choices=["log", "linear"], default="linear", help="Scale for the x-axis.")
    parser.add_argument(
        "--loss-decomposition",
        choices=["tasks", "tasks_dependencies_ready", "quanta"],
        default="tasks",
        help="Loss curves to display for scaling laws runs.",
    )
    parser.add_argument(
        "--weighted-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use distribution-weighted aggregate loss instead of mean task loss.",
    )
    parser.add_argument("--record-train-loss", action="store_true", help="Plot train loss when available.")
    parser.add_argument("--plot-total-loss", action="store_true", help="Plot the average loss curve when available.")
    return parser.parse_args()


def plot_from_pkl(
    results_path: str | os.PathLike[str],
    *,
    output: str | os.PathLike[str] | None = None,
    window: int = 100,
    smoothing: float = 0.0,
    x_start: float = 10.0,
    x_lim: float | None = None,
    ylim: float | None = None,
    x_axis: str = "steps",
    x_scale: str = "linear",
    loss_decomposition: str = "tasks",
    weighted_loss: bool = True,
    record_train_loss: bool = False,
    plot_total_loss: bool = False,
    save_pdf: bool = False,
) -> list[str]:
    results_pkl = _resolve_results_pkl(results_path)
    results = load_results(results_pkl)

    if _is_scaling_laws_aggregate(results):
        output_path = output or os.path.join(os.path.dirname(results_pkl), "figure.png")
        pair_summaries = results["pair_summaries"]
        if results.get("runs"):
            pair_summaries = aggregate_scaling_results(
                select_best_runs(results["runs"], weighted_loss=weighted_loss),
                weighted_loss=weighted_loss,
            )
        plot_scaling_laws(
            str(output_path),
            pair_summaries,
            task_name=results.get("config", {}).get("task"),
            model_name=_model_name(results.get("config", {})),
            depth=_model_depth(results.get("config", {})),
            ylim=ylim,
            save_pdf=save_pdf,
        )
        return [str(output_path)]

    if _is_scaling_laws_run(results):
        output_path = output or os.path.join(os.path.dirname(results_pkl), "figure.png")
        config = load_config_metadata(results_pkl)
        import numpy as np
        is_cxor = all(isinstance(code, (int, np.integer)) for code in results.get("codes", []))
        effective_sample_data = (
            effective_sample_data_from_results(results, config)
            if x_axis == "effective_samples"
            else None
        )
        plot_discovery_trajectories(
            output_image_path=str(output_path),
            codes=results["codes"],
            subtask_losses=results["subtask_losses"],
            overall_loss_bits=_overall_loss_bits(results, weighted_loss),
            quantum_subtask_losses=results.get("quantum_subtask_losses"),
            mean_quantum_subtask_losses=results.get(
                "mean_quantum_subtask_losses"
            ),
            samples=_optimization_steps(results, config),
            effective_samples=(
                effective_sample_data.axes
                if effective_sample_data is not None
                else None
            ),
            effective_sample_data=effective_sample_data,
            subtask_train_losses=results.get("subtask_train_losses") if record_train_loss else None,
            graph_dependencies=_int_nested_mapping(
                results.get("graph_dependencies", config.get("graph_dependencies", {}))
            ),
            model_name=_model_name(config),
            depth=_model_depth(config),
            width=_model_width(config),
            task_name=config.get("task", "CXOR").upper(),
            window=window,
            smoothing=smoothing,
            x_start=x_start,
            x_lim=x_lim,
            ylim=ylim,
            x_axis=x_axis,
            x_scale=x_scale,
            loss_decomposition=loss_decomposition,
            weighted_loss=weighted_loss,
            record_train_loss=record_train_loss,
            plot_total_loss=plot_total_loss,
            discreteness_transition_error=config.get("discreteness_transition_error"),
            discreteness_value=results.get("dte"),
            discreteness_area=results.get("dte_area"),
            discreteness_candidate_fraction=_candidate_fraction(results, len(results.get("codes", []))),
            save_pdf=save_pdf,
            legend=False,
        )
        return [str(output_path)]


    raise ValueError(f"Could not infer plot type from results file: {results_pkl}")


def _resolve_results_pkl(path: str | os.PathLike[str]) -> str:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "results.pkl"
    if not candidate.exists():
        raise FileNotFoundError(f"Results file not found: {candidate}")
    return str(candidate)


def _is_scaling_laws_run(results: dict[str, Any]) -> bool:
    return "subtask_losses" in results and "samples" in results


def _is_scaling_laws_aggregate(results: dict[str, Any]) -> bool:
    return results.get("experiment") == "scaling_laws" and "pair_summaries" in results


def _int_nested_mapping(mapping: dict[Any, Any]) -> dict[int, list[int]]:
    return {int(node): [int(parent) for parent in parents] for node, parents in (mapping or {}).items()}


def _model_name(config: dict[str, Any]) -> str:
    raw_name = str(config.get("model") or config.get("architecture") or "MLP")
    return "Transformer" if "transformer" in raw_name.lower() else raw_name


def _model_depth(config: dict[str, Any]):
    return config.get("depth", "unknown")


def _model_width(config: dict[str, Any]):
    return config.get("width", "unknown")


def _candidate_fraction(results: dict[str, Any], total_codes: int) -> float | None:
    metrics = results.get("discreteness_transition_error")
    if isinstance(metrics, dict) and metrics.get("candidate_quanta_fraction") is not None:
        return float(metrics["candidate_quanta_fraction"])
    total_quanta = results.get("total_quanta")
    if total_quanta is None or total_codes <= 0:
        return None
    return float(total_quanta) / float(total_codes)


def _overall_loss_bits(
    results: dict[str, Any],
    weighted_loss: bool,
) -> list[float] | None:
    if not weighted_loss:
        if results.get("mean_task_losses_bits") is not None:
            return [float(value) for value in results["mean_task_losses_bits"]]
        diagnostics_history = results.get("eval_diagnostics_history")
        if diagnostics_history is not None:
            return [
                float(diagnostics["mean_task_loss_bits"])
                for diagnostics in diagnostics_history
            ]
        return None
    if results.get("eval_losses_bits") is not None:
        return [float(value) for value in results["eval_losses_bits"]]
    if results.get("eval_losses") is not None:
        import math

        return [float(value) / math.log(2.0) for value in results["eval_losses"]]
    return None


def _optimization_steps(results: dict[str, Any], config: dict[str, Any]):
    if results.get("eval_steps") is not None:
        return [float(value) for value in results["eval_steps"]]
    n_points = len(results["subtask_losses"][0]) if results.get("subtask_losses") else 0
    eval_steps = int(config.get("eval_steps", 100))
    return [max(eval_steps * index, 1) for index in range(n_points)]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output_paths = plot_from_pkl(
        args.results,
        output=args.output,
        window=args.window,
        smoothing=args.smoothing,
        x_start=args.x_start,
        x_lim=args.x_lim,
        ylim=args.ylim,
        x_axis=args.x_axis,
        x_scale=args.x_scale,
        loss_decomposition=args.loss_decomposition,
        weighted_loss=args.weighted_loss,
        record_train_loss=args.record_train_loss,
        plot_total_loss=args.plot_total_loss,
        save_pdf=False,
    )
    for path in output_paths:
        logging.info("figure=%s", path)


if __name__ == "__main__":
    main()
