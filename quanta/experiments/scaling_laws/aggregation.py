from __future__ import annotations

import json
import logging
import os
import pickle
import re
from typing import Any

import numpy as np

from quanta.experiments.demand import select_theoretical_alpha
from quanta.experiments.demand_solver import compute_actual_tail_alpha
from quanta.experiments.scaling_laws.metrics import (
    TAIL_MEDIAN_EVAL_POINTS,
    tail_median,
)
from quanta.utils import theoretical_alpha


def _delta_for_pair(saved_config: dict[str, Any], pair_index: int) -> float:
    delta = saved_config.get("delta", 0.0)
    if isinstance(delta, list):
        if not delta:
            return 0.0
        return float(delta[pair_index] if pair_index < len(delta) else delta[0])
    return float(delta)


def _pair_values_from_config(saved_config: dict[str, Any], pair_index: int) -> tuple[float, float, float] | None:
    triplets = saved_config.get("rho_beta_delta")
    if isinstance(triplets, list) and triplets:
        triplet = triplets[pair_index] if pair_index < len(triplets) else triplets[0]
        if isinstance(triplet, list) and len(triplet) == 3:
            return float(triplet[0]), float(triplet[1]), float(triplet[2])

    rho = saved_config.get("rho")
    beta = saved_config.get("beta")
    if rho is None or beta is None:
        return None

    rho_values = rho if isinstance(rho, list) else [rho]
    beta_values = beta if isinstance(beta, list) else [beta]
    rho_value = rho_values[pair_index] if pair_index < len(rho_values) else rho_values[0]
    beta_value = beta_values[pair_index] if pair_index < len(beta_values) else beta_values[0]
    return float(rho_value), float(beta_value), _delta_for_pair(saved_config, pair_index)


def _pair_values_from_dir(parent_dir: str) -> tuple[int, float, float] | None:
    match = re.search(r"pair(\d+)-rho([\w\d]+)-beta([\w\d]+)", parent_dir)
    if not match:
        return None
    return (
        int(match.group(1)),
        float(match.group(2).replace("p", ".").replace("m", "-")),
        float(match.group(3).replace("p", ".").replace("m", "-")),
    )


def _demand_metadata(
    saved_config: dict[str, Any],
    rho: float,
    beta: float,
    delta: float,
) -> dict[str, Any]:
    diagnostics = saved_config.get("quanta_demand_diagnostics") or {}
    mode = str(saved_config.get("quanta_demand", diagnostics.get("mode", "shortcut")))
    actual_beta = float(diagnostics.get("actual_induced_beta", beta))
    close_fit = bool(diagnostics.get("close_fit", mode == "shortcut"))
    comparison_beta = float(
        diagnostics.get(
            "comparison_beta",
            beta if mode == "shortcut" or close_fit else actual_beta,
        )
    )
    actual_tail_alpha = diagnostics.get("actual_tail_alpha")
    saved_induced_demand = diagnostics.get("induced_quanta_demand_raw") or diagnostics.get(
        "induced_quanta_demand"
    )
    if actual_tail_alpha is None and saved_induced_demand:
        actual_tail_alpha = compute_actual_tail_alpha(
            np.asarray(
                list(saved_induced_demand.values()),
                dtype=float,
            )
        )
    demand_for_theory = {
        **diagnostics,
        "mode": mode,
        "close_fit": close_fit,
        "comparison_beta": comparison_beta,
        "actual_tail_alpha": actual_tail_alpha,
    }
    selected_alpha, theory_alpha_source = select_theoretical_alpha(
        demand=demand_for_theory,
        rho=rho,
        beta=beta,
        delta=delta,
    )
    return {
        "quanta_demand": mode,
        "trace_sampling": str(saved_config.get("trace_sampling", "principal")),
        "desired_beta": float(beta),
        "actual_induced_beta": actual_beta,
        "actual_tail_alpha": (
            float(actual_tail_alpha) if actual_tail_alpha is not None else None
        ),
        "comparison_beta": comparison_beta,
        "quanta_demand_relative_rmse": diagnostics.get("relative_rmse"),
        "quanta_demand_close_fit": close_fit,
        "quanta_demand_diagnostics": diagnostics or None,
        "desired_theoretical_alpha": theoretical_alpha(rho, beta, delta),
        "theoretical_alpha": float(selected_alpha),
        "theory_alpha_source": theory_alpha_source,
    }


def discover_all_runs(base_save_dir: str) -> list[dict[str, Any]]:
    runs_dir = os.path.join(base_save_dir, "runs")
    if not os.path.exists(runs_dir):
        return []

    discovered_records = []
    for root, dirs, files in os.walk(runs_dir):
        if "config.json" in files and "results.pkl" in files:
            config_path = os.path.join(root, "config.json")
            results_path = os.path.join(root, "results.pkl")
            try:
                with open(config_path, "r") as f:
                    saved_config = json.load(f)
                with open(results_path, "rb") as f:
                    result = pickle.load(f)
                steps_run = int(result.get("steps_run", 0))
                target_steps = int(saved_config.get("steps", result.get("target_steps", steps_run)))
                legacy_converged = bool(result.get("converged", False))
                if steps_run < target_steps and not legacy_converged:
                    logging.debug(
                        "Skipping incomplete compositional scaling checkpoint at %s (%s/%s steps).",
                        root,
                        steps_run,
                        target_steps,
                    )
                    continue

                # Prefer saved config metadata. Older archived runs encoded rho/beta in the parent directory.
                parent_dir = os.path.basename(os.path.dirname(root))
                pair_index_match = re.search(r"pair(\d+)", parent_dir)
                pair_index = int(pair_index_match.group(1)) if pair_index_match else 0
                config_pair_values = _pair_values_from_config(saved_config, pair_index)
                legacy_pair_values = _pair_values_from_dir(parent_dir)
                if config_pair_values is not None:
                    rho, beta, delta = config_pair_values
                elif legacy_pair_values is not None:
                    pair_index, rho, beta = legacy_pair_values
                    delta = _delta_for_pair(saved_config, pair_index)
                else:
                    rho = 2.0
                    beta = 2.53
                    delta = _delta_for_pair(saved_config, pair_index)

                width = int(saved_config["width"])
                architecture = saved_config.get("architecture", "mlp")
                effective_width = int(saved_config.get("width", width))
                seed = int(saved_config["seed"])
                lr = float(saved_config["lr"])
                n_parameters = int(result["n_parameters"])
                n_embedding_parameters = int(result.get("n_embedding_parameters", 0))
                n_non_embedding_parameters = int(
                    result.get(
                        "n_non_embedding_parameters",
                        n_parameters - n_embedding_parameters,
                    )
                )
                demand_metadata = _demand_metadata(saved_config, rho, beta, delta)

                record = {
                    "pair_index": pair_index,
                    "rho": rho,
                    "beta": beta,
                    "delta": delta,
                    "eval_loss_formula": saved_config.get(
                        "eval_loss_formula", "quanta_weighted"
                    ),
                    "loss_supervision": saved_config.get("loss_supervision", "all"),
                    **demand_metadata,
                    "seed": seed,
                    "width": width,
                    "lr": lr,
                    "depth": int(saved_config["depth"]),
                    "architecture": architecture,
                    "effective_width": effective_width,
                    "n_heads": int(saved_config.get("n_heads", 0) or 0),
                    "n_tasks": int(saved_config.get("n_tasks", len(result.get("codes", [])))),
                    "n_bits": int(saved_config.get("n_bits", len(result.get("Ss", [])))),
                    "n_parameters": n_parameters,
                    "n_embedding_parameters": n_embedding_parameters,
                    "n_non_embedding_parameters": n_non_embedding_parameters,
                    "model_scale": _model_scale_key(saved_config, width, n_parameters),
                    "final_eval_loss_nats": float(result["final_eval_loss_nats"]),
                    "final_eval_loss_bits": float(result["final_eval_loss_bits"]),
                    "final_mean_task_loss_bits": _final_mean_task_loss_bits(result),
                    "tail_median_eval_loss_bits": _tail_median_result_loss_bits(
                        result,
                        weighted_loss=True,
                    ),
                    "tail_median_mean_task_loss_bits": _tail_median_result_loss_bits(
                        result,
                        weighted_loss=False,
                    ),
                    "tail_median_eval_points": int(
                        result.get(
                            "tail_median_eval_points",
                            min(
                                TAIL_MEDIAN_EVAL_POINTS,
                                len(result.get("eval_steps", [])),
                            ),
                        )
                    ),
                    "tail_median_start_step": result.get("tail_median_start_step"),
                    "tail_median_end_step": result.get("tail_median_end_step"),
                    "steps_run": steps_run,
                    "save_dir": root,
                }
                discovered_records.append(record)
            except Exception as e:
                logging.warning("Error reading run directory %s: %r", root, e)

    return discovered_records


def select_best_runs(
    run_records: list[dict[str, Any]],
    *,
    weighted_loss: bool = True,
) -> list[dict[str, Any]]:
    # 1. Group runs by (pair_index, delta, model scale, lr) to compute seed averages
    lr_groups = {}
    for record in run_records:
        key = (
            record["pair_index"],
            record.get("quanta_demand", "shortcut"),
            record.get("trace_sampling", "principal"),
            record.get("eval_loss_formula", "quanta_weighted"),
            record.get("loss_supervision", "all"),
            record.get("delta", 0.0),
            _record_scale_key(record),
            record["lr"],
        )
        if key not in lr_groups:
            lr_groups[key] = []
        lr_groups[key].append(_record_loss_bits(record, weighted_loss))

    # 2. For each (pair_index, delta, model scale), find which lr has the lowest mean loss
    best_lr_for_scale = {}
    for (
        pair_index,
        quanta_demand,
        trace_sampling,
        eval_loss_formula,
        loss_supervision,
        delta,
        scale_key,
        lr,
    ), losses in lr_groups.items():
        mean_loss = sum(losses) / len(losses)
        group_key = (
            pair_index,
            quanta_demand,
            trace_sampling,
            eval_loss_formula,
            loss_supervision,
            delta,
            scale_key,
        )
        if group_key not in best_lr_for_scale or mean_loss < best_lr_for_scale[group_key]["mean_loss"]:
            best_lr_for_scale[group_key] = {
                "lr": lr,
                "mean_loss": mean_loss
            }

    # 3. Filter the original run_records to keep only those matching the best_lr for their model scale
    filtered_records = []
    for record in run_records:
        group_key = (
            record["pair_index"],
            record.get("quanta_demand", "shortcut"),
            record.get("trace_sampling", "principal"),
            record.get("eval_loss_formula", "quanta_weighted"),
            record.get("loss_supervision", "all"),
            record.get("delta", 0.0),
            _record_scale_key(record),
        )
        if group_key in best_lr_for_scale:
            if record["lr"] == best_lr_for_scale[group_key]["lr"]:
                filtered_records.append(record)

    return filtered_records


def aggregate_scaling_results(
    run_records: list[dict[str, Any]],
    *,
    weighted_loss: bool = True,
) -> list[dict[str, Any]]:
    summaries = []
    pair_keys = sorted(
        {
            (
                record["pair_index"],
                record.get("quanta_demand", "shortcut"),
                record.get("trace_sampling", "principal"),
                record.get("eval_loss_formula", "quanta_weighted"),
                record.get("loss_supervision", "all"),
                record["rho"],
                record["beta"],
                record.get("delta", 0.0),
            )
            for record in run_records
        }
    )
    for pair_index, quanta_demand, trace_sampling, eval_loss_formula, loss_supervision, rho, beta, delta in pair_keys:
        pair_records = [
            record
            for record in run_records
            if record["pair_index"] == pair_index and record["rho"] == rho and record["beta"] == beta
            and record.get("quanta_demand", "shortcut") == quanta_demand
            and record.get("trace_sampling", "principal") == trace_sampling
            and record.get("eval_loss_formula", "quanta_weighted") == eval_loss_formula
            and record.get("loss_supervision", "all") == loss_supervision
            and record.get("delta", 0.0) == delta
        ]
        width_summaries = []
        for scale_key in sorted({_record_scale_key(record) for record in pair_records}, key=str):
            width_records = [record for record in pair_records if _record_scale_key(record) == scale_key]
            losses = np.array(
                [_record_loss_bits(record, weighted_loss) for record in width_records],
                dtype=float,
            )
            tail_losses = np.array(
                [_record_tail_loss_bits(record, weighted_loss) for record in width_records],
                dtype=float,
            )
            params = np.array([record["n_parameters"] for record in width_records], dtype=float)
            embedding_params = np.array(
                [record.get("n_embedding_parameters", 0) for record in width_records],
                dtype=float,
            )
            non_embedding_params = np.array(
                [
                    record.get("n_non_embedding_parameters", record["n_parameters"])
                    for record in width_records
                ],
                dtype=float,
            )

            n_runs = len(width_records)
            loss_stats = _distribution_summary(losses)
            tail_loss_stats = _distribution_summary(tail_losses)

            width_summaries.append(
                {
                    "width": int(width_records[0].get("effective_width", width_records[0]["width"])),
                    "model_scale": str(scale_key),
                    "n_parameters_mean": float(np.mean(params)),
                    "n_parameters_std": float(np.std(params)),
                    "n_embedding_parameters_mean": float(np.mean(embedding_params)),
                    "n_embedding_parameters_std": float(np.std(embedding_params)),
                    "n_non_embedding_parameters_mean": float(np.mean(non_embedding_params)),
                    "n_non_embedding_parameters_std": float(np.std(non_embedding_params)),
                    **{
                        f"final_eval_loss_bits_{key}": value
                        for key, value in loss_stats.items()
                    },
                    **{
                        f"tail_median_eval_loss_bits_{key}": value
                        for key, value in tail_loss_stats.items()
                    },
                    "n_runs": n_runs,
                    "seeds": sorted({int(record.get("seed", 0)) for record in width_records}),
                    "selected_lr": (
                        float(width_records[0]["lr"])
                        if width_records[0].get("lr") is not None
                        else None
                    ),
                }
            )

        # Average seeds at each model scale before fitting the width trend.
        endpoint_all_fit = _fit_power_law(
            width_summaries,
            x_key="n_parameters_mean",
            y_key="final_eval_loss_bits_mean",
        )
        endpoint_non_embedding_fit = _fit_power_law(
            width_summaries,
            x_key="n_non_embedding_parameters_mean",
            y_key="final_eval_loss_bits_mean",
        )
        tail_all_fit = _fit_power_law(
            width_summaries,
            x_key="n_parameters_mean",
            y_key="tail_median_eval_loss_bits_mean",
        )

        actual_betas = [
            float(record.get("actual_induced_beta", beta))
            for record in pair_records
        ]
        comparison_betas = [
            float(record.get("comparison_beta", beta))
            for record in pair_records
        ]
        actual_tail_alphas = [
            float(record["actual_tail_alpha"])
            for record in pair_records
            if record.get("actual_tail_alpha") is not None
        ]
        fit_errors = [
            float(record["quanta_demand_relative_rmse"])
            for record in pair_records
            if record.get("quanta_demand_relative_rmse") is not None
        ]
        theoretical_alphas = []
        theory_alpha_sources = set()
        for record, comparison_beta in zip(pair_records, comparison_betas):
            source = record.get("theory_alpha_source")
            if source is None:
                source = (
                    "actual_induced_tail"
                    if (
                        quanta_demand == "composition"
                        and not record.get("quanta_demand_close_fit", False)
                        and float(delta) == 0.0
                        and record.get("actual_tail_alpha") is not None
                    )
                    else "depth_beta"
                )
            alpha = record.get("theoretical_alpha")
            if alpha is None:
                alpha = (
                    record["actual_tail_alpha"]
                    if source == "actual_induced_tail"
                    else theoretical_alpha(rho, comparison_beta, delta)
                )
            theoretical_alphas.append(float(alpha))
            theory_alpha_sources.add(str(source))
        theory_alpha_source = (
            next(iter(theory_alpha_sources))
            if len(theory_alpha_sources) == 1
            else "mixed"
        )
        theoretical_alpha_mean = float(np.mean(theoretical_alphas))
        for fit in (
            endpoint_all_fit,
            endpoint_non_embedding_fit,
            tail_all_fit,
        ):
            fit["relative_error"] = _relative_fit_error(
                fit["empirical_alpha"],
                theoretical_alpha_mean,
            )
        summaries.append(
            {
                "pair_index": int(pair_index),
                "rho": float(rho),
                "beta": float(beta),
                "delta": float(delta),
                "quanta_demand": quanta_demand,
                "trace_sampling": trace_sampling,
                "eval_loss_formula": eval_loss_formula,
                "loss_supervision": loss_supervision,
                "weighted_loss": bool(weighted_loss),
                "desired_beta": float(beta),
                "actual_induced_beta": float(np.mean(actual_betas)),
                "actual_tail_alpha": (
                    float(np.mean(actual_tail_alphas))
                    if actual_tail_alphas
                    else None
                ),
                "comparison_beta": float(np.mean(comparison_betas)),
                "quanta_demand_relative_rmse": (
                    float(np.mean(fit_errors)) if fit_errors else None
                ),
                "quanta_demand_close_fit": all(
                    record.get("quanta_demand_close_fit", quanta_demand == "shortcut")
                    for record in pair_records
                ),
                "desired_theoretical_alpha": theoretical_alpha(rho, beta, delta),
                "theoretical_alpha": theoretical_alpha_mean,
                "theory_alpha_source": theory_alpha_source,
                # Preserve the original top-level names for plot/archive compatibility.
                "empirical_alpha": endpoint_all_fit["empirical_alpha"],
                "fit_intercept": endpoint_all_fit["fit_intercept"],
                "fit_r_squared": endpoint_all_fit["r_squared"],
                "non_embedding_empirical_alpha": endpoint_non_embedding_fit[
                    "empirical_alpha"
                ],
                "non_embedding_fit_intercept": endpoint_non_embedding_fit[
                    "fit_intercept"
                ],
                "non_embedding_fit_r_squared": endpoint_non_embedding_fit[
                    "r_squared"
                ],
                "tail_median_empirical_alpha": tail_all_fit["empirical_alpha"],
                "tail_median_fit_intercept": tail_all_fit["fit_intercept"],
                "tail_median_fit_r_squared": tail_all_fit["r_squared"],
                "fits": {
                    "endpoint_all_parameters": endpoint_all_fit,
                    "endpoint_non_embedding_parameters": endpoint_non_embedding_fit,
                    "tail_median_all_parameters": tail_all_fit,
                },
                "width_summaries": width_summaries,
                "n_runs": len(pair_records),
            }
        )
    return summaries


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    n_values = int(values.size)
    mean = float(np.mean(values))
    std = float(np.std(values))
    se = float(std / np.sqrt(n_values)) if n_values else 0.0
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci95": float(1.96 * se),
    }


def _fit_power_law(
    points: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> dict[str, float | None]:
    fit_points = [
        point
        for point in points
        if float(point.get(x_key, float("nan"))) > 0
        and float(point.get(y_key, float("nan"))) > 0
    ]
    if len({float(point[x_key]) for point in fit_points}) < 2:
        return {
            "empirical_alpha": None,
            "fit_intercept": None,
            "r_squared": None,
            "relative_error": None,
        }
    x = np.log([float(point[x_key]) for point in fit_points])
    y = np.log([float(point[y_key]) for point in fit_points])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else None
    return {
        "empirical_alpha": float(-slope),
        "fit_intercept": float(intercept),
        "r_squared": float(r_squared) if r_squared is not None else None,
        "relative_error": None,
    }


def _relative_fit_error(
    empirical_alpha: float | None,
    theoretical_alpha_value: float,
) -> float | None:
    if empirical_alpha is None or theoretical_alpha_value == 0:
        return None
    return abs(float(empirical_alpha) - theoretical_alpha_value) / abs(
        theoretical_alpha_value
    )


def _final_mean_task_loss_bits(result: dict[str, Any]) -> float:
    if result.get("final_mean_task_loss_bits") is not None:
        return float(result["final_mean_task_loss_bits"])
    history = result.get("mean_task_losses_bits")
    if history:
        return float(history[-1])
    diagnostics = result.get("eval_diagnostics")
    if diagnostics and diagnostics.get("mean_task_loss_bits") is not None:
        return float(diagnostics["mean_task_loss_bits"])
    diagnostics_history = result.get("eval_diagnostics_history")
    if diagnostics_history:
        return float(diagnostics_history[-1]["mean_task_loss_bits"])
    return float("nan")


def _tail_median_result_loss_bits(
    result: dict[str, Any],
    *,
    weighted_loss: bool,
) -> float:
    explicit_key = (
        "tail_median_eval_loss_bits"
        if weighted_loss
        else "tail_median_mean_task_loss_bits"
    )
    explicit = result.get(explicit_key)
    if explicit is not None and np.isfinite(float(explicit)):
        return float(explicit)
    history = (
        result.get("eval_losses_bits")
        if weighted_loss
        else result.get("mean_task_losses_bits")
    )
    value = tail_median(history)
    if np.isfinite(value):
        return value
    return (
        float(result["final_eval_loss_bits"])
        if weighted_loss
        else _final_mean_task_loss_bits(result)
    )


def _record_loss_bits(record: dict[str, Any], weighted_loss: bool) -> float:
    key = "final_eval_loss_bits" if weighted_loss else "final_mean_task_loss_bits"
    value = float(record.get(key, float("nan")))
    if not np.isfinite(value):
        mode = "weighted" if weighted_loss else "mean-task"
        raise ValueError(f"Run record is missing a finite {mode} loss: {record.get('save_dir')}")
    return value


def _record_tail_loss_bits(record: dict[str, Any], weighted_loss: bool) -> float:
    key = (
        "tail_median_eval_loss_bits"
        if weighted_loss
        else "tail_median_mean_task_loss_bits"
    )
    value = float(record.get(key, _record_loss_bits(record, weighted_loss)))
    if not np.isfinite(value):
        mode = "weighted" if weighted_loss else "mean-task"
        raise ValueError(
            f"Run record is missing a finite {mode} tail-median loss: "
            f"{record.get('save_dir')}"
        )
    return value


def _record_scale_key(record: dict[str, Any]) -> str:
    if "model_scale" in record:
        return str(record["model_scale"])
    architecture = record.get("architecture", "mlp")
    depth = int(record.get("depth", 0) or 0)
    if architecture == "transformer":
        return f"params{int(record['n_parameters'])}"
    return f"width{int(record['width'])}-depth{depth}"


def _model_scale_key(config: dict[str, Any], width: int, n_parameters: int) -> str:
    arch = config.get("architecture", "mlp")
    depth = int(config.get("depth", 0) or 0)
    if arch == "transformer":
        width_val = int(config.get("width", width) or width)
        return (
            f"width{width_val}"
            f"-depth{depth}"
            f"-heads{int(config.get('n_heads', 0) or 0)}"
            f"-params{int(n_parameters)}"
        )
    return f"width{int(width)}-depth{depth}"
