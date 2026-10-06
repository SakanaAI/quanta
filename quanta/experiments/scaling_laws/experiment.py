from __future__ import annotations

import fcntl
import json
import logging
import os
import pickle
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import wandb
from accelerate import Accelerator

from quanta.config import ScalingLawsConfig, PlotConfig
from quanta.figures import plot_discovery_trajectories, plot_rank_demand, plot_scaling_laws
from quanta.effective_samples import effective_sample_data_from_results
from quanta.experiments.common import log_event

from ..base import Experiment
from .aggregation import aggregate_scaling_results, discover_all_runs, select_best_runs
from .jobs import build_scaling_run_jobs
from .training import run_scaling_job
from quanta.utils import config_pair_slug, input_slug, _jsonable, _slug_number


class ScalingLawsExperiment(Experiment):
    def __init__(self, config: ScalingLawsConfig, plot_config: PlotConfig):
        self.config = config
        self.plot_config = plot_config
        self.config.save_dir = self.config.save_dir or self._default_save_dir()

    def run(self) -> str:
        accelerator = Accelerator(mixed_precision=self.config.mixed_precision)
        if accelerator.is_main_process:
            self._start_wandb()
        if accelerator.is_main_process:
            log_event(
                "scaling_laws",
                "run_start",
                task=self.config.task,
                architecture=self.config.architecture,
                trace_sampling=self.config.trace_sampling,
                jobs="pending",
                save_dir=self.config.save_dir,
            )
            os.makedirs(self.config.save_dir, exist_ok=True)
            if accelerator.num_processes > 1:
                log_event(
                    "scaling_laws",
                    "parallelism",
                    mode="accelerate_data",
                    processes=accelerator.num_processes,
                    global_batch_size=self.config.batch_size,
                )
        accelerator.wait_for_everyone()
        try:
            graph_cache_dir = os.path.join(
                os.path.dirname(self.config.save_dir),
                "graph_cache",
            )
            jobs = (
                build_scaling_run_jobs(
                    self.config,
                    graph_cache_dir=graph_cache_dir,
                )
                if accelerator.is_main_process
                else None
            )
            accelerator.wait_for_everyone()
            if jobs is None:
                jobs = build_scaling_run_jobs(
                    self.config,
                    graph_cache_dir=graph_cache_dir,
                )
            if accelerator.is_main_process:
                self._plot_demand_diagnostics(jobs)
            current_records = self._run_jobs(jobs, accelerator)
            accelerator.wait_for_everyone()

            output_image = os.path.join(self.config.save_dir, self.plot_config.output_filename)
            if not accelerator.is_main_process:
                return output_image

            records_to_plot = current_records
            for record in records_to_plot:
                self._plot_loss_decomposition_run(record)

            # Array elements may finish simultaneously.  Serialize discovery,
            # aggregation, and figure writes so a slower task cannot overwrite
            # a newer summary with an older filesystem snapshot.
            with _aggregate_lock(self.config.save_dir):
                logging.debug("Discovering all existing runs in save directory")
                unique_records = {
                    _record_identity(record): record
                    for record in discover_all_runs(self.config.save_dir)
                }
                for record in current_records:
                    unique_records[_record_identity(record)] = record
                combined_records = list(unique_records.values())
                logging.debug("unique run records=%s", len(combined_records))

                completed_variants = {
                    (
                        int(record["pair_index"]),
                        int(record["seed"]),
                        int(record["width"]),
                        float(record["lr"]),
                    )
                    for record in combined_records
                }
                expected_variants = {
                    (
                        int(job["pair_index"]),
                        int(job["seed"]),
                        int(job["width"]),
                        float(job["lr"]),
                    )
                    for job in jobs
                }
                missing_variants = expected_variants - completed_variants

                # Learning-rate selection continues to use endpoint loss.  The
                # selected runs are then reused consistently by all three fits.
                filtered_records = select_best_runs(
                    combined_records,
                    weighted_loss=self.plot_config.weighted_loss,
                )
                pair_summaries = aggregate_scaling_results(
                    filtered_records,
                    weighted_loss=self.plot_config.weighted_loss,
                )
                results = {
                    "experiment": "scaling_laws",
                    "config": _jsonable(asdict(self.config)),
                    "progress": {
                        "stage_completed": len(expected_variants) - len(missing_variants),
                        "stage_expected": len(expected_variants),
                        "stage_complete": not missing_variants,
                    },
                    "runs": combined_records,
                    "pair_summaries": pair_summaries,
                }
                self._save_aggregate(results)
                plot_scaling_laws(
                    output_image,
                    pair_summaries,
                    save_pdf=self.plot_config.save_pdf,
                    task_name=self.config.task,
                    model_name="Transformer" if self.config.architecture == "transformer" else "MLP",
                    depth=self.config.depth,
                    ylim=self.plot_config.ylim,
                )
            if wandb.run is not None:
                wandb.log({"figures/scaling_laws": wandb.Image(output_image)})
            log_event(
                "scaling_laws",
                "aggregate_updated",
                runs=len(combined_records),
                stage_completed=len(expected_variants) - len(missing_variants),
                stage_expected=len(expected_variants),
                stage_complete=not missing_variants,
                figure=output_image,
                save_dir=self.config.save_dir,
            )
            return output_image
        finally:
            if accelerator.is_main_process and wandb.run is not None:
                wandb.finish()

    def _run_jobs(self, jobs: list[dict[str, Any]], accelerator: Accelerator) -> list[dict[str, Any]]:
        return [
            run_scaling_job(
                self.config,
                self.config.save_dir,
                job,
                accelerator=accelerator,
            )
            for job in jobs
        ]

    def _save_aggregate(self, results: dict[str, Any]) -> None:
        _atomic_write_pickle(os.path.join(self.config.save_dir, "results.pkl"), results)
        _atomic_write_json(
            os.path.join(self.config.save_dir, "summary.json"),
            _jsonable(_compact_summary(results)),
        )

    def _plot_loss_decomposition_run(self, record: dict[str, Any]) -> None:
        results_path = os.path.join(record["save_dir"], "results.pkl")
        if not os.path.exists(results_path):
            return
        with open(results_path, "rb") as handle:
            results = pickle.load(handle)
        output_path = os.path.join(
            record["save_dir"],
            self.plot_config.loss_decomposition_filename,
        )
        effective_sample_data = (
            effective_sample_data_from_results(
                results,
                batch_size=record["batch_size"],
            )
            if self.plot_config.x_axis == "effective_samples"
            else None
        )
        common = {
            "codes": results["codes"],
            "subtask_losses": results["subtask_losses"],
            "overall_loss_bits": (
                results.get("eval_losses_bits")
                if self.plot_config.weighted_loss
                else _mean_task_loss_history_bits(results)
            ),
            "samples": results["eval_steps"],
            "effective_samples": (
                effective_sample_data.axes
                if effective_sample_data is not None
                else None
            ),
            "effective_sample_data": effective_sample_data,
            "subtask_train_losses": (
                results.get("subtask_train_losses")
                if self.plot_config.record_train_loss
                else None
            ),
            "graph_dependencies": results["graph_dependencies"],
            "task_frequencies": results.get("task_frequencies"),
            "model_name": "Transformer" if record["architecture"] == "transformer" else "MLP",
            "depth": record["depth"],
            "width": record["width"],
            "task_name": self.config.task.upper(),
            "quantum_subtask_losses": results.get("quantum_subtask_losses"),
            "mean_quantum_subtask_losses": results.get(
                "mean_quantum_subtask_losses"
            ),
            **self.plot_config.trajectory_kwargs(),
        }
        plot_discovery_trajectories(
            output_image_path=output_path,
            legend=False,
            **common,
        )
        animation_path = None
        if self.plot_config.animate_loss_decomposition:
            animation_path = os.path.splitext(output_path)[0] + ".gif"
            plot_discovery_trajectories(
                output_image_path=animation_path,
                animate=True,
                legend=False,
                **common,
            )
        log_event(
            "scaling_laws",
            "loss_decomposition_saved",
            figure=output_path,
            animation=animation_path,
        )

    def _plot_demand_diagnostics(self, jobs: list[dict[str, Any]]) -> None:
        unique_graphs: dict[tuple[int, int], dict[str, Any]] = {}
        for job in jobs:
            key = (int(job["pair_index"]), int(job["seed"]))
            unique_graphs.setdefault(key, job["graph"])
        filename_root, filename_extension = os.path.splitext(
            self.plot_config.demand_diagnostics_filename
        )
        for (pair_index, seed), graph in unique_graphs.items():
            suffix = "" if len(unique_graphs) == 1 else f"-pair{pair_index}-seed{seed}"
            output_path = os.path.join(
                self.config.save_dir,
                f"{filename_root}{suffix}{filename_extension}",
            )
            plot_rank_demand(
                output_path,
                graph["quanta_demand"],
                save_pdf=self.plot_config.save_pdf,
            )

    def _start_wandb(self) -> None:
        if not self.config.wandb_project or wandb.run is not None:
            return
        wandb.init(
            project=self.config.wandb_project,
            mode=self.config.wandb_mode,
            config=_jsonable(asdict(self.config)),
        )

    def _default_save_dir(self) -> str:
        pairs = "_".join(
            config_pair_slug(
                rho,
                beta,
                self.config.delta[index] if index < len(self.config.delta) else (self.config.delta[0] if self.config.delta else 0.0),
            )
            for index, (rho, beta) in enumerate(zip(self.config.rho, self.config.beta))
        )
        attention_masking = getattr(self.config, "attention_masking", None) or "none"
        mask_slug = f"-mask{attention_masking}" if attention_masking != "none" else ""
        lut_slug = f"-lut{self.config.n_lut_functions}" if getattr(self.config, "n_lut_functions", None) is not None else ""
        eval_slug = (
            f"-eval{self.config.eval_loss_formula}"
            if self.config.eval_loss_formula != "quanta_weighted"
            else ""
        )
        supervision_slug = (
            f"-sup{self.config.loss_supervision}"
            if self.config.loss_supervision != "all"
            else ""
        )
        trace_slug = (
            f"-trace{self.config.trace_sampling}"
            if self.config.trace_sampling != "principal"
            else ""
        )
        if self.config.graph_family == "exponential_paired":
            graph_slug = f"-graphexppaired-alpha{_slug_number(self.config.target_alpha)}"
        elif self.config.graph_family == "exponential_depth_paired":
            graph_slug = "-graphexpdepthpaired"
        elif self.config.graph_family == "flat_roots":
            graph_slug = f"-graphflatroots-exp{_slug_number(self.config.flat_frequency_exponent)}"
        else:
            graph_slug = ""
        run_tag_slug = (
            f"-tag{_slug_run_tag(self.config.run_tag)}"
            if self.config.run_tag
            else ""
        )
        accumulation_slug = (
            f"-accum{self.config.gradient_accumulation_steps}"
            if int(self.config.gradient_accumulation_steps) != 1
            else ""
        )
        run_slug = (
            f"depth{self.config.depth}-steps{self.config.steps}-batch{self.config.batch_size}"
            f"{mask_slug}{lut_slug}{eval_slug}{supervision_slug}{trace_slug}{graph_slug}"
            f"{accumulation_slug}{run_tag_slug}"
        )
        return os.path.join(
            ".experiments",
            self.config.task.lower(),
            "scaling_laws",
            (
                f"{pairs}-base{self.config.base_tasks}-maxdepth{self.config.max_depth}"
                f"-m{self.config.m}-demand{self.config.quanta_demand}"
            ),
            run_slug,
        )

    def _run_save_dir(self, pair_index: int, rho: float, beta: float, seed: int, width: int) -> str:
        return os.path.join(
            self.config.save_dir,
            "runs",
            f"pair{pair_index}-rho{_slug_number(rho)}-beta{_slug_number(beta)}",
            f"seed{seed}-width{width}",
        )


def _mean_task_loss_history_bits(results: dict[str, Any]) -> list[float] | None:
    if results.get("mean_task_losses_bits") is not None:
        return [float(value) for value in results["mean_task_losses_bits"]]
    diagnostics_history = results.get("eval_diagnostics_history")
    if diagnostics_history is None:
        return None
    return [
        float(diagnostics["mean_task_loss_bits"])
        for diagnostics in diagnostics_history
    ]


def _record_identity(record: dict[str, Any]) -> tuple:
    return (
        record["pair_index"],
        record.get("quanta_demand", "shortcut"),
        record.get("trace_sampling", "principal"),
        record.get("eval_loss_formula", "quanta_weighted"),
        record.get("loss_supervision", "all"),
        record["seed"],
        record.get("model_scale", f"width{record['width']}"),
        record["lr"],
    )


def _slug_run_tag(value: str) -> str:
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return slug.strip("-") or "run"


def _compact_summary(results: dict[str, Any]) -> dict[str, Any]:
    runs = results.get("runs", [])
    config = results.get("config", {})
    config_keys = (
        "task",
        "architecture",
        "width",
        "depth",
        "n_heads",
        "mlp_ratio",
        "rho_beta_delta",
        "seed",
        "eval_seed",
        "base_tasks",
        "max_depth",
        "n_local_bits",
        "graph_family",
        "flat_frequency_exponent",
        "quanta_demand",
        "trace_sampling",
        "attention_masking",
        "steps",
        "batch_size",
        "eval_samples_per_task",
        "lr",
        "scheduler",
        "warmup_phase",
        "weight_decay",
        "eval_steps",
        "save_steps",
        "run_tag",
    )
    run_keys = (
        "pair_index",
        "seed",
        "width",
        "lr",
        "depth",
        "architecture",
        "n_parameters",
        "n_embedding_parameters",
        "n_non_embedding_parameters",
        "final_eval_loss_nats",
        "final_eval_loss_bits",
        "final_mean_task_loss_bits",
        "tail_median_eval_loss_bits",
        "tail_median_mean_task_loss_bits",
        "tail_median_eval_points",
        "tail_median_start_step",
        "tail_median_end_step",
        "steps_run",
        "save_dir",
    )
    compact_runs = [
        {key: run[key] for key in run_keys if key in run}
        for run in sorted(
            runs,
            key=lambda run: (
                int(run.get("seed", 0)),
                int(run.get("width", 0)),
                float(run.get("lr", 0.0)),
            ),
        )
    ]
    return {
        "experiment": results.get("experiment", "scaling_laws"),
        "progress": {
            **results.get("progress", {}),
            "completed_runs": len(compact_runs),
            "widths": sorted(
                {int(run["width"]) for run in compact_runs if "width" in run}
            ),
            "seeds": sorted(
                {int(run["seed"]) for run in compact_runs if "seed" in run}
            ),
            "learning_rates": sorted(
                {float(run["lr"]) for run in compact_runs if "lr" in run}
            ),
        },
        "config": {key: config[key] for key in config_keys if key in config},
        "runs": compact_runs,
        "pair_summaries": results.get("pair_summaries", []),
    }


@contextmanager
def _aggregate_lock(save_dir: str):
    lock_path = os.path.join(save_dir, ".aggregate.lock")
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_pickle(path: str, value: Any) -> None:
    _atomic_write(path, "wb", lambda handle: pickle.dump(value, handle))


def _atomic_write_json(path: str, value: Any) -> None:
    _atomic_write(path, "w", lambda handle: json.dump(value, handle, indent=4))


def _atomic_write(path: str, mode: str, write) -> None:
    directory = os.path.dirname(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=directory, delete=False) as handle:
            temporary_path = handle.name
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
