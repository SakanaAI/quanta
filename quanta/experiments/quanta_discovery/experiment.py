from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import torch

from quanta.config import PlotConfig, QuantaDiscoveryConfig
from quanta.experiments.base import Experiment
from quanta.experiments.common import log_event, save_config
from quanta.utils import _jsonable, get_device, set_seeds

from .priority import (
    analyze_event_priority,
    collect_full_batch_gd_trajectory,
)
from .cross_fitted_priority import analyze_cross_fitted_event_priority
from .factorization import build_simple_closure_graph
from .progress import QuantaDiscoveryProgress
from .timing import RunTimingRecorder
from .trajectory import (
    build_model_and_task,
    build_prediction_event_panel,
    write_trajectory_artifacts,
    save_functional_residual_writes,
)


ALGORITHM_NAME = "quanta_discovery_v0"


class QuantaDiscoveryExperiment(Experiment):
    """Measure event-level GD priority and expose binary-owner candidates."""

    def __init__(
        self,
        config: QuantaDiscoveryConfig,
        plot_config: PlotConfig | None = None,
    ) -> None:
        self.config = config
        self.plot_config = plot_config
        self.config.save_dir = self.config.save_dir or _default_save_dir(config)

    def run(self) -> str:
        save_dir = str(self.config.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        console_progress = QuantaDiscoveryProgress(save_dir)
        timing = RunTimingRecorder(
            save_dir,
            experiment_name="quanta_discovery",
            emit_log_events=False,
            observer=console_progress,
        )
        try:
            return self._run(timing)
        except Exception as error:
            elapsed = timing.fail(error)
            logging.exception("quanta discovery failed: save_dir=%s", save_dir)
            _write_status(
                save_dir,
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                total_elapsed_seconds=elapsed,
            )
            log_event(
                "quanta_discovery",
                "run_failed",
                error_type=type(error).__name__,
                error=str(error),
                total_elapsed_seconds=elapsed,
                save_dir=save_dir,
            )
            raise

    def _run(self, timing: RunTimingRecorder) -> str:
        save_dir = str(self.config.save_dir)
        setup = timing.start_phase("configuration_setup", total=1)
        set_seeds(int(self.config.seed))
        device = torch.device(get_device(self.config.device))
        setup.update(1, device=device)
        timing.finish_phase(setup)

        panel_progress = timing.start_phase("model_and_event_panel", total=1)
        _write_status(save_dir, status="constructing_event_panel")
        task, model = build_model_and_task(self.config, device=device)
        train_panel = build_prediction_event_panel(task, split="train")
        eval_panel = build_prediction_event_panel(task, split="eval")
        panel = build_prediction_event_panel(task, split="all")
        save_config(self.config, save_dir)
        panel_progress.update(
            1,
            train_examples=len(task.train),
            eval_examples=len(task.eval_examples),
            train_prediction_events=len(train_panel),
            eval_prediction_events=len(panel) - len(train_panel),
            transformer_layers=len(model.layers),
        )
        timing.finish_phase(panel_progress)
        _write_metadata(
            save_dir,
            self.config,
            model=model,
            panel=panel,
            train_panel=train_panel,
            device=device,
        )

        total_optimizer_steps = int(self.config.steps or 0)
        trajectory_progress = timing.start_phase(
            "training_trajectory",
            total=total_optimizer_steps,
            checkpoint_every=int(self.config.checkpoint_every),
        )

        def checkpoint_complete(
            checkpoint: int,
            step: int,
            total_checkpoints: int,
            mean_event_losses: dict[str, float],
        ) -> None:
            progress = trajectory_progress.update(
                int(step),
                checkpoint_index=int(checkpoint),
                total_checkpoints=int(total_checkpoints),
                mean_train_event_loss=float(mean_event_losses["train"]),
                mean_eval_event_loss=float(mean_event_losses["eval"]),
            )
            _write_status(
                save_dir,
                status="collecting_training_trajectory",
                optimizer_step=int(step),
                total_optimizer_steps=total_optimizer_steps,
                **progress,
            )

        trajectory = collect_full_batch_gd_trajectory(
            model,
            task,
            panel,
            checkpoint_every=int(self.config.checkpoint_every),
            save_dir=save_dir,
            device=device,
            on_checkpoint=checkpoint_complete,
        )
        timing.finish_phase(
            trajectory_progress, checkpoints=trajectory.num_checkpoints
        )
        serialization = timing.start_phase("trajectory_serialization", total=1)
        write_trajectory_artifacts(save_dir, panel, trajectory)
        # This is intentionally pre-closure and uses the source block writes,
        # not MLP-neuron update packets.
        save_functional_residual_writes(
            save_dir, model, train_panel, trajectory, device=device
        )
        serialization.update(1, checkpoints=trajectory.num_checkpoints)
        timing.finish_phase(serialization)

        priority_progress = timing.start_phase(
            "event_priority_analysis", total=trajectory.num_checkpoints - 1
        )

        def priority_checkpoint(
            completed: int,
            total: int,
            optimizer_step: int,
            metrics: dict[str, float],
        ) -> None:
            progress = priority_progress.update(
                int(completed),
                optimizer_step=int(optimizer_step),
                **metrics,
            )
            _write_status(
                save_dir,
                status="analyzing_event_priority",
                **progress,
            )

        common_priority_arguments = {
            "max_components": int(self.config.max_candidates_per_layer),
            "alternating_steps": int(self.config.candidate_fit_steps),
            "minimum_component_gain_fraction": float(
                self.config.minimum_candidate_gain
            ),
            "save_dir": save_dir,
            "device": device,
            "on_checkpoint": priority_checkpoint,
        }
        if self.config.priority_estimator in {
            "cross_fitted_checkpoint",
            "cross_fitted_full_complement",
        }:
            analysis = analyze_cross_fitted_event_priority(
                model,
                task,
                train_panel,
                eval_panel,
                trajectory,
                folds=int(self.config.priority_crossfit_folds),
                direction_examples=int(self.config.priority_direction_examples),
                direction_replicates=int(
                    self.config.priority_direction_replicates
                ),
                split_seed=int(self.config.priority_split_seed),
                full_complement=(
                    self.config.priority_estimator
                    == "cross_fitted_full_complement"
                ),
                **common_priority_arguments,
            )
        else:
            analysis = analyze_event_priority(
                model,
                task,
                train_panel,
                eval_panel,
                trajectory,
                **common_priority_arguments,
            )
        timing.finish_phase(
            priority_progress,
            maximum_priority_identity_error=analysis.summary.get(
                "maximum_priority_identity_error"
            ),
        )

        factorization_progress = timing.start_phase(
            "composition_graph", total=1
        )
        graph_summary = build_simple_closure_graph(
            Path(save_dir),
            maximum_closure_cost=float(self.config.maximum_edge_closure_cost),
        )
        factorization_progress.update(1, artifact=str(graph_summary))
        timing.finish_phase(factorization_progress)

        summary = {
            "method": _algorithm_name(self.config),
            "scientific_status": (
                "Full-batch GD dynamics with binary event support, explicit "
                "residual priority, source-relative candidate retention, and the "
                "canonical simple-closure composition graph."
            ),
            "priority_analysis": analysis.summary,
            "artifacts": {
                "priority_analysis": "priority_analysis.npz",
                "priority_events": "priority_events.json",
                "priority_eval_events": "priority_eval_events.json",
                "factorization": "factorization/summary.json",
            },
        }
        _write_json(os.path.join(save_dir, "summary.json"), summary)
        total_elapsed = timing.complete(
            analyzed_layers=int(self.config.n_layers),
            candidate_slot_ceiling=(
                int(self.config.n_layers)
                * int(self.config.max_candidates_per_layer)
            ),
        )
        _write_status(
            save_dir,
            status="complete",
            analyzed_layer_count=int(self.config.n_layers),
            candidate_slot_ceiling=(
                int(self.config.n_layers)
                * int(self.config.max_candidates_per_layer)
            ),
            total_elapsed_seconds=total_elapsed,
        )
        return save_dir


def _write_metadata(
    save_dir: str,
    config: QuantaDiscoveryConfig,
    *,
    model: torch.nn.Module,
    panel: Any,
    train_panel: Any,
    device: torch.device,
) -> None:
    payload = {
        "experiment": "quanta_discovery",
        "algorithm": _algorithm_name(config),
        "scientific_status": (
            "Full-batch GD priority with binary event supports, no event "
            "amplitudes, and a source-relative candidate threshold."
        ),
        "num_prediction_events": len(panel),
        "num_train_prediction_events": len(train_panel),
        "num_eval_prediction_events": len(panel) - len(train_panel),
        "num_transformer_layers": len(model.layers),
        "checkpoint_every": int(config.checkpoint_every),
        "max_candidates_per_layer": int(config.max_candidates_per_layer),
        "candidate_fit_steps": int(config.candidate_fit_steps),
        "fixed_conventions": {
            "optimizer": str(config.optimizer),
            "dynamics": str(config.priority_estimator),
            "observation_panel": "all train and evaluation prediction events",
            "data_atom": "independent_supervised_prediction_event",
            "priority": str(config.priority_estimator),
            "candidate_model": "binary_event_support_times_shared_priority_curve",
            "event_amplitudes": "absent",
            "count": "retained only above minimum gain; maximum is a ceiling",
            "residual": "explicit_positive_and_negative_unexplained_priority",
            "cumulative_priority": "timing_diagnostic_not_existence",
            "composition": "canonical one-parent simple-closure graph",
            "discreteness": "cumulative curves measured explicitly",
        },
        "device": str(device),
    }
    _write_json(os.path.join(save_dir, "metadata.json"), payload)


def _default_save_dir(config: QuantaDiscoveryConfig) -> str:
    payload = {
        key: value for key, value in asdict(config).items() if key != "save_dir"
    }
    digest = hashlib.sha256(
        json.dumps(
            _jsonable(payload), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:12]
    training_length = (
        f"steps{int(config.steps)}"
        if config.steps is not None
        else f"epochs{float(config.epochs):g}"
    )
    label = (
        f"{config.data_splits}-layers{int(config.n_layers)}-"
        f"d{int(config.d_model)}-heads{int(config.n_heads)}-"
        f"{training_length}-seed{int(config.seed)}"
    )
    label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in label
    )
    return os.path.join(
        ".experiments",
        "quanta_discovery",
        "number_naming",
        ALGORITHM_NAME,
        str(config.priority_estimator),
        f"{label[:72]}-{digest}",
    )


def _algorithm_name(config: QuantaDiscoveryConfig) -> str:
    del config
    return ALGORITHM_NAME


def _write_status(save_dir: str, **payload: Any) -> None:
    _write_json(
        os.path.join(save_dir, "status.json"),
        {"updated_at_unix": time.time(), **payload},
    )


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
    os.replace(temporary, path)
