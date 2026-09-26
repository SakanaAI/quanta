from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from accelerate import Accelerator

from quanta.config import ScalingLawsConfig
from quanta.experiments.common import log_event
from quanta.utils import ForkRNG
from quanta.experiments.scaling_laws.training.trainer_state import (
    _checkpoint_diagnostics,
    _save_training_state,
    _should_save_checkpoint,
    save_process_rng_state,
)


def training_save_kwargs(
    *,
    save_dir: str,
    train_losses: list[float],
    eval_losses: list[float],
    eval_losses_bits: list[float],
    eval_accuracies: list[float],
    eval_steps: list[int],
    eval_diagnostics_history: list[dict[str, Any]],
    samples: list[int],
    effective_samples: dict[int, list[int]],
    curriculum_threshold_bits: float,
    resume_state: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "save_dir": save_dir,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "eval_losses_bits": eval_losses_bits,
        "eval_accuracies": eval_accuracies,
        "eval_steps": eval_steps,
        "eval_diagnostics_history": eval_diagnostics_history,
        "samples": samples,
        "effective_samples": effective_samples,
        "curriculum_threshold_bits": curriculum_threshold_bits,
        "resume_state": resume_state,
    }


def checkpoint_if_due(
    *,
    step: int,
    model,
    optimizer,
    loss_fn,
    config: ScalingLawsConfig,
    task_spec,
    task_probabilities,
    node_depths: dict[int, int],
    batch_cache: dict[str, Any],
    device,
    accelerator: Accelerator,
    latest_diagnostics: dict[str, Any] | None,
    save_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    if not _should_save_checkpoint(config, step):
        return latest_diagnostics
    with ForkRNG() if config.task == "cnand" else nullcontext():
        latest_diagnostics = _checkpoint_diagnostics(
            model=accelerator.unwrap_model(model),
            loss_fn=loss_fn,
            config=config,
            task_spec=task_spec,
            probabilities=task_probabilities,
            node_depths=node_depths,
            batch_cache=batch_cache,
            device=device,
            accelerator=accelerator,
            latest_diagnostics=latest_diagnostics,
        )
    if config.task == "cnand":
        save_process_rng_state(
            save_kwargs["save_dir"],
            process_index=int(accelerator.process_index),
        )
    if accelerator.is_main_process:
        _save_training_state(
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            config=config,
            task_spec=task_spec,
            task_probabilities=task_probabilities,
            node_depths=node_depths,
            diagnostics=latest_diagnostics,
            n_parameters=None,
            checkpoint=True,
            **save_kwargs,
        )
    accelerator.wait_for_everyone()
    return latest_diagnostics


def save_final_results(
    *,
    model,
    optimizer,
    config: ScalingLawsConfig,
    task_spec,
    task_probabilities,
    node_depths: dict[int, int],
    latest_diagnostics: dict[str, Any] | None,
    save_kwargs: dict[str, Any],
    accelerator: Accelerator,
    run_start_time: float,
) -> dict[str, Any] | None:
    saved_model = accelerator.unwrap_model(model)
    if config.task == "cnand":
        save_process_rng_state(
            save_kwargs["save_dir"],
            process_index=int(accelerator.process_index),
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None
    results = _save_training_state(
        model=saved_model,
        optimizer=optimizer,
        config=config,
        task_spec=task_spec,
        task_probabilities=task_probabilities,
        node_depths=node_depths,
        diagnostics=latest_diagnostics,
        n_parameters=None,
        checkpoint=False,
        **save_kwargs,
    )
    log_event(
        "scaling_laws",
        "job_complete",
        runtime_seconds=time.monotonic() - run_start_time,
        width=config.width,
        learning_rate=config.lr,
        seed=config.seed,
        steps=len(save_kwargs["train_losses"]),
        total_steps=config.steps,
        save_dir=save_kwargs["save_dir"],
    )
    return results
