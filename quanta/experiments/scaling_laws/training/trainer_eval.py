from __future__ import annotations

import logging
from typing import Any

from accelerate import Accelerator
import torch.nn as nn

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.batch_evaluation import evaluate_weighted_task_loss
from quanta.experiments.scaling_laws.batch_records import prediction_sample_records
from quanta.experiments.scaling_laws.training.trainer_curriculum import learned_task_percentages_by_depth
from quanta.experiments.scaling_laws.training.trainer_format import (
    _format_prediction_sample,
)
from quanta.experiments.scaling_laws.wandb_logging import log_scaling_wandb
from quanta.experiments.common import log_event


def evaluate_and_log_step(
    *,
    model: nn.Module,
    loss_fn,
    config: ScalingLawsConfig,
    task_spec,
    task_probabilities,
    node_depths: dict[int, int],
    batch_cache: dict[str, Any],
    device,
    accelerator: Accelerator,
    step: int,
    curriculum_threshold_bits: float,
    eval_losses: list[float],
    eval_losses_bits: list[float],
    eval_accuracies: list[float],
    eval_steps: list[int],
    eval_diagnostics_history: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics = evaluate_weighted_task_loss(
        model=accelerator.unwrap_model(model),
        loss_fn=loss_fn,
        config=config,
        task_spec=task_spec,
        probabilities=task_probabilities,
        node_depths=node_depths,
        batch_cache=batch_cache,
        device=device,
        accelerator=accelerator,
    )
    diagnostics["depth_learned_percentages"] = learned_task_percentages_by_depth(
        diagnostics,
        curriculum_threshold_bits,
    )
    eval_loss_bits = diagnostics["eval_loss_bits"]
    eval_losses.append(float(diagnostics["eval_loss_nats"]))
    eval_losses_bits.append(float(eval_loss_bits))
    eval_accuracies.append(float(diagnostics["weighted_accuracy"]))
    eval_steps.append(int(step))
    eval_diagnostics_history.append(diagnostics)
    if accelerator.is_main_process:
        _log_step_diagnostics(
            model=accelerator.unwrap_model(model),
            config=config,
            task_spec=task_spec,
            probabilities=task_probabilities,
            batch_cache=batch_cache,
            node_depths=node_depths,
            device=device,
            step=step,
            diagnostics=diagnostics,
        )
    return diagnostics


def _log_step_diagnostics(
    *,
    model: nn.Module,
    config: ScalingLawsConfig,
    task_spec,
    probabilities,
    batch_cache: dict[str, Any],
    node_depths: dict[int, int],
    device,
    step: int,
    diagnostics: dict[str, Any],
) -> None:
    log_event(
        "scaling_laws",
        "evaluation",
        step=step,
        total_steps=config.steps,
        eval_loss_formula=diagnostics["eval_loss_formula"],
        eval_loss_bits=diagnostics["eval_loss_bits"],
        weighted_accuracy=diagnostics["weighted_accuracy"],
    )
    sample_records = prediction_sample_records(
        model=model,
        config=config,
        task_spec=task_spec,
        probabilities=probabilities,
        batch_cache=batch_cache,
        node_depths=node_depths,
        device=device,
        sample_count=1,
    )
    logging.debug("qualitative_sample %s", _format_prediction_sample(sample_records[0] if sample_records else None))
    log_scaling_wandb(
        config=config,
        step=step,
        samples_seen=int(config.batch_size) * int(step),
        diagnostics=diagnostics,
    )
