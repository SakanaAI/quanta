from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator

from quanta.config import ScalingLawsConfig
from quanta.metrics import learned_threshold_bits
from quanta.utils import ForkRNG, set_seeds
from quanta.experiments.common import build_adam_optimizer, log_event

from ..batches import (
    build_seeded_cnand_batch_cache,
    build_sampled_cnand_batch,
    task_probability_tensor,
    uses_masked_token_supervision,
)
from ..distributed import broadcast_object
from .trainer_batching import set_optimizer_learning_rate
from .trainer_curriculum import (
    scheduled_learning_rate,
)
from .trainer_eval import evaluate_and_log_step
from .trainer_finalize import checkpoint_if_due, save_final_results, training_save_kwargs
from .trainer_state import restore_process_rng_state
from .trainer_step import process_local_batch_size, run_training_step
from quanta.utils import _jsonable


def train_until_budget_or_convergence(
    *,
    model: nn.Module,
    loss_fn,
    config: ScalingLawsConfig,
    task_spec,
    save_dir: str,
    node_depths: dict[int, int],
    accelerator: Accelerator | None = None,
    resume_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accelerator = accelerator or Accelerator(mixed_precision=config.mixed_precision)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    device = accelerator.device
    start_step = 0
    previous_results = None
    if resume_state is not None:
        previous_results = resume_state["results"]
        start_step = int(resume_state["steps_run"])
        state_dict = torch.load(resume_state["model_path"], map_location=device)
        model.load_state_dict(state_dict)
    optimizer = build_adam_optimizer(model, lr=config.lr, weight_decay=config.weight_decay)
    optimizer_path = resume_state.get("optimizer_path") if resume_state is not None else None
    if optimizer_path and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    model, optimizer = accelerator.prepare(model, optimizer)
    if accelerator.num_processes > 1:
        set_seeds(int(config.seed) + 1_000_003 * int(accelerator.process_index))

    train_losses = list(previous_results.get("train_losses", [])) if previous_results else []
    eval_losses = list(previous_results.get("eval_losses", [])) if previous_results else []
    eval_losses_bits = list(previous_results.get("eval_losses_bits", [])) if previous_results else []
    eval_accuracies = list(previous_results.get("eval_accuracies", [])) if previous_results else []
    eval_steps = list(previous_results.get("eval_steps", [])) if previous_results else []
    eval_diagnostics_history = (
        list(previous_results.get("eval_diagnostics_history", []))
        if previous_results
        else []
    )
    samples = (
        list(previous_results.get("training_samples", previous_results.get("samples", [])))
        if previous_results
        else []
    )
    effective_sample_history = (
        {
            int(code): [int(value) for value in values]
            for code, values in previous_results.get("effective_samples", {}).items()
        }
        if previous_results and isinstance(previous_results.get("effective_samples"), dict)
        else {}
    )
    latest_diagnostics = previous_results.get("eval_diagnostics") if previous_results else None
    task_probabilities = task_probability_tensor(task_spec.codes, config.task_frequencies, device)
    batch_cache = build_seeded_cnand_batch_cache(config, task_spec, device)
    if config.task == "cnand":
        process_seed = int(config.seed) + 1_000_003 * int(accelerator.process_index)
        set_seeds(process_seed)
    if config.task == "cnand" and resume_state is not None:
        restore_process_rng_state(
            resume_state["run_dir"],
            process_index=int(accelerator.process_index),
            device=device,
        )
    curriculum_threshold_bits = learned_threshold_bits()
    local_batch_size = process_local_batch_size(config, accelerator)
    fixed_microbatches = None
    if not config.dynamic:
        fixed_microbatches = [
            build_sampled_cnand_batch(
                config=config, task_spec=task_spec, probabilities=task_probabilities,
                batch_cache=batch_cache, device=device, batch_size=local_batch_size,
            )
            for _ in range(int(config.gradient_accumulation_steps))
        ]
    cumulative_task_samples = _effective_sample_totals_from_history(
        task_spec.codes,
        effective_sample_history,
    )
    if uses_masked_token_supervision(config):
        loss_sum_fn = None
    else:
        loss_sum_fn = nn.CrossEntropyLoss(reduction="sum")
    run_start_time = time.monotonic()
    evaluation_interval = int(config.eval_steps)
    progress_interval = min(evaluation_interval, 1_000)
    if accelerator.is_main_process and start_step:
        log_event(
            "scaling_laws",
            "training_resumed",
            step=start_step,
            total_steps=config.steps,
        )

    if start_step == 0 and 0 not in eval_steps:
        with _evaluation_rng(config):
            latest_diagnostics = evaluate_and_log_step(
                model=model,
                loss_fn=loss_fn,
                config=config,
                task_spec=task_spec,
                task_probabilities=task_probabilities,
                node_depths=node_depths,
                batch_cache=batch_cache,
                device=device,
                accelerator=accelerator,
                step=0,
                curriculum_threshold_bits=curriculum_threshold_bits,
                eval_losses=eval_losses,
                eval_losses_bits=eval_losses_bits,
                eval_accuracies=eval_accuracies,
                eval_steps=eval_steps,
                eval_diagnostics_history=eval_diagnostics_history,
            )
            _append_effective_sample_snapshot(
                effective_sample_history,
                task_spec.codes,
                cumulative_task_samples,
            )

    for step in range(start_step + 1, int(config.steps) + 1):
        current_lr = scheduled_learning_rate(
            float(config.lr),
            int(step),
            int(config.steps),
            config.scheduler,
            warmup_phase=float(config.warmup_phase),
            plateau_phase=float(config.plateau_phase),
        )
        set_optimizer_learning_rate(optimizer, current_lr)
        model.train()
        train_loss, task_sample_counts = run_training_step(
            model=model,
            optimizer=optimizer,
            loss_sum_fn=loss_sum_fn,
            config=config,
            task_spec=task_spec,
            task_probabilities=task_probabilities,
            batch_cache=batch_cache,
            local_batch_size=local_batch_size,
            device=device,
            accelerator=accelerator,
            fixed_microbatches=fixed_microbatches,
            return_task_counts=True,
        )
        train_losses.append(train_loss)
        _add_task_sample_counts(cumulative_task_samples, task_spec.codes, task_sample_counts)
        samples.append(int(config.batch_size))

        if accelerator.is_main_process and (
            step == start_step + 1 or step % progress_interval == 0
        ):
            elapsed_seconds = max(time.monotonic() - run_start_time, 1e-9)
            log_event(
                "scaling_laws",
                "training_progress",
                step=step,
                total_steps=config.steps,
                train_loss=train_loss,
                learning_rate=current_lr,
                steps_per_second=(step - start_step) / elapsed_seconds,
            )

        if step % evaluation_interval == 0 or step == config.steps:
            with _evaluation_rng(config):
                latest_diagnostics = evaluate_and_log_step(
                    model=model,
                    loss_fn=loss_fn,
                    config=config,
                    task_spec=task_spec,
                    task_probabilities=task_probabilities,
                    node_depths=node_depths,
                    batch_cache=batch_cache,
                    device=device,
                    accelerator=accelerator,
                    step=step,
                    curriculum_threshold_bits=curriculum_threshold_bits,
                    eval_losses=eval_losses,
                    eval_losses_bits=eval_losses_bits,
                    eval_accuracies=eval_accuracies,
                    eval_steps=eval_steps,
                    eval_diagnostics_history=eval_diagnostics_history,
                )
                _append_effective_sample_snapshot(
                    effective_sample_history,
                    task_spec.codes,
                    cumulative_task_samples,
                )

        latest_diagnostics = checkpoint_if_due(
            step=step,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            task_spec=task_spec,
            task_probabilities=task_probabilities,
            node_depths=node_depths,
            batch_cache=batch_cache,
            device=device,
            accelerator=accelerator,
            latest_diagnostics=latest_diagnostics,
            save_kwargs=training_save_kwargs(
                save_dir=save_dir,
                train_losses=train_losses,
                eval_losses=eval_losses,
                eval_losses_bits=eval_losses_bits,
                eval_accuracies=eval_accuracies,
                eval_steps=eval_steps,
                eval_diagnostics_history=eval_diagnostics_history,
                samples=samples,
                effective_samples=effective_sample_history,
                curriculum_threshold_bits=curriculum_threshold_bits,
                resume_state=resume_state,
            ),
        )

    results = save_final_results(
        model=model,
        optimizer=optimizer,
        config=config,
        task_spec=task_spec,
        task_probabilities=task_probabilities,
        node_depths=node_depths,
        latest_diagnostics=latest_diagnostics,
        accelerator=accelerator,
        run_start_time=run_start_time,
        save_kwargs=training_save_kwargs(
            save_dir=save_dir,
            train_losses=train_losses,
            eval_losses=eval_losses,
            eval_losses_bits=eval_losses_bits,
            eval_accuracies=eval_accuracies,
            eval_steps=eval_steps,
            eval_diagnostics_history=eval_diagnostics_history,
            samples=samples,
            effective_samples=effective_sample_history,
            curriculum_threshold_bits=curriculum_threshold_bits,
            resume_state=resume_state,
        ),
    )
    results = broadcast_object(accelerator, _jsonable(results) if accelerator.is_main_process else None)
    accelerator.wait_for_everyone()
    return results


@contextmanager
def _evaluation_rng(config: ScalingLawsConfig):
    """Isolate evaluation RNG and reuse a fixed held-out panel where available."""
    if config.task not in {"cnand", "multitask_sparse_parity"}:
        yield
        return
    with ForkRNG():
        if config.eval_seed is not None:
            set_seeds(int(config.eval_seed))
        yield


def _effective_sample_totals_from_history(
    codes: list[int],
    effective_sample_history: dict[int, list[int]],
) -> dict[int, int]:
    return {
        int(code): (
            int(effective_sample_history.get(int(code), [0])[-1])
            if effective_sample_history.get(int(code))
            else 0
        )
        for code in codes
    }


def _add_task_sample_counts(
    totals: dict[int, int],
    codes: list[int],
    counts: list[int],
) -> None:
    for code, count in zip(codes, counts):
        totals[int(code)] = int(totals.get(int(code), 0)) + int(count)


def _append_effective_sample_snapshot(
    history: dict[int, list[int]],
    codes: list[int],
    totals: dict[int, int],
) -> None:
    for code in codes:
        history.setdefault(int(code), []).append(int(totals.get(int(code), 0)))
