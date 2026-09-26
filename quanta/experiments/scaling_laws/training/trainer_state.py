from __future__ import annotations

import json
import logging
import os
import pickle
import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator

from quanta.config import ScalingLawsConfig
from quanta.experiments.common import log_event
from quanta.experiments.scaling_laws.batch_evaluation import evaluate_weighted_task_loss
from quanta.experiments.scaling_laws.metrics import (
    TAIL_MEDIAN_EVAL_POINTS,
    tail_median,
    trainable_parameter_counts,
)
from quanta.utils import _jsonable


def save_process_rng_state(
    save_dir: str,
    *,
    process_index: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    torch.save(state, _rng_state_path(save_dir, process_index))


def restore_process_rng_state(
    save_dir: str,
    *,
    process_index: int,
    device,
) -> bool:
    path = _rng_state_path(save_dir, process_index)
    if not os.path.exists(path):
        logging.warning(
            "Resume RNG state missing for process=%s at %s; continuation will be deterministic "
            "but may not match an uninterrupted run.",
            process_index,
            path,
        )
        return False
    # RNG APIs consume CPU ByteTensors even when they restore CUDA generators.
    # Loading this checkpoint onto the training device turns the saved CUDA
    # states into CUDA tensors, which torch.cuda.set_rng_state_all rejects.
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])
    if state.get("mps") is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    return True


def _rng_state_path(save_dir: str, process_index: int) -> str:
    return os.path.join(save_dir, f"rng_state_rank{int(process_index)}.pt")


def _should_save_checkpoint(config: ScalingLawsConfig, step: int) -> bool:
    save_steps = getattr(config, "save_steps", None)
    if save_steps is None:
        return False
    return int(step) > 0 and int(step) % int(save_steps) == 0


def _checkpoint_diagnostics(
    *,
    model: nn.Module,
    loss_fn,
    config: ScalingLawsConfig,
    task_spec,
    probabilities: torch.Tensor,
    node_depths: dict[int, int],
    batch_cache: dict[str, torch.Tensor],
    device,
    accelerator: Accelerator,
    latest_diagnostics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    diagnostics = latest_diagnostics
    if diagnostics is None:
        diagnostics = evaluate_weighted_task_loss(
            model=model,
            loss_fn=loss_fn,
            config=config,
            task_spec=task_spec,
            probabilities=probabilities,
            node_depths=node_depths,
            batch_cache=batch_cache,
            device=device,
            accelerator=accelerator,
        )
    return diagnostics


def _save_training_state(
    *,
    model: nn.Module,
    optimizer,
    config: ScalingLawsConfig,
    save_dir: str,
    train_losses: list[float],
    eval_losses: list[float],
    eval_losses_bits: list[float],
    eval_accuracies: list[float],
    eval_steps: list[int],
    eval_diagnostics_history: list[dict[str, Any]],
    samples: list[int],
    effective_samples: dict[int, list[int]],
    task_spec,
    task_probabilities: torch.Tensor,
    node_depths: dict[int, int],
    diagnostics: dict[str, Any] | None,
    n_parameters: int | None,
    curriculum_threshold_bits: float,
    resume_state: dict[str, Any] | None,
    checkpoint: bool,
) -> dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    parameter_counts = trainable_parameter_counts(model)
    if n_parameters is not None and int(n_parameters) != parameter_counts["n_parameters"]:
        raise ValueError(
            "Saved trainable parameter count does not match the current model: "
            f"{n_parameters} != {parameter_counts['n_parameters']}."
        )
    final_eval_loss_nats = float(eval_losses[-1]) if eval_losses else (
        float(diagnostics["eval_loss_nats"]) if diagnostics is not None else float("nan")
    )
    final_eval_loss_bits = float(eval_losses_bits[-1]) if eval_losses_bits else (
        float(diagnostics["eval_loss_bits"]) if diagnostics is not None else float("nan")
    )
    steps_run = int(len(train_losses))
    subtask_losses = [
        [
            float(diagnostics["task_losses"][int(code)]["loss_nats"])
            for diagnostics in eval_diagnostics_history
        ]
        for code in task_spec.codes
    ]
    quantum_subtask_losses = [
        [
            float(
                diagnostics.get("quantum_losses", {})
                .get(int(code), {})
                .get("loss_nats", float("nan"))
            )
            for diagnostics in eval_diagnostics_history
        ]
        for code in task_spec.codes
    ]
    mean_quantum_subtask_losses = [
        [
            float(
                diagnostics.get("quantum_mean_losses", {})
                .get(int(code), {})
                .get("loss_nats", float("nan"))
            )
            for diagnostics in eval_diagnostics_history
        ]
        for code in task_spec.codes
    ]
    mean_task_losses_bits = [
        float(diagnostics["mean_task_loss_bits"])
        for diagnostics in eval_diagnostics_history
    ]
    tail_points = min(TAIL_MEDIAN_EVAL_POINTS, len(eval_steps))
    evaluation_samples = [
        int(config.batch_size) * max(
            0,
            int(step) - int(eval_steps[index - 1] if index else 0),
        )
        for index, step in enumerate(eval_steps)
    ]
    results = {
        "experiment": "scaling_laws_run",
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "eval_losses_bits": eval_losses_bits,
        "mean_task_losses_bits": mean_task_losses_bits,
        "eval_accuracies": eval_accuracies,
        "eval_steps": eval_steps,
        "eval_diagnostics_history": eval_diagnostics_history,
        "samples": evaluation_samples,
        "training_samples": samples,
        "effective_samples": {
            int(code): [int(value) for value in effective_samples.get(int(code), [])]
            for code in task_spec.codes
        },
        "subtask_losses": subtask_losses,
        "quantum_subtask_losses": quantum_subtask_losses,
        "mean_quantum_subtask_losses": mean_quantum_subtask_losses,
        "subtask_train_losses": [],
        "alternative_decomposition": config.alternative_decomposition,
        "alternative_decomposition_losses": [],
        "train_batch_sampling": (
            "threshold_ideal_mixture"
            if config.trace_sampling == "ideal_threshold"
            else (
                "independent_compact_ideal_paths"
                if config.trace_sampling == "ideal_path"
                else "categorical_task_frequency"
            )
        ),
        "trace_sampling": config.trace_sampling,
        "eval_loss_formula": config.eval_loss_formula,
        "loss_supervision": config.loss_supervision,
        "curriculum_threshold_bits": float(curriculum_threshold_bits),
        "codes": task_spec.codes,
        "Ss": task_spec.Ss_atomic,
        "graph_dependencies": task_spec.graph_dependencies,
        "task_frequencies": config.task_frequencies,
        "quanta_demand": config.quanta_demand,
        "quanta_demand_diagnostics": config.quanta_demand_diagnostics,
        "actual_tail_alpha": (
            config.quanta_demand_diagnostics.get("actual_tail_alpha")
            if config.quanta_demand_diagnostics
            else None
        ),
        "theoretical_alpha": (
            config.quanta_demand_diagnostics.get("theoretical_alpha")
            if config.quanta_demand_diagnostics
            else None
        ),
        "theory_alpha_source": (
            config.quanta_demand_diagnostics.get("theory_alpha_source")
            if config.quanta_demand_diagnostics
            else None
        ),
        "task_probabilities": {
            int(code): float(probability)
            for code, probability in zip(task_spec.codes, task_probabilities.detach().cpu().tolist())
        },
        "node_depths": {int(node): int(depth) for node, depth in node_depths.items()},
        "eval_diagnostics": diagnostics,
        **parameter_counts,
        "final_eval_loss_nats": final_eval_loss_nats,
        "final_eval_loss_bits": final_eval_loss_bits,
        "final_mean_task_loss_bits": (
            float(mean_task_losses_bits[-1])
            if mean_task_losses_bits
            else float("nan")
        ),
        "tail_median_eval_loss_bits": tail_median(eval_losses_bits),
        "tail_median_mean_task_loss_bits": tail_median(mean_task_losses_bits),
        "tail_median_eval_points": int(tail_points),
        "tail_median_start_step": (
            int(eval_steps[-tail_points]) if tail_points else None
        ),
        "tail_median_end_step": int(eval_steps[-1]) if tail_points else None,
        "steps_run": steps_run,
        "target_steps": int(config.steps),
        "checkpoint": bool(checkpoint),
        "resumed_from": resume_state["run_dir"] if resume_state is not None else None,
    }
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    with open(os.path.join(save_dir, "results.pkl"), "wb") as handle:
        pickle.dump(results, handle)

    config_dict = _jsonable(asdict(config))
    config_dict["model"] = model.__class__.__name__
    with open(os.path.join(save_dir, "config.json"), "w") as handle:
        json.dump(config_dict, handle, indent=4)
    if checkpoint:
        log_event(
            "scaling_laws",
            "checkpoint",
            step=steps_run,
            total_steps=config.steps,
            save_dir=save_dir,
        )
    return results
