from __future__ import annotations

import copy
import json
import os
import pickle
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator

from quanta.config import ScalingLawsConfig
from quanta.experiments.common import log_event
from quanta.experiments.demand import select_theoretical_alpha
from quanta.experiments.task_specs import TaskSpecBuilder
from quanta.utils import set_seeds
from .batches import build_seeded_cnand_batch_cache
from .batches import masked_token_cross_entropy, uses_masked_token_supervision
from .model import CNANDTransformerModel, MultitaskSparseParityMLP
from .metrics import tail_median, trainable_parameter_counts
from quanta.tools.masked_cnand import MaskedCNANDTransformer
from .training.trainer import train_until_budget_or_convergence

from .distributed import broadcast_object
from .resume import _find_existing_or_resume_state, scaling_run_save_dir
from quanta.utils import _jsonable, theoretical_alpha


def run_scaling_job(
    config: ScalingLawsConfig,
    base_save_dir: str,
    job: dict[str, Any],
    *,
    accelerator: Accelerator,
) -> dict[str, Any]:
    pair_index = int(job["pair_index"])
    rho = float(job["rho"])
    beta = float(job["beta"])
    delta = float(job.get("delta", 0.0))
    seed = int(job["seed"])
    width = int(job["width"])
    lr_val = float(job["lr"])
    graph = job["graph"]
    run_config = copy.deepcopy(config)
    run_config.width = width
    run_config.seed = seed
    run_config.lr = lr_val
    run_config.rho = [rho]
    run_config.beta = [beta]
    run_config.delta = [delta]
    for key, value in job.get("overrides", {}).items():
        setattr(run_config, key, value)

    run_config.graph_dependencies = graph["graph_dependencies"]
    run_config.task_frequencies = graph["task_frequencies"]
    run_config.quanta_demand_diagnostics = dict(graph["quanta_demand"])
    run_theoretical_alpha, theory_alpha_source = select_theoretical_alpha(
        demand=run_config.quanta_demand_diagnostics,
        rho=rho,
        beta=beta,
        delta=delta,
    )
    run_config.quanta_demand_diagnostics.update(
        {
            "theoretical_alpha": float(run_theoretical_alpha),
            "theory_alpha_source": theory_alpha_source,
        }
    )
    run_config.save_dir = scaling_run_save_dir(base_save_dir, pair_index, rho, beta, seed, width, lr_val, config, run_config)
    task_spec = TaskSpecBuilder().build(run_config)
    run_config.n_tasks = task_spec.n_tasks

    resume_state = _find_existing_or_resume_state(
        base_save_dir=base_save_dir,
        pair_index=pair_index,
        rho=rho,
        beta=beta,
        seed=seed,
        width=width,
        lr_val=lr_val,
        config=config,
        run_config=run_config,
        accelerator=accelerator,
    )
    existing_record = None
    if resume_state is not None and resume_state["mode"] == "complete":
        result = resume_state["results"]
        existing_record = {
            "pair_index": pair_index,
            "rho": rho,
            "beta": beta,
            "delta": delta,
            "eval_loss_formula": run_config.eval_loss_formula,
            "loss_supervision": run_config.loss_supervision,
            "trace_sampling": run_config.trace_sampling,
            **_demand_record(graph, rho, beta, delta),
            "seed": seed,
            "width": width,
            "effective_width": _effective_width(run_config, width),
            "lr": lr_val,
            "depth": int(run_config.depth),
            "architecture": run_config.architecture,
            "n_heads": int(run_config.n_heads),
            "n_tasks": int(run_config.n_tasks),
            "n_bits": int(run_config.n_bits),
            "n_parameters": int(result["n_parameters"]),
            "n_embedding_parameters": int(result.get("n_embedding_parameters", 0)),
            "n_non_embedding_parameters": int(
                result.get("n_non_embedding_parameters", result["n_parameters"])
            ),
            "model_scale": _model_scale_key(run_config, width, int(result["n_parameters"])),
            "final_eval_loss_nats": float(result["final_eval_loss_nats"]),
            "final_eval_loss_bits": float(result["final_eval_loss_bits"]),
            "final_mean_task_loss_bits": _final_mean_task_loss_bits(result),
            "tail_median_eval_loss_bits": _tail_median_eval_loss_bits(result),
            "tail_median_mean_task_loss_bits": _tail_median_mean_task_loss_bits(result),
            "tail_median_eval_points": int(result.get("tail_median_eval_points", 0)),
            "tail_median_start_step": result.get("tail_median_start_step"),
            "tail_median_end_step": result.get("tail_median_end_step"),
            "steps_run": int(result["steps_run"]),
            "save_dir": run_config.save_dir,
            "graph": graph,
        }
    existing_record = broadcast_object(accelerator, _jsonable(existing_record) if existing_record is not None else None)
    if existing_record is not None:
        return existing_record

    set_seeds(seed)
    batch_cache = build_seeded_cnand_batch_cache(run_config, task_spec, "cpu")
    if run_config.task == "cnand":
        set_seeds(seed)
    if run_config.task == "cnand":
        attention_masking = getattr(run_config, "attention_masking", None) or "none"
        run_config.attention_masking = attention_masking
        if attention_masking == "none":
            model = CNANDTransformerModel(config=run_config, n_slots=int(batch_cache["slot_ids"].shape[0]))
        else:
            model = MaskedCNANDTransformer(
                config=run_config,
                n_slots=int(batch_cache["slot_ids"].shape[0]),
                tokens_per_node=int(batch_cache["tokens_per_node"].item()),
                mask_mode=attention_masking,
                parent_indices=batch_cache["parent_indices"],
                parent_masks=batch_cache["parent_masks"],
            )
        input_dim = model.input_dim
    elif run_config.task == "multitask_sparse_parity":
        model = MultitaskSparseParityMLP(config=run_config)
        input_dim = model.input_dim

    parameter_counts = trainable_parameter_counts(model)
    if accelerator.is_main_process:
        demand = graph["quanta_demand"]
        log_event(
            "scaling_laws",
            "job_start",
            pair=pair_index,
            rho=rho,
            beta=beta,
            seed=seed,
            architecture=run_config.architecture,
            depth=run_config.depth,
            width=width,
            parameters=parameter_counts["n_parameters"],
            non_embedding_parameters=parameter_counts["n_non_embedding_parameters"],
            learning_rate=lr_val,
            steps=run_config.steps,
            batch_size=run_config.batch_size,
            gradient_accumulation_steps=run_config.gradient_accumulation_steps,
            microbatch_size=(
                int(run_config.batch_size) // int(run_config.gradient_accumulation_steps)
            ),
            demand=demand["mode"],
            trace_sampling=run_config.trace_sampling,
            demand_fit_rmse=demand["relative_rmse"],
            device=accelerator.device,
            processes=accelerator.num_processes,
        )

    loss_fn = masked_token_mean_loss if uses_masked_token_supervision(run_config) else nn.CrossEntropyLoss()
    result = train_until_budget_or_convergence(
        model=model,
        loss_fn=loss_fn,
        config=run_config,
        task_spec=task_spec,
        save_dir=run_config.save_dir,
        node_depths=graph["node_depths"],
        accelerator=accelerator,
        resume_state=resume_state if resume_state is not None and resume_state["mode"] == "resume" else None,
    )
    record = {
        "pair_index": pair_index,
        "rho": rho,
        "beta": beta,
        "delta": delta,
        "eval_loss_formula": run_config.eval_loss_formula,
        "loss_supervision": run_config.loss_supervision,
        "trace_sampling": run_config.trace_sampling,
        **_demand_record(graph, rho, beta, delta),
        "seed": seed,
        "width": width,
        "effective_width": _effective_width(run_config, width),
        "lr": lr_val,
        "depth": int(run_config.depth),
        "architecture": run_config.architecture,
        "n_heads": int(run_config.n_heads),
        "n_tasks": int(run_config.n_tasks),
        "n_bits": int(run_config.n_bits),
        "n_parameters": int(result["n_parameters"]),
        "n_embedding_parameters": int(result.get("n_embedding_parameters", 0)),
        "n_non_embedding_parameters": int(
            result.get("n_non_embedding_parameters", result["n_parameters"])
        ),
        "model_scale": _model_scale_key(run_config, width, int(result["n_parameters"])),
        "final_eval_loss_nats": float(result["final_eval_loss_nats"]),
        "final_eval_loss_bits": float(result["final_eval_loss_bits"]),
        "final_mean_task_loss_bits": _final_mean_task_loss_bits(result),
        "tail_median_eval_loss_bits": _tail_median_eval_loss_bits(result),
        "tail_median_mean_task_loss_bits": _tail_median_mean_task_loss_bits(result),
        "tail_median_eval_points": int(result.get("tail_median_eval_points", 0)),
        "tail_median_start_step": result.get("tail_median_start_step"),
        "tail_median_end_step": result.get("tail_median_end_step"),
        "steps_run": int(result["steps_run"]),
        "save_dir": run_config.save_dir,
        "graph": graph,
    }
    return _jsonable(record)


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


def _tail_median_eval_loss_bits(result: dict[str, Any]) -> float:
    if result.get("tail_median_eval_loss_bits") is not None:
        return float(result["tail_median_eval_loss_bits"])
    value = tail_median(result.get("eval_losses_bits"))
    if torch.isfinite(torch.tensor(value)):
        return value
    return float(result["final_eval_loss_bits"])


def _tail_median_mean_task_loss_bits(result: dict[str, Any]) -> float:
    if result.get("tail_median_mean_task_loss_bits") is not None:
        return float(result["tail_median_mean_task_loss_bits"])
    value = tail_median(result.get("mean_task_losses_bits"))
    if torch.isfinite(torch.tensor(value)):
        return value
    return _final_mean_task_loss_bits(result)


def _demand_record(
    graph: dict[str, Any],
    rho: float,
    beta: float,
    delta: float,
) -> dict[str, Any]:
    demand = graph["quanta_demand"]
    comparison_beta = float(demand["comparison_beta"])
    selected_alpha, theory_alpha_source = select_theoretical_alpha(
        demand=demand,
        rho=rho,
        beta=beta,
        delta=delta,
    )
    return {
        "quanta_demand": demand["mode"],
        "desired_beta": float(beta),
        "actual_induced_beta": float(demand["actual_induced_beta"]),
        "actual_tail_alpha": float(demand["actual_tail_alpha"]),
        "comparison_beta": comparison_beta,
        "quanta_demand_relative_rmse": float(demand["relative_rmse"]),
        "quanta_demand_close_fit": bool(demand["close_fit"]),
        "desired_theoretical_alpha": theoretical_alpha(rho, beta, delta),
        "theoretical_alpha": float(selected_alpha),
        "theory_alpha_source": theory_alpha_source,
        "quanta_demand_diagnostics": demand,
    }


def masked_token_mean_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return masked_token_cross_entropy(logits, batch)


def _model_scale_key(config: ScalingLawsConfig, width: int, n_parameters: int) -> str:
    if config.architecture == "transformer":
        return f"width{int(width)}-depth{int(config.depth)}-heads{int(config.n_heads)}-params{int(n_parameters)}"
    return f"width{int(width)}-depth{int(config.depth)}"


def _effective_width(config: ScalingLawsConfig, width: int) -> int:
    return int(width)
