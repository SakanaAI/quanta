from __future__ import annotations

import json
import logging
import os
import pickle
import re
from dataclasses import asdict
from typing import Any

from accelerate import Accelerator

from quanta.config import ScalingLawsConfig
from quanta.experiments.common import log_event

from .distributed import broadcast_object
from quanta.utils import _jsonable, _slug_number


def scaling_run_save_dir(
    base_save_dir: str,
    pair_index: int,
    rho: float,
    beta: float,
    seed: int,
    width: int,
    lr: float,
    config: ScalingLawsConfig,
    run_config: ScalingLawsConfig | None = None,
) -> str:
    pair_dir = os.path.join(base_save_dir, "runs", f"pair{pair_index}")
    variant = _run_variant_slug(run_config or config, width)
    if _uses_lr_subdirectories(config):
        return os.path.join(
            pair_dir,
            f"seed{seed}-{variant}-lr{_slug_number(lr)}",
        )
    return os.path.join(
        pair_dir,
        f"seed{seed}-{variant}",
    )


def _find_existing_or_resume_state(
    *,
    base_save_dir: str,
    pair_index: int,
    rho: float,
    beta: float,
    seed: int,
    width: int,
    lr_val: float,
    config: ScalingLawsConfig,
    run_config: ScalingLawsConfig,
    accelerator: Accelerator,
) -> dict[str, Any] | None:
    state = None
    if accelerator.is_main_process:
        candidates = _resume_candidate_dirs(
            base_save_dir=base_save_dir,
            pair_index=pair_index,
            rho=rho,
            beta=beta,
            seed=seed,
            width=width,
            lr_val=lr_val,
            config=config,
            run_config=run_config,
        )
        state = _select_resume_state(candidates, run_config)
    return broadcast_object(accelerator, state)


def _resume_candidate_dirs(
    *,
    base_save_dir: str,
    pair_index: int,
    rho: float,
    beta: float,
    seed: int,
    width: int,
    lr_val: float,
    config: ScalingLawsConfig,
    run_config: ScalingLawsConfig | None = None,
) -> list[str]:
    candidate_bases = _budget_sibling_base_dirs(base_save_dir, config)
    legacy_base = _legacy_null_mask_base_dir(base_save_dir, config)
    if legacy_base is not None:
        candidate_bases.extend(
            candidate
            for candidate in _budget_sibling_base_dirs(legacy_base, config)
            if os.path.isdir(candidate)
        )
    candidate_bases = list(dict.fromkeys(candidate_bases))
    candidates = [
        scaling_run_save_dir(
            candidate_base,
            pair_index,
            rho,
            beta,
            seed,
            width,
            lr_val,
            config,
            run_config,
        )
        for candidate_base in candidate_bases
    ]
    if _uses_lr_subdirectories(config):
        legacy_candidates = [
            _legacy_single_lr_run_save_dir(
                candidate_base,
                pair_index,
                seed,
                width,
                run_config or config,
            )
            for candidate_base in candidate_bases
        ]
        candidates.extend(path for path in legacy_candidates if os.path.isdir(path))
    return list(dict.fromkeys(candidates))


def _legacy_single_lr_run_save_dir(
    base_save_dir: str,
    pair_index: int,
    seed: int,
    width: int,
    config: ScalingLawsConfig,
) -> str:
    return os.path.join(
        base_save_dir,
        "runs",
        f"pair{pair_index}",
        f"seed{seed}-{_run_variant_slug(config, width)}",
    )


def _legacy_null_mask_base_dir(
    base_save_dir: str,
    config: ScalingLawsConfig,
) -> str | None:
    if (getattr(config, "attention_masking", None) or "none") != "none":
        return None
    normalized = os.path.normpath(base_save_dir)
    parent = os.path.dirname(normalized)
    basename = os.path.basename(normalized)
    if "-maskNone" in basename:
        return None
    suffix_positions = [
        position
        for marker in ("-nsf-", "-eval")
        if (position := basename.find(marker)) >= 0
    ]
    insertion = min(suffix_positions, default=len(basename))
    legacy_basename = basename[:insertion] + "-maskNone" + basename[insertion:]
    return os.path.join(parent, legacy_basename)


def _budget_sibling_base_dirs(
    base_save_dir: str,
    config: ScalingLawsConfig,
) -> list[str]:
    """Return the current run root and sibling roots differing only in step budget."""
    current = os.path.normpath(base_save_dir)
    parent = os.path.dirname(current)
    basename = os.path.basename(current)
    step_token = f"steps{int(config.steps)}"
    if step_token not in basename or not os.path.isdir(parent):
        return [current]

    pattern = re.compile(
        "^"
        + re.escape(basename).replace(
            re.escape(step_token),
            r"steps[0-9]+",
            1,
        )
        + "$"
    )
    siblings = [
        os.path.join(parent, name)
        for name in os.listdir(parent)
        if pattern.fullmatch(name) and os.path.isdir(os.path.join(parent, name))
    ]
    return [current, *sorted(path for path in siblings if os.path.normpath(path) != current)]


def _run_variant_slug(config: ScalingLawsConfig, width: int) -> str:
    if config.architecture == "transformer":
        return f"width{int(width)}-depth{int(config.depth)}-heads{int(config.n_heads)}"
    return f"width{int(width)}-depth{int(config.depth)}"


def _list_values(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def _uses_lr_subdirectories(config: ScalingLawsConfig) -> bool:
    widths = {int(value) for value in _list_values(config.width)}
    learning_rates = {float(value) for value in _list_values(config.lr)}
    return len(widths) == 1 or len(learning_rates) > 1


def _select_resume_state(candidate_dirs: list[str], run_config: ScalingLawsConfig) -> dict[str, Any] | None:
    current_config = _jsonable(asdict(run_config))
    best_resume = None
    for candidate_dir in candidate_dirs:
        config_json_path = os.path.join(candidate_dir, "config.json")
        results_pkl_path = os.path.join(candidate_dir, "results.pkl")
        model_path = os.path.join(candidate_dir, "model.pt")
        if not (os.path.exists(config_json_path) and os.path.exists(results_pkl_path)):
            continue
        try:
            with open(config_json_path, "r") as handle:
                saved_config = json.load(handle)
            if not _configs_match_for_resume(saved_config, current_config):
                continue
            with open(results_pkl_path, "rb") as handle:
                result = pickle.load(handle)
        except Exception as error:
            logging.warning("Error reading existing run at %s; ignoring for resume. Error: %r", candidate_dir, error)
            continue

        saved_steps = int(saved_config.get("steps", result.get("steps_run", 0)))
        steps_run = int(result.get("steps_run", 0))
        current_steps = int(current_config["steps"])
        if steps_run >= current_steps:
            log_event("scaling_laws", "run_reused", run_dir=candidate_dir, steps=steps_run)
            return {"mode": "complete", "run_dir": candidate_dir, "results": _jsonable(result)}
        scheduler = current_config.get("scheduler", "constant")
        budget_extension_is_valid = saved_steps == current_steps or scheduler == "constant"
        if (
            saved_steps <= current_steps
            and steps_run < current_steps
            and budget_extension_is_valid
            and os.path.exists(model_path)
        ):
            if best_resume is None or steps_run > int(best_resume["steps_run"]):
                best_resume = {
                    "mode": "resume",
                    "run_dir": candidate_dir,
                    "model_path": model_path,
                    "optimizer_path": os.path.join(candidate_dir, "optimizer.pt"),
                    "results": _jsonable(result),
                    "steps_run": steps_run,
                    "saved_steps": saved_steps,
                }
        elif saved_steps != current_steps and scheduler != "constant":
            logging.debug(
                "Ignoring resume candidate %s because extending a %s schedule from %s to %s "
                "steps would change its prior learning-rate trajectory.",
                candidate_dir,
                scheduler,
                saved_steps,
                current_steps,
            )

    if best_resume is not None:
        log_event(
            "scaling_laws",
            "resuming",
            run_dir=best_resume["run_dir"],
            step=best_resume["steps_run"],
            saved_steps=best_resume["saved_steps"],
            total_steps=current_config["steps"],
        )
    return best_resume


def _configs_match_for_resume(saved_config: dict[str, Any], current_config: dict[str, Any]) -> bool:
    keys_to_compare = [
        "depth",
        "width",
        "seed",
        "lr",
        "weight_decay",
        "scheduler",
        "warmup_phase",
        "plateau_phase",
        "batch_size",
        "gradient_accumulation_steps",
        "eval_steps",
        "eval_samples_per_task",
        "eval_seed",
        "eval_loss_formula",
        "loss_supervision",
        "trace_sampling",
        "mixed_precision",
        "dynamic",
        "layernorm",
        "task",
        "n_atomic_task_bits",
        "n_local_bits",
        "n_noise_bits",
        "task_bits",
        "base_tasks",
        "base_freq",
        "max_depth",
        "m",
        "graph_family",
        "target_alpha",
        "rho",
        "beta",
        "delta",
        "quanta_demand",
        "quanta_demand_fit_tolerance",
        "activation",
        "architecture",
        "n_heads",
        "mlp_ratio",
        "dropout",
        "transformer_activation",
        "attention",
        "readout",
        "attention_masking",
        "n_lut_functions",
        "lut_family",
        "function_seed",
        "exhaustive_local_eval",
        "run_tag",
    ]
    defaults = {
        "scheduler": "constant",
        "gradient_accumulation_steps": 1,
        "eval_seed": None,
        "eval_loss_formula": "quanta_weighted",
        "loss_supervision": "all",
        "trace_sampling": "principal",
        "warmup_phase": 0.1,
        "plateau_phase": 0.0,
        "task_bits": [-1, 1],
        "activation": "relu",
        "architecture": "mlp",
        "n_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "transformer_activation": "gelu",
        "attention": "full_bidirectional",
        "readout": "query_token",
        "attention_masking": "none",
        "n_lut_functions": None,
        "lut_family": "random_balanced",
        "function_seed": None,
        "exhaustive_local_eval": False,
        "graph_family": "exponential_random",
        "target_alpha": None,
    }
    def comparable_value(config: dict[str, Any], key: str):
        value = config.get(key, defaults.get(key))
        if key == "attention_masking":
            return value or "none"
        return value

    return all(
        comparable_value(saved_config, key) == comparable_value(current_config, key)
        for key in keys_to_compare
    )
