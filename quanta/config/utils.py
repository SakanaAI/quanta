from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from quanta.config.types import TrainingConfig


def _validate_task_name(config: TrainingConfig, path: str | Path) -> None:
    config.task = config.task.lower()
    if config.task not in {"cnand", "multitask_sparse_parity"}:
        raise ValueError(f"Unsupported task in {path}: {config.task!r}")


def _validate_discreteness_transition_error(config: TrainingConfig) -> None:
    if not isinstance(config.discreteness_transition_error, bool):
        raise ValueError("discreteness_transition_error must be true or false.")


def resolve_steps_and_epochs(config: TrainingConfig, dataset_size: int | None) -> TrainingConfig:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.steps is not None and config.epochs is not None:
        raise ValueError("Set either steps or epochs in a training config, not both.")

    if config.dynamic:
        if config.epochs is not None:
            raise ValueError("epochs is only supported when dynamic is false; use steps for dynamic training.")
        if config.steps is None:
            config.steps = 200000
        if config.steps <= 0:
            raise ValueError("steps must be positive for dynamic training.")
        return config

    if config.offline_dataset_size is None and config.samples_per_task is None:
        raise ValueError("offline_dataset_size or samples_per_task is required when dynamic is false.")
    if config.offline_dataset_size is not None and config.offline_dataset_size <= 0:
        raise ValueError("offline_dataset_size must be positive when dynamic is false.")
    if config.samples_per_task is not None and config.samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive when dynamic is false.")
    if dataset_size is None or dataset_size <= 0:
        raise ValueError("Cannot resolve steps/epochs for an empty fixed training dataset.")
    if config.steps is None and config.epochs is None:
        config.steps = 200000

    if config.steps is None:
        if config.epochs is None or config.epochs <= 0:
            raise ValueError("epochs must be positive when steps is omitted.")
        config.steps = int(math.ceil(config.epochs * dataset_size / config.batch_size))
    else:
        if config.steps <= 0:
            raise ValueError("steps must be positive when epochs is omitted.")
        config.epochs = config.steps * config.batch_size / dataset_size
    return config


def normalize_graph_dependencies(raw: Any) -> dict[int, list[int]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("graph_dependencies must be a mapping of node -> parent list.")
    return {int(node): [int(parent) for parent in parents] for node, parents in raw.items()}


def normalize_nested_int_lists(raw: Any, field_name: str) -> list[list[int]]:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of integer lists.")
    normalized = []
    for item in raw:
        if not isinstance(item, list):
            raise ValueError(f"{field_name} must be a list of integer lists.")
        normalized.append([int(value) for value in item])
    return normalized


def normalize_int_list(raw: Any, field_name: str) -> list[int]:
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of integers.")
    return [int(value) for value in raw]


def normalize_float_list(raw: Any, field_name: str) -> list[float]:
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of numbers.")
    return [float(value) for value in raw]


def normalize_str_list(raw: Any, field_name: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    return [str(value) for value in raw]


def normalize_int_float_mapping(raw: Any, field_name: str) -> dict[int, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a mapping of task node -> frequency.")
    normalized = {int(key): float(value) for key, value in raw.items()}
    if any(value < 0 for value in normalized.values()) or sum(normalized.values()) <= 0:
        raise ValueError(f"{field_name} must contain non-negative frequencies with positive total weight.")
    return normalized


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _resolve_source_config_path(analysis_path: str | Path, source_path: str | Path) -> Path:
    source_path = Path(source_path)
    if source_path.exists():
        return source_path
    analysis_relative = Path(analysis_path).parent / source_path
    if analysis_relative.exists():
        return analysis_relative
    experiment_relative = (
        Path("configs") / "scaling_laws" / source_path
    )
    if experiment_relative.suffix != ".yaml":
        experiment_relative = experiment_relative.with_suffix(".yaml")
    if experiment_relative.exists():
        return experiment_relative
    raise FileNotFoundError(f"Source scaling laws config not found: {source_path}")


def _normalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def _flatten_training_data(data: dict[str, Any], extra_sections: set[str] | None = None) -> dict[str, Any]:
    data = _normalize_keys(data)
    sections = {"model", "task", "training"} | (extra_sections or set())
    flattened = {
        key: value
        for key, value in data.items()
        if key not in sections
    }

    if "model" in data:
        _merge_section(flattened, "model", data["model"], key_aliases={})
    if isinstance(data.get("task"), dict):
        _merge_section(flattened, "task", data["task"], key_aliases={"name": "task"})
    elif "task" in data:
        _set_unique(flattened, "task", data["task"], "task")
    if "training" in data:
        _merge_section(flattened, "training", data["training"], key_aliases={})
    for section_name in sorted(extra_sections or set()):
        if section_name in data:
            _merge_section(flattened, section_name, data[section_name], key_aliases={})

    return flattened


def _merge_section(
    flattened: dict[str, Any],
    section_name: str,
    section: Any,
    key_aliases: dict[str, str],
) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{section_name} must be a YAML mapping.")
    for raw_key, value in _normalize_keys(section).items():
        key = key_aliases.get(raw_key, raw_key)
        _set_unique(flattened, key, value, f"{section_name}.{raw_key}")


def _set_unique(flattened: dict[str, Any], key: str, value: Any, source: str) -> None:
    if key in flattened:
        raise ValueError(f"Duplicate training config value for {key!r} from {source}.")
    flattened[key] = value
