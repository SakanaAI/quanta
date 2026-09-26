from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

from quanta.config import ScalingLawsConfig

from .graph import generate_layered_poset


SWEEP_FIELDS = (
    "rho",
    "beta",
    "delta",
    "width",
    "lr",
    "depth",
    "batch_size",
    "weight_decay",
    "scheduler",
    "warmup_phase",
    "plateau_phase",
    "activation",
    "layernorm",
    "architecture",
    "n_heads",
    "mlp_ratio",
    "dropout",
    "max_depth",
    "base_tasks",
    "base_freq",
)

MLP_ONLY_FIELDS = {
    "activation",
    "layernorm",
}

TRANSFORMER_ONLY_FIELDS = {
    "n_heads",
    "mlp_ratio",
    "dropout",
}

GRAPH_CACHE_VERSION = 6


def build_scaling_run_jobs(
    config: ScalingLawsConfig,
    *,
    graph_cache_dir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    rows = _sweep_rows(config)
    jobs = []
    pair_indices: dict[tuple[float, float, float], int] = {}
    graph_cache: dict[tuple, dict[str, Any]] = {}
    seeds = config.seed if isinstance(config.seed, list) else [config.seed]
    # Exhaust all widths for one seed before advancing to the next. Widths and
    # learning rates remain paired by row rather than forming a Cartesian product.
    for seed in seeds:
        for row in rows:
            rho = float(row["rho"])
            beta = float(row["beta"])
            delta = float(row["delta"])
            pair_key = (rho, beta, delta)
            if pair_key not in pair_indices:
                pair_indices[pair_key] = len(pair_indices)
            pair_index = pair_indices[pair_key]
            width = int(row["width"])
            lr_val = float(row["lr"])
            graph_key = (
                rho,
                beta,
                delta,
                int(row["base_tasks"]),
                float(row["base_freq"]),
                int(row["max_depth"]),
                int(config.m),
                int(seed),
                config.quanta_demand,
                float(config.quanta_demand_fit_tolerance),
                config.trace_sampling,
                config.graph_family,
                config.flat_frequency_exponent,
                config.target_alpha,
            )
            if graph_key not in graph_cache:
                graph_inputs = {
                    "rho": rho,
                    "beta": beta,
                    "delta": delta,
                    "base_tasks": int(row["base_tasks"]),
                    "base_freq": float(row["base_freq"]),
                    "max_depth": int(row["max_depth"]),
                    "m": int(config.m),
                    "seed": int(seed),
                    "quanta_demand": config.quanta_demand,
                    "quanta_demand_fit_tolerance": float(config.quanta_demand_fit_tolerance),
                    "trace_sampling": config.trace_sampling,
                    "graph_family": config.graph_family,
                    "flat_frequency_exponent": config.flat_frequency_exponent,
                    "target_alpha": config.target_alpha,
                }
                graph_cache[graph_key] = _load_or_generate_graph(
                    graph_inputs,
                    graph_cache_dir=graph_cache_dir,
                )
            graph = graph_cache[graph_key]
            jobs.append(
                {
                    "pair_index": int(pair_index),
                    "rho": rho,
                    "beta": beta,
                    "delta": delta,
                    "seed": seed,
                    "width": width,
                    "lr": lr_val,
                    "graph": graph,
                    "overrides": row["overrides"],
                }
            )
    return jobs


def _load_or_generate_graph(
    graph_inputs: dict[str, Any],
    *,
    graph_cache_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    if graph_cache_dir is None:
        return generate_layered_poset(**graph_inputs)

    cache_path = _graph_cache_path(graph_cache_dir, graph_inputs)
    cached = _read_graph_cache(cache_path, graph_inputs)
    if cached is not None:
        logging.debug("graph_cache=hit path=%s", cache_path)
        return cached

    logging.debug("graph_cache=miss path=%s", cache_path)
    graph = generate_layered_poset(**graph_inputs)
    _write_graph_cache(cache_path, graph_inputs, graph)
    return graph


def _graph_cache_path(
    graph_cache_dir: str | os.PathLike[str],
    graph_inputs: dict[str, Any],
) -> Path:
    payload = _graph_cache_metadata(graph_inputs)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return Path(graph_cache_dir) / f"seed{int(graph_inputs['seed'])}-{digest}.pkl"


def _graph_cache_metadata(graph_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": GRAPH_CACHE_VERSION,
        "graph_inputs": graph_inputs,
    }


def _read_graph_cache(
    cache_path: Path,
    graph_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            record = pickle.load(handle)
        if record.get("metadata") != _graph_cache_metadata(graph_inputs):
            return None
        graph = record["graph"]
        if not isinstance(graph, dict) or "graph_dependencies" not in graph:
            return None
        return graph
    except (EOFError, OSError, pickle.PickleError, AttributeError, KeyError, TypeError):
        logging.warning("graph_cache=invalid path=%s", cache_path)
        return None


def _write_graph_cache(
    cache_path: Path,
    graph_inputs: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "metadata": _graph_cache_metadata(graph_inputs),
        "graph": graph,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
        logging.debug("graph_cache=saved path=%s", cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sweep_rows(config: ScalingLawsConfig) -> list[dict[str, Any]]:
    architecture_values = _sweep_values("architecture", config.architecture)
    relevant_fields = {
        field for architecture in architecture_values for field in SWEEP_FIELDS if _field_relevant(field, str(architecture), config.task)
    }
    values_by_field = {
        field: _sweep_values(field, getattr(config, field))
        if field in relevant_fields
        else _sweep_values(field, getattr(config, field))[:1]
        for field in SWEEP_FIELDS
    }
    sweep_lengths = {
        field: len(values)
        for field, values in values_by_field.items()
        if len(values) > 1
    }
    if len(set(sweep_lengths.values())) > 1:
        details = ", ".join(f"{field}={length}" for field, length in sorted(sweep_lengths.items()))
        raise ValueError(
            "Compositional scaling sweep parameters with multiple values must have the same length "
            f"({details})."
        )
    sweep_size = next(iter(sweep_lengths.values()), 1)
    rows = []
    for index in range(sweep_size):
        row = {
            field: values[index] if len(values) > 1 else values[0]
            for field, values in values_by_field.items()
        }
        relevant_row_fields = {field for field in SWEEP_FIELDS if _field_relevant(field, str(row["architecture"]), config.task)}
        rows.append(
            {
                "rho": row["rho"],
                "beta": row["beta"],
                "delta": row["delta"],
                "width": row["width"],
                "lr": row["lr"],
                "base_tasks": row["base_tasks"],
                "base_freq": row["base_freq"],
                "max_depth": row["max_depth"],
                "overrides": {
                    _config_field_name(field): value
                    for field, value in row.items()
                    if field in relevant_row_fields and field not in {"rho", "beta", "delta", "width", "lr"}
                },
            }
        )
    return rows


def _sweep_values(field: str, value: Any) -> list[Any]:
    if isinstance(value, list):
        if field == "delta" and not value:
            return [0.0]
        return value
    return [value]


def _config_field_name(field: str) -> str:
    return "width" if field == "width" else field


def _field_relevant(field: str, architecture: str, task: str) -> bool:
    del task
    if architecture == "transformer":
        return field not in MLP_ONLY_FIELDS
    if architecture == "mlp":
        return field not in TRANSFORMER_ONLY_FIELDS
    return True
