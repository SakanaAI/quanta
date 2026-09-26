"""Scalable controlled loss-decomposition tasks for an ordinary shared MLP.

The legacy configuration is the three-factor parallel NAND pilot: a task
one-hot concatenated with dense sensors, legacy disjoint support allocation,
and primitive Walsh-mode tracking.  The scalable path is algebraically
equivalent at the first layer but uses a task embedding lookup, compact support
allocation, gradient microbatching, and streamed exact evaluation.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CompositionalMLPConfig:
    n_tasks: int = 16
    group_size: int = 4
    subgroup_size: int = 2
    degree_a: int = 2
    degree_b: int = 2
    degree_c: int = 2
    composition: str = "parallel_nand"
    event_scope: str = "primitive"
    support_layout: str = "legacy"
    support_pool_bits: int = 0
    support_seed: int = 0
    task_conditioning: str = "onehot"
    sensor_bits: int = 128
    distractor_bits: int = 0
    feature_mode: str = "dense"
    frequency_exponent: float = 0.75
    rank_seed: int = 0
    model_seed: int = 0
    data_seed: int = 1_000
    eval_seed: int = 10_000
    width: int = 256
    hidden_layers: int = 2
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    batch_size: int = 512
    microbatch_size: int = 512
    steps: int = 65_000
    eval_every: int = 200
    eval_samples_per_task: int = 128
    eval_task_chunk: int = 16
    sampling: str = "stratified"
    precision: str = "fp32"
    device: str = "auto"
    output_dir: str = ".experiments/cnand/compositional-mlp"


@dataclass(frozen=True)
class FactorSpec:
    name: str
    degree: int
    sharing: int


def factor_specs(config: CompositionalMLPConfig) -> tuple[FactorSpec, ...]:
    """Return active factors while preserving the original A/B/C semantics."""
    specs: list[FactorSpec] = []
    if config.degree_a:
        specs.append(FactorSpec("A", config.degree_a, config.group_size))
    if config.degree_b:
        # The original two-stage task made B private when C was absent.
        sharing = config.subgroup_size if config.degree_c else 1
        specs.append(FactorSpec("B", config.degree_b, sharing))
    if config.degree_c:
        specs.append(FactorSpec("C", config.degree_c, 1))
    return tuple(specs)


def minimum_support_pool_bits(n_supports: int, degree: int) -> int:
    """Return the smallest bit pool containing enough distinct degree-d supports."""
    pool_bits = max(1, degree)
    while math.comb(pool_bits, degree) < n_supports:
        pool_bits += 1
    return pool_bits


def factor_support_pool_bits(config: CompositionalMLPConfig, spec: FactorSpec) -> int:
    n_supports = config.n_tasks // spec.sharing
    return config.support_pool_bits or minimum_support_pool_bits(n_supports, spec.degree)


def support_span(config: CompositionalMLPConfig) -> int:
    if config.support_layout == "legacy":
        return sum(config.n_tasks * spec.degree for spec in factor_specs(config))
    if config.support_layout == "compact":
        return sum(
            (config.n_tasks // spec.sharing) * spec.degree
            for spec in factor_specs(config)
        )
    if config.support_layout == "shared_pool":
        return sum(
            factor_support_pool_bits(config, spec) for spec in factor_specs(config)
        )
    raise ValueError("support_layout must be 'legacy', 'compact', or 'shared_pool'")


def resolved_sensor_bits(config: CompositionalMLPConfig) -> int:
    required = support_span(config)
    requested = config.sensor_bits or required + config.distractor_bits
    if requested < required:
        raise ValueError(
            f"sensor_bits={requested} is smaller than required support span {required}"
        )
    return int(requested)


def validate_config(config: CompositionalMLPConfig) -> CompositionalMLPConfig:
    specs = factor_specs(config)
    if not specs:
        raise ValueError("at least one factor degree must be positive")
    if any(degree < 0 for degree in (config.degree_a, config.degree_b, config.degree_c)):
        raise ValueError("factor degrees must be nonnegative")
    if config.hidden_layers < 1:
        raise ValueError("hidden_layers must be positive")
    if config.optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    if config.group_size <= 0 or config.n_tasks % config.group_size:
        raise ValueError("group_size must be positive and divide n_tasks")
    if config.subgroup_size <= 0 or config.n_tasks % config.subgroup_size:
        raise ValueError("subgroup_size must be positive and divide n_tasks")
    if config.group_size % config.subgroup_size:
        raise ValueError("subgroup_size must divide group_size")
    if config.composition not in {"parallel_nand", "nested_nand"}:
        raise ValueError("composition must be 'parallel_nand' or 'nested_nand'")
    if config.event_scope not in {"primitive", "all_modes"}:
        raise ValueError("event_scope must be 'primitive' or 'all_modes'")
    if config.support_pool_bits < 0:
        raise ValueError("support_pool_bits must be nonnegative")
    if config.support_layout == "shared_pool":
        for spec in specs:
            pool_bits = factor_support_pool_bits(config, spec)
            n_supports = config.n_tasks // spec.sharing
            if math.comb(pool_bits, spec.degree) < n_supports:
                raise ValueError(
                    f"support_pool_bits={pool_bits} cannot provide {n_supports} "
                    f"distinct degree-{spec.degree} supports for factor {spec.name}"
                )
    if config.task_conditioning not in {"onehot", "embedding"}:
        raise ValueError("task_conditioning must be 'onehot' or 'embedding'")
    if config.feature_mode not in {"dense", "sparse"}:
        raise ValueError("feature_mode must be 'dense' or 'sparse'")
    if config.sampling not in {"iid", "stratified"}:
        raise ValueError("sampling must be 'iid' or 'stratified'")
    if config.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be 'fp32' or 'bf16'")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if config.batch_size <= 0 or config.microbatch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.microbatch_size > config.batch_size:
        raise ValueError("microbatch_size cannot exceed batch_size")
    if config.sampling == "stratified" and config.batch_size % config.n_tasks:
        raise ValueError("stratified sampling requires batch_size divisible by n_tasks")
    if config.eval_samples_per_task % (2 ** sum(spec.degree for spec in specs)):
        raise ValueError(
            "eval_samples_per_task must be divisible by the exact relevant-bit panel"
        )
    if config.eval_task_chunk <= 0:
        raise ValueError("eval_task_chunk must be positive")
    sensor_bits = resolved_sensor_bits(config)
    return replace(config, sensor_bits=sensor_bits)


def task_probabilities(config: CompositionalMLPConfig) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.arange(1, config.n_tasks + 1, dtype=np.float64)
    ranked = ranks ** (-config.frequency_exponent)
    ranked /= ranked.sum()
    permutation = np.random.default_rng(config.rank_seed).permutation(config.n_tasks)
    probabilities = np.empty(config.n_tasks, dtype=np.float64)
    probabilities[permutation] = ranked
    rank_by_task = np.empty(config.n_tasks, dtype=np.int64)
    rank_by_task[permutation] = np.arange(1, config.n_tasks + 1)
    return probabilities, rank_by_task


@lru_cache(maxsize=None)
def shared_pool_supports(config: CompositionalMLPConfig) -> tuple[np.ndarray, ...]:
    """Sample fixed, unique sparse-parity supports from one bit pool per role."""
    tables: list[np.ndarray] = []
    offset = 0
    for factor_index, spec in enumerate(factor_specs(config)):
        n_supports = config.n_tasks // spec.sharing
        pool_bits = factor_support_pool_bits(config, spec)
        n_candidates = math.comb(pool_bits, spec.degree)
        generator = np.random.default_rng(config.support_seed + 104_729 * factor_index)
        if n_candidates <= 500_000:
            candidates = np.asarray(
                list(itertools.combinations(range(pool_bits), spec.degree)),
                dtype=np.int64,
            )
            selected = generator.choice(n_candidates, size=n_supports, replace=False)
            table = candidates[selected]
        else:
            selected_supports: set[tuple[int, ...]] = set()
            while len(selected_supports) < n_supports:
                candidate = tuple(
                    sorted(generator.choice(pool_bits, size=spec.degree, replace=False))
                )
                selected_supports.add(candidate)
            table = np.asarray(sorted(selected_supports), dtype=np.int64)
            generator.shuffle(table)
        tables.append(table + offset)
        offset += pool_bits
    return tuple(tables)


def support_indices(
    config: CompositionalMLPConfig, tasks: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    supports: list[torch.Tensor] = []
    offset = 0
    pooled = shared_pool_supports(config) if config.support_layout == "shared_pool" else ()
    for factor_index, spec in enumerate(factor_specs(config)):
        group = tasks // spec.sharing
        if config.support_layout == "legacy":
            start = group * spec.sharing * spec.degree
            block = config.n_tasks * spec.degree
        elif config.support_layout == "compact":
            start = group * spec.degree
            block = (config.n_tasks // spec.sharing) * spec.degree
        else:
            table = torch.as_tensor(pooled[factor_index], device=tasks.device)
            supports.append(table[group])
            offset += factor_support_pool_bits(config, spec)
            continue
        support = offset + start[:, None] + torch.arange(
            spec.degree, device=tasks.device
        )[None, :]
        supports.append(support)
        offset += block
    return tuple(supports)


def latent_signs_from_raw(
    config: CompositionalMLPConfig, raw: torch.Tensor
) -> torch.Tensor:
    signs = []
    offset = 0
    for spec in factor_specs(config):
        signs.append(raw[..., offset : offset + spec.degree].prod(dim=-1))
        offset += spec.degree
    return torch.stack(signs, dim=-1)


def fourier_basis(signs: torch.Tensor) -> torch.Tensor:
    modes = []
    for mask in range(2 ** signs.shape[-1]):
        mode = torch.ones_like(signs[..., 0])
        for index in range(signs.shape[-1]):
            if mask & (1 << index):
                mode = mode * signs[..., index]
        modes.append(mode)
    return torch.stack(modes, dim=-1)


def labels_from_signs(signs: torch.Tensor, composition: str) -> torch.Tensor:
    predicates = signs < 0
    if composition == "parallel_nand":
        labels = ~predicates.all(dim=-1)
    elif predicates.shape[-1] == 1:
        labels = ~predicates[..., 0]
    else:
        labels = ~(predicates[..., 0] & predicates[..., 1])
        for index in range(2, predicates.shape[-1]):
            labels = ~(labels & predicates[..., index])
    return labels.long()


def target_mode_coefficients(config: CompositionalMLPConfig) -> np.ndarray:
    n_factors = len(factor_specs(config))
    signs = torch.tensor(
        [
            [1.0 if mask & (1 << bit) else -1.0 for bit in range(n_factors)]
            for mask in range(2**n_factors)
        ]
    )
    signed_target = 2.0 * labels_from_signs(signs, config.composition).float() - 1.0
    return (signed_target[:, None] * fourier_basis(signs)).mean(dim=0).numpy()


def conditional_bayes_losses(config: CompositionalMLPConfig) -> dict[int, float]:
    n_factors = len(factor_specs(config))
    signs = torch.tensor(
        [
            [1.0 if mask & (1 << bit) else -1.0 for bit in range(n_factors)]
            for mask in range(2**n_factors)
        ]
    )
    labels = labels_from_signs(signs, config.composition).numpy()
    losses: dict[int, float] = {}
    for known_mask in range(2**n_factors):
        groups: dict[tuple[float, ...], list[int]] = {}
        for row, sign_row in enumerate(signs.numpy()):
            key = tuple(
                float(sign_row[index])
                for index in range(n_factors)
                if known_mask & (1 << index)
            )
            groups.setdefault(key, []).append(row)
        entropy = 0.0
        for rows in groups.values():
            probability = float(labels[rows].mean())
            if probability in {0.0, 1.0}:
                group_entropy = 0.0
            else:
                group_entropy = -probability * math.log2(probability) - (
                    1.0 - probability
                ) * math.log2(1.0 - probability)
            entropy += len(rows) * group_entropy / len(labels)
        losses[known_mask] = entropy
    return losses


class OneHotSharedMLP(nn.Module):
    """The original concatenated task-one-hot MLP."""

    def __init__(self, n_tasks: int, sensor_bits: int, width: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        current = n_tasks + sensor_bits
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current, width), nn.ReLU()))
            current = width
        layers.append(nn.Linear(current, 2))
        self.net = nn.Sequential(*layers)
        self.n_tasks = n_tasks

    def forward(self, tasks: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        inputs = torch.zeros(
            len(tasks), self.n_tasks + sensors.shape[1], device=sensors.device
        )
        inputs[torch.arange(len(tasks), device=tasks.device), tasks] = 1.0
        inputs[:, self.n_tasks :] = sensors
        return self.net(inputs)


class EmbeddedSharedMLP(nn.Module):
    """Exact first-layer algebra without materializing task one-hots."""

    def __init__(
        self,
        n_tasks: int,
        sensor_bits: int,
        width: int,
        hidden_layers: int,
        readout_cls: type[nn.Linear] = nn.Linear,
    ):
        super().__init__()
        self.task_embedding = nn.Embedding(n_tasks, width)
        self.sensor_projection = nn.Linear(sensor_bits, width)
        self.hidden = nn.ModuleList(
            nn.Linear(width, width) for _ in range(hidden_layers - 1)
        )
        self.readout = readout_cls(width, 2)
        bound = 1.0 / math.sqrt(n_tasks + sensor_bits)
        nn.init.uniform_(self.task_embedding.weight, -bound, bound)
        nn.init.uniform_(self.sensor_projection.weight, -bound, bound)
        nn.init.uniform_(self.sensor_projection.bias, -bound, bound)

    def forward(self, tasks: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.task_embedding(tasks) + self.sensor_projection(sensors))
        for layer in self.hidden:
            hidden = F.relu(layer(hidden))
        return self.readout(hidden)


def build_model(config: CompositionalMLPConfig) -> nn.Module:
    cls = OneHotSharedMLP if config.task_conditioning == "onehot" else EmbeddedSharedMLP
    return cls(config.n_tasks, config.sensor_bits, config.width, config.hidden_layers)


def make_sensor_batch(
    config: CompositionalMLPConfig,
    tasks: torch.Tensor,
    generator: torch.Generator,
    raw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sensors = 2.0 * torch.randint(
        0,
        2,
        (len(tasks), config.sensor_bits),
        generator=generator,
        dtype=torch.int64,
        device=tasks.device,
    ).float() - 1.0
    supports = support_indices(config, tasks)
    if config.feature_mode == "sparse":
        sparse = torch.zeros_like(sensors)
        relevant = torch.cat(supports, dim=1)
        sparse.scatter_(1, relevant, sensors.gather(1, relevant))
        sensors = sparse
    if raw is not None:
        offset = 0
        for support, spec in zip(supports, factor_specs(config)):
            sensors.scatter_(1, support, raw[:, offset : offset + spec.degree])
            offset += spec.degree
    signs = torch.stack(
        [sensors.gather(1, support).prod(dim=1) for support in supports], dim=1
    )
    labels = labels_from_signs(signs, config.composition)
    return sensors, labels, signs


def autocast_context(config: CompositionalMLPConfig, device: torch.device):
    if config.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    config: CompositionalMLPConfig,
    probabilities: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    specs = factor_specs(config)
    n_raw = sum(spec.degree for spec in specs)
    patterns = torch.tensor(
        [
            [1.0 if mask & (1 << bit) else -1.0 for bit in range(n_raw)]
            for mask in range(2**n_raw)
        ],
        device=device,
    )
    repeats = config.eval_samples_per_task // len(patterns)
    raw_one_task = patterns.repeat(repeats, 1)
    generator = torch.Generator(device=device.type).manual_seed(config.eval_seed)
    task_losses = np.empty(config.n_tasks, dtype=np.float64)
    coefficients = np.empty((config.n_tasks, 2 ** len(specs)), dtype=np.float64)
    residuals = np.empty(config.n_tasks, dtype=np.float64)
    exact = np.empty(config.n_tasks, dtype=bool)
    for start in range(0, config.n_tasks, config.eval_task_chunk):
        stop = min(start + config.eval_task_chunk, config.n_tasks)
        chunk_tasks = torch.arange(start, stop, device=device)
        tasks = chunk_tasks.repeat_interleave(config.eval_samples_per_task)
        raw = raw_one_task.repeat(stop - start, 1)
        sensors, labels, signs = make_sensor_batch(config, tasks, generator, raw=raw)
        with autocast_context(config, device):
            logits = model(tasks, sensors)
        losses = F.cross_entropy(logits.float(), labels, reduction="none") / math.log(2.0)
        signed_probability = 2.0 * logits.float().softmax(dim=-1)[:, 1] - 1.0
        basis = fourier_basis(signs)
        chunk_size = stop - start
        loss_matrix = losses.reshape(chunk_size, config.eval_samples_per_task)
        probability_matrix = signed_probability.reshape(
            chunk_size, config.eval_samples_per_task
        )
        basis_matrix = basis.reshape(
            chunk_size, config.eval_samples_per_task, -1
        )
        chunk_coefficients = (
            probability_matrix[:, :, None] * basis_matrix
        ).mean(dim=1)
        reconstruction = (chunk_coefficients[:, None, :] * basis_matrix).sum(dim=2)
        predictions = logits.argmax(dim=-1).reshape(
            chunk_size, config.eval_samples_per_task
        )
        target_matrix = labels.reshape(chunk_size, config.eval_samples_per_task)
        task_losses[start:stop] = loss_matrix.mean(dim=1).cpu().numpy()
        coefficients[start:stop] = chunk_coefficients.cpu().numpy()
        residuals[start:stop] = (
            (probability_matrix - reconstruction).square().mean(dim=1).sqrt().cpu().numpy()
        )
        exact[start:stop] = (predictions == target_matrix).all(dim=1).cpu().numpy()
    return {
        "task_losses_bits": task_losses,
        "weighted_loss_bits": float(probabilities @ task_losses),
        "mean_loss_bits": float(task_losses.mean()),
        "mode_coefficients": coefficients,
        "mode_residual_rms": residuals,
        "task_exact": exact,
    }


def first_upward_crossing(steps: np.ndarray, curve: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(curve >= threshold)
    return float(steps[indices[0]]) if len(indices) else None


def first_downward_crossings(
    steps: np.ndarray, curves: np.ndarray, threshold: float
) -> list[float | None]:
    output: list[float | None] = []
    for task in range(curves.shape[1]):
        indices = np.flatnonzero(curves[:, task] <= threshold)
        output.append(float(steps[indices[0]]) if len(indices) else None)
    return output


def event_masks(config: CompositionalMLPConfig, targets: np.ndarray) -> list[int]:
    primitive = [1 << index for index in range(len(factor_specs(config)))]
    if config.event_scope == "primitive":
        return primitive
    return [mask for mask in range(1, len(targets)) if abs(targets[mask]) > 1e-8]


def mode_name(mask: int, specs: tuple[FactorSpec, ...]) -> str:
    return "".join(spec.name for index, spec in enumerate(specs) if mask & (1 << index))


def fit_log_clock(events: list[dict[str, object]]) -> dict[str, float | None]:
    completed = [event for event in events if event["time_80pct"] is not None]
    if len(completed) < 3:
        return {"gamma": None, "r2": None}
    demand = np.asarray([event["demand"] for event in completed], dtype=float)
    times = np.asarray([event["time_80pct"] for event in completed], dtype=float)
    slope, intercept = np.polyfit(np.log(demand), np.log(times), 1)
    prediction = intercept + slope * np.log(demand)
    total = ((np.log(times) - np.log(times).mean()) ** 2).sum()
    residual = ((np.log(times) - prediction) ** 2).sum()
    return {
        "gamma": float(-slope),
        "r2": float(1.0 - residual / total) if total else None,
    }


def fit_ranked_demand(events: list[dict[str, object]]) -> dict[str, float | None]:
    if len(events) < 3:
        return {"exponent": None, "r2": None}
    demand = np.sort(np.asarray([event["demand"] for event in events], dtype=float))[::-1]
    rank = np.arange(1, len(demand) + 1, dtype=float)
    slope, intercept = np.polyfit(np.log(rank), np.log(demand), 1)
    prediction = intercept + slope * np.log(rank)
    total = ((np.log(demand) - np.log(demand).mean()) ** 2).sum()
    residual = ((np.log(demand) - prediction) ** 2).sum()
    return {
        "exponent": float(-slope),
        "r2": float(1.0 - residual / total) if total else None,
    }


def analyze(
    config: CompositionalMLPConfig,
    probabilities: np.ndarray,
    rank_by_task: np.ndarray,
    steps: np.ndarray,
    losses: np.ndarray,
    coefficients: np.ndarray,
    residuals: np.ndarray,
    exact: np.ndarray,
    parameter_counts: dict[str, int],
) -> tuple[dict[str, object], np.ndarray]:
    specs = factor_specs(config)
    targets = target_mode_coefficients(config)
    events: list[dict[str, object]] = []
    for mask in event_masks(config, targets):
        selected_specs = [spec for index, spec in enumerate(specs) if mask & (1 << index)]
        sharing = min(spec.sharing for spec in selected_specs)
        degree = sum(spec.degree for spec in selected_specs)
        for group in range(config.n_tasks // sharing):
            start = group * sharing
            stop = start + sharing
            curve = coefficients[:, start:stop, mask].mean(axis=1) * np.sign(targets[mask])
            crossings = {
                fraction: first_upward_crossing(
                    steps, curve, fraction / 100.0 * abs(targets[mask])
                )
                for fraction in (20, 50, 80)
            }
            relative_width = None
            if all(crossings[fraction] is not None for fraction in (20, 50, 80)):
                relative_width = (
                    float(crossings[80]) - float(crossings[20])
                ) / float(crossings[50])
            events.append(
                {
                    "mode_mask": mask,
                    "mode": mode_name(mask, specs),
                    "order": mask.bit_count(),
                    "degree": degree,
                    "sharing": sharing,
                    "group": group,
                    "demand": float(probabilities[start:stop].sum()),
                    "target_coefficient": float(targets[mask]),
                    "time_20pct": crossings[20],
                    "time_50pct": crossings[50],
                    "time_80pct": crossings[80],
                    "relative_width_20_to_80": relative_width,
                }
            )
    completed = [event for event in events if event["time_80pct"] is not None]
    demand = np.asarray([event["demand"] for event in completed], dtype=float)
    times = np.asarray([event["time_80pct"] for event in completed], dtype=float)
    global_constant = float(np.median(demand * times)) if len(times) else None
    global_error = (
        float(np.median(np.abs(global_constant / demand - times) / times))
        if global_constant is not None
        else None
    )
    clock_by_mode: dict[str, dict[str, float | int | None]] = {}
    for name in sorted({str(event["mode"]) for event in events}):
        selected = [event for event in completed if event["mode"] == name]
        if not selected:
            clock_by_mode[name] = {"completed": 0, "constant": None, "median_relative_error": None}
            continue
        selected_demand = np.asarray([event["demand"] for event in selected], dtype=float)
        selected_time = np.asarray([event["time_80pct"] for event in selected], dtype=float)
        constant = float(np.median(selected_demand * selected_time))
        clock_by_mode[name] = {
            "completed": len(selected),
            "constant": constant,
            "median_relative_error": float(
                np.median(np.abs(constant / selected_demand - selected_time) / selected_time)
            ),
        }
    complexity_fit: dict[str, float | None] = {
        "demand_gamma": None,
        "degree_coefficient": None,
        "r2": None,
    }
    if len(completed) >= 4 and len({event["degree"] for event in completed}) > 1:
        degrees = np.asarray([event["degree"] for event in completed], dtype=float)
        design = np.column_stack((np.ones(len(times)), np.log(demand), degrees))
        response = np.log(times)
        parameters = np.linalg.lstsq(design, response, rcond=None)[0]
        prediction = design @ parameters
        total = ((response - response.mean()) ** 2).sum()
        residual = ((response - prediction) ** 2).sum()
        complexity_fit = {
            "demand_gamma": float(-parameters[1]),
            "degree_coefficient": float(parameters[2]),
            "r2": float(1.0 - residual / total) if total else None,
        }
    bayes = conditional_bayes_losses(config)
    known_masks = np.zeros((len(steps), config.n_tasks), dtype=np.int64)
    for index, spec in enumerate(specs):
        mask = 1 << index
        normalized = coefficients[:, :, mask] * np.sign(targets[mask])
        known_masks[normalized >= 0.8 * abs(targets[mask])] |= mask
    reconstructed = np.empty(len(steps), dtype=np.float64)
    for step_index in range(len(steps)):
        per_task = np.asarray([bayes[int(mask)] for mask in known_masks[step_index]])
        reconstructed[step_index] = float(probabilities @ per_task)
    exact_crossings = first_downward_crossings(steps, 1.0 - exact.astype(float), 0.0)
    widths = [
        float(event["relative_width_20_to_80"])
        for event in completed
        if event["relative_width_20_to_80"] is not None
    ]
    raw_spearman = (
        float(spearmanr(np.log(demand), times).statistic) if len(times) >= 3 else None
    )
    fit = fit_log_clock(events)
    acquired = {id(event) for event in completed}
    ordered = sorted(events, key=lambda event: float(event["demand"]), reverse=True)
    top_n = ordered[: len(completed)]
    prefix_fraction = (
        sum(id(event) in acquired for event in top_n) / len(top_n) if top_n else None
    )
    summary: dict[str, object] = {
        "config": asdict(config),
        "factor_specs": [asdict(spec) for spec in specs],
        "parameter_counts": parameter_counts,
        "support_span": support_span(config),
        "frequency_range": float(probabilities.max() / probabilities.min()),
        "probabilities": probabilities.tolist(),
        "rank_by_task": rank_by_task.tolist(),
        "target_mode_coefficients": targets.tolist(),
        "conditional_bayes_losses_bits": {str(key): value for key, value in bayes.items()},
        "event_records": events,
        "ranked_event_demand_fit": fit_ranked_demand(events),
        "total_events": len(events),
        "completed_events": len(completed),
        "final_exact_tasks": int(exact[-1].sum()),
        "best_exact_tasks": int(exact.sum(axis=1).max()),
        "crossing_exact": exact_crossings,
        "final_weighted_loss_bits": float(probabilities @ losses[-1]),
        "final_mean_loss_bits": float(losses[-1].mean()),
        "final_reconstructed_loss_bits": float(reconstructed[-1]),
        "final_mean_mode_residual_rms": float(residuals[-1].mean()),
        "event_demand_time_spearman": raw_spearman,
        "event_time_gamma": fit["gamma"],
        "event_time_r2": fit["r2"],
        "global_clock_constant": global_constant,
        "global_clock_median_relative_error": global_error,
        "clock_by_mode": clock_by_mode,
        "complexity_adjusted_clock": complexity_fit,
        "median_event_relative_width_20_to_80": float(np.median(widths)) if widths else None,
        "demand_prefix_fraction_at_completed_count": prefix_fraction,
    }
    return summary, reconstructed


def plot_run(
    output: Path,
    config: CompositionalMLPConfig,
    steps: np.ndarray,
    losses: np.ndarray,
    weighted: np.ndarray,
    reconstructed: np.ndarray,
    coefficients: np.ndarray,
    summary: dict[str, object],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    stride = max(1, config.n_tasks // 128)
    for task in range(0, config.n_tasks, stride):
        axes[0, 0].plot(steps, losses[:, task], color="0.55", alpha=0.18, linewidth=0.6)
    axes[0, 0].plot(steps, weighted, color="crimson", linewidth=2.2, label="neural aggregate")
    axes[0, 0].plot(steps, reconstructed, color="black", linestyle="--", linewidth=1.4, label="acquired-factor reconstruction")
    axes[0, 0].set(xlabel="optimizer step", ylabel="cross-entropy (bits)")
    axes[0, 0].legend(frameon=False)

    completed = [event for event in summary["event_records"] if event["time_80pct"] is not None]
    names = sorted({event["mode"] for event in completed})
    colors = {name: plt.cm.tab10(index % 10) for index, name in enumerate(names)}
    for name in names:
        selected = [event for event in completed if event["mode"] == name]
        axes[0, 1].scatter(
            [event["demand"] for event in selected],
            [event["time_80pct"] for event in selected],
            s=20,
            alpha=0.7,
            color=colors[name],
            label=name,
        )
    if completed:
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].set(xlabel="event demand D", ylabel="80% acquisition step")

    targets = np.asarray(summary["target_mode_coefficients"])
    shown = sorted(completed, key=lambda event: float(event["demand"]), reverse=True)
    if len(shown) > 32:
        indices = np.linspace(0, len(shown) - 1, 32, dtype=int)
        shown = [shown[index] for index in indices]
    for event in shown:
        start = int(event["group"]) * int(event["sharing"])
        stop = start + int(event["sharing"])
        mask = int(event["mode_mask"])
        curve = coefficients[:, start:stop, mask].mean(axis=1) / targets[mask]
        axes[1, 0].plot(
            steps,
            curve,
            color=colors[str(event["mode"])],
            alpha=0.45,
            linewidth=0.8,
        )
    axes[1, 0].axhline(0.2, color="black", linestyle=":", linewidth=0.7)
    axes[1, 0].axhline(0.8, color="black", linestyle=":", linewidth=0.7)
    axes[1, 0].set(
        xlabel="optimizer step",
        ylabel="target-normalized functional mode",
        ylim=(-0.2, 1.35),
    )

    ranks = np.arange(1, config.n_tasks + 1)
    sorted_probabilities = np.sort(np.asarray(summary["probabilities"]))[::-1]
    axes[1, 1].loglog(ranks, sorted_probabilities, marker=".", linestyle="none")
    axes[1, 1].set(xlabel="task-frequency rank", ylabel="task probability")
    axes[1, 1].set_title(
        f"configured exponent={config.frequency_exponent:g}; range={summary['frequency_range']:.1f}x"
    )
    figure.suptitle(
        f"{config.composition}, Q={config.n_tasks}, width={config.width}, "
        f"{config.sampling}; completed {summary['completed_events']}/{summary['total_events']} events"
    )
    figure.tight_layout()
    figure.savefig(output / "trajectory.png", dpi=180)
    figure.savefig(output / "trajectory.pdf")
    plt.close(figure)


def run_name(config: CompositionalMLPConfig) -> str:
    return (
        f"{config.composition}-{config.event_scope}-q{config.n_tasks}-"
        f"g{config.group_size}-{config.subgroup_size}-"
        f"d{config.degree_a}-{config.degree_b}-{config.degree_c}-"
        f"{config.support_layout}-pool{config.support_pool_bits}-"
        f"ps{config.support_seed}-{config.feature_mode}-{config.task_conditioning}-"
        f"s{config.sensor_bits}-"
        f"w{config.width}-h{config.hidden_layers}-exp{config.frequency_exponent:g}-"
        f"{config.optimizer}{config.learning_rate:g}-"
        f"b{config.batch_size}-mb{config.microbatch_size}-"
        f"m{config.model_seed}-r{config.rank_seed}-{config.sampling}-n{config.steps}"
    ).replace(".", "p")


def parameter_counts(model: nn.Module, config: CompositionalMLPConfig) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    task = (
        config.n_tasks * config.width
        if config.task_conditioning in {"onehot", "embedding"}
        else 0
    )
    return {"total": int(total), "task_conditioning": int(task), "shared_trunk": int(total - task)}


def task_chunks(
    config: CompositionalMLPConfig,
    probabilities: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> Iterable[torch.Tensor]:
    if config.sampling == "stratified":
        examples_per_task = config.batch_size // config.n_tasks
        all_tasks = torch.arange(config.n_tasks, device=device).repeat_interleave(examples_per_task)
        permutation = torch.randperm(len(all_tasks), generator=generator, device=device)
        all_tasks = all_tasks[permutation]
        for start in range(0, config.batch_size, config.microbatch_size):
            yield all_tasks[start : start + config.microbatch_size]
    else:
        remaining = config.batch_size
        while remaining:
            size = min(config.microbatch_size, remaining)
            yield torch.multinomial(
                probabilities, size, replacement=True, generator=generator
            )
            remaining -= size


def run(config: CompositionalMLPConfig) -> Path:
    config = validate_config(config)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    resolved_device = (
        "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    random.seed(config.model_seed)
    np.random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)
    probabilities, rank_by_task = task_probabilities(config)
    probability_tensor = torch.tensor(probabilities, dtype=torch.float32, device=device)
    model = build_model(config).to(device)
    optimizer_class = torch.optim.SGD if config.optimizer == "sgd" else torch.optim.Adam
    optimizer = optimizer_class(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    output = Path(config.output_dir) / run_name(config)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "status.json").open("w") as handle:
        json.dump({"status": "running", "config": asdict(config)}, handle, indent=2)
    generator = torch.Generator(device=device.type).manual_seed(config.data_seed)
    records: list[dict[str, np.ndarray | float]] = []
    record_steps: list[int] = []

    def record(step: int) -> None:
        model.eval()
        records.append(evaluate(model, config, probabilities, device))
        record_steps.append(step)
        model.train()

    try:
        record(0)
        for step in range(1, config.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for tasks in task_chunks(config, probability_tensor, generator, device):
                sensors, labels, _ = make_sensor_batch(config, tasks, generator)
                with autocast_context(config, device):
                    logits = model(tasks, sensors)
                    per_example = F.cross_entropy(logits, labels, reduction="none")
                    if config.sampling == "iid":
                        loss = per_example.sum() / config.batch_size
                    else:
                        examples_per_task = config.batch_size // config.n_tasks
                        loss = (
                            per_example * probability_tensor[tasks] / examples_per_task
                        ).sum()
                loss.backward()
            optimizer.step()
            if step % config.eval_every == 0 or step == config.steps:
                record(step)
                progress_every = max(config.eval_every, config.steps // 20)
                if step == config.steps or step % progress_every < config.eval_every:
                    latest = records[-1]
                    print(
                        f"step={step} weighted_bits={latest['weighted_loss_bits']:.6f} "
                        f"exact={int(np.asarray(latest['task_exact']).sum())}/{config.n_tasks}",
                        flush=True,
                    )
        steps = np.asarray(record_steps)
        task_losses = np.stack([record["task_losses_bits"] for record in records])
        weighted = np.asarray([record["weighted_loss_bits"] for record in records])
        mean_losses = np.asarray([record["mean_loss_bits"] for record in records])
        coefficients = np.stack([record["mode_coefficients"] for record in records])
        residuals = np.stack([record["mode_residual_rms"] for record in records])
        exact = np.stack([record["task_exact"] for record in records])
        summary, reconstructed = analyze(
            config,
            probabilities,
            rank_by_task,
            steps,
            task_losses,
            coefficients,
            residuals,
            exact,
            parameter_counts(model, config),
        )
        np.savez_compressed(
            output / "trajectory.npz",
            steps=steps,
            task_losses_bits=task_losses,
            weighted_losses_bits=weighted,
            mean_losses_bits=mean_losses,
            mode_coefficients=coefficients,
            mode_residual_rms=residuals,
            task_exact=exact,
            reconstructed_losses_bits=reconstructed,
            probabilities=probabilities,
            rank_by_task=rank_by_task,
        )
        with (output / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
        plot_run(output, config, steps, task_losses, weighted, reconstructed, coefficients, summary)
        with (output / "status.json").open("w") as handle:
            json.dump({"status": "complete", "config": asdict(config)}, handle, indent=2)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"config", "probabilities", "rank_by_task", "event_records"}
                },
                indent=2,
            )
        )
        print(f"saved={output}")
        return output
    except Exception as error:
        with (output / "status.json").open("w") as handle:
            json.dump(
                {"status": "failed", "error": repr(error), "config": asdict(config)},
                handle,
                indent=2,
            )
        raise
