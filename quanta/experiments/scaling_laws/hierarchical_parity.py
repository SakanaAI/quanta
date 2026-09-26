"""Hierarchical fixed-pool sparse parity for compositional learning dynamics.

Every node owns one degree-two parity drawn from a fixed dense bit pool.  Root
targets are their local parities; a child target is the left-associated NAND of
its parent's target and its own parity.  Training examples query one node, while
the known ancestor path defines the candidate computational prerequisites and
the closure demand of every node.
"""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mup
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from quanta.experiments.scaling_laws.compositional_mlp import EmbeddedSharedMLP


@dataclass(frozen=True)
class HierarchicalParityConfig:
    base_tasks: int = 4
    branching_factor: int = 2
    max_depth: int = 3
    depth_mass_decay: float = 1.0
    demand_law: str = "depth_mass"
    beta: float = 0.0
    root_frequency_exponent: float = 0.5
    rank_seed: int = 0
    support_pool_bits: int = 64
    support_seed: int = 0
    model_seed: int = 0
    data_seed: int = 1_000
    eval_seed: int = 10_000
    width: int = 256
    hidden_layers: int = 3
    task_conditioning: str = "early"
    parameterization: str = "standard"
    mup_base_width: int = 8
    mup_delta_width: int = 16
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    batch_size: int = 1_024
    microbatch_size: int = 256
    steps: int = 20_000
    eval_every: int = 100
    checkpoint_every: int = 0
    eval_samples: int = 256
    eval_task_chunk: int = 64
    sampling: str = "iid"
    precision: str = "fp32"
    device: str = "auto"
    output_dir: str = ".experiments/cnand/hierarchical-parity"
    resume_checkpoint: str = ""


def validate_config(config: HierarchicalParityConfig) -> HierarchicalParityConfig:
    if config.base_tasks < 2:
        raise ValueError("base_tasks must be at least two")
    if config.branching_factor < 1:
        raise ValueError("branching_factor must be positive")
    if config.max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    if config.depth_mass_decay <= 0:
        raise ValueError("depth_mass_decay must be positive")
    if config.demand_law not in {"depth_mass", "rho_beta"}:
        raise ValueError("demand_law must be 'depth_mass' or 'rho_beta'")
    if config.demand_law == "rho_beta":
        if config.beta <= config.branching_factor:
            raise ValueError(
                "rho_beta demand requires beta > branching_factor (rho)"
            )
        if config.root_frequency_exponent != 0:
            raise ValueError(
                "rho_beta demand requires root_frequency_exponent=0 so the "
                "requested depth law is not mixed with a second rank law"
            )
    elif config.beta != 0:
        raise ValueError("beta is only used when demand_law='rho_beta'")
    if config.root_frequency_exponent < 0:
        raise ValueError("root_frequency_exponent must be nonnegative")
    n_nodes = sum(
        config.base_tasks * config.branching_factor**depth
        for depth in range(config.max_depth + 1)
    )
    if math.comb(config.support_pool_bits, 2) < n_nodes:
        raise ValueError("support_pool_bits cannot provide one unique pair per node")
    if config.hidden_layers < config.max_depth:
        raise ValueError(
            "hidden_layers must be at least max_depth so the carrier is not "
            "shallower than the declared composition"
        )
    if config.hidden_layers < 2 and config.task_conditioning == "late":
        raise ValueError("late task conditioning requires at least two hidden layers")
    if config.task_conditioning not in {"early", "late"}:
        raise ValueError("task_conditioning must be 'early' or 'late'")
    if config.parameterization not in {"standard", "mup"}:
        raise ValueError("parameterization must be 'standard' or 'mup'")
    if config.mup_base_width <= 0 or config.mup_delta_width <= 0:
        raise ValueError("mup base and delta widths must be positive")
    if config.mup_base_width == config.mup_delta_width:
        raise ValueError("mup base and delta widths must differ")
    if config.optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if config.batch_size <= 0 or config.microbatch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.microbatch_size > config.batch_size:
        raise ValueError("microbatch_size cannot exceed batch_size")
    if config.sampling not in {"iid", "stratified"}:
        raise ValueError("sampling must be 'iid' or 'stratified'")
    if config.sampling == "stratified" and config.batch_size % n_nodes:
        raise ValueError("stratified sampling requires batch_size divisible by n_nodes")
    if config.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be 'fp32' or 'bf16'")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if config.eval_samples < 32 or config.eval_task_chunk <= 0:
        raise ValueError("evaluation sizes must be positive and eval_samples >= 32")
    if config.steps <= 0 or config.eval_every <= 0:
        raise ValueError("steps and eval_every must be positive")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    return replace(config)


def build_hierarchy(config: HierarchicalParityConfig) -> dict[str, np.ndarray | list[list[int]]]:
    nodes_by_depth: list[list[int]] = []
    node_depths: list[int] = []
    parents: list[int] = []
    roots: list[int] = []
    next_node = 0
    for depth in range(config.max_depth + 1):
        count = config.base_tasks * config.branching_factor**depth
        nodes = list(range(next_node, next_node + count))
        nodes_by_depth.append(nodes)
        node_depths.extend([depth] * count)
        if depth == 0:
            parents.extend([-1] * count)
            roots.extend(nodes)
        else:
            previous = nodes_by_depth[depth - 1]
            for local_index in range(count):
                parent = previous[local_index // config.branching_factor]
                parents.append(parent)
                roots.append(roots[parent])
        next_node += count

    n_nodes = next_node
    paths = np.full((n_nodes, config.max_depth + 1), -1, dtype=np.int64)
    for node in range(n_nodes):
        reverse_path: list[int] = []
        current = node
        while current >= 0:
            reverse_path.append(current)
            current = parents[current]
        path = reverse_path[::-1]
        paths[node, : len(path)] = path

    ranks = np.arange(1, config.base_tasks + 1, dtype=np.float64)
    ranked = ranks ** (-config.root_frequency_exponent)
    ranked /= ranked.sum()
    permutation = np.random.default_rng(config.rank_seed).permutation(config.base_tasks)
    root_probabilities = np.empty(config.base_tasks, dtype=np.float64)
    root_probabilities[permutation] = ranked
    root_ranks = np.empty(config.base_tasks, dtype=np.int64)
    root_ranks[permutation] = np.arange(1, config.base_tasks + 1)

    depths = np.asarray(node_depths, dtype=np.int64)
    root_ids = np.asarray(roots, dtype=np.int64)
    rho = float(config.branching_factor)
    if config.demand_law == "rho_beta":
        requested_marginal_demand = (
            root_probabilities[root_ids] * config.beta ** (-depths)
        )
        terminal_probabilities = requested_marginal_demand.copy()
        for depth in range(1, config.max_depth + 1):
            for node in nodes_by_depth[depth]:
                terminal_probabilities[parents[node]] -= requested_marginal_demand[
                    node
                ]
        depth_masses = np.asarray(
            [
                terminal_probabilities[depths == depth].sum()
                for depth in range(config.max_depth + 1)
            ],
            dtype=np.float64,
        )
    else:
        depth_indices = np.arange(config.max_depth + 1, dtype=np.float64)
        depth_masses = config.depth_mass_decay ** (-depth_indices)
        depth_masses /= depth_masses.sum()
        terminal_probabilities = (
            root_probabilities[root_ids]
            * depth_masses[depths]
            / config.branching_factor**depths
        )
    marginal_demand = terminal_probabilities.copy()
    for depth in range(config.max_depth, 0, -1):
        for node in nodes_by_depth[depth]:
            marginal_demand[parents[node]] += marginal_demand[node]
    node_counts = np.asarray(
        [len(nodes) for nodes in nodes_by_depth], dtype=np.float64
    )
    per_node_probabilities = np.asarray(
        [
            terminal_probabilities[depths == depth].mean()
            for depth in range(config.max_depth + 1)
        ],
        dtype=np.float64,
    )
    per_node_marginal_demand = np.asarray(
        [
            marginal_demand[depths == depth].mean()
            for depth in range(config.max_depth + 1)
        ],
        dtype=np.float64,
    )
    realized_rho = node_counts[1:] / node_counts[:-1]
    realized_beta = per_node_marginal_demand[:-1] / per_node_marginal_demand[1:]
    realized_direct_beta = per_node_probabilities[:-1] / per_node_probabilities[1:]
    requested_beta = (
        config.beta
        if config.demand_law == "rho_beta"
        else rho * config.depth_mass_decay
    )
    rho_relative_error = (
        np.abs(realized_rho - rho) / rho if len(realized_rho) else np.asarray([])
    )
    beta_relative_error = (
        np.abs(realized_beta - requested_beta) / requested_beta
        if len(realized_beta)
        else np.asarray([])
    )
    demand_law_diagnostics = {
        "mode": config.demand_law,
        "requested_rho": rho,
        "requested_beta": float(requested_beta),
        "implied_alpha": float(math.log(requested_beta, rho) - 1.0)
        if rho > 1.0
        else None,
        "internal_terminal_depth_mass_decay": float(requested_beta / rho),
        "realized_rho_by_depth": realized_rho.tolist(),
        "realized_beta_by_depth": realized_beta.tolist(),
        "realized_direct_beta_by_depth": realized_direct_beta.tolist(),
        "terminal_depth_masses": depth_masses.tolist(),
        "max_rho_relative_error": float(rho_relative_error.max())
        if len(rho_relative_error)
        else 0.0,
        "max_beta_relative_error": float(beta_relative_error.max())
        if len(beta_relative_error)
        else 0.0,
    }
    return {
        "nodes_by_depth": nodes_by_depth,
        "node_depths": depths,
        "parents": np.asarray(parents, dtype=np.int64),
        "root_ids": root_ids,
        "rank_by_root": root_ranks,
        "root_ranks": root_ranks[root_ids],
        "paths": paths,
        "depth_masses": depth_masses,
        "demand_law_diagnostics": demand_law_diagnostics,
        "marginal_demand": marginal_demand,
        "terminal_probabilities": terminal_probabilities,
    }


def gf2_rank(masks: list[int]) -> int:
    basis: dict[int, int] = {}
    for value in masks:
        reduced = value
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = reduced
                break
            reduced ^= basis[pivot]
    return len(basis)


def build_supports(
    config: HierarchicalParityConfig, hierarchy: dict[str, object]
) -> np.ndarray:
    parents = np.asarray(hierarchy["parents"], dtype=np.int64)
    n_nodes = len(parents)
    generator = np.random.default_rng(config.support_seed)
    used: set[tuple[int, int]] = set()
    path_masks: list[list[int]] = []
    supports = np.empty((n_nodes, 2), dtype=np.int64)
    for node in range(n_nodes):
        ancestor_masks = [] if parents[node] < 0 else path_masks[int(parents[node])]
        for _ in range(100_000):
            sampled = generator.choice(config.support_pool_bits, size=2, replace=False)
            pair = tuple(sorted((int(sampled[0]), int(sampled[1]))))
            if pair in used:
                continue
            mask = (1 << pair[0]) | (1 << pair[1])
            if gf2_rank([*ancestor_masks, mask]) != len(ancestor_masks) + 1:
                continue
            supports[node] = pair
            used.add(pair)
            path_masks.append([*ancestor_masks, mask])
            break
        else:
            raise RuntimeError("could not sample independent unique path supports")
    return supports


def nested_nand_sign(parent_sign: torch.Tensor, local_sign: torch.Tensor) -> torch.Tensor:
    """Compose Boolean NAND while using +1 for label true and -1 for false."""
    return (1.0 - parent_sign - local_sign - parent_sign * local_sign) / 2.0


def target_signs(
    tasks: torch.Tensor,
    sensors: torch.Tensor,
    paths: torch.Tensor,
    supports: torch.Tensor,
) -> torch.Tensor:
    selected_paths = paths[tasks]
    state = torch.zeros(len(tasks), device=sensors.device)
    for path_index in range(selected_paths.shape[1]):
        nodes = selected_paths[:, path_index]
        active = nodes >= 0
        safe_nodes = nodes.clamp_min(0)
        indices = supports[safe_nodes]
        local = sensors.gather(1, indices).prod(dim=1)
        if path_index == 0:
            updated = local
        else:
            updated = nested_nand_sign(state, local)
        state = torch.where(active, updated, state)
    return state


class LateConditionedSharedMLP(nn.Module):
    """Ordinary sensor trunk with task identity entering after one nonlinearity."""

    def __init__(
        self,
        n_tasks: int,
        sensor_bits: int,
        width: int,
        hidden_layers: int,
        readout_cls: type[nn.Linear] = nn.Linear,
    ):
        super().__init__()
        self.sensor_layer = nn.Linear(sensor_bits, width)
        self.conditioned_layer = nn.Linear(width, width)
        self.task_embedding = nn.Embedding(n_tasks, width)
        self.hidden = nn.ModuleList(
            nn.Linear(width, width) for _ in range(hidden_layers - 2)
        )
        self.readout = readout_cls(width, 2)
        bound = 1.0 / math.sqrt(width + n_tasks)
        nn.init.uniform_(self.conditioned_layer.weight, -bound, bound)
        nn.init.uniform_(self.conditioned_layer.bias, -bound, bound)
        nn.init.uniform_(self.task_embedding.weight, -bound, bound)

    def forward(self, tasks: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.sensor_layer(sensors))
        hidden = F.relu(self.conditioned_layer(hidden) + self.task_embedding(tasks))
        for layer in self.hidden:
            hidden = F.relu(layer(hidden))
        return self.readout(hidden)


def _build_raw_model(
    config: HierarchicalParityConfig, n_tasks: int, width: int
) -> nn.Module:
    readout_cls = mup.MuReadout if config.parameterization == "mup" else nn.Linear
    if config.task_conditioning == "early":
        return EmbeddedSharedMLP(
            n_tasks,
            config.support_pool_bits,
            width,
            config.hidden_layers,
            readout_cls=readout_cls,
        )
    return LateConditionedSharedMLP(
        n_tasks,
        config.support_pool_bits,
        width,
        config.hidden_layers,
        readout_cls=readout_cls,
    )


def build_model(config: HierarchicalParityConfig, n_tasks: int) -> nn.Module:
    model = _build_raw_model(config, n_tasks, config.width)
    if config.parameterization == "standard":
        return model
    base = _build_raw_model(config, n_tasks, config.mup_base_width)
    delta = _build_raw_model(config, n_tasks, config.mup_delta_width)
    return mup.set_base_shapes(model, base, delta=delta)


def build_optimizer(
    model: nn.Module, config: HierarchicalParityConfig
) -> torch.optim.Optimizer:
    if config.parameterization == "mup":
        optimizer_class = mup.MuSGD if config.optimizer == "sgd" else mup.MuAdam
    else:
        optimizer_class = torch.optim.SGD if config.optimizer == "sgd" else torch.optim.Adam
    return optimizer_class(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def autocast_context(config: HierarchicalParityConfig, device: torch.device):
    if config.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def task_chunks(
    config: HierarchicalParityConfig,
    probabilities: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> Iterable[torch.Tensor]:
    n_tasks = len(probabilities)
    if config.sampling == "stratified":
        examples_per_task = config.batch_size // n_tasks
        tasks = torch.arange(n_tasks, device=device).repeat_interleave(examples_per_task)
        tasks = tasks[torch.randperm(len(tasks), generator=generator, device=device)]
        for start in range(0, config.batch_size, config.microbatch_size):
            yield tasks[start : start + config.microbatch_size]
        return
    remaining = config.batch_size
    while remaining:
        size = min(remaining, config.microbatch_size)
        yield torch.multinomial(probabilities, size, replacement=True, generator=generator)
        remaining -= size


def make_batch(
    config: HierarchicalParityConfig,
    tasks: torch.Tensor,
    generator: torch.Generator,
    paths: torch.Tensor,
    supports: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sensors = 2.0 * torch.randint(
        0,
        2,
        (len(tasks), config.support_pool_bits),
        generator=generator,
        device=tasks.device,
    ).float() - 1.0
    signs = target_signs(tasks, sensors, paths, supports)
    labels = (signs > 0).long()
    return sensors, labels, signs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    config: HierarchicalParityConfig,
    n_tasks: int,
    terminal_probabilities: np.ndarray,
    device: torch.device,
    paths: torch.Tensor,
    supports: torch.Tensor,
) -> dict[str, np.ndarray | float]:
    generator = torch.Generator(device=device.type).manual_seed(config.eval_seed)
    sensors = 2.0 * torch.randint(
        0,
        2,
        (config.eval_samples, config.support_pool_bits),
        generator=generator,
        device=device,
    ).float() - 1.0
    losses = np.empty(n_tasks, dtype=np.float64)
    accuracy = np.empty(n_tasks, dtype=np.float64)
    coefficients = np.empty(n_tasks, dtype=np.float64)
    entropies = np.empty(n_tasks, dtype=np.float64)
    for start in range(0, n_tasks, config.eval_task_chunk):
        stop = min(start + config.eval_task_chunk, n_tasks)
        tasks = torch.arange(start, stop, device=device).repeat_interleave(
            config.eval_samples
        )
        repeated_sensors = sensors.repeat(stop - start, 1)
        target = target_signs(tasks, repeated_sensors, paths, supports)
        labels = (target > 0).long()
        with autocast_context(config, device):
            logits = model(tasks, repeated_sensors)
        probabilities = logits.float().softmax(dim=-1)[:, 1]
        signed_probability = 2.0 * probabilities - 1.0
        per_example_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        per_example_loss /= math.log(2.0)
        chunk_size = stop - start
        target_matrix = target.reshape(chunk_size, config.eval_samples)
        prediction_matrix = signed_probability.reshape(chunk_size, config.eval_samples)
        target_mean = target_matrix.mean(dim=1, keepdim=True)
        centered = target_matrix - target_mean
        coefficient = (prediction_matrix * centered).mean(dim=1) / centered.square().mean(
            dim=1
        ).clamp_min(1e-8)
        positive_rate = (target_matrix > 0).float().mean(dim=1)
        safe_rate = positive_rate.clamp(1e-6, 1.0 - 1e-6)
        entropy = -safe_rate * torch.log2(safe_rate) - (
            1.0 - safe_rate
        ) * torch.log2(1.0 - safe_rate)
        losses[start:stop] = per_example_loss.reshape(
            chunk_size, config.eval_samples
        ).mean(dim=1).cpu().numpy()
        accuracy[start:stop] = (
            (logits.argmax(dim=-1) == labels)
            .reshape(chunk_size, config.eval_samples)
            .float()
            .mean(dim=1)
            .cpu()
            .numpy()
        )
        coefficients[start:stop] = coefficient.cpu().numpy()
        entropies[start:stop] = entropy.cpu().numpy()
    return {
        "task_losses_bits": losses,
        "task_accuracy": accuracy,
        "functional_coefficients": coefficients,
        "task_entropies_bits": entropies,
        "weighted_loss_bits": float(terminal_probabilities @ losses),
    }


def first_sustained_crossing(
    steps: np.ndarray, curve: np.ndarray, threshold: float, window: int = 5
) -> float | None:
    """Return the first threshold crossing sustained for several evaluations."""
    if window <= 0:
        raise ValueError("window must be positive")
    good = np.asarray(curve) >= threshold
    if len(good) < window:
        return None
    sustained = np.convolve(good.astype(np.int64), np.ones(window, dtype=np.int64), "valid")
    indices = np.flatnonzero(sustained == window)
    return float(steps[indices[0]]) if len(indices) else None


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


def exact_depth_entropies(max_depth: int) -> np.ndarray:
    """Exact target entropy at each depth under independent local parities."""
    positive_probability = 0.5
    entropies = [binary_entropy(positive_probability)]
    for _ in range(max_depth):
        positive_probability = 1.0 - 0.5 * positive_probability
        entropies.append(binary_entropy(positive_probability))
    return np.asarray(entropies, dtype=np.float64)


def fit_log_clock(
    probabilities: np.ndarray, times: np.ndarray
) -> dict[str, float | None]:
    selected = np.isfinite(times) & (probabilities > 0)
    if selected.sum() < 3 or len(np.unique(probabilities[selected])) < 2:
        return {"gamma": None, "intercept": None, "r2": None}
    x = np.log(probabilities[selected])
    y = np.log(times[selected])
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    total = np.square(y - y.mean()).sum()
    residual = np.square(y - prediction).sum()
    return {
        "gamma": float(-slope),
        "intercept": float(intercept),
        "r2": float(1.0 - residual / total) if total > 0 else None,
    }


def fit_depth_adjusted_clock(
    demand: np.ndarray, depths: np.ndarray, times: np.ndarray
) -> dict[str, float | None]:
    """Fit log T = intercept - gamma log demand + kappa * depth."""
    selected = np.isfinite(times) & (demand > 0)
    if selected.sum() < 4 or len(np.unique(depths[selected])) < 2:
        return {
            "gamma": None,
            "depth_kappa": None,
            "per_depth_multiplier": None,
            "intercept": None,
            "r2": None,
        }
    design = np.column_stack(
        (
            np.ones(selected.sum()),
            np.log(demand[selected]),
            depths[selected].astype(np.float64),
        )
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return {
            "gamma": None,
            "depth_kappa": None,
            "per_depth_multiplier": None,
            "intercept": None,
            "r2": None,
        }
    target = np.log(times[selected])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    total = np.square(target - target.mean()).sum()
    residual = np.square(target - prediction).sum()
    return {
        "gamma": float(-coefficients[1]),
        "depth_kappa": float(coefficients[2]),
        "per_depth_multiplier": float(math.exp(coefficients[2])),
        "intercept": float(coefficients[0]),
        "r2": float(1.0 - residual / total) if total > 0 else None,
    }


def constant_clock(
    demand: np.ndarray, times: np.ndarray
) -> dict[str, float | int | None]:
    selected = np.isfinite(times) & (demand > 0)
    if not selected.any():
        return {"completed": 0, "constant": None, "median_relative_error": None}
    constant = float(np.median(demand[selected] * times[selected]))
    prediction = constant / demand[selected]
    error = np.abs(prediction - times[selected]) / times[selected]
    return {
        "completed": int(selected.sum()),
        "constant": constant,
        "median_relative_error": float(np.median(error)),
    }


def parameter_counts(
    model: nn.Module, config: HierarchicalParityConfig, n_tasks: int
) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    task_conditioning = n_tasks * config.width
    return {
        "total": int(total),
        "task_conditioning": int(task_conditioning),
        "shared_trunk": int(total - task_conditioning),
    }


def analyze(
    config: HierarchicalParityConfig,
    hierarchy: dict[str, object],
    steps: np.ndarray,
    losses: np.ndarray,
    accuracy: np.ndarray,
    coefficients: np.ndarray,
    model_parameter_counts: dict[str, int],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    depths = np.asarray(hierarchy["node_depths"], dtype=np.int64)
    parents = np.asarray(hierarchy["parents"], dtype=np.int64)
    paths = np.asarray(hierarchy["paths"], dtype=np.int64)
    marginal_demand = np.asarray(hierarchy["marginal_demand"], dtype=np.float64)
    terminal = np.asarray(hierarchy["terminal_probabilities"], dtype=np.float64)
    n_tasks = len(depths)
    crossing_window = min(5, len(steps))
    events: list[dict[str, object]] = []
    crossing_80 = np.full(n_tasks, np.nan, dtype=np.float64)
    widths: list[float] = []
    for node in range(n_tasks):
        crossing = {
            fraction: first_sustained_crossing(
                steps, coefficients[:, node], fraction / 100.0, crossing_window
            )
            for fraction in (20, 50, 80)
        }
        if crossing[80] is not None:
            crossing_80[node] = float(crossing[80])
        relative_width = None
        if all(crossing[fraction] is not None for fraction in (20, 50, 80)):
            denominator = max(float(crossing[50]), float(config.eval_every))
            relative_width = (
                float(crossing[80]) - float(crossing[20])
            ) / denominator
            widths.append(relative_width)
        events.append(
            {
                "node": node,
                "parent": int(parents[node]),
                "depth": int(depths[node]),
                "terminal_probability": float(terminal[node]),
                "closure_demand": float(marginal_demand[node]),
                "time_20pct": crossing[20],
                "time_50pct": crossing[50],
                "time_80pct": crossing[80],
                "relative_width_20_to_80": relative_width,
                "final_functional_coefficient": float(coefficients[-1, node]),
                "final_accuracy": float(accuracy[-1, node]),
            }
        )

    acquired = coefficients >= 0.8
    exact_entropies = exact_depth_entropies(config.max_depth)[depths]
    direct_reconstruction = np.sum(
        terminal[None, :] * exact_entropies[None, :] * (~acquired), axis=1
    )
    path_available = np.ones_like(acquired, dtype=bool)
    for node in range(n_tasks):
        node_path = paths[node][paths[node] >= 0]
        path_available[:, node] = acquired[:, node_path].all(axis=1)
    poset_reconstruction = np.sum(
        terminal[None, :] * exact_entropies[None, :] * (~path_available), axis=1
    )

    completed = np.isfinite(crossing_80)
    acquired_nodes = set(np.flatnonzero(completed).tolist())
    ordered = np.argsort(-marginal_demand)
    prefix = ordered[: int(completed.sum())]
    prefix_fraction = (
        float(sum(int(node) in acquired_nodes for node in prefix) / len(prefix))
        if len(prefix)
        else None
    )
    timing_prerequisite_violations = 0
    timing_comparable_edges = 0
    for node in range(n_tasks):
        parent = int(parents[node])
        if parent < 0 or not np.isfinite(crossing_80[node]):
            continue
        timing_comparable_edges += 1
        if not np.isfinite(crossing_80[parent]) or crossing_80[node] < crossing_80[parent]:
            timing_prerequisite_violations += 1
    final_comparable_edges = 0
    final_prerequisite_violations = 0
    for node in range(n_tasks):
        parent = int(parents[node])
        if parent < 0 or not acquired[-1, node]:
            continue
        final_comparable_edges += 1
        if not acquired[-1, parent]:
            final_prerequisite_violations += 1

    closure_fit = fit_log_clock(marginal_demand, crossing_80)
    direct_fit = fit_log_clock(terminal, crossing_80)
    closure_depth_fit = fit_depth_adjusted_clock(
        marginal_demand, depths, crossing_80
    )
    direct_depth_fit = fit_depth_adjusted_clock(terminal, depths, crossing_80)
    closure_clock = constant_clock(marginal_demand, crossing_80)
    direct_clock = constant_clock(terminal, crossing_80)
    clock_by_depth = {
        str(depth): constant_clock(
            marginal_demand[depths == depth], crossing_80[depths == depth]
        )
        for depth in range(config.max_depth + 1)
    }
    completed_by_depth = {
        str(depth): int((completed & (depths == depth)).sum())
        for depth in range(config.max_depth + 1)
    }
    nodes_by_depth = {
        str(depth): int((depths == depth).sum())
        for depth in range(config.max_depth + 1)
    }
    closure_spearman = None
    if completed.sum() >= 3 and len(np.unique(marginal_demand[completed])) >= 2:
        closure_spearman = float(
            spearmanr(
                np.log(marginal_demand[completed]), crossing_80[completed]
            ).statistic
        )
    direct_spearman = None
    if completed.sum() >= 3 and len(np.unique(terminal[completed])) >= 2:
        direct_spearman = float(
            spearmanr(np.log(terminal[completed]), crossing_80[completed]).statistic
        )
    summary: dict[str, object] = {
        "config": asdict(config),
        "n_tasks": n_tasks,
        "nodes_by_depth": nodes_by_depth,
        "demand_law_diagnostics": hierarchy["demand_law_diagnostics"],
        "parameter_counts": model_parameter_counts,
        "terminal_frequency_range": float(terminal.max() / terminal.min()),
        "closure_demand_range": float(marginal_demand.max() / marginal_demand.min()),
        "exact_target_entropy_bits_by_depth": exact_depth_entropies(
            config.max_depth
        ).tolist(),
        "completed_tasks": int(completed.sum()),
        "completed_by_depth": completed_by_depth,
        "final_acquired_tasks": int(acquired[-1].sum()),
        "final_solved_tasks_99pct": int((accuracy[-1] >= 0.99).sum()),
        "final_weighted_loss_bits": float(terminal @ losses[-1]),
        "final_mean_loss_bits": float(losses[-1].mean()),
        "final_direct_reconstructed_loss_bits": float(direct_reconstruction[-1]),
        "final_poset_reconstructed_loss_bits": float(poset_reconstruction[-1]),
        "closure_demand_time_spearman": closure_spearman,
        "direct_frequency_time_spearman": direct_spearman,
        "closure_log_clock_fit": closure_fit,
        "direct_log_clock_fit": direct_fit,
        "depth_adjusted_closure_clock_fit": closure_depth_fit,
        "depth_adjusted_direct_clock_fit": direct_depth_fit,
        "closure_constant_clock": closure_clock,
        "direct_constant_clock": direct_clock,
        "closure_constant_clock_by_depth": clock_by_depth,
        "median_relative_width_20_to_80": float(np.median(widths)) if widths else None,
        "closure_demand_prefix_fraction": prefix_fraction,
        "timing_prerequisite_violations": timing_prerequisite_violations,
        "timing_comparable_edges": timing_comparable_edges,
        "final_prerequisite_violations": final_prerequisite_violations,
        "final_comparable_edges": final_comparable_edges,
        "event_records": events,
    }
    return summary, direct_reconstruction, poset_reconstruction


def plot_run(
    output: Path,
    config: HierarchicalParityConfig,
    hierarchy: dict[str, object],
    steps: np.ndarray,
    losses: np.ndarray,
    weighted: np.ndarray,
    coefficients: np.ndarray,
    direct_reconstruction: np.ndarray,
    poset_reconstruction: np.ndarray,
    summary: dict[str, object],
) -> None:
    depths = np.asarray(hierarchy["node_depths"], dtype=np.int64)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    stride = max(1, len(depths) // 128)
    for node in range(0, len(depths), stride):
        axes[0, 0].plot(
            steps, losses[:, node], color="0.55", alpha=0.14, linewidth=0.55
        )
    axes[0, 0].plot(steps, weighted, color="crimson", linewidth=2.1, label="neural")
    axes[0, 0].plot(
        steps,
        direct_reconstruction,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="direct acquisition",
    )
    axes[0, 0].plot(
        steps,
        poset_reconstruction,
        color="royalblue",
        linestyle=":",
        linewidth=1.5,
        label="ancestor-closed",
    )
    axes[0, 0].set(xlabel="optimizer step", ylabel="cross-entropy (bits)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    colors = {
        depth: plt.cm.viridis(depth / max(config.max_depth, 1))
        for depth in range(config.max_depth + 1)
    }
    completed = [
        event for event in summary["event_records"] if event["time_80pct"] is not None
    ]
    for depth in range(config.max_depth + 1):
        selected = [event for event in completed if event["depth"] == depth]
        if not selected:
            continue
        axes[0, 1].scatter(
            [event["terminal_probability"] for event in selected],
            [event["time_80pct"] for event in selected],
            s=22,
            alpha=0.75,
            color=colors[depth],
            label=f"depth {depth}",
        )
    if completed:
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_yscale("log")
    direct_fit = summary["direct_log_clock_fit"]
    if completed and direct_fit["gamma"] is not None:
        shown_probability = np.geomspace(
            min(float(event["terminal_probability"]) for event in completed),
            max(float(event["terminal_probability"]) for event in completed),
            100,
        )
        prediction = np.exp(float(direct_fit["intercept"])) * shown_probability ** (
            -float(direct_fit["gamma"])
        )
        axes[0, 1].plot(
            shown_probability,
            prediction,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="frequency-only fit",
        )
    if completed:
        axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].set(xlabel="direct task probability", ylabel="80% acquisition step")

    for depth in range(config.max_depth + 1):
        nodes = np.flatnonzero(depths == depth)
        shown = nodes
        if len(shown) > 24:
            shown = shown[np.linspace(0, len(shown) - 1, 24, dtype=int)]
        for node in shown:
            axes[1, 0].plot(
                steps,
                coefficients[:, node],
                color=colors[depth],
                alpha=0.24,
                linewidth=0.65,
            )
    axes[1, 0].axhline(0.2, color="black", linestyle=":", linewidth=0.7)
    axes[1, 0].axhline(0.8, color="black", linestyle=":", linewidth=0.7)
    axes[1, 0].set(
        xlabel="optimizer step",
        ylabel="centered target coefficient",
        ylim=(-0.15, 1.15),
    )

    for depth in range(config.max_depth + 1):
        count = (coefficients[:, depths == depth] >= 0.8).sum(axis=1)
        axes[1, 1].step(
            steps,
            count,
            where="post",
            color=colors[depth],
            linewidth=1.7,
            label=f"depth {depth} / {(depths == depth).sum()}",
        )
    axes[1, 1].set(xlabel="optimizer step", ylabel="acquired tasks (coefficient >= 0.8)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    depth_fit = summary["depth_adjusted_direct_clock_fit"]
    gamma = depth_fit["gamma"]
    multiplier = depth_fit["per_depth_multiplier"]
    fit_label = (
        f"gamma={float(gamma):.2f}, x{float(multiplier):.2f}/depth"
        if gamma is not None and multiplier is not None
        else "timing fit unavailable"
    )
    figure.suptitle(
        f"hierarchical fixed-pool parity, Q={len(depths)}, depth={config.max_depth}; "
        f"completed {summary['completed_tasks']}/{len(depths)}; {fit_label}"
    )
    figure.tight_layout()
    figure.savefig(output / "trajectory.png", dpi=180)
    figure.savefig(output / "trajectory.pdf")
    plt.close(figure)


def run_name(config: HierarchicalParityConfig) -> str:
    demand_law = (
        f"lawrho-beta{config.beta:g}-"
        if config.demand_law == "rho_beta"
        else f"dmass{config.depth_mass_decay:g}-exp{config.root_frequency_exponent:g}-"
    )
    parameterization = (
        f"-mup{config.mup_base_width}x{config.mup_delta_width}"
        if config.parameterization == "mup"
        else ""
    )
    checkpointing = (
        f"-ckpt{config.checkpoint_every}" if config.checkpoint_every else ""
    )
    return (
        f"qtree-r{config.base_tasks}-b{config.branching_factor}-d{config.max_depth}-"
        f"{demand_law}"
        f"pool{config.support_pool_bits}-ps{config.support_seed}-{config.task_conditioning}-"
        f"w{config.width}-h{config.hidden_layers}{parameterization}-"
        f"{config.optimizer}{config.learning_rate:g}-"
        f"b{config.batch_size}-mb{config.microbatch_size}-m{config.model_seed}-"
        f"r{config.rank_seed}-ds{config.data_seed}-es{config.eval_seed}-"
        f"{config.sampling}-n{config.steps}{checkpointing}"
    ).replace(".", "p")


def save_training_checkpoint(
    output: Path,
    config: HierarchicalParityConfig,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    records: list[dict[str, np.ndarray | float]],
    record_steps: list[int],
) -> Path:
    checkpoint_path = output / f"checkpoint_step_{step:07d}.pt"
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    payload = {
        "format_version": 1,
        "step": step,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_generator_state": generator.get_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "record_steps": list(record_steps),
        "records": records,
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: Path,
    config: HierarchicalParityConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, list[dict[str, np.ndarray | float]], list[int]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    # RNG states must remain CPU byte tensors. Model and optimizer loading copy
    # their state to the devices of the already-constructed parameters.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")

    saved_config = dict(checkpoint["config"])
    current_config = asdict(config)
    # Microbatching changes only how one fixed effective batch is partitioned
    # for forward/backward passes.  It is a permitted continuation branch, but
    # remains recorded in the new run's status and checkpoint metadata.
    for ignored in ("steps", "output_dir", "resume_checkpoint", "microbatch_size"):
        saved_config.pop(ignored, None)
        current_config.pop(ignored, None)
    if saved_config != current_config:
        differing = sorted(
            key
            for key in set(saved_config) | set(current_config)
            if saved_config.get(key) != current_config.get(key)
        )
        raise ValueError(
            "resume checkpoint configuration differs in: " + ", ".join(differing)
        )

    step = int(checkpoint["step"])
    if step >= config.steps:
        raise ValueError(
            f"resume checkpoint step {step} must be below target steps {config.steps}"
        )
    record_steps = [int(value) for value in checkpoint["record_steps"]]
    if not record_steps or record_steps[-1] != step:
        raise ValueError("resume checkpoint evaluation history does not end at its step")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    generator.set_state(checkpoint["training_generator_state"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    cuda_states = checkpoint.get("cuda_random_states")
    if device.type == "cuda" and cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return step, list(checkpoint["records"]), record_steps


def run(config: HierarchicalParityConfig) -> Path:
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
    hierarchy = build_hierarchy(config)
    supports_numpy = build_supports(config, hierarchy)
    depths = np.asarray(hierarchy["node_depths"], dtype=np.int64)
    n_tasks = len(depths)
    terminal = np.asarray(hierarchy["terminal_probabilities"], dtype=np.float64)
    probabilities = torch.tensor(terminal, dtype=torch.float32, device=device)
    paths = torch.tensor(hierarchy["paths"], dtype=torch.long, device=device)
    supports = torch.tensor(supports_numpy, dtype=torch.long, device=device)
    model = build_model(config, n_tasks).to(device)
    optimizer = build_optimizer(model, config)

    generator = torch.Generator(device=device.type).manual_seed(config.data_seed)
    start_step = 0
    records: list[dict[str, np.ndarray | float]] = []
    record_steps: list[int] = []
    resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint else None
    if resume_path is not None:
        start_step, records, record_steps = load_training_checkpoint(
            resume_path,
            config,
            model,
            optimizer,
            generator,
            device,
        )

    output = Path(config.output_dir) / run_name(config)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "status.json").open("w") as handle:
        json.dump(
            {
                "status": "running",
                "config": asdict(config),
                "resumed_from_step": start_step if resume_path is not None else None,
            },
            handle,
            indent=2,
        )

    def record(step: int) -> None:
        model.eval()
        records.append(
            evaluate(
                model,
                config,
                n_tasks,
                terminal,
                device,
                paths,
                supports,
            )
        )
        record_steps.append(step)
        model.train()

    try:
        if resume_path is None:
            record(0)
        else:
            print(f"resumed={resume_path} step={start_step}", flush=True)
        for step in range(start_step + 1, config.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for tasks in task_chunks(config, probabilities, generator, device):
                sensors, labels, _ = make_batch(
                    config, tasks, generator, paths, supports
                )
                with autocast_context(config, device):
                    logits = model(tasks, sensors)
                    per_example = F.cross_entropy(logits, labels, reduction="none")
                    if config.sampling == "iid":
                        loss = per_example.sum() / config.batch_size
                    else:
                        examples_per_task = config.batch_size // n_tasks
                        loss = (
                            per_example * probabilities[tasks] / examples_per_task
                        ).sum()
                loss.backward()
            optimizer.step()
            if step % config.eval_every == 0 or step == config.steps:
                record(step)
                progress_every = max(config.eval_every, config.steps // 20)
                if step == config.steps or step % progress_every < config.eval_every:
                    latest = records[-1]
                    acquired = int(
                        (np.asarray(latest["functional_coefficients"]) >= 0.8).sum()
                    )
                    print(
                        f"step={step} weighted_bits={latest['weighted_loss_bits']:.6f} "
                        f"acquired={acquired}/{n_tasks}",
                        flush=True,
                    )
            if config.checkpoint_every and step % config.checkpoint_every == 0:
                checkpoint_path = save_training_checkpoint(
                    output,
                    config,
                    step,
                    model,
                    optimizer,
                    generator,
                    records,
                    record_steps,
                )
                with (output / "status.json").open("w") as handle:
                    json.dump(
                        {
                            "status": "running",
                            "config": asdict(config),
                            "step": step,
                            "latest_checkpoint": checkpoint_path.name,
                        },
                        handle,
                        indent=2,
                    )
                print(f"checkpoint={checkpoint_path}", flush=True)

        steps = np.asarray(record_steps, dtype=np.int64)
        losses = np.stack([record["task_losses_bits"] for record in records])
        accuracy = np.stack([record["task_accuracy"] for record in records])
        coefficients = np.stack(
            [record["functional_coefficients"] for record in records]
        )
        empirical_entropies = np.stack(
            [record["task_entropies_bits"] for record in records]
        )
        weighted = np.asarray(
            [record["weighted_loss_bits"] for record in records], dtype=np.float64
        )
        summary, direct_reconstruction, poset_reconstruction = analyze(
            config,
            hierarchy,
            steps,
            losses,
            accuracy,
            coefficients,
            parameter_counts(model, config, n_tasks),
        )
        summary["optimizer_diagnostics"] = {
            "parameterization": config.parameterization,
            "global_learning_rate": config.learning_rate,
            "parameter_group_learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
        }
        summary["resume_diagnostics"] = {
            "checkpoint": str(resume_path) if resume_path is not None else None,
            "step": start_step if resume_path is not None else None,
        }
        np.savez_compressed(
            output / "trajectory.npz",
            steps=steps,
            task_losses_bits=losses,
            task_accuracy=accuracy,
            functional_coefficients=coefficients,
            empirical_task_entropies_bits=empirical_entropies,
            weighted_losses_bits=weighted,
            direct_reconstructed_losses_bits=direct_reconstruction,
            poset_reconstructed_losses_bits=poset_reconstruction,
            node_depths=depths,
            parents=np.asarray(hierarchy["parents"]),
            root_ids=np.asarray(hierarchy["root_ids"]),
            rank_by_root=np.asarray(hierarchy["rank_by_root"]),
            root_ranks=np.asarray(hierarchy["root_ranks"]),
            paths=np.asarray(hierarchy["paths"]),
            supports=supports_numpy,
            depth_masses=np.asarray(hierarchy["depth_masses"]),
            marginal_demand=np.asarray(hierarchy["marginal_demand"]),
            terminal_probabilities=terminal,
        )
        with (output / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
        plot_run(
            output,
            config,
            hierarchy,
            steps,
            losses,
            weighted,
            coefficients,
            direct_reconstruction,
            poset_reconstruction,
            summary,
        )
        with (output / "status.json").open("w") as handle:
            json.dump({"status": "complete", "config": asdict(config)}, handle, indent=2)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"config", "event_records"}
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
