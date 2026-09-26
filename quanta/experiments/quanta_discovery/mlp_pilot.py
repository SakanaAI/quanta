"""Exact-GD Quanta Discovery calibration for an ordinary shared MLP."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from quanta.experiments.quanta_discovery.priority import (
    EPSILON,
    _factorization_summary,
    _support_improving_residual,
)
from quanta.experiments.scaling_laws.compositional_mlp import (
    CompositionalMLPConfig,
    build_model,
    evaluate,
    factor_support_pool_bits,
    factor_specs,
    make_sensor_batch,
    task_probabilities,
    validate_config,
)


@dataclass(frozen=True)
class MLPDiscoveryPilotConfig:
    n_tasks: int = 8
    function_group_size: int = 1
    parity_degree: int = 3
    degree_b: int = 0
    degree_c: int = 0
    subgroup_size: int = 1
    composition: str = "parallel_nand"
    event_granularity: str = "task"
    support_pool_bits: int = 16
    samples_per_task: int = 8
    feature_mode: str = "sparse"
    frequency_exponent: float = 0.5
    width: int = 64
    hidden_layers: int = 2
    learning_rate: float = 0.1
    steps: int = 8_000
    checkpoint_every: int = 100
    max_components: int = 12
    alternating_steps: int = 50
    minimum_component_gain_fraction: float = 0.01
    model_seed: int = 0
    rank_seed: int = 0
    support_seed: int = 0
    train_seed: int = 1_000
    eval_seed: int = 10_000
    output_dir: str = ".experiments/quanta_discovery/mlp_flat_parity_cpu_v1"


def factorize_priority_field(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    max_components: int,
    alternating_steps: int,
    minimum_component_gain_fraction: float = 0.05,
    interval_steps: np.ndarray | None = None,
    nonnegative_curves: bool = True,
) -> dict[str, Any]:
    """Pilot-only factorization supporting signed interaction features."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise ValueError("priority field and event weights have incompatible shapes")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("event weights must be finite, nonnegative, and nonzero")
    weights = weights / weights.sum()
    steps = (
        np.arange(1, values.shape[1] + 1, dtype=np.int64)
        if interval_steps is None
        else np.asarray(interval_steps, dtype=np.int64)
    )
    if steps.shape != (values.shape[1],):
        raise ValueError("interval_steps must have one entry per feature")

    residual = values.copy()
    original_sse = float(np.sum(weights[:, None] * np.square(residual)))
    supports: list[np.ndarray] = []
    curves: list[np.ndarray] = []
    improvements: list[float] = []
    gain_fractions: list[float] = []
    for component in range(int(max_components)):
        candidate = _fit_pilot_binary_component(
            residual,
            weights,
            alternating_steps=int(alternating_steps),
            seed=104_729 + 1_009 * component,
            nonnegative_curve=bool(nonnegative_curves),
        )
        if candidate is None:
            break
        support, curve = candidate
        estimate = support[:, None] * curve[None]
        before = float(np.sum(weights[:, None] * np.square(residual)))
        after = float(np.sum(weights[:, None] * np.square(residual - estimate)))
        improvement = before - after
        gain_fraction = improvement / max(original_sse, EPSILON)
        if gain_fraction + EPSILON < float(minimum_component_gain_fraction):
            break
        residual -= estimate
        supports.append(support)
        curves.append(curve)
        improvements.append(improvement)
        gain_fractions.append(gain_fraction)

    support_array = (
        np.stack(supports, axis=1)
        if supports
        else np.empty((values.shape[0], 0), dtype=bool)
    )
    curve_array = (
        np.stack(curves)
        if curves
        else np.empty((0, values.shape[1]), dtype=np.float64)
    )
    return {
        "support": support_array,
        "temporal_priority": curve_array,
        "residual": residual,
        "improvements": improvements,
        "gain_fractions_of_original_sse": gain_fractions,
        "summary": _factorization_summary(
            values, residual, support_array, curve_array, weights, steps
        ),
    }


def _fit_pilot_binary_component(
    residual: np.ndarray,
    weights: np.ndarray,
    *,
    alternating_steps: int,
    seed: int,
    nonnegative_curve: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    weighted = residual * np.sqrt(weights[:, None])
    _, _, right = np.linalg.svd(weighted, full_matrices=False)
    generator = np.random.default_rng(int(seed))
    if nonnegative_curve:
        initializations = (
            np.maximum(weights @ residual, 0.0),
            np.maximum(right[0], 0.0),
            np.maximum(-right[0], 0.0),
            generator.random(residual.shape[1]),
        )
        row_curves = np.maximum(residual, 0.0)
    else:
        initializations = (
            weights @ residual,
            right[0],
            -right[0],
            generator.standard_normal(residual.shape[1]),
        )
        row_curves = residual
    row_energy = weights * np.square(row_curves).sum(axis=1)
    row_order = np.argsort(-row_energy, kind="stable")
    initializations += tuple(
        row_curves[index]
        for index in row_order[: min(len(row_order), 32)]
        if row_energy[index] > EPSILON
    )
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_objective = math.inf
    for initial in initializations:
        curve = np.asarray(initial, dtype=np.float64)
        if float(np.square(curve).sum()) <= EPSILON:
            continue
        support = np.zeros(residual.shape[0], dtype=bool)
        for _ in range(int(alternating_steps)):
            updated_support = _support_improving_residual(residual, curve)
            active_mass = float(weights @ updated_support)
            if active_mass <= EPSILON:
                break
            updated_curve = np.sum(
                weights[:, None] * updated_support[:, None] * residual, axis=0
            ) / active_mass
            if nonnegative_curve:
                updated_curve = np.maximum(updated_curve, 0.0)
            if float(np.square(updated_curve).sum()) <= EPSILON:
                break
            converged = np.array_equal(updated_support, support) and np.allclose(
                updated_curve, curve, rtol=1.0e-8, atol=1.0e-10
            )
            support, curve = updated_support, updated_curve
            if converged:
                break
        support = _support_improving_residual(residual, curve)
        if not np.any(support):
            continue
        objective = float(
            np.sum(
                weights[:, None]
                * np.square(residual - support[:, None] * curve[None])
            )
        )
        if objective < best_objective:
            best_objective = objective
            best = support, curve
    return best


def _mlp_config(config: MLPDiscoveryPilotConfig) -> CompositionalMLPConfig:
    return validate_config(
        CompositionalMLPConfig(
            n_tasks=config.n_tasks,
            group_size=config.function_group_size,
            subgroup_size=config.subgroup_size,
            degree_a=config.parity_degree,
            degree_b=config.degree_b,
            degree_c=config.degree_c,
            composition=config.composition,
            event_scope="primitive",
            support_layout="shared_pool",
            support_pool_bits=config.support_pool_bits,
            support_seed=config.support_seed,
            task_conditioning="onehot",
            sensor_bits=0,
            feature_mode=config.feature_mode,
            frequency_exponent=config.frequency_exponent,
            rank_seed=config.rank_seed,
            model_seed=config.model_seed,
            data_seed=config.train_seed,
            eval_seed=config.eval_seed,
            width=config.width,
            hidden_layers=config.hidden_layers,
            optimizer="sgd",
            learning_rate=config.learning_rate,
            weight_decay=0.0,
            batch_size=config.n_tasks * config.samples_per_task,
            microbatch_size=config.n_tasks * config.samples_per_task,
            steps=config.steps,
            eval_every=config.checkpoint_every,
            eval_samples_per_task=config.samples_per_task,
            eval_task_chunk=config.n_tasks,
            sampling="stratified",
            precision="fp32",
            device="cpu",
            output_dir=config.output_dir,
        )
    )


def _fixed_panel(
    config: CompositionalMLPConfig,
    *,
    samples_per_task: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    relevant_bits = sum(spec.degree for spec in factor_specs(config))
    patterns = torch.tensor(
        [
            [1.0 if mask & (1 << bit) else -1.0 for bit in range(relevant_bits)]
            for mask in range(2**relevant_bits)
        ]
    )
    if samples_per_task % len(patterns):
        raise ValueError("samples_per_task must be divisible by the parity truth table")
    raw_one_task = patterns.repeat(samples_per_task // len(patterns), 1)
    tasks = torch.arange(config.n_tasks).repeat_interleave(samples_per_task)
    raw = raw_one_task.repeat(config.n_tasks, 1)
    generator = torch.Generator().manual_seed(int(seed))
    sensors, labels, _ = make_sensor_batch(config, tasks, generator, raw=raw)
    return tasks, sensors, labels


def _expected_primitive_supports(
    config: CompositionalMLPConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in factor_specs(config):
        for group in range(config.n_tasks // spec.sharing):
            start = group * spec.sharing
            records.append(
                {
                    "role": spec.name,
                    "group": group,
                    "tasks": list(range(start, start + spec.sharing)),
                }
            )
    return records


def _conditional_losses(
    model: torch.nn.Module,
    tasks: torch.Tensor,
    sensors: torch.Tensor,
    labels: torch.Tensor,
    n_tasks: int,
) -> torch.Tensor:
    logits = model(tasks, sensors)
    losses = F.cross_entropy(logits, labels, reduction="none")
    return torch.stack([losses[tasks == task].mean() for task in range(n_tasks)])


def _task_gradients(
    model: torch.nn.Module,
    tasks: torch.Tensor,
    sensors: torch.Tensor,
    labels: torch.Tensor,
    n_tasks: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    parameters = tuple(model.parameters())
    gradients: list[tuple[torch.Tensor, ...]] = []
    for task in range(n_tasks):
        model.zero_grad(set_to_none=True)
        mask = tasks == task
        loss = F.cross_entropy(model(tasks[mask], sensors[mask]), labels[mask])
        observed = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradients.append(
            tuple(
                torch.zeros_like(parameter) if value is None else value.detach()
                for parameter, value in zip(parameters, observed)
            )
        )
    return tuple(gradients)


def _example_gradients(
    model: torch.nn.Module,
    tasks: torch.Tensor,
    sensors: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    parameters = tuple(model.parameters())
    gradients: list[tuple[torch.Tensor, ...]] = []
    for event in range(len(tasks)):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(tasks[event : event + 1], sensors[event : event + 1]),
            labels[event : event + 1],
        )
        observed = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradients.append(
            tuple(
                torch.zeros_like(parameter) if value is None else value.detach()
                for parameter, value in zip(parameters, observed)
            )
        )
    return tuple(gradients)


def _discovery_events(
    model: torch.nn.Module,
    panel: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    probabilities: np.ndarray,
    granularity: str,
) -> tuple[tuple[tuple[torch.Tensor, ...], ...], np.ndarray, np.ndarray]:
    tasks, sensors, labels = panel
    if granularity == "task":
        gradients = _task_gradients(
            model, tasks, sensors, labels, len(probabilities)
        )
        return gradients, probabilities.copy(), np.arange(len(probabilities))
    if granularity == "example":
        gradients = _example_gradients(model, tasks, sensors, labels)
        event_tasks = tasks.cpu().numpy().astype(np.int64)
        counts = np.bincount(event_tasks, minlength=len(probabilities))
        weights = probabilities[event_tasks] / counts[event_tasks]
        return gradients, weights, event_tasks
    raise ValueError("event_granularity must be 'task' or 'example'")


def _weighted_direction(
    gradients: tuple[tuple[torch.Tensor, ...], ...],
    probabilities: np.ndarray,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        sum(float(probabilities[task]) * gradients[task][parameter]
            for task in range(len(gradients)))
        for parameter in range(len(gradients[0]))
    )


def _gradient_inner_products(
    gradients: tuple[tuple[torch.Tensor, ...], ...],
    direction: tuple[torch.Tensor, ...],
) -> np.ndarray:
    return np.asarray(
        [
            float(sum((left * right).sum() for left, right in zip(task, direction)))
            for task in gradients
        ],
        dtype=np.float64,
    )


def _gradient_gram(
    gradients: tuple[tuple[torch.Tensor, ...], ...],
) -> np.ndarray:
    n_events = len(gradients)
    gram = np.empty((n_events, n_events), dtype=np.float64)
    for left in range(n_events):
        for right in range(left, n_events):
            value = float(
                sum(
                    (left_value * right_value).sum()
                    for left_value, right_value in zip(
                        gradients[left], gradients[right]
                    )
                )
            )
            gram[left, right] = value
            gram[right, left] = value
    return gram


def _group_gradient_grams(
    gradients: tuple[tuple[torch.Tensor, ...], ...],
    config: CompositionalMLPConfig,
) -> dict[str, np.ndarray]:
    """Resolve gradient geometry by ordinary-MLP parameter role."""

    def inner_product(
        left: tuple[torch.Tensor, ...],
        right: tuple[torch.Tensor, ...],
        group: str,
    ) -> float:
        if group == "task_conditioning":
            return float(
                (left[0][:, : config.n_tasks] * right[0][:, : config.n_tasks]).sum()
            )
        if group == "sensor_input":
            return float(
                (left[0][:, config.n_tasks :] * right[0][:, config.n_tasks :]).sum()
            )
        if group == "hidden_shared":
            return float(
                sum((a * b).sum() for a, b in zip(left[1:-2], right[1:-2]))
            )
        if group == "readout":
            return float(
                sum((a * b).sum() for a, b in zip(left[-2:], right[-2:]))
            )
        if group.startswith("sensor_role_"):
            requested = group.removeprefix("sensor_role_")
            offset = config.n_tasks
            for spec in factor_specs(config):
                span = factor_support_pool_bits(config, spec)
                if spec.name == requested:
                    return float(
                        (
                            left[0][:, offset : offset + span]
                            * right[0][:, offset : offset + span]
                        ).sum()
                    )
                offset += span
        raise ValueError(f"unknown parameter group {group!r}")

    output: dict[str, np.ndarray] = {}
    groups = ["task_conditioning", "sensor_input", "hidden_shared", "readout"]
    groups.extend(f"sensor_role_{spec.name}" for spec in factor_specs(config))
    for group in groups:
        gram = np.empty((len(gradients), len(gradients)), dtype=np.float64)
        for left in range(len(gradients)):
            for right in range(left, len(gradients)):
                value = inner_product(gradients[left], gradients[right], group)
                gram[left, right] = value
                gram[right, left] = value
        output[group] = gram
    return output


def _interaction_factorization_summary(
    interaction_credit: np.ndarray,
    weights: np.ndarray,
    *,
    config: MLPDiscoveryPilotConfig,
    n_events: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    flattened = interaction_credit.reshape(n_events, -1)
    fit = factorize_priority_field(
        flattened,
        weights,
        max_components=config.max_components,
        alternating_steps=config.alternating_steps,
        minimum_component_gain_fraction=config.minimum_component_gain_fraction,
        interval_steps=np.arange(1, flattened.shape[1] + 1),
        nonnegative_curves=False,
    )
    supports = [
        np.flatnonzero(fit["support"][:, candidate]).astype(int).tolist()
        for candidate in range(fit["support"].shape[1])
    ]
    singleton_tasks = sorted(
        support[0] for support in supports if len(support) == 1
    )
    summary = {
        "candidate_count": int(fit["support"].shape[1]),
        "field_explained_fraction": float(
            fit["summary"]["field_explained_fraction"]
        ),
        "supports": supports,
        "singleton_tasks": singleton_tasks,
        "singleton_task_count": len(singleton_tasks),
        "universal_support_count": sum(
            len(support) == n_events for support in supports
        ),
        "gain_fractions_of_original_sse": fit[
            "gain_fractions_of_original_sse"
        ],
    }
    return summary, fit


def _support_records(support: np.ndarray, curves: np.ndarray, steps: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in range(support.shape[1]):
        curve = np.maximum(curves[candidate], 0.0)
        cumulative = np.cumsum(curve)
        midpoint = None
        if len(cumulative) and float(cumulative[-1]) > 0:
            midpoint = int(steps[min(int(np.searchsorted(cumulative, 0.5 * cumulative[-1])), len(steps) - 1)])
        records.append(
            {
                "candidate": int(candidate),
                "tasks": np.flatnonzero(support[:, candidate]).astype(int).tolist(),
                "support_size": int(support[:, candidate].sum()),
                "priority_midpoint_step": midpoint,
            }
        )
    return records


def _hasse_edges(records: list[dict[str, Any]]) -> list[list[int]]:
    supports = [set(record["tasks"]) for record in records]
    ordered = [
        (left, right)
        for left, parent in enumerate(supports)
        for right, child in enumerate(supports)
        if left != right and parent > child
    ]
    return [
        [left, right]
        for left, right in ordered
        if not any(
            left != middle != right
            and supports[left] > supports[middle] > supports[right]
            for middle in range(len(supports))
        )
    ]


def _plot(
    output: Path,
    steps: np.ndarray,
    losses: np.ndarray,
    support: np.ndarray,
    curves: np.ndarray,
    interaction_support: np.ndarray | None = None,
) -> None:
    columns = 3 if interaction_support is not None else 2
    figure, axes = plt.subplots(1, columns, figsize=(5.5 * columns, 4.2))
    for task in range(losses.shape[1]):
        axes[0].plot(steps, losses[:, task], linewidth=1.1, label=f"task {task}")
    axes[0].set(xlabel="SGD step", ylabel="held-out loss (bits)", title="Ordinary MLP task losses")
    axes[0].set_ylim(bottom=0)
    axes[1].imshow(support.astype(float), aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[1].set(
        xlabel="discovered candidate",
        ylabel="task event",
        title=f"Binary priority supports (K={support.shape[1]})",
    )
    if interaction_support is not None:
        axes[2].imshow(
            interaction_support.astype(float),
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        axes[2].set(
            xlabel="interaction-resolved candidate",
            ylabel="task event",
            title=f"Sensor-input interactions (K={interaction_support.shape[1]})",
        )
    figure.tight_layout()
    figure.savefig(output / "discovery.png", dpi=180)
    plt.close(figure)


def run(config: MLPDiscoveryPilotConfig) -> Path:
    if config.n_tasks < 2 or config.steps < 1 or config.checkpoint_every < 1:
        raise ValueError("pilot requires at least two tasks and positive training/checkpoint steps")
    if config.steps % config.checkpoint_every:
        raise ValueError("steps must be divisible by checkpoint_every")
    if config.function_group_size < 1 or config.n_tasks % config.function_group_size:
        raise ValueError("function_group_size must be positive and divide n_tasks")
    if config.event_granularity not in {"task", "example"}:
        raise ValueError("event_granularity must be 'task' or 'example'")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    with (output / "status.json").open("w") as handle:
        json.dump({"status": "running", "config": asdict(config)}, handle, indent=2)

    mlp_config = _mlp_config(config)
    expected_supports = _expected_primitive_supports(mlp_config)
    random.seed(config.model_seed)
    np.random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)
    probabilities, rank_by_task = task_probabilities(mlp_config)
    train = _fixed_panel(mlp_config, samples_per_task=config.samples_per_task, seed=config.train_seed)
    eval_panel = _fixed_panel(mlp_config, samples_per_task=config.samples_per_task, seed=config.eval_seed)
    model = build_model(mlp_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    checkpoint_steps = np.arange(0, config.steps + 1, config.checkpoint_every, dtype=np.int64)
    loss_history: list[np.ndarray] = []

    try:
        for step in range(config.steps + 1):
            if step % config.checkpoint_every == 0:
                torch.save(model.state_dict(), checkpoint_dir / f"step_{step:08d}.pt")
                model.eval()
                with torch.no_grad():
                    conditional = _conditional_losses(model, *eval_panel, config.n_tasks)
                loss_history.append((conditional / math.log(2.0)).cpu().numpy())
                model.train()
            if step == config.steps:
                break
            optimizer.zero_grad(set_to_none=True)
            conditional = _conditional_losses(model, *train, config.n_tasks)
            loss = sum(float(probabilities[task]) * conditional[task] for task in range(config.n_tasks))
            loss.backward()
            optimizer.step()

        train_scores: list[np.ndarray] = []
        eval_scores: list[np.ndarray] = []
        interaction_scores: list[np.ndarray] = []
        group_interaction_scores: dict[str, list[np.ndarray]] = {
            group: []
            for group in (
                "task_conditioning",
                "sensor_input",
                "hidden_shared",
                "readout",
                *(f"sensor_role_{spec.name}" for spec in factor_specs(mlp_config)),
            )
        }
        identity_errors: list[float] = []
        for index, step in enumerate(checkpoint_steps[:-1]):
            state = torch.load(checkpoint_dir / f"step_{int(step):08d}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            train_gradients, discovery_weights, event_tasks = _discovery_events(
                model, train, probabilities, config.event_granularity
            )
            direction = _weighted_direction(train_gradients, discovery_weights)
            train_score = _gradient_inner_products(train_gradients, direction)
            interaction_score = (
                _gradient_gram(train_gradients) * discovery_weights[None, :]
            )
            group_interactions = _group_gradient_grams(
                train_gradients, mlp_config
            )
            eval_gradients, _, _ = _discovery_events(
                model, eval_panel, probabilities, config.event_granularity
            )
            eval_score = _gradient_inner_products(eval_gradients, direction)
            energy = float(sum(value.square().sum() for value in direction))
            identity_errors.append(
                abs(float(discovery_weights @ train_score) - energy)
            )
            interval_mass = config.learning_rate * int(checkpoint_steps[index + 1] - step)
            train_scores.append(train_score * interval_mass)
            eval_scores.append(eval_score * interval_mass)
            interaction_scores.append(interaction_score * interval_mass)
            for group, group_gram in group_interactions.items():
                group_interaction_scores[group].append(
                    group_gram * discovery_weights[None, :] * interval_mass
                )

        train_credit = np.stack(train_scores, axis=1)
        eval_credit = np.stack(eval_scores, axis=1)
        interaction_credit = np.stack(interaction_scores, axis=1)
        group_interaction_credit = {
            group: np.stack(scores, axis=1)
            for group, scores in group_interaction_scores.items()
        }
        interaction_scalar_error = float(
            np.max(np.abs(interaction_credit.sum(axis=2) - train_credit))
        )
        fit = factorize_priority_field(
            train_credit,
            discovery_weights,
            max_components=config.max_components,
            alternating_steps=config.alternating_steps,
            minimum_component_gain_fraction=config.minimum_component_gain_fraction,
            interval_steps=checkpoint_steps[1:],
        )
        eval_support = np.zeros_like(fit["support"])
        eval_residual = eval_credit.copy()
        for candidate, curve in enumerate(fit["temporal_priority"]):
            curve_energy = float(np.square(curve).sum())
            selected = 2.0 * (eval_residual @ curve) - curve_energy > 0
            eval_support[:, candidate] = selected
            eval_residual -= selected[:, None] * curve[None]

        records = _support_records(fit["support"], fit["temporal_priority"], checkpoint_steps[1:])
        edges = _hasse_edges(records)
        interaction_population, interaction_population_fit = (
            _interaction_factorization_summary(
                interaction_credit,
                discovery_weights,
                config=config,
                n_events=len(discovery_weights),
            )
        )
        interaction_uniform, interaction_uniform_fit = (
            _interaction_factorization_summary(
                interaction_credit,
                np.ones(len(discovery_weights), dtype=np.float64),
                config=config,
                n_events=len(discovery_weights),
            )
        )
        parameter_group_audit: dict[str, dict[str, Any]] = {}
        parameter_group_fits: dict[str, dict[str, np.ndarray]] = {}
        group_energy = {
            group: float(np.square(credit).sum())
            for group, credit in group_interaction_credit.items()
        }
        total_group_energy = sum(group_energy.values())
        for group, credit in group_interaction_credit.items():
            group_summary, group_fit = _interaction_factorization_summary(
                credit,
                np.ones(len(discovery_weights), dtype=np.float64),
                config=config,
                n_events=len(discovery_weights),
            )
            group_summary["standalone_energy_fraction_across_groups"] = (
                group_energy[group] / total_group_energy
                if total_group_energy
                else 0.0
            )
            parameter_group_audit[group] = group_summary
            parameter_group_fits[group] = group_fit
        threshold_sensitivity = []
        for threshold in sorted(
            {0.05, 0.02, 0.01, 0.005, 0.001, float(config.minimum_component_gain_fraction)},
            reverse=True,
        ):
            threshold_fit = factorize_priority_field(
                train_credit,
                discovery_weights,
                max_components=config.max_components,
                alternating_steps=config.alternating_steps,
                minimum_component_gain_fraction=threshold,
                interval_steps=checkpoint_steps[1:],
            )
            threshold_sensitivity.append(
                {
                    "minimum_gain_fraction": float(threshold),
                    "candidate_count": int(threshold_fit["support"].shape[1]),
                    "field_explained_fraction": float(
                        threshold_fit["summary"]["field_explained_fraction"]
                    ),
                    "supports": [
                        np.flatnonzero(threshold_fit["support"][:, candidate])
                        .astype(int)
                        .tolist()
                        for candidate in range(threshold_fit["support"].shape[1])
                    ],
                }
            )
        losses = np.stack(loss_history)
        final_exact = (losses[-1] < 0.01).astype(bool)
        sensor_supports = parameter_group_audit["sensor_input"]["supports"]
        for record in expected_supports:
            record["events"] = np.flatnonzero(
                np.isin(event_tasks, record["tasks"])
            ).astype(int).tolist()
        expected_support_sets = {
            tuple(record["events"]) for record in expected_supports
        }
        recovered_expected_supports = sorted(
            expected_support_sets & {tuple(support) for support in sensor_supports}
        )
        role_recovered = 0
        for record in expected_supports:
            role_supports = parameter_group_audit[
                f"sensor_role_{record['role']}"
            ]["supports"]
            if record["events"] in role_supports:
                role_recovered += 1
        summary = {
            "status": "complete",
            "scientific_status": (
                "CPU calibration of architecture-agnostic Stage-B discovery on exact "
                "task-level population-GD priority from one complete ordinary-MLP host block. "
                "Task-level events make this a count/support calibration, not evidence of an "
                "internally implemented runtime graph."
            ),
            "expected_functional_quanta": len(expected_supports),
            "expected_geometry": (
                "antichain"
                if len(factor_specs(mlp_config)) == 1
                else "nested primitive support family"
            ),
            "expected_primitive_supports": expected_supports,
            "discovery_event_count": len(discovery_weights),
            "discovery_event_tasks": event_tasks.tolist(),
            "discovery_event_weights": discovery_weights.tolist(),
            "sensor_recovered_expected_supports": [
                list(support) for support in recovered_expected_supports
            ],
            "sensor_expected_support_recall": (
                len(recovered_expected_supports) / len(expected_supports)
                if expected_supports
                else 0.0
            ),
            "sensor_role_expected_support_recall": (
                role_recovered / len(expected_supports) if expected_supports else 0.0
            ),
            "discovered_candidates": int(fit["support"].shape[1]),
            "candidates": records,
            "support_hasse_edges": edges,
            "singleton_candidates": int(sum(record["support_size"] == 1 for record in records)),
            "unique_singleton_tasks": sorted(
                {record["tasks"][0] for record in records if record["support_size"] == 1}
            ),
            "final_tasks_below_0p01_bits": int(final_exact.sum()),
            "final_task_losses_bits": losses[-1].tolist(),
            "task_probabilities": probabilities.tolist(),
            "rank_by_task": rank_by_task.astype(int).tolist(),
            "minimum_component_gain_fraction_of_original_sse": float(config.minimum_component_gain_fraction),
            "maximum_priority_identity_error": float(max(identity_errors, default=0.0)),
            "maximum_interaction_to_scalar_priority_error": interaction_scalar_error,
            "factorization": fit["summary"],
            "candidate_gain_fractions_of_original_sse": fit["gain_fractions_of_original_sse"],
            "threshold_sensitivity": threshold_sensitivity,
            "interaction_resolved_priority": {
                "definition": (
                    "I[e,r,t] = p[r] M_t <grad L_e, grad L_r>; summing r "
                    "recovers scalar priority. Signed event-reference-time features "
                    "retain the coordinate-invariant event-gradient geometry."
                ),
                "population_weighted_events": interaction_population,
                "uniform_event_types": interaction_uniform,
                "parameter_group_audit_uniform_event_types": parameter_group_audit,
            },
            "config": asdict(config),
        }
        np.savez_compressed(
            output / "priority_analysis.npz",
            checkpoint_steps=checkpoint_steps,
            task_losses_bits=losses,
            train_credit=train_credit,
            eval_credit=eval_credit,
            train_support=fit["support"],
            eval_support=eval_support,
            temporal_priority=fit["temporal_priority"],
            train_residual=fit["residual"],
            eval_residual=eval_residual,
            interaction_credit=interaction_credit,
            interaction_population_support=interaction_population_fit["support"],
            interaction_uniform_support=interaction_uniform_fit["support"],
            interaction_uniform_feature_curves=interaction_uniform_fit[
                "temporal_priority"
            ],
            **{
                f"interaction_{group}_credit": credit
                for group, credit in group_interaction_credit.items()
            },
            **{
                f"interaction_{group}_support": fit["support"]
                for group, fit in parameter_group_fits.items()
            },
            task_probabilities=probabilities,
        )
        with (output / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        _plot(
            output,
            checkpoint_steps,
            losses,
            fit["support"],
            fit["temporal_priority"],
            parameter_group_fits["sensor_input"]["support"],
        )
        with (output / "status.json").open("w") as handle:
            json.dump({"status": "complete", "summary": "summary.json"}, handle, indent=2)
        return output
    except Exception as error:
        with (output / "status.json").open("w") as handle:
            json.dump({"status": "failed", "error": repr(error), "config": asdict(config)}, handle, indent=2)
        raise
