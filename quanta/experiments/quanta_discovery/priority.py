from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.number_naming.task import NumberNamingTask
from quanta.experiments.scaling_laws.training.trainer_curriculum import (
    scheduled_learning_rate,
)

from .dynamics import (
    block_parameters_by_layer,
    flattened_block_parameters,
    flattened_mlp_parameters,
)
from .trajectory import (
    PredictionEventPanel,
    TrainingTrajectory,
    checkpoint_path,
    evaluate_prediction_events,
)


EPSILON = 1.0e-12
@dataclass(frozen=True)
class PriorityAnalysis:
    """Binary event-support decomposition of measured learning priority."""

    checkpoint_steps: np.ndarray
    interval_steps: np.ndarray
    interval_learning_mass: np.ndarray
    train_credit: np.ndarray
    eval_credit: np.ndarray
    train_supports: tuple[np.ndarray, ...]
    eval_supports: tuple[np.ndarray, ...]
    temporal_priority: tuple[np.ndarray, ...]
    train_residuals: tuple[np.ndarray, ...]
    eval_residuals: tuple[np.ndarray, ...]
    summary: dict[str, Any]


def collect_full_batch_gd_trajectory(
    model: DecoderTransformerLM,
    task: NumberNamingTask,
    panel: PredictionEventPanel,
    *,
    checkpoint_every: int,
    save_dir: str,
    device: torch.device,
    on_checkpoint: Callable[..., None] | None = None,
) -> TrainingTrajectory:
    """Train with full-batch GD and retain exact block updates and checkpoints."""

    if str(task.config.optimizer).lower() != "sgd":
        raise ValueError("exact priority discovery requires optimizer=sgd")
    if float(task.config.weight_decay) != 0.0:
        raise ValueError("exact priority discovery requires weight_decay=0")
    if int(task.config.batch_size) != len(task.train):
        raise ValueError(
            "exact priority discovery requires batch_size=training_size"
        )

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(task.config.lr))
    train_batch = task.encode_examples(list(task.train), device=device)
    total_steps = int(task.config.steps or 0)
    expected_checkpoints = 1 + int(
        math.ceil(total_steps / int(checkpoint_every))
    )
    checkpoint_steps: list[int] = []
    checkpoint_paths: list[str] = []
    losses: list[np.ndarray] = []
    optimizer_steps: list[int] = []
    learning_rates: list[float] = []
    layer_updates: list[np.ndarray] = []
    mlp_layer_updates: list[np.ndarray] = []
    training_losses: list[float] = []
    gd_identity_max_abs_error = 0.0
    split_array = np.asarray(panel.splits)
    split_indices = {
        name: np.flatnonzero(split_array == name)
        for name in sorted(set(panel.splits))
    }

    def record(step: int) -> None:
        checkpoint_index = len(checkpoint_steps)
        relative_path = os.path.join(
            "checkpoints",
            f"checkpoint_{checkpoint_index:04d}_step_{int(step):08d}.pt",
        )
        torch.save(
            {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            os.path.join(save_dir, relative_path),
        )
        event_losses = evaluate_prediction_events(model, panel, device=device)
        checkpoint_steps.append(int(step))
        checkpoint_paths.append(relative_path)
        losses.append(event_losses)
        if on_checkpoint is not None:
            split_losses = {
                name: float(event_losses[indices].mean())
                for name, indices in split_indices.items()
            }
            on_checkpoint(
                checkpoint_index,
                int(step),
                expected_checkpoints,
                {"overall": float(event_losses.mean()), **split_losses},
            )

    record(0)
    for step in range(1, total_steps + 1):
        learning_rate = scheduled_learning_rate(
            float(task.config.lr),
            step,
            total_steps,
            str(task.config.scheduler),
            warmup_phase=float(task.config.warmup_phase),
            plateau_phase=float(task.config.plateau_phase),
        )
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        before = flattened_block_parameters(model).detach().clone()
        before_mlp = flattened_mlp_parameters(model).detach().clone()
        logits = model(**train_batch.model_inputs)
        loss = task.compute_loss(logits, train_batch)
        training_losses.append(float(loss.detach().cpu()))
        loss.backward()
        expected_update = -float(learning_rate) * _flatten_block_gradients(model)
        optimizer.step()
        actual_update = flattened_block_parameters(model).detach() - before
        actual_mlp_update = flattened_mlp_parameters(model).detach() - before_mlp
        gd_identity_max_abs_error = max(
            gd_identity_max_abs_error,
            float((actual_update - expected_update).abs().max().cpu()),
        )
        layer_updates.append(actual_update.float().cpu().numpy())
        mlp_layer_updates.append(actual_mlp_update.float().cpu().numpy())
        optimizer_steps.append(int(step))
        learning_rates.append(float(learning_rate))
        if step % int(checkpoint_every) == 0 or step == total_steps:
            record(step)

    return TrainingTrajectory(
        checkpoint_steps=tuple(checkpoint_steps),
        checkpoint_paths=tuple(checkpoint_paths),
        losses=np.stack(losses, axis=1).astype(np.float32, copy=False),
        optimizer_steps=tuple(optimizer_steps),
        learning_rates=np.asarray(learning_rates, dtype=np.float32),
        layer_updates=np.stack(layer_updates).astype(np.float32, copy=False),
        mlp_layer_updates=np.stack(mlp_layer_updates).astype(np.float32, copy=False),
        training_losses=np.asarray(training_losses, dtype=np.float32),
        gd_identity_max_abs_error=float(gd_identity_max_abs_error),
    )


def analyze_event_priority(
    model: DecoderTransformerLM,
    task: NumberNamingTask,
    train_panel: PredictionEventPanel,
    eval_panel: PredictionEventPanel,
    trajectory: TrainingTrajectory,
    *,
    max_components: int,
    alternating_steps: int,
    minimum_component_gain_fraction: float = 0.05,
    save_dir: str,
    device: torch.device,
    on_checkpoint: Callable[[int, int, int, dict[str, float]], None] | None = None,
) -> PriorityAnalysis:
    """Measure event GD priority and expose a binary-only candidate basis.

    The factorization is ``P(e,t) ~= sum_q a_q(e) rho_q(t) + R(e,t)``.
    ``a_q`` is exactly binary and there is no event amplitude. A candidate is
    retained only when its marginal gain clears the configured fraction of the
    host layer's original SSE; ``max_components`` is only a ceiling.
    """

    if trajectory.learning_rates is None:
        raise ValueError("priority analysis requires recorded learning rates")
    checkpoint_steps = np.asarray(trajectory.checkpoint_steps, dtype=np.int64)
    learning_rates = np.asarray(trajectory.learning_rates, dtype=np.float64)
    interval_mass = np.asarray(
        [
            learning_rates[int(start) : int(end)].sum()
            for start, end in zip(checkpoint_steps[:-1], checkpoint_steps[1:])
        ],
        dtype=np.float64,
    )
    train_batch = task.encode_examples(list(task.train), device=device)
    eval_batch = task.encode_examples(list(task.eval_examples), device=device)
    train_scores: list[np.ndarray] = []
    eval_scores: list[np.ndarray] = []
    closure_errors: list[float] = []

    for checkpoint_index, optimizer_step in enumerate(checkpoint_steps[:-1]):
        state = torch.load(
            checkpoint_path(save_dir, trajectory, checkpoint_index),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        population_gradients = _population_block_gradients(
            model, task, train_batch
        )
        train_field = _event_gradient_inner_products(
            model,
            train_batch,
            train_panel,
            population_gradients,
        )
        eval_field = _event_gradient_inner_products(
            model,
            eval_batch,
            eval_panel,
            population_gradients,
        )
        population_energy = torch.stack(
            [
                sum(value.square().sum() for value in layer)
                for layer in population_gradients
            ]
        )
        recovered = train_field.mean(dim=1)
        closure_errors.append(
            float((recovered - population_energy).abs().max().cpu())
        )
        train_scores.append(train_field.cpu().numpy())
        eval_scores.append(eval_field.cpu().numpy())
        if on_checkpoint is not None:
            on_checkpoint(
                checkpoint_index + 1,
                len(checkpoint_steps) - 1,
                int(optimizer_step),
                {"priority_closure_error": closure_errors[-1]},
            )

    train_credit = np.transpose(np.stack(train_scores), (1, 2, 0))
    eval_credit = np.transpose(np.stack(eval_scores), (1, 2, 0))
    train_credit *= interval_mass[None, None]
    eval_credit *= interval_mass[None, None]
    train_weights = np.full(
        train_credit.shape[1], 1.0 / train_credit.shape[1], dtype=np.float64
    )
    eval_weights = np.full(
        eval_credit.shape[1], 1.0 / eval_credit.shape[1], dtype=np.float64
    )

    train_supports: list[np.ndarray] = []
    eval_supports: list[np.ndarray] = []
    temporal_priority: list[np.ndarray] = []
    train_residuals: list[np.ndarray] = []
    eval_residuals: list[np.ndarray] = []
    layer_summaries: list[dict[str, Any]] = []
    for layer in range(train_credit.shape[0]):
        fit = _binary_priority_factorization(
            train_credit[layer],
            train_weights,
            max_components=int(max_components),
            alternating_steps=int(alternating_steps),
            minimum_component_gain_fraction=float(
                minimum_component_gain_fraction
            ),
        )
        curves = fit["temporal_priority"]
        train_support = fit["support"]
        train_residual = fit["residual"]
        eval_support, eval_residual = _infer_binary_support(
            eval_credit[layer], curves
        )
        train_supports.append(train_support)
        eval_supports.append(eval_support)
        temporal_priority.append(curves)
        train_residuals.append(train_residual)
        eval_residuals.append(eval_residual)
        layer_summaries.append(
            {
                "host_layer": int(layer),
                "candidate_slot_ceiling": int(max_components),
                "nonzero_candidate_slots": int(curves.shape[0]),
                "count_status": "retained by source-relative marginal gain",
                "train": _factorization_summary(
                    train_credit[layer],
                    train_residual,
                    train_support,
                    curves,
                    train_weights,
                    checkpoint_steps[1:],
                ),
                "eval_diagnostic": _factorization_summary(
                    eval_credit[layer],
                    eval_residual,
                    eval_support,
                    curves,
                    eval_weights,
                    checkpoint_steps[1:],
                ),
                "candidate_improvements": fit["improvements"],
                "candidate_gain_fractions_of_original_sse": fit[
                    "gain_fractions_of_original_sse"
                ],
            }
        )

    summary = {
        "method": "quanta_dynamics_v0",
        "scientific_status": (
            "Exact full-batch GD event-gradient alignment for complete transformer "
            "blocks at saved checkpoints. "
            "Candidates have binary event support and shared temporal curves; no "
            "event amplitudes are fitted. The experiment then writes the "
            "canonical simple-closure composition graph."
        ),
        "priority_identity": (
            "mean_e <grad L_event(e), grad L_train> = ||grad L_train||^2 "
            "for every host layer and saved checkpoint"
        ),
        "priority_host_parameters": (
            "Each host is one complete transformer block: attention, MLP, and "
            "both local LayerNorm affine parameter pairs. Shared embeddings and "
            "the final normalization/readout are excluded."
        ),
        "factorization": "P(e,t) ~= sum_q a_q(e) rho_q(t) + R(e,t)",
        "ownership": "a_q(e) is exactly 0 or 1; no event amplitude exists",
        "candidate_count": (
            "Candidates are retained when their marginal improvement is at least "
            "the configured fraction of the host layer's original SSE; the maximum "
            "component count is only a ceiling."
        ),
        "discreteness": (
            "cumulative priority is the measured acquisition-time curve used as "
            "the Q-model existence schedule"
        ),
        "maximum_priority_identity_error": float(max(closure_errors, default=0.0)),
        "minimum_component_gain_fraction_of_original_sse": float(
            minimum_component_gain_fraction
        ),
        "layers": layer_summaries,
    }
    result = PriorityAnalysis(
        checkpoint_steps=checkpoint_steps,
        interval_steps=checkpoint_steps[1:],
        interval_learning_mass=interval_mass,
        train_credit=train_credit.astype(np.float32),
        eval_credit=eval_credit.astype(np.float32),
        train_supports=tuple(train_supports),
        eval_supports=tuple(eval_supports),
        temporal_priority=tuple(temporal_priority),
        train_residuals=tuple(train_residuals),
        eval_residuals=tuple(eval_residuals),
        summary=summary,
    )
    write_priority_artifacts(save_dir, train_panel, eval_panel, result)
    return result


def _binary_priority_factorization(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    max_components: int,
    alternating_steps: int,
    minimum_component_gain_fraction: float = 0.05,
) -> dict[str, Any]:
    """Greedy binary factorization with a minimum marginal explained fraction."""

    residual = np.asarray(values, dtype=np.float64).copy()
    original_sse = float(np.sum(weights[:, None] * np.square(residual)))
    supports: list[np.ndarray] = []
    curves: list[np.ndarray] = []
    improvements: list[float] = []
    gain_fractions: list[float] = []
    for component in range(int(max_components)):
        candidate = _fit_binary_component(
            residual,
            weights,
            alternating_steps=int(alternating_steps),
            seed=104_729 + 1_009 * component,
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
        improvements.append(float(improvement))
        gain_fractions.append(float(gain_fraction))
    return {
        "support": (
            np.stack(supports, axis=1)
            if supports
            else np.empty((values.shape[0], 0), dtype=bool)
        ),
        "temporal_priority": (
            np.stack(curves)
            if curves
            else np.empty((0, values.shape[1]), dtype=np.float64)
        ),
        "residual": residual,
        "improvements": improvements,
        "gain_fractions_of_original_sse": gain_fractions,
    }


def _fit_binary_component(
    residual: np.ndarray,
    weights: np.ndarray,
    *,
    alternating_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    weighted = residual * np.sqrt(weights[:, None])
    _, _, right = np.linalg.svd(weighted, full_matrices=False)
    generator = np.random.default_rng(int(seed))
    initializations = (
        np.maximum(weights @ residual, 0.0),
        np.maximum(right[0], 0.0),
        np.maximum(-right[0], 0.0),
        generator.random(residual.shape[1]),
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
            updated_curve = np.maximum(
                np.sum(
                    weights[:, None]
                    * updated_support[:, None]
                    * residual,
                    axis=0,
                )
                / active_mass,
                0.0,
            )
            if float(np.square(updated_curve).sum()) <= EPSILON:
                break
            converged = np.array_equal(updated_support, support) and np.allclose(
                updated_curve, curve, rtol=1.0e-8, atol=1.0e-10
            )
            support = updated_support
            curve = updated_curve
            if converged:
                break
        support = _support_improving_residual(residual, curve)
        if not np.any(support):
            continue
        estimate = support[:, None] * curve[None]
        objective = float(
            np.sum(weights[:, None] * np.square(residual - estimate))
        )
        if objective < best_objective:
            best_objective = objective
            best = support, curve
    return best


def _support_improving_residual(
    residual: np.ndarray, curve: np.ndarray
) -> np.ndarray:
    """Assign an event iff the binary assignment strictly lowers its SSE."""

    curve_energy = float(np.square(curve).sum())
    gain = 2.0 * (residual @ curve) - curve_energy
    return gain > 0.0


def _infer_binary_support(
    values: np.ndarray, curves: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(values, dtype=np.float64).copy()
    supports = np.zeros((values.shape[0], len(curves)), dtype=bool)
    for component, curve in enumerate(curves):
        support = _support_improving_residual(residual, curve)
        supports[:, component] = support
        residual -= support[:, None] * curve[None]
    return supports, residual


def _population_block_gradients(
    model: DecoderTransformerLM,
    task: NumberNamingTask,
    batch: Any,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(**batch.model_inputs)
    task.compute_loss(logits, batch).backward()
    return tuple(
        tuple(
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in parameters
        )
        for parameters in block_parameters_by_layer(model)
    )


def _event_gradient_inner_products(
    model: DecoderTransformerLM,
    event_batch: Any,
    event_panel: PredictionEventPanel,
    directions: tuple[tuple[torch.Tensor, ...], ...],
) -> torch.Tensor:
    """Return ``<grad L_event, grad L_train>`` for each layer and event."""

    model.eval()
    shift_labels = event_batch.labels[:, 1:]
    valid = shift_labels != -100
    if int(valid.sum()) != len(event_panel):
        raise ValueError("batch and prediction-event panel have different sizes")
    rows, positions = torch.nonzero(valid, as_tuple=True)
    if not torch.equal(
        shift_labels[valid].detach().cpu(), event_panel.target_ids.detach().cpu()
    ):
        raise ValueError("prediction-event targets are not aligned")
    if not torch.equal(
        positions.detach().cpu(), event_panel.prediction_positions.detach().cpu()
    ):
        raise ValueError("prediction-event positions are not aligned")

    parameter_names = {id(value): name for name, value in model.named_parameters()}
    scores: list[torch.Tensor] = []
    for parameters, tangents in zip(block_parameters_by_layer(model), directions):
        names = tuple(parameter_names[id(parameter)] for parameter in parameters)

        def event_losses(*values: torch.Tensor) -> torch.Tensor:
            replacements = dict(zip(names, values))
            with sdpa_kernel(SDPBackend.MATH):
                logits = functional_call(
                    model, replacements, (), event_batch.model_inputs
                )
            return F.cross_entropy(
                logits[rows, positions], shift_labels[valid], reduction="none"
            )

        _, directional_losses = jvp(event_losses, parameters, tangents)
        scores.append(directional_losses.detach())
    return torch.stack(scores)


def _factorization_summary(
    values: np.ndarray,
    residual: np.ndarray,
    support: np.ndarray,
    curves: np.ndarray,
    weights: np.ndarray,
    interval_steps: np.ndarray,
) -> dict[str, Any]:
    total = max(float(np.sum(weights[:, None] * np.square(values))), EPSILON)
    residual_energy = float(np.sum(weights[:, None] * np.square(residual)))
    aggregate = weights @ values
    aggregate_residual = weights @ residual
    aggregate_total = max(float(np.square(aggregate).sum()), EPSILON)
    factors = []
    for component, curve in enumerate(curves):
        factors.append(
            {
                "candidate_slot": int(component),
                "demand": float(weights @ support[:, component]),
                "total_priority": float(curve.sum()),
                "priority_peak_step": int(interval_steps[int(np.argmax(curve))]),
                "priority_10_90_width_steps": float(
                    _quantile_width(curve, interval_steps, 0.1, 0.9)
                ),
            }
        )
    return {
        "field_explained_fraction": float(1.0 - residual_energy / total),
        "aggregate_priority_explained_fraction": float(
            1.0 - np.square(aggregate_residual).sum() / aggregate_total
        ),
        "mean_active_candidates_per_event": float(
            weights @ support.sum(axis=1)
        ),
        "median_active_candidates_per_event": float(
            np.median(support.sum(axis=1))
        ),
        "residual_positive_energy_fraction": float(
            np.sum(weights[:, None] * np.square(np.maximum(residual, 0.0)))
            / total
        ),
        "residual_negative_energy_fraction": float(
            np.sum(weights[:, None] * np.square(np.minimum(residual, 0.0)))
            / total
        ),
        "factors": factors,
    }


def _quantile_width(
    curve: np.ndarray,
    interval_steps: np.ndarray,
    low: float,
    high: float,
) -> float:
    values = np.maximum(np.asarray(curve, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= EPSILON:
        return 0.0
    cumulative = np.cumsum(values) / total
    low_index = min(int(np.searchsorted(cumulative, low)), len(values) - 1)
    high_index = min(int(np.searchsorted(cumulative, high)), len(values) - 1)
    return float(interval_steps[high_index] - interval_steps[low_index])


def write_priority_artifacts(
    save_dir: str,
    train_panel: PredictionEventPanel,
    eval_panel: PredictionEventPanel,
    result: PriorityAnalysis,
) -> None:
    np.savez_compressed(
        os.path.join(save_dir, "priority_analysis.npz"),
        checkpoint_steps=result.checkpoint_steps,
        interval_steps=result.interval_steps,
        interval_learning_mass=result.interval_learning_mass,
        train_credit=result.train_credit,
        eval_credit=result.eval_credit,
        train_supports=_object_array(result.train_supports),
        eval_supports=_object_array(result.eval_supports),
        temporal_priority=_object_array(result.temporal_priority),
        train_residuals=_object_array(result.train_residuals),
        eval_residuals=_object_array(result.eval_residuals),
    )
    _write_json(
        os.path.join(save_dir, "priority_events.json"), train_panel.metadata()
    )
    _write_json(
        os.path.join(save_dir, "priority_eval_events.json"), eval_panel.metadata()
    )
    _write_json(os.path.join(save_dir, "priority_summary.json"), result.summary)


def _flatten_block_gradients(model: DecoderTransformerLM) -> torch.Tensor:
    return torch.stack(
        [
            torch.cat(
                [
                    torch.zeros_like(parameter).flatten()
                    if parameter.grad is None
                    else parameter.grad.flatten()
                    for parameter in parameters
                ]
            )
            for parameters in block_parameters_by_layer(model)
        ]
    )


def _object_array(values: Sequence[np.ndarray]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
