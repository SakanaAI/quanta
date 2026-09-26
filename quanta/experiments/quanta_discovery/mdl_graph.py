from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EPSILON = 1.0e-7
PriorityEdge = tuple[int, int, int, int]
EdgeKey = tuple[int, int]


@dataclass(frozen=True)
class MdlOwnershipGraphFit:
    """Saved arrays and diagnostics from constrained MDL graph discovery."""

    dynamics_supports: tuple[np.ndarray, ...]
    local_supports: tuple[np.ndarray, ...]
    inherited_supports: tuple[np.ndarray, ...]
    effective_supports: tuple[np.ndarray, ...]
    local_support_probabilities: tuple[np.ndarray, ...]
    dynamics_support_probabilities: tuple[np.ndarray, ...]
    effective_support_probabilities: tuple[np.ndarray, ...]
    local_temporal_priority: tuple[np.ndarray, ...]
    temporal_priority: tuple[np.ndarray, ...]
    edge_probabilities: dict[EdgeKey, np.ndarray]
    edges: tuple[PriorityEdge, ...]
    history: tuple[dict[str, float], ...]
    summary: dict[str, object]


@dataclass(frozen=True)
class MdlOwnershipGraphConfig:
    """Settings for fixed-dynamics, one-run MDL graph discovery."""

    steps: int = 3_000
    learning_rate: float = 3.0e-2
    initial_edge_probability: float = 0.10
    temperature_start: float = 1.0
    temperature_end: float = 0.10
    edge_threshold: float = 0.50
    fidelity_tolerance: float = 0.01
    dual_learning_rate: float = 50.0
    augmented_weight: float = 5_000.0
    seed: int = 0
    log_every: int = 100

    def validate(self) -> None:
        if self.steps <= 0 or self.learning_rate <= 0.0:
            raise ValueError("steps and learning_rate must be positive")
        if not 0.0 < self.initial_edge_probability < 1.0:
            raise ValueError("initial_edge_probability must lie in (0, 1)")
        if not 0.0 < self.temperature_end <= self.temperature_start:
            raise ValueError("temperatures must satisfy 0 < end <= start")
        if not 0.0 < self.edge_threshold < 1.0:
            raise ValueError("edge_threshold must lie in (0, 1)")
        if self.fidelity_tolerance < 0.0:
            raise ValueError("fidelity_tolerance must be nonnegative")
        if self.dual_learning_rate <= 0.0 or self.augmented_weight <= 0.0:
            raise ValueError("constraint optimizer weights must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")


def ownership_provenance(
    dynamics: Sequence[torch.Tensor],
    edge_gates: Mapping[EdgeKey, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Split fixed dynamics into local, inherited, and closed participation."""

    local: list[torch.Tensor | None] = [None] * len(dynamics)
    inherited: list[torch.Tensor | None] = [None] * len(dynamics)
    effective: list[torch.Tensor | None] = [None] * len(dynamics)
    for parent_layer in reversed(range(len(dynamics))):
        observed = dynamics[parent_layer].clamp(0.0, 1.0)
        inherited_score = torch.zeros_like(observed)
        for child_layer in range(parent_layer + 1, len(dynamics)):
            gate = edge_gates.get((parent_layer, child_layer))
            if gate is None:
                continue
            child = effective[child_layer]
            assert child is not None
            inherited_score = inherited_score + child @ gate.transpose(0, 1)
        cause = 1.0 - torch.exp(-inherited_score.clamp_min(0.0))
        local[parent_layer] = observed * (1.0 - cause)
        inherited[parent_layer] = cause
        effective[parent_layer] = observed + (1.0 - observed) * cause
    return (
        tuple(value for value in local if value is not None),
        tuple(value for value in inherited if value is not None),
        tuple(value for value in effective if value is not None),
    )


def hard_ownership_provenance(
    dynamics_supports: Sequence[np.ndarray], edges: Sequence[PriorityEdge]
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return exact local, inherited, and downward-closed ownership."""

    dynamics = tuple(np.asarray(value, dtype=bool) for value in dynamics_supports)
    effective = [value.copy() for value in dynamics]
    inherited = [np.zeros_like(value) for value in dynamics]
    for parent_layer in reversed(range(len(dynamics))):
        for edge_parent, parent, child_layer, child in edges:
            if edge_parent == parent_layer:
                inherited[parent_layer][:, parent] |= effective[child_layer][:, child]
        effective[parent_layer] |= inherited[parent_layer]
    local = [value & ~cause for value, cause in zip(dynamics, inherited)]
    return tuple(local), tuple(inherited), tuple(effective)


def prerequisite_closure(
    local_probabilities: Sequence[torch.Tensor],
    edge_gates: Mapping[EdgeKey, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Propagate later demand to earlier prerequisites with differentiable noisy-OR."""

    hazards: list[torch.Tensor | None] = [None] * len(local_probabilities)
    for parent_layer in reversed(range(len(local_probabilities))):
        local = local_probabilities[parent_layer].clamp(0.0, 1.0 - EPSILON)
        hazard = -torch.log1p(-local)
        for child_layer in range(parent_layer + 1, len(local_probabilities)):
            gate = edge_gates.get((parent_layer, child_layer))
            if gate is None:
                continue
            child_hazard = hazards[child_layer]
            assert child_hazard is not None
            hazard = hazard + child_hazard @ gate.transpose(0, 1)
        hazards[parent_layer] = hazard
    return tuple(1.0 - torch.exp(-value) for value in hazards if value is not None)


def _array_metrics(
    fields: Sequence[np.ndarray],
    supports: Sequence[np.ndarray],
    curves: Sequence[np.ndarray],
) -> dict[str, object]:
    residual_sse = []
    explained = []
    for observed, support, curve in zip(fields, supports, curves):
        prediction = np.asarray(support, dtype=np.float64) @ np.asarray(
            curve, dtype=np.float64
        )
        residual = float(np.square(np.asarray(observed) - prediction).sum())
        total = float(np.square(observed).sum())
        residual_sse.append(residual)
        explained.append(1.0 - residual / max(total, EPSILON))
    return {
        "residual_sse_by_layer": residual_sse,
        "field_explained_fraction_by_layer": explained,
        "mean_normalized_residual": float(np.mean([1.0 - x for x in explained])),
        "mean_active_candidates": float(
            np.concatenate(supports, axis=1).sum(axis=1).mean()
        ),
    }


def _hard_edges(
    probabilities: Mapping[EdgeKey, np.ndarray], threshold: float
) -> tuple[PriorityEdge, ...]:
    result = []
    for (parent_layer, child_layer), values in probabilities.items():
        rows, columns = np.nonzero(np.asarray(values) >= threshold)
        result.extend(
            (parent_layer, int(parent), child_layer, int(child))
            for parent, child in zip(rows, columns)
        )
    return tuple(sorted(result))


def _temporal_margins(
    curves: Sequence[np.ndarray], edges: Sequence[PriorityEdge]
) -> list[float]:
    cumulative = []
    for value in curves:
        positive = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
        cumulative.append(
            np.cumsum(positive, axis=1)
            / np.maximum(positive.sum(axis=1, keepdims=True), EPSILON)
        )
    return [
        float(
            np.min(
                cumulative[parent_layer][parent]
                - cumulative[child_layer][child]
            )
        )
        for parent_layer, parent, child_layer, child in edges
    ]


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


def profiled_bernoulli_nll(
    successes: torch.Tensor, trials: torch.Tensor
) -> torch.Tensor:
    """Jeffreys-smoothed Bernoulli codelength in nats."""

    probability = (successes + 0.5) / (trials + 1.0)
    probability = probability.clamp(EPSILON, 1.0 - EPSILON)
    return -successes * torch.log(probability) - (
        trials - successes
    ) * torch.log1p(-probability)


def ownership_codelength(
    dynamics: Sequence[torch.Tensor], inherited: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Encode local innovations and closure exceptions under optimal rates."""

    total = dynamics[0].new_zeros(())
    for observed, cause in zip(dynamics, inherited):
        observed = observed.clamp(0.0, 1.0)
        cause = cause.clamp(0.0, 1.0)
        local_trials = (1.0 - cause).sum(dim=0)
        local_ones = (observed * (1.0 - cause)).sum(dim=0)
        inherited_trials = cause.sum(dim=0)
        exceptions = ((1.0 - observed) * cause).sum(dim=0)
        total = total + profiled_bernoulli_nll(
            local_ones, local_trials
        ).sum()
        total = total + profiled_bernoulli_nll(
            exceptions, inherited_trials
        ).sum()
    return total


def temporal_edge_codelengths(
    curves: Sequence[np.ndarray], event_count: int
) -> dict[EdgeKey, np.ndarray]:
    """Return per-edge precedence surprisal plus a two-part MDL edge cost."""

    counts = tuple(value.shape[0] for value in curves)
    possible_edges = sum(
        counts[parent] * counts[child]
        for parent in range(len(counts))
        for child in range(parent + 1, len(counts))
    )
    identifier_cost = math.log(max(possible_edges, 2))
    parameter_cost = 0.5 * math.log(max(int(event_count), 2))
    result: dict[EdgeKey, np.ndarray] = {}
    normalized = []
    for values in curves:
        positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        normalized.append(
            positive / np.maximum(positive.sum(axis=1, keepdims=True), EPSILON)
        )
    for parent_layer in range(len(counts)):
        parent_cdf = np.cumsum(normalized[parent_layer], axis=1)
        for child_layer in range(parent_layer + 1, len(counts)):
            child_density = normalized[child_layer]
            precedence_probability = parent_cdf @ child_density.T
            precedence_surprisal = -np.log(
                np.clip(precedence_probability, EPSILON, 1.0)
            )
            result[(parent_layer, child_layer)] = (
                identifier_cost + parameter_cost + precedence_surprisal
            )
    return result


class _MdlEdgeModel(nn.Module):
    def __init__(
        self,
        counts: Sequence[int],
        *,
        initial_probability: float,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.counts = tuple(int(value) for value in counts)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        initial = _logit(float(initial_probability))
        self.edge_logits = nn.ParameterDict()
        for parent in range(len(self.counts)):
            for child in range(parent + 1, len(self.counts)):
                shape = (self.counts[parent], self.counts[child])
                noise = 0.02 * torch.randn(shape, generator=generator, device=device)
                self.edge_logits[f"layer_{parent}_to_{child}"] = nn.Parameter(
                    initial + noise
                )

    def probabilities(self, temperature: float) -> dict[EdgeKey, torch.Tensor]:
        return {
            (parent, child): torch.sigmoid(
                self.edge_logits[f"layer_{parent}_to_{child}"] / temperature
            )
            for parent in range(len(self.counts))
            for child in range(parent + 1, len(self.counts))
        }


def _temperature(config: MdlOwnershipGraphConfig, step: int) -> float:
    if config.steps <= 1:
        return float(config.temperature_end)
    progress = step / float(config.steps - 1)
    return config.temperature_start * (
        config.temperature_end / config.temperature_start
    ) ** progress


def _distortion(
    fields: Sequence[torch.Tensor],
    supports: Sequence[torch.Tensor],
    curves: Sequence[torch.Tensor],
    energies: Sequence[torch.Tensor],
) -> torch.Tensor:
    return torch.stack(
        [
            torch.square(field - support @ curve).sum()
            / energy.clamp_min(EPSILON)
            for field, support, curve, energy in zip(
                fields, supports, curves, energies
            )
        ]
    ).mean()


def _closed_temporal_curves(
    curves: Sequence[np.ndarray],
    counts: Sequence[int],
    edges: Sequence[PriorityEdge],
    device: torch.device,
) -> tuple[np.ndarray, ...]:
    gates: dict[EdgeKey, torch.Tensor] = {}
    for parent in range(len(counts)):
        for child in range(parent + 1, len(counts)):
            gate = torch.zeros((counts[parent], counts[child]), device=device)
            for edge_parent, parent_id, edge_child, child_id in edges:
                if (edge_parent, edge_child) == (parent, child):
                    gate[parent_id, child_id] = 1.0
            gates[(parent, child)] = gate
    cumulative = tuple(
        torch.as_tensor(
            np.cumsum(value, axis=1)
            / np.maximum(value.sum(axis=1, keepdims=True), EPSILON),
            dtype=torch.float32,
            device=device,
        ).transpose(0, 1)
        for value in curves
    )
    closed = prerequisite_closure(cumulative, gates)
    result = []
    for original, value in zip(curves, closed):
        density = torch.diff(
            F.pad(value.transpose(0, 1), (1, 0), value=0.0), dim=1
        ).clamp_min(0.0)
        mass = np.asarray(original).sum(axis=1)
        result.append(density.cpu().numpy() * mass[:, None])
    return tuple(result)


def fit_mdl_ownership_graph(
    fields: Sequence[np.ndarray],
    dynamics_supports: Sequence[np.ndarray],
    temporal_priority: Sequence[np.ndarray],
    *,
    config: MdlOwnershipGraphConfig,
    device: torch.device,
) -> MdlOwnershipGraphFit:
    """Discover a fixed-dynamics graph under a fidelity constraint."""

    config.validate()
    torch.manual_seed(config.seed)
    dynamics_np = tuple(np.asarray(value, dtype=bool) for value in dynamics_supports)
    curves_np = tuple(np.asarray(value, dtype=np.float32) for value in temporal_priority)
    fields_np = tuple(np.asarray(value, dtype=np.float32) for value in fields)
    counts = tuple(value.shape[1] for value in dynamics_np)
    event_count = dynamics_np[0].shape[0]
    interval_count = curves_np[0].shape[1]
    dynamics = tuple(
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in dynamics_np
    )
    curves = tuple(torch.as_tensor(value, device=device) for value in curves_np)
    observed = tuple(torch.as_tensor(value, device=device) for value in fields_np)
    energies = tuple(value.square().sum().detach() for value in observed)
    baseline = _array_metrics(fields_np, dynamics_np, curves_np)
    baseline_distortion = float(baseline["mean_normalized_residual"])
    fidelity_limit = baseline_distortion * (1.0 + config.fidelity_tolerance)
    edge_cost_np = temporal_edge_codelengths(curves_np, event_count)
    edge_cost = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in edge_cost_np.items()
    }
    model = _MdlEdgeModel(
        counts,
        initial_probability=config.initial_edge_probability,
        seed=config.seed,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dual = 0.0
    history: list[dict[str, float]] = []

    _, empty_inherited, _ = hard_ownership_provenance(dynamics_np, ())
    with torch.no_grad():
        baseline_code_nats = float(
            ownership_codelength(
                dynamics,
                tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in empty_inherited),
            ).cpu()
        )
    best_edges: tuple[PriorityEdge, ...] = ()
    best_code_nats = baseline_code_nats
    best_step = 0
    best_probabilities: dict[EdgeKey, np.ndarray] | None = None
    last_hard_masks: dict[EdgeKey, torch.Tensor] | None = None
    last_hard_values: tuple[
        float, float, tuple[PriorityEdge, ...], bool
    ] | None = None

    def evaluate_hard(
        probabilities: Mapping[EdgeKey, np.ndarray], step: int
    ) -> tuple[float, float, tuple[PriorityEdge, ...], bool]:
        nonlocal best_edges, best_code_nats, best_step, best_probabilities
        edges = _hard_edges(probabilities, config.edge_threshold)
        local_np, inherited_np, effective_np = hard_ownership_provenance(
            dynamics_np, edges
        )
        metrics = _array_metrics(fields_np, effective_np, curves_np)
        distortion = float(metrics["mean_normalized_residual"])
        inherited_t = tuple(
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in inherited_np
        )
        with torch.no_grad():
            ownership_nats = float(
                ownership_codelength(dynamics, inherited_t).cpu()
            )
        graph_nats = sum(
            float(edge_cost_np[(parent_layer, child_layer)][parent, child])
            for parent_layer, parent, child_layer, child in edges
        )
        code_nats = ownership_nats + graph_nats
        feasible = distortion <= fidelity_limit + 1.0e-12
        if feasible and code_nats < best_code_nats - 1.0e-9:
            best_edges = edges
            best_code_nats = code_nats
            best_step = int(step)
            best_probabilities = {
                key: np.asarray(value).copy() for key, value in probabilities.items()
            }
        return distortion, code_nats, edges, feasible

    for step in range(config.steps):
        temperature = _temperature(config, step)
        probabilities = model.probabilities(temperature)
        _, inherited, effective = ownership_provenance(dynamics, probabilities)
        ownership_nats = ownership_codelength(dynamics, inherited)
        graph_nats = sum(
            (probabilities[key] * edge_cost[key]).sum() for key in probabilities
        )
        distortion = _distortion(observed, effective, curves, energies)
        excess = F.relu(distortion - fidelity_limit)
        loss = (
            (ownership_nats + graph_nats) / event_count
            + dual * excess
            + 0.5 * config.augmented_weight * excess.square()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        dual = max(
            0.0,
            dual + config.dual_learning_rate * float(excess.detach().cpu()),
        )

        hard_values: tuple[float, float, tuple[PriorityEdge, ...], bool] | None = None
        with torch.no_grad():
            updated_probabilities = model.probabilities(temperature)
            current_masks = {
                key: value >= config.edge_threshold
                for key, value in updated_probabilities.items()
            }
            hard_state_changed = last_hard_masks is None or any(
                not torch.equal(current_masks[key], last_hard_masks[key])
                for key in current_masks
            )
            if hard_state_changed:
                probability_np = {
                    key: value.detach().cpu().numpy()
                    for key, value in updated_probabilities.items()
                }
                hard_values = evaluate_hard(probability_np, step + 1)
                last_hard_masks = {
                    key: value.clone() for key, value in current_masks.items()
                }
                last_hard_values = hard_values
        if (
            step == 0
            or step + 1 == config.steps
            or (step + 1) % config.log_every == 0
        ):
            if hard_values is None:
                assert last_hard_values is not None
                hard_values = last_hard_values
            hard_distortion, hard_code, hard_edges, hard_feasible = hard_values
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "soft_distortion": float(distortion.detach().cpu()),
                    "soft_ownership_code_nats": float(ownership_nats.detach().cpu()),
                    "soft_graph_code_nats": float(graph_nats.detach().cpu()),
                    "dual": float(dual),
                    "temperature": float(temperature),
                    "hard_distortion": float(hard_distortion),
                    "hard_code_nats": float(hard_code),
                    "hard_edge_count": float(len(hard_edges)),
                    "hard_feasible": float(hard_feasible),
                    "best_feasible_code_nats": float(best_code_nats),
                    "best_feasible_edge_count": float(len(best_edges)),
                }
            )

    if best_probabilities is None:
        with torch.no_grad():
            best_probabilities = {
                key: value.cpu().numpy()
                for key, value in model.probabilities(config.temperature_end).items()
            }
    local_np, inherited_np, effective_np = hard_ownership_provenance(
        dynamics_np, best_edges
    )
    closed_curves = _closed_temporal_curves(
        curves_np, counts, best_edges, device
    )
    final_metrics = _array_metrics(fields_np, effective_np, curves_np)
    local_active = np.concatenate(local_np, axis=1).sum(axis=1)
    inherited_active = np.concatenate(inherited_np, axis=1).sum(axis=1)
    dynamics_active = np.concatenate(dynamics_np, axis=1).sum(axis=1)
    effective_active = np.concatenate(effective_np, axis=1).sum(axis=1)
    margins = _temporal_margins(closed_curves, best_edges)
    edge_records = []
    for parent_layer, parent, child_layer, child in best_edges:
        child_mask = effective_np[child_layer][:, child]
        parent_dynamics = dynamics_np[parent_layer][:, parent]
        edge_records.append(
            {
                "edge": [parent_layer, parent, child_layer, child],
                "probability_at_selection": float(
                    best_probabilities[(parent_layer, child_layer)][parent, child]
                ),
                "matched_inherited_events": int(
                    (child_mask & parent_dynamics).sum()
                ),
                "closure_additions": int(
                    (child_mask & ~parent_dynamics).sum()
                ),
                "temporal_and_graph_code_nats": float(
                    edge_cost_np[(parent_layer, child_layer)][parent, child]
                ),
            }
        )
    summary: dict[str, object] = {
        "method": "quanta_factorization_v4_fixed_dynamics_mdl_provenance",
        "scientific_status": (
            "The initial dynamics supports and contribution curves are fixed. "
            "One differentiable run minimizes ownership and graph codelength "
            "while a learned dual variable enforces the configured fidelity "
            "tolerance. The saved graph is the lowest-code hardened state visited "
            "by the relaxation that satisfies that constraint."
        ),
        "config": asdict(config),
        "candidate_counts": list(counts),
        "event_count": event_count,
        "interval_count": interval_count,
        "edges": [list(edge) for edge in best_edges],
        "edge_count": len(best_edges),
        "edge_records": edge_records,
        "baseline": baseline,
        "final": final_metrics,
        "fidelity": {
            "tolerance": config.fidelity_tolerance,
            "limit": fidelity_limit,
            "satisfied": bool(
                float(final_metrics["mean_normalized_residual"])
                <= fidelity_limit + 1.0e-12
            ),
        },
        "description_length": {
            "baseline_nats": baseline_code_nats,
            "selected_nats": best_code_nats,
            "saved_nats": baseline_code_nats - best_code_nats,
            "saved_bits": (baseline_code_nats - best_code_nats) / math.log(2.0),
            "best_step": best_step,
        },
        "activity": {
            "mean_dynamics_participation": float(dynamics_active.mean()),
            "mean_local_requests": float(local_active.mean()),
            "mean_inherited_requests": float(inherited_active.mean()),
            "mean_closure_additions": float(
                (effective_active - dynamics_active).mean()
            ),
            "mean_effective_active": float(effective_active.mean()),
            "maximum_effective_active": int(effective_active.max(initial=0)),
        },
        "closure": {
            "support_violation_count": 0,
            "minimum_temporal_margin": min(margins) if margins else None,
            "temporal_margins": margins,
        },
        "history": history,
    }
    hard_float = tuple(value.astype(np.float32) for value in local_np)
    effective_float = tuple(value.astype(np.float32) for value in effective_np)
    return MdlOwnershipGraphFit(
        dynamics_supports=dynamics_np,
        local_supports=local_np,
        inherited_supports=inherited_np,
        effective_supports=effective_np,
        local_support_probabilities=hard_float,
        dynamics_support_probabilities=tuple(
            value.astype(np.float32) for value in dynamics_np
        ),
        effective_support_probabilities=effective_float,
        local_temporal_priority=curves_np,
        temporal_priority=closed_curves,
        edge_probabilities=best_probabilities,
        edges=best_edges,
        history=tuple(history),
        summary=summary,
    )
