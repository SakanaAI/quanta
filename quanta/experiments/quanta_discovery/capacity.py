from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch


EPSILON = 1.0e-12


@dataclass(frozen=True)
class DynamicsWriterMetrics:
    """Dynamics-only capacity diagnostics for one host layer."""

    demand: np.ndarray
    tau_50_steps: np.ndarray
    packet_effective_groups: np.ndarray
    packet_groups_for_90_percent_energy: np.ndarray
    priority_weighted_instantaneous_groups: np.ndarray


@dataclass(frozen=True)
class FunctionalWriterMetrics:
    """Functional residual-write complexity of one host layer."""

    complexity: np.ndarray
    eigenvalues: tuple[np.ndarray, ...]
    zero_energy: np.ndarray


@dataclass(frozen=True)
class WriterBudgetAllocation:
    """A ragged writer allocation that fits an executable parameter budget."""

    writer_counts: tuple[np.ndarray, ...]
    target_parameter_count: int
    fixed_parameter_count: int
    parameters_per_writer: int
    total_writers: int
    executable_parameter_count: int
    unused_parameter_count: int


def parameter_count(modules: Sequence[torch.nn.Module]) -> int:
    """Count trainable and frozen parameters stored by a module collection."""

    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
    )


def fixed_qmodel_parameter_count(
    quantum_counts: Sequence[int],
    *,
    d_model: int,
    vocabulary_size: int,
    attention_rank: int,
    attention_value_dim: int,
    embedding_parameter_count: int,
    attention_direct_residual: bool = False,
    share_attention_projections: bool = False,
    attention_query_offsets: bool = False,
) -> tuple[int, int]:
    """Return fixed Q-model parameters and the size of one writer.

    The optional direct attention projection is counted per quantum whenever
    the controlled attention residual path is enabled.
    """

    d = int(d_model)
    rank = int(attention_rank)
    value = int(attention_value_dim)
    per_quantum = d + value + 1
    if bool(attention_direct_residual):
        per_quantum += value * d
    projection_parameters = 2 * rank * d + value * d
    if bool(share_attention_projections):
        quantum_fixed = len(quantum_counts) * projection_parameters
    else:
        quantum_fixed = sum(int(count) * projection_parameters for count in quantum_counts)
    quantum_fixed += sum(int(count) * per_quantum for count in quantum_counts)
    if bool(attention_query_offsets):
        quantum_fixed += sum(int(count) * rank for count in quantum_counts)
    readout = 2 * d + d * int(vocabulary_size) + int(vocabulary_size)
    parameters_per_writer = 2 * d + value + 1
    return (
        int(embedding_parameter_count) + quantum_fixed + readout,
        parameters_per_writer,
    )


def _writer_group_energies(
    directions: np.ndarray,
    *,
    d_model: int,
    hidden_size: int,
) -> np.ndarray:
    """Energy in source-MLP units: incoming row, bias, and outgoing column."""

    values = np.asarray(directions, dtype=np.float64)
    w1_size = int(hidden_size * d_model)
    b1_size = int(hidden_size)
    w2_size = int(d_model * hidden_size)
    expected = w1_size + b1_size + w2_size + int(d_model)
    if values.shape[-1] != expected:
        raise ValueError(
            f"expected {expected} MLP parameters, got {values.shape[-1]}"
        )
    w1 = values[..., :w1_size].reshape(
        *values.shape[:-1], hidden_size, d_model
    )
    offset = w1_size
    b1 = values[..., offset : offset + b1_size]
    offset += b1_size
    w2 = values[..., offset : offset + w2_size].reshape(
        *values.shape[:-1], d_model, hidden_size
    )
    offset += w2_size
    b2 = values[..., offset : offset + d_model]
    hidden_energy = (
        np.square(w1).sum(axis=-1)
        + np.square(b1)
        + np.square(w2).sum(axis=-2)
    )
    return np.concatenate(
        (hidden_energy, np.square(b2).sum(axis=-1, keepdims=True)), axis=-1
    )


def _effective_count(energies: np.ndarray) -> np.ndarray:
    values = np.asarray(energies, dtype=np.float64)
    numerator = np.square(values.sum(axis=-1))
    denominator = np.square(values).sum(axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > EPSILON,
    )


def _count_at_coverage(energies: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    sorted_energy = np.sort(np.asarray(energies), axis=-1)[..., ::-1]
    cumulative = np.cumsum(sorted_energy, axis=-1, dtype=np.float64)
    total = cumulative[..., -1:]
    normalized = np.divide(
        cumulative,
        total,
        out=np.zeros_like(cumulative),
        where=total > EPSILON,
    )
    return (np.argmax(normalized >= float(coverage), axis=-1) + 1).astype(int)


def _aggregate_interval_updates(
    layer_updates: np.ndarray, checkpoint_steps: np.ndarray
) -> np.ndarray:
    updates = np.asarray(layer_updates)
    steps = np.asarray(checkpoint_steps, dtype=int)
    if updates.ndim != 3:
        raise ValueError("layer_updates must have shape [step, layer, parameter]")
    if steps.ndim != 1 or len(steps) < 2 or steps[0] != 0:
        raise ValueError("checkpoint_steps must begin at zero and contain intervals")
    if np.any(np.diff(steps) <= 0) or steps[-1] > len(updates):
        raise ValueError("checkpoint_steps must increase within layer_updates")
    return np.stack(
        [updates[start:end].sum(axis=0) for start, end in zip(steps[:-1], steps[1:])],
        axis=0,
    )


def _quantile_step(curve: np.ndarray, steps: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(np.maximum(np.asarray(curve), 0.0), dtype=np.float64)
    if cumulative[-1] <= EPSILON:
        return float(steps[-1])
    index = int(np.searchsorted(cumulative / cumulative[-1], float(quantile)))
    return float(steps[min(index, len(steps) - 1)])


def dynamics_writer_metrics(
    layer_updates: np.ndarray,
    checkpoint_steps: np.ndarray,
    temporal_priorities: Sequence[np.ndarray],
    supports: Sequence[np.ndarray],
    *,
    d_model: int,
    mlp_ratio: float,
) -> tuple[DynamicsWriterMetrics, ...]:
    """Measure candidate complexity from source writer-group update packets.

    A candidate's acquisition packet is the source MLP parameter update summed
    over time with its normalized nonnegative priority curve. Its effective
    writer-group count is an inverse-concentration statistic over source hidden
    units. Demand is reported independently and never enters the complexity.
    """

    if len(temporal_priorities) != len(supports):
        raise ValueError("temporal priorities and supports must have the same layers")
    interval_updates = _aggregate_interval_updates(layer_updates, checkpoint_steps)
    if interval_updates.shape[1] != len(temporal_priorities):
        raise ValueError("update and priority layer counts do not match")
    interval_steps = np.asarray(checkpoint_steps, dtype=float)[1:]
    hidden_size = max(1, int(round(int(d_model) * float(mlp_ratio))))
    records = []
    for layer, (priority, owner_support) in enumerate(
        zip(temporal_priorities, supports)
    ):
        curves = np.asarray(priority, dtype=np.float64)
        owner_support = np.asarray(owner_support, dtype=bool)
        if curves.ndim != 2 or curves.shape[1] != len(interval_updates):
            raise ValueError("each priority array must have shape [quantum, interval]")
        if owner_support.ndim != 2 or owner_support.shape[1] != len(curves):
            raise ValueError("each support array must have shape [event, quantum]")
        if not np.isfinite(curves).all() or np.any(curves < -EPSILON):
            raise ValueError("priority curves must be finite and nonnegative")
        curve_mass = np.maximum(curves, 0.0).sum(axis=1, keepdims=True)
        if np.any(curve_mass <= EPSILON):
            raise ValueError("every candidate must have positive priority mass")
        normalized_priority = np.maximum(curves, 0.0) / curve_mass
        packet_directions = normalized_priority @ interval_updates[:, layer]
        packet_energies = _writer_group_energies(
            packet_directions, d_model=int(d_model), hidden_size=hidden_size
        )
        instantaneous_energies = _writer_group_energies(
            interval_updates[:, layer],
            d_model=int(d_model),
            hidden_size=hidden_size,
        )
        instantaneous_complexity = _effective_count(instantaneous_energies)
        records.append(
            DynamicsWriterMetrics(
                demand=owner_support.mean(axis=0, dtype=np.float64),
                tau_50_steps=np.asarray(
                    [
                        _quantile_step(curve, interval_steps, 0.5)
                        for curve in curves
                    ],
                    dtype=np.float64,
                ),
                packet_effective_groups=_effective_count(packet_energies),
                packet_groups_for_90_percent_energy=_count_at_coverage(
                    packet_energies
                ),
                priority_weighted_instantaneous_groups=(
                    normalized_priority @ instantaneous_complexity
                ),
            )
        )
    return tuple(records)


def functional_writer_metrics(
    residual_writes: np.ndarray,
    temporal_priorities: Sequence[np.ndarray],
    supports: Sequence[np.ndarray],
    *,
    energy: float = 0.90,
) -> tuple[FunctionalWriterMetrics, ...]:
    """Measure uncentered, support-priority weighted write covariance.

    ``residual_writes`` is the saved source write ``u_l(e, t)`` with shape
    ``[checkpoint, event, layer, d_model]``. Capacity deliberately uses the
    original discovery supports and curves, before graph closure.
    """
    values = np.asarray(residual_writes, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] < 2:
        raise ValueError("residual_writes must be [checkpoint, event, layer, d_model]")
    if not 0.0 < float(energy) <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    changes = np.diff(values, axis=0).transpose(1, 0, 2, 3)
    if len(temporal_priorities) != values.shape[2] or len(supports) != values.shape[2]:
        raise ValueError("residual writes, priorities, and supports must have matching layers")
    result = []
    for layer, (priority, support) in enumerate(zip(temporal_priorities, supports)):
        curves = np.maximum(np.asarray(priority, dtype=np.float64), 0.0)
        active = np.asarray(support, dtype=bool)
        if curves.ndim != 2 or curves.shape[1] != changes.shape[1]:
            raise ValueError("priority must be [quantum, checkpoint_interval]")
        if active.shape != (changes.shape[0], curves.shape[0]):
            raise ValueError("support must be [event, quantum]")
        ranks, spectra, zeros = [], [], []
        layer_changes = changes[:, :, layer, :]
        for quantum in range(curves.shape[0]):
            weights = active[:, quantum, None] * curves[quantum][None, :]
            total = float(weights.sum())
            covariance = np.einsum("et,eti,etj->ij", weights, layer_changes, layer_changes)
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1].clip(min=0.0)
            zero = total <= EPSILON or float(eigenvalues.sum()) <= EPSILON
            if zero:
                ranks.append(1)
            else:
                ranks.append(int(np.searchsorted(np.cumsum(eigenvalues) / eigenvalues.sum(), energy) + 1))
            spectra.append(eigenvalues)
            zeros.append(zero)
        result.append(FunctionalWriterMetrics(np.asarray(ranks, dtype=int), tuple(spectra), np.asarray(zeros, dtype=bool)))
    return tuple(result)


def allocate_overcomplete_writer_counts(
    complexities: Sequence[float], *, minimum: int = 4, factor: float = 4.0, maximum: int = 32
) -> np.ndarray:
    """Independently size each dictionary; no global budget is consumed."""
    values = np.asarray(tuple(complexities), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("complexities must be a nonempty positive vector")
    if int(minimum) <= 0 or int(maximum) < int(minimum) or not math.isfinite(float(factor)) or float(factor) <= 0:
        raise ValueError("invalid overcomplete writer allocation controls")
    return np.clip(np.ceil(float(factor) * values).astype(int), int(minimum), int(maximum))


def allocate_writer_counts(
    complexities: Sequence[float],
    total_writers: int,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Allocate a global integer budget after giving every quantum one writer."""

    values = np.asarray(tuple(complexities), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("complexities must be a nonempty vector")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("complexities must be finite and positive")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if int(total_writers) < len(values):
        raise ValueError("the total writer budget must assign at least one per quantum")
    remaining = int(total_writers) - len(values)
    logits = np.log(values) / float(temperature)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    fractional = remaining * probabilities
    extras = np.floor(fractional).astype(int)
    unassigned = remaining - int(extras.sum())
    if unassigned:
        remainders = fractional - extras
        order = np.lexsort((np.arange(len(values)), -remainders))
        extras[order[:unassigned]] += 1
    return extras + 1


def allocate_parameter_budget(
    complexities_by_layer: Sequence[Sequence[float]],
    *,
    source_parameter_count: int,
    parameter_budget_fraction: float,
    fixed_parameter_count: int,
    parameters_per_writer: int,
    temperature: float = 1.0,
) -> WriterBudgetAllocation:
    """Spend all complete writer units that fit beneath a source-relative budget."""

    if int(source_parameter_count) <= 0 or int(fixed_parameter_count) < 0:
        raise ValueError("source and fixed parameter counts are invalid")
    if int(parameters_per_writer) <= 0:
        raise ValueError("parameters_per_writer must be positive")
    if not math.isfinite(float(parameter_budget_fraction)) or not (
        0.0 < float(parameter_budget_fraction) <= 1.0
    ):
        raise ValueError("parameter_budget_fraction must be in (0, 1]")
    shapes = [len(tuple(values)) for values in complexities_by_layer]
    if not shapes or any(count <= 0 for count in shapes):
        raise ValueError("every layer must contain a quantum")
    flat_complexity = np.concatenate(
        [np.asarray(tuple(values), dtype=np.float64) for values in complexities_by_layer]
    )
    target = int(math.floor(float(parameter_budget_fraction) * source_parameter_count))
    available = target - int(fixed_parameter_count)
    total_writers = available // int(parameters_per_writer)
    quantum_count = len(flat_complexity)
    if total_writers < quantum_count:
        minimum = int(fixed_parameter_count) + quantum_count * int(
            parameters_per_writer
        )
        raise ValueError(
            "parameter budget cannot assign one writer per quantum: "
            f"target={target}, required={minimum}"
        )
    flat_counts = allocate_writer_counts(
        flat_complexity, total_writers, temperature=float(temperature)
    )
    counts = []
    offset = 0
    for size in shapes:
        counts.append(flat_counts[offset : offset + size])
        offset += size
    executable = int(fixed_parameter_count) + int(total_writers) * int(
        parameters_per_writer
    )
    return WriterBudgetAllocation(
        writer_counts=tuple(counts),
        target_parameter_count=target,
        fixed_parameter_count=int(fixed_parameter_count),
        parameters_per_writer=int(parameters_per_writer),
        total_writers=int(total_writers),
        executable_parameter_count=executable,
        unused_parameter_count=target - executable,
    )


def allocate_uniform_writer_ceiling(
    quantum_counts: Sequence[int],
    *,
    writers_per_quantum: int,
    source_parameter_count: int,
    parameter_budget_fraction: float,
    fixed_parameter_count: int,
    parameters_per_writer: int,
) -> WriterBudgetAllocation:
    """Give every quantum the same small bank, clipped only by a global ceiling."""

    counts = tuple(int(value) for value in quantum_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("quantum_counts must be nonempty and positive")
    if int(writers_per_quantum) <= 0:
        raise ValueError("writers_per_quantum must be positive")
    if not 0.0 < float(parameter_budget_fraction) <= 1.0:
        raise ValueError("parameter_budget_fraction must be in (0, 1]")
    target = int(math.floor(float(parameter_budget_fraction) * source_parameter_count))
    available = target - int(fixed_parameter_count)
    maximum_total = available // int(parameters_per_writer)
    quantum_total = sum(counts)
    if maximum_total < quantum_total:
        raise ValueError(
            "parameter ceiling cannot assign one writer per quantum: "
            f"target={target}, fixed={fixed_parameter_count}, "
            f"quantum_count={quantum_total}"
        )
    per_quantum = min(int(writers_per_quantum), maximum_total // quantum_total)
    writer_counts = tuple(
        np.full(count, per_quantum, dtype=int) for count in counts
    )
    total_writers = per_quantum * quantum_total
    executable = int(fixed_parameter_count) + total_writers * int(parameters_per_writer)
    return WriterBudgetAllocation(
        writer_counts=writer_counts,
        target_parameter_count=target,
        fixed_parameter_count=int(fixed_parameter_count),
        parameters_per_writer=int(parameters_per_writer),
        total_writers=total_writers,
        executable_parameter_count=executable,
        unused_parameter_count=target - executable,
    )
