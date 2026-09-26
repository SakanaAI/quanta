from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from .errors import CompilationError
from .types import AlignedEventTrace, CompiledQProgram, EventTrace


@dataclass(frozen=True)
class ModuleParameterCount:
    name: str
    total: int
    trainable: int
    shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ParameterAudit:
    per_node: tuple[ModuleParameterCount, ...]
    shared: tuple[ModuleParameterCount, ...]
    per_node_parameter_count: int
    quantum_parameters: int
    shared_parameters: int
    total_parameters: int


def audit_homogeneous_parameters(
    node_modules: Mapping[str, nn.Module],
    shared_modules: Mapping[str, nn.Module | torch.Tensor | nn.Parameter],
    *,
    per_node_tensors: Mapping[str, tuple[torch.Tensor, ...]] | None = None,
) -> ParameterAudit:
    """Verify equal node allocation and count shared infrastructure once."""
    if not node_modules:
        raise CompilationError("a homogeneous parameter audit requires at least one quantum module")
    per_node_tensors = per_node_tensors or {}
    node_counts = tuple(
        _module_count(name, module, extra=per_node_tensors.get(name, ()))
        for name, module in sorted(node_modules.items())
    )
    signatures = {(item.total, item.trainable, item.shapes) for item in node_counts}
    if len(signatures) != 1:
        details = ", ".join(f"{item.name}={item.total}" for item in node_counts)
        raise CompilationError(f"quantum modules do not have equal parameter allocation: {details}")
    shared_counts = tuple(
        _value_count(name, value) for name, value in sorted(shared_modules.items())
    )
    node_parameter_count = node_counts[0].total
    quantum_parameters = sum(item.total for item in node_counts)
    shared_parameters = sum(item.total for item in shared_counts)
    return ParameterAudit(
        per_node=node_counts,
        shared=shared_counts,
        per_node_parameter_count=node_parameter_count,
        quantum_parameters=quantum_parameters,
        shared_parameters=shared_parameters,
        total_parameters=quantum_parameters + shared_parameters,
    )


def activity_tensors(
    compiled: CompiledQProgram,
    *,
    split: str,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    traces = _split_traces(compiled, split)
    width = len(compiled.nodes)
    if not traces:
        empty = torch.empty((0, width), dtype=torch.float32, device=device)
        return empty, empty.clone()
    targets = [list(trace.activity_targets) for trace in traces]
    masks = [list(trace.activity_mask) for trace in traces]
    return (
        torch.tensor(targets, dtype=torch.float32, device=device),
        torch.tensor(masks, dtype=torch.float32, device=device),
    )


def _split_traces(
    compiled: CompiledQProgram,
    split: str,
) -> tuple[EventTrace, ...] | tuple[AlignedEventTrace, ...]:
    if split == "validation":
        return compiled.validation_traces
    if split == "train":
        return compiled.training_traces
    if split in {"eval", "evaluation"}:
        return compiled.evaluation_traces
    raise ValueError("split must be 'validation', 'train', or 'evaluation'")


def _module_count(
    name: str,
    module: nn.Module,
    *,
    extra: tuple[torch.Tensor, ...] = (),
) -> ModuleParameterCount:
    parameters = (*tuple(module.parameters()), *extra)
    return ModuleParameterCount(
        name=name,
        total=sum(parameter.numel() for parameter in parameters),
        trainable=sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )


def _value_count(
    name: str, value: nn.Module | torch.Tensor | nn.Parameter
) -> ModuleParameterCount:
    if isinstance(value, nn.Module):
        return _module_count(name, value)
    return ModuleParameterCount(
        name=name,
        total=value.numel(),
        trainable=value.numel() if value.requires_grad else 0,
        shapes=(tuple(value.shape),),
    )
