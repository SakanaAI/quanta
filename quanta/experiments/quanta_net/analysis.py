from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantanet import QCoreTrace


@dataclass(frozen=True)
class PostHocProbeResult:
    probe: nn.Linear
    train_accuracy: float
    evaluation_accuracy: float


def q_node_features(trace: QCoreTrace, node_index: int) -> torch.Tensor:
    return trace.messages[..., int(node_index), :].detach()


def q_depth_features(trace: QCoreTrace, depth_index: int) -> torch.Tensor:
    return trace.depth_updates[..., int(depth_index), :].detach()


def train_posthoc_linear_probe(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    evaluation_features: torch.Tensor,
    evaluation_labels: torch.Tensor,
    classes: int,
    steps: int = 200,
    lr: float = 1.0e-2,
) -> PostHocProbeResult:
    """Fit a removable linear diagnostic without modifying source representations."""
    train_features = train_features.detach()
    evaluation_features = evaluation_features.detach()
    probe = nn.Linear(train_features.shape[-1], int(classes)).to(train_features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(train_features).argmax(-1) == train_labels).float().mean())
        evaluation_accuracy = float(
            (probe(evaluation_features).argmax(-1) == evaluation_labels).float().mean()
        )
    return PostHocProbeResult(probe, train_accuracy, evaluation_accuracy)
