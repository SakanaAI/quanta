from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from quanta.experiments.number_naming.model import DecoderTransformerLM, TransformerResidualTrace

from .quantanet import QComputer, QCoreTrace


AlignmentTargetControl = str


@dataclass(frozen=True)
class AlignmentTrace:
    teacher: QCoreTrace
    transformer: TransformerResidualTrace
    projected_block_updates: torch.Tensor
    teacher_depth_updates: torch.Tensor
    alignment_loss: torch.Tensor
    alignment_loss_by_depth: torch.Tensor
    cumulative_state_mse: torch.Tensor
    cumulative_state_mse_by_depth: torch.Tensor
    target_control: AlignmentTargetControl


class FixedQInterface(nn.Module):
    """Fixed map U from Q-message space into the transformer residual stream."""

    def __init__(self, d_quantum: int, d_model: int, *, seed: int) -> None:
        super().__init__()
        if int(d_model) < int(d_quantum):
            raise ValueError("transformer d_model must be at least d_quantum.")
        if int(d_model) == int(d_quantum):
            matrix = torch.eye(int(d_quantum))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            random = torch.randn(int(d_model), int(d_quantum), generator=generator)
            matrix, _ = torch.linalg.qr(random, mode="reduced")
        self.register_buffer("matrix", matrix, persistent=True)

    @property
    def is_identity(self) -> bool:
        rows, columns = self.matrix.shape
        return rows == columns and bool(torch.equal(self.matrix, torch.eye(rows, device=self.matrix.device)))

    def to_model(self, value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...q,mq->...m", value, self.matrix)

    def to_quantum(self, value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...m,mq->...q", value, self.matrix)


class FrozenQTeacher(nn.Module):
    def __init__(self, q_computer: QComputer) -> None:
        super().__init__()
        self.q_computer = q_computer
        self.q_computer.eval()
        for parameter in self.q_computer.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenQTeacher:
        super().train(False)
        self.q_computer.eval()
        return self

    def forward(self, **kwargs) -> QCoreTrace:
        with torch.no_grad():
            trace = self.q_computer(routing="oracle", **kwargs)
        return _detach_q_trace(trace)


class LayerwiseQAlignment(nn.Module):
    """Training-only Q targets around an ordinary deployment transformer."""

    def __init__(
        self,
        *,
        transformer: DecoderTransformerLM,
        teacher: FrozenQTeacher,
        interface_seed: int,
        depth_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.teacher = teacher
        self.depth = int(teacher.q_computer.core.depth)
        if len(transformer.layers) != self.depth:
            raise ValueError(
                "layerwise Q alignment requires exactly one complete transformer block per Q depth: "
                f"transformer_layers={len(transformer.layers)}, q_depth={self.depth}."
            )
        self.interface = FixedQInterface(
            teacher.q_computer.core.d_quantum,
            transformer.d_model,
            seed=int(interface_seed),
        )
        scales = torch.ones(self.depth) if depth_scales is None else torch.as_tensor(depth_scales).float()
        if tuple(scales.shape) != (self.depth,):
            raise ValueError("depth_scales must contain exactly one value per Q depth.")
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError("depth_scales must be finite and strictly positive.")
        self.register_buffer("depth_scales", scales, persistent=True)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        activity_targets: torch.Tensor,
        valid_prediction_mask: torch.Tensor,
        target_control: AlignmentTargetControl = "normal",
        control_seed: int = 0,
    ) -> AlignmentTrace:
        if target_control not in {"normal", "depth_shuffled", "example_shuffled"}:
            raise ValueError(
                "target_control must be 'normal', 'depth_shuffled', or 'example_shuffled'."
            )
        teacher = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            activity_targets=activity_targets,
        )
        transformer = self.transformer.residual_trace(input_ids, attention_mask)
        targets = _controlled_targets(
            teacher.depth_updates,
            control=target_control,
            seed=int(control_seed),
        )
        targets = _pad_teacher_updates(targets, input_ids.shape[1]).detach()

        block_boundaries = transformer.block_boundaries
        block_updates = torch.stack(
            [after - before for before, after in zip(block_boundaries, block_boundaries[1:])],
            dim=-2,
        )
        projected = self.interface.to_quantum(block_updates)
        valid = valid_prediction_mask.to(torch.bool)
        alignment_by_depth = layerwise_alignment_loss_by_depth(
            projected,
            targets,
            valid_mask=valid,
            depth_scales=self.depth_scales,
        )
        alignment = alignment_by_depth.mean()

        initial = block_boundaries[0]
        cumulative_transformer = torch.stack(
            [self.interface.to_quantum(boundary - initial) for boundary in block_boundaries[1:]],
            dim=-2,
        )
        cumulative_teacher = targets.cumsum(dim=-2)
        cumulative_mse_by_depth = _unscaled_depth_mse_by_depth(
            cumulative_transformer,
            cumulative_teacher,
            valid_mask=valid,
        )
        cumulative_mse = cumulative_mse_by_depth.mean()
        return AlignmentTrace(
            teacher=teacher,
            transformer=transformer,
            projected_block_updates=projected,
            teacher_depth_updates=targets,
            alignment_loss=alignment,
            alignment_loss_by_depth=alignment_by_depth,
            cumulative_state_mse=cumulative_mse,
            cumulative_state_mse_by_depth=cumulative_mse_by_depth,
            target_control=target_control,
        )

    def deployment_model(self) -> DecoderTransformerLM:
        return self.transformer


def alignment_objective(
    trace: AlignmentTrace,
    *,
    labels: torch.Tensor,
    lambda_align: float,
) -> dict[str, torch.Tensor]:
    task_loss = _causal_ce(trace.transformer.logits, labels)
    total = task_loss + float(lambda_align) * trace.alignment_loss
    return {
        "loss": total,
        "task_loss": task_loss,
        "alignment_loss": trace.alignment_loss,
        "alignment_loss_by_depth": trace.alignment_loss_by_depth,
        "cumulative_state_mse": trace.cumulative_state_mse,
        "cumulative_state_mse_by_depth": trace.cumulative_state_mse_by_depth,
    }


def rms_depth_scales(
    squared_norm_sums: torch.Tensor,
    event_counts: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return fixed sqrt(E[||Z_l||^2]) scales from accumulated statistics."""

    if squared_norm_sums.ndim != 1 or event_counts.shape != squared_norm_sums.shape:
        raise ValueError("RMS statistics must contain one scalar per Q depth.")
    if float(epsilon) <= 0:
        raise ValueError("RMS scale epsilon must be positive.")
    return (squared_norm_sums / event_counts.clamp_min(1)).sqrt().clamp_min(float(epsilon))


def accumulate_depth_scale_statistics(
    depth_updates: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth_updates.ndim != 4:
        raise ValueError("depth_updates must have shape [batch, position, depth, d_quantum].")
    if valid_mask.shape != depth_updates.shape[:2]:
        raise ValueError("valid_mask must match the depth-update batch and position dimensions.")
    valid = valid_mask.to(depth_updates.dtype).unsqueeze(-1)
    squared_norms = depth_updates.pow(2).sum(dim=-1)
    sums = (squared_norms * valid).sum(dim=(0, 1))
    counts = valid.sum(dim=(0, 1)).expand(depth_updates.shape[-2])
    return sums, counts


def depth_shuffled_control(depth_updates: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=depth_updates.device)
    generator.manual_seed(int(seed))
    permutation = torch.randperm(depth_updates.shape[-2], generator=generator, device=depth_updates.device)
    return depth_updates.index_select(-2, permutation)


def example_shuffled_control(depth_updates: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=depth_updates.device)
    generator.manual_seed(int(seed))
    permutation = torch.randperm(depth_updates.shape[0], generator=generator, device=depth_updates.device)
    return depth_updates.index_select(0, permutation)


def _controlled_targets(
    depth_updates: torch.Tensor,
    *,
    control: AlignmentTargetControl,
    seed: int,
) -> torch.Tensor:
    if control == "depth_shuffled":
        return depth_shuffled_control(depth_updates, seed=seed)
    if control == "example_shuffled":
        return example_shuffled_control(depth_updates, seed=seed)
    return depth_updates


def layerwise_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    depth_scales: torch.Tensor,
) -> torch.Tensor:
    return layerwise_alignment_loss_by_depth(
        predicted,
        target,
        valid_mask=valid_mask,
        depth_scales=depth_scales,
    ).mean()


def layerwise_alignment_loss_by_depth(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    depth_scales: torch.Tensor,
) -> torch.Tensor:
    squared_l2 = (predicted - target).pow(2).sum(dim=-1)
    scaled = squared_l2 / depth_scales.square().view(1, 1, -1)
    valid = valid_mask.to(scaled.dtype).unsqueeze(-1)
    counts = valid.sum(dim=(0, 1)).clamp_min(1)
    return (scaled * valid).sum(dim=(0, 1)) / counts


def _unscaled_depth_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return _unscaled_depth_mse_by_depth(
        predicted,
        target,
        valid_mask=valid_mask,
    ).mean()


def _unscaled_depth_mse_by_depth(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    squared_l2 = (predicted - target).pow(2).sum(dim=-1)
    valid = valid_mask.to(squared_l2.dtype).unsqueeze(-1)
    counts = valid.sum(dim=(0, 1)).clamp_min(1)
    return (squared_l2 * valid).sum(dim=(0, 1)) / counts


def _causal_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _pad_teacher_updates(updates: torch.Tensor, length: int) -> torch.Tensor:
    if updates.shape[1] > int(length):
        raise ValueError("Q teacher exposes more prediction positions than transformer residuals.")
    if updates.shape[1] == int(length):
        return updates
    padding = updates.new_zeros(updates.shape[0], int(length) - updates.shape[1], *updates.shape[2:])
    return torch.cat((updates, padding), dim=1)


def _detach_q_trace(trace: QCoreTrace) -> QCoreTrace:
    return QCoreTrace(**{name: value.detach() for name, value in trace.__dict__.items()})
