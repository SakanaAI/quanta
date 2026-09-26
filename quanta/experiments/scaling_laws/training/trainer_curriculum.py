from __future__ import annotations

import math


def scheduled_learning_rate(
    base_lr: float,
    step: int,
    total_steps: int,
    scheduler: str,
    *,
    warmup_phase: float = 0.1,
    plateau_phase: float = 0.0,
) -> float:
    if scheduler == "constant":
        return float(base_lr)
    if scheduler == "linear_after_plateau":
        plateau_phase = min(max(float(plateau_phase), 0.0), 1.0)
        plateau_steps = int(round(int(total_steps) * plateau_phase))
        if int(step) <= plateau_steps:
            return float(base_lr)
        decay_steps = int(total_steps) - plateau_steps
        if decay_steps <= 0:
            return float(base_lr)
        progress = min(max((int(step) - plateau_steps) / decay_steps, 0.0), 1.0)
        return float(base_lr * (1.0 - progress))
    warmup_phase = min(max(float(warmup_phase), 0.0), 1.0)
    warmup_steps = int(round(int(total_steps) * warmup_phase))

    if warmup_steps > 0 and int(step) <= warmup_steps:
        return float(base_lr * min(max(int(step) / warmup_steps, 0.0), 1.0))

    if scheduler == "constant_with_warmup":
        return float(base_lr)

    if total_steps <= warmup_steps + 1:
        progress = 0.0
    elif total_steps <= 1:
        progress = 0.0
    else:
        progress = min(
            max((int(step) - warmup_steps - 1) / (int(total_steps) - warmup_steps - 1), 0.0),
            1.0,
        )
    if scheduler == "linear":
        return float(base_lr * (1.0 - progress))
    if scheduler == "cosine":
        return float(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(
        "scheduler must be 'constant', 'constant_with_warmup', 'linear', "
        "'cosine', or 'linear_after_plateau'."
    )


def learned_task_percentages_by_depth(
    diagnostics: dict,
    threshold_bits: float,
) -> dict[int, float]:
    counts_by_depth: dict[int, int] = {}
    learned_by_depth: dict[int, int] = {}
    for values in diagnostics.get("task_losses", {}).values():
        depth = int(values["depth"])
        counts_by_depth[depth] = counts_by_depth.get(depth, 0) + 1
        if float(values["loss_bits"]) <= float(threshold_bits):
            learned_by_depth[depth] = learned_by_depth.get(depth, 0) + 1
    return {
        depth: 100.0 * learned_by_depth.get(depth, 0) / count
        for depth, count in sorted(counts_by_depth.items())
    }
