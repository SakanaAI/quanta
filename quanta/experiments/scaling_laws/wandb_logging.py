from __future__ import annotations

from typing import Any

import wandb

from quanta.config import ScalingLawsConfig


def log_scaling_wandb(
    *,
    config: ScalingLawsConfig,
    step: int,
    samples_seen: int,
    diagnostics: dict[str, Any],
) -> None:
    if wandb.run is None:
        return

    width = int(config.width)
    learning_rate = float(config.lr)
    lr_slug = f"{learning_rate:.0e}".replace("-", "m").replace("+", "p")
    namespace = f"runs/width{width}_lr{lr_slug}"
    payload = {
        f"{namespace}/loss_bits": float(diagnostics["eval_loss_bits"]),
        f"{namespace}/weighted_accuracy": float(diagnostics["weighted_accuracy"]),
        f"{namespace}/samples": int(samples_seen),
        f"{namespace}/step": int(step),
    }
    depth_values = sorted(
        {
            int(depth)
            for field in (
                "mean_depth_loss",
                "mean_depth_accuracy",
                "weighted_depth_loss",
                "weighted_depth_accuracy",
                "depth_learned_percentages",
            )
            for depth in diagnostics.get(field, {})
        }
    )
    for depth in depth_values:
        mean_loss = diagnostics.get("mean_depth_loss", {}).get(depth)
        if mean_loss is not None:
            payload[f"{namespace}/loss_by_depth/depth_{depth}"] = float(
                mean_loss["loss_bits"]
            )
        if depth in diagnostics.get("depth_learned_percentages", {}):
            payload[f"{namespace}/learned_by_depth/depth_{depth}"] = float(
                diagnostics["depth_learned_percentages"][depth]
            )

    # Let W&B own a monotone global history index.  Each sequential width/LR
    # variant has its own namespace and carries its physical training step, so
    # later variants are not rejected when their local step restarts at zero.
    wandb.log(payload)
