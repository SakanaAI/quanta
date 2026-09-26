"""Reproduce the composed-task loss-decomposition panel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from quanta.experiments.scaling_laws.compositional_mlp import (
    CompositionalMLPConfig,
    run,
)


def experiment_config(*, device: str) -> CompositionalMLPConfig:
    return CompositionalMLPConfig(
        n_tasks=32,
        group_size=8,
        subgroup_size=2,
        degree_a=3,
        degree_b=1,
        degree_c=0,
        composition="parallel_nand",
        event_scope="primitive",
        support_layout="shared_pool",
        support_pool_bits=32,
        support_seed=0,
        task_conditioning="embedding",
        sensor_bits=96,
        frequency_exponent=0.75,
        rank_seed=3,
        model_seed=0,
        data_seed=1_000,
        eval_seed=10_000,
        width=256,
        hidden_layers=2,
        optimizer="adam",
        learning_rate=1.0e-3,
        batch_size=512,
        microbatch_size=512,
        steps=30_000,
        eval_every=50,
        eval_samples_per_task=256,
        eval_task_chunk=16,
        sampling="iid",
        precision="fp32",
        device=device,
        output_dir=".experiments/loss_decomposition",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = experiment_config(device=args.device)
    if args.dry_run:
        print(json.dumps(asdict(config), indent=2))
        return
    print(run(config))


if __name__ == "__main__":
    main()
