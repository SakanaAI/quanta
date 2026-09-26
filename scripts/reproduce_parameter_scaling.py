"""Run the selected HSP parameter-scaling protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from quanta.experiments.scaling_laws.hierarchical_parity import (
    HierarchicalParityConfig,
    run,
)


WIDTHS = (8, 16, 24, 32, 48, 64, 96, 128, 256, 512)
ALPHAS = {
    "0.2": ("alpha020", 2.29739671),
    "0.25": ("alpha025", 2.37841423),
    "0.339137": ("alpha0339", 2.53),
}


def experiment_config(alpha: str, width: int, *, device: str) -> HierarchicalParityConfig:
    label, beta = ALPHAS[alpha]
    return HierarchicalParityConfig(
        base_tasks=8,
        branching_factor=2,
        max_depth=5,
        demand_law="rho_beta",
        beta=beta,
        root_frequency_exponent=0.0,
        rank_seed=0,
        support_pool_bits=100,
        support_seed=0,
        model_seed=0,
        data_seed=7_000,
        eval_seed=10_000,
        width=width,
        hidden_layers=5,
        task_conditioning="early",
        parameterization="mup",
        mup_base_width=8,
        mup_delta_width=16,
        optimizer="sgd",
        learning_rate=0.0075,
        batch_size=16_384,
        microbatch_size=16_384,
        steps=20_000_000,
        eval_every=5_000,
        checkpoint_every=100_000,
        eval_samples=256,
        eval_task_chunk=8,
        sampling="iid",
        precision="bf16",
        device=device,
        output_dir=f".experiments/parameter_scaling/{label}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", choices=tuple(ALPHAS))
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.all:
        if args.alpha is not None or args.width is not None:
            parser.error("--all cannot be combined with --alpha or --width")
        selections = [(alpha, width) for alpha in ALPHAS for width in WIDTHS]
    else:
        if args.alpha is None or args.width is None:
            parser.error("provide --alpha and --width, or use --all")
        selections = [(args.alpha, args.width)]
    configs = [experiment_config(alpha, width, device=args.device) for alpha, width in selections]
    if args.dry_run:
        print(json.dumps([asdict(config) for config in configs], indent=2))
        return
    for config in configs:
        print(run(config))


if __name__ == "__main__":
    main()
