from __future__ import annotations

import argparse
import logging

from quanta.config import load_experiment_config, load_plot_config
from quanta.config.paths import resolve_run_config
from quanta.experiments import create_experiment
from quanta.experiments.common import log_event


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an experiment from one YAML config. Paths may be explicit or "
            "relative to configs/, and the .yaml suffix is optional."
        )
    )
    parser.add_argument(
        "config",
        help="Experiment config, for example scaling_laws/cxor/default.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional plotting config path or name under configs/plot/.",
    )
    return parser.parse_args()


def run_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    resolved = resolve_run_config(
        args.config,
        plot_name=args.plot,
    )
    log_event(
        resolved.experiment_name,
        "resolved",
        config=resolved.experiment_path,
        plot=resolved.plot_path,
    )

    plot_config = load_plot_config(
        resolved.plot_path,
        experiment_name=resolved.experiment_name,
    )
    experiment_config = load_experiment_config(
        resolved.experiment_name,
        resolved.experiment_path,
    )
    log_event(resolved.experiment_name, "starting", config=resolved.experiment_path)
    create_experiment(resolved.experiment_name, experiment_config, plot_config).run()


if __name__ == "__main__":
    run_cli()
