"""Run NumberNaming discovery and fit its executable Q-model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from quanta.config import load_experiment_config, load_plot_config
from quanta.experiments import create_experiment


CONFIG_PATH = Path("configs/quanta_discovery/number_naming/main.yaml")
PLOT_PATH = Path("configs/plot/default.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config("quanta_discovery", CONFIG_PATH)
    config.device = args.device
    if args.dry_run:
        print(json.dumps(asdict(config), indent=2))
        return
    plot_config = load_plot_config(PLOT_PATH, experiment_name="quanta_discovery")
    source_run = Path(create_experiment("quanta_discovery", config, plot_config).run())
    output_dir = source_run / "qmodel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_quanta_model",
            str(source_run),
            "--priority-dir",
            str(source_run / "factorization"),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
        ],
        check=True,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
