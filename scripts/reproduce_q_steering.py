"""Compile, train, and compare the Q-aligned Transformer with its control."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from quanta.config import load_experiment_config, load_plot_config
from quanta.experiments import create_experiment


Q_CONFIG_PATH = Path("configs/quanta_net/number_naming/main.yaml")
CONTROL_CONFIG_PATH = Path("configs/quanta_steering/number_naming/control.yaml")
ALIGNED_CONFIG_PATH = Path("configs/quanta_steering/number_naming/aligned.yaml")
PLOT_PATH = Path("configs/plot/default.yaml")


def compile_program() -> Path:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "quanta.experiments.number_naming.compile_qprogram",
            str(Q_CONFIG_PATH),
            "--program",
            "factorized",
            "--audit-samples",
            "4096",
            "--audit-seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("the compiler did not report its artifact directory")
    return Path(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    q_config = load_experiment_config("quanta_net", Q_CONFIG_PATH)
    control = load_experiment_config("quanta_steering", CONTROL_CONFIG_PATH)
    aligned = load_experiment_config("quanta_steering", ALIGNED_CONFIG_PATH)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "q_model": asdict(q_config),
                    "control": asdict(control),
                    "aligned": asdict(aligned),
                },
                indent=2,
            )
        )
        return
    compiled = compile_program()
    q_config.compiled_program_path = str(compiled)
    q_config.device = args.device
    q_plot = load_plot_config(PLOT_PATH, experiment_name="quanta_net")
    teacher_run = Path(create_experiment("quanta_net", q_config, q_plot).run())
    outputs: dict[str, str] = {"teacher": str(teacher_run)}
    steering_plot = load_plot_config(PLOT_PATH, experiment_name="quanta_steering")
    for label, config in (("control", control), ("aligned", aligned)):
        config.q_checkpoint = str(teacher_run / "model.pt")
        config.device = args.device
        outputs[label] = create_experiment(
            "quanta_steering", config, steering_plot
        ).run()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
