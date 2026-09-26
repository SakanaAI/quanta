from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quanta.config import load_quanta_net_config
from quanta.experiments.number_naming.task import NumberNamingTask
from quanta.qprogram import CompiledQProgram, CompiledSupervisionIndex
from quanta.utils import get_device, set_seeds

from .experiment import _build_q_computer, _evaluate_q


def evaluate_checkpoint(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    parent_audit_size: int = 512,
    include_full_decode: bool = True,
) -> Path:
    config = load_quanta_net_config(config_path)
    set_seeds(int(config.seed))
    device = torch.device(get_device(config.device))
    compiled = CompiledQProgram.read(config.compiled_program_path)
    supervision = CompiledSupervisionIndex(compiled)
    task = NumberNamingTask(config)
    model = _build_q_computer(config, compiled, task).to(device)

    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics = _evaluate_q(
        model,
        task,
        supervision,
        compiled,
        device=device,
        include_full_decode=bool(include_full_decode),
        parent_audit_size=int(parent_audit_size),
    )
    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "compiled_program_fingerprint": compiled.metadata.program_fingerprint,
        "compiler_version": compiled.metadata.compiler_version,
        "parent_audit_size": int(parent_audit_size),
        "full_decode": bool(include_full_decode),
        "metrics": metrics,
    }
    output = Path(output_path) if output_path is not None else checkpoint.parent / "detailed_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a standalone compiled Q checkpoint.")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--parent-audit-size", type=int, default=512)
    parser.add_argument("--skip-greedy", action="store_true")
    args = parser.parse_args(argv)
    output = evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        parent_audit_size=args.parent_audit_size,
        include_full_decode=not args.skip_greedy,
    )
    print(output)


if __name__ == "__main__":
    main()
