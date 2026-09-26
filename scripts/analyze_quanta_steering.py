#!/usr/bin/env python3
"""Compare ordinary, aligned, and lambda=0 transformers against one Q teacher."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from quanta.config import PosetsProbingConfig, QuantaSteeringConfig
from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.number_naming.task import NumberNamingTask
from quanta.experiments.quanta_net.experiment import (
    _batch_supervision,
    _build_q_computer,
    _evaluate_transformer,
    _load_teacher_run,
    _prediction_mask,
    _validate_q_teacher_metadata,
)
from quanta.experiments.quanta_net.steering import FrozenQTeacher, _pad_teacher_updates
from quanta.qprogram import CompiledSupervisionIndex
from quanta.utils import _jsonable, get_device, set_seeds


class BlockAblatedTransformer(nn.Module):
    """Deployment transformer with selected complete attention-plus-MLP blocks skipped."""

    def __init__(self, model: DecoderTransformerLM, skipped_blocks: set[int]) -> None:
        super().__init__()
        self.model = model
        self.skipped_blocks = frozenset(int(index) for index in skipped_blocks)
        invalid = self.skipped_blocks - set(range(len(model.layers)))
        if invalid:
            raise ValueError(f"invalid skipped block indices: {sorted(invalid)}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.model.token_embedding(input_ids) + self.model.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding_mask = (
            attention_mask == 0
            if attention_mask is not None
            else input_ids == self.model.pad_id
        )
        for index, layer in enumerate(self.model.layers):
            if index in self.skipped_blocks:
                continue
            hidden = layer.attention_write(
                hidden,
                causal_mask=causal_mask,
                padding_mask=padding_mask,
            )
            hidden = layer.mlp_write(hidden)
        return self.model.head(self.model.final_norm(hidden))


class AlignmentAccumulator:
    def __init__(self, depth: int) -> None:
        self.depth = int(depth)
        self.count = 0
        self.error = torch.zeros(depth, depth, dtype=torch.float64)
        self.dot = torch.zeros(depth, depth, dtype=torch.float64)
        self.update_sq = torch.zeros(depth, dtype=torch.float64)
        self.target_sq = torch.zeros(depth, dtype=torch.float64)
        self.cumulative_error = torch.zeros(depth, dtype=torch.float64)
        self.cumulative_dot = torch.zeros(depth, dtype=torch.float64)
        self.cumulative_update_sq = torch.zeros(depth, dtype=torch.float64)
        self.cumulative_target_sq = torch.zeros(depth, dtype=torch.float64)
        self.active_count = torch.zeros(depth, dtype=torch.float64)
        self.active_error = torch.zeros(depth, dtype=torch.float64)
        self.active_dot = torch.zeros(depth, dtype=torch.float64)
        self.active_update_sq = torch.zeros(depth, dtype=torch.float64)
        self.active_target_sq = torch.zeros(depth, dtype=torch.float64)
        self.inactive_count = torch.zeros(depth, dtype=torch.float64)
        self.inactive_update_sq = torch.zeros(depth, dtype=torch.float64)

    def add(self, updates: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
        update_events = updates[valid].to(torch.float64).cpu()
        target_events = targets[valid].to(torch.float64).cpu()
        if not len(update_events):
            return
        self.count += len(update_events)
        difference = update_events[:, :, None, :] - target_events[:, None, :, :]
        self.error += difference.square().sum(dim=-1).sum(dim=0)
        self.dot += torch.einsum("nbq,ndq->bd", update_events, target_events)
        self.update_sq += update_events.square().sum(dim=-1).sum(dim=0)
        self.target_sq += target_events.square().sum(dim=-1).sum(dim=0)
        target_norm_sq = target_events.square().sum(dim=-1)
        update_norm_sq = update_events.square().sum(dim=-1)
        diagonal_error = (update_events - target_events).square().sum(dim=-1)
        diagonal_dot = (update_events * target_events).sum(dim=-1)
        active = target_norm_sq > 1.0e-12
        inactive = ~active
        self.active_count += active.sum(dim=0)
        self.active_error += (diagonal_error * active).sum(dim=0)
        self.active_dot += (diagonal_dot * active).sum(dim=0)
        self.active_update_sq += (update_norm_sq * active).sum(dim=0)
        self.active_target_sq += (target_norm_sq * active).sum(dim=0)
        self.inactive_count += inactive.sum(dim=0)
        self.inactive_update_sq += (update_norm_sq * inactive).sum(dim=0)

        cumulative_updates = update_events.cumsum(dim=1)
        cumulative_targets = target_events.cumsum(dim=1)
        self.cumulative_error += (
            cumulative_updates - cumulative_targets
        ).square().sum(dim=-1).sum(dim=0)
        self.cumulative_dot += (cumulative_updates * cumulative_targets).sum(dim=-1).sum(dim=0)
        self.cumulative_update_sq += cumulative_updates.square().sum(dim=-1).sum(dim=0)
        self.cumulative_target_sq += cumulative_targets.square().sum(dim=-1).sum(dim=0)

    def summary(self) -> dict[str, Any]:
        count = max(self.count, 1)
        mse = self.error / count
        cosine = self.dot / (
            self.update_sq.sqrt()[:, None] * self.target_sq.sqrt()[None, :]
        ).clamp_min(1.0e-12)
        diagonal = torch.arange(self.depth)
        off_diagonal = ~torch.eye(self.depth, dtype=torch.bool)
        best_targets = mse.argmin(dim=1)
        ranks = torch.argsort(torch.argsort(mse, dim=1), dim=1)[diagonal, diagonal] + 1
        cumulative_cosine = self.cumulative_dot / (
            self.cumulative_update_sq.sqrt() * self.cumulative_target_sq.sqrt()
        ).clamp_min(1.0e-12)
        per_depth = []
        for index in range(self.depth):
            target_energy = float(self.target_sq[index] / count)
            per_depth.append(
                {
                    "depth": index + 1,
                    "mse": float(mse[index, index]),
                    "relative_mse": float(mse[index, index] / max(target_energy, 1.0e-12)),
                    "cosine": float(cosine[index, index]),
                    "transformer_update_rms_norm": float((self.update_sq[index] / count).sqrt()),
                    "q_target_rms_norm": float((self.target_sq[index] / count).sqrt()),
                    "active_target_fraction": float(self.active_count[index] / count),
                    "active_target_mse": float(
                        self.active_error[index] / self.active_count[index].clamp_min(1)
                    ),
                    "active_target_cosine": float(
                        self.active_dot[index]
                        / (
                            self.active_update_sq[index].sqrt()
                            * self.active_target_sq[index].sqrt()
                        ).clamp_min(1.0e-12)
                    ),
                    "inactive_target_update_mse": float(
                        self.inactive_update_sq[index]
                        / self.inactive_count[index].clamp_min(1)
                    ),
                    "best_matching_q_depth": int(best_targets[index]) + 1,
                    "correct_depth_rank_by_mse": int(ranks[index]),
                    "cumulative_mse": float(self.cumulative_error[index] / count),
                    "cumulative_cosine": float(cumulative_cosine[index]),
                }
            )
        return {
            "prediction_events": self.count,
            "diagonal_mean_mse": float(mse[diagonal, diagonal].mean()),
            "off_diagonal_mean_mse": float(mse[off_diagonal].mean()),
            "diagonal_mean_cosine": float(cosine[diagonal, diagonal].mean()),
            "off_diagonal_mean_cosine": float(cosine[off_diagonal].mean()),
            "correct_depth_is_best_fraction": float((best_targets == diagonal).float().mean()),
            "mean_correct_depth_rank_by_mse": float(ranks.float().mean()),
            "cross_depth_mse": mse.tolist(),
            "cross_depth_cosine": cosine.tolist(),
            "per_depth": per_depth,
        }


def _config_from_json(path: Path, config_type):
    payload = json.loads(path.read_text())
    accepted = {field.name for field in fields(config_type) if field.init}
    return config_type(**{key: value for key, value in payload.items() if key in accepted})


def _transformer_from_baseline_config(config: PosetsProbingConfig, task: NumberNamingTask):
    return DecoderTransformerLM(
        vocab_size=task.tokenizer.vocab_size,
        max_seq_len=int(config.max_seq_len),
        d_model=int(config.d_model),
        n_layers=int(config.n_layers),
        n_heads=int(config.n_heads),
        dropout=float(config.dropout),
        pad_id=task.tokenizer.pad_id,
        mlp_ratio=float(config.mlp_ratio),
    )


def _transformer_from_alignment_config(config: QuantaSteeringConfig, task: NumberNamingTask):
    return DecoderTransformerLM(
        vocab_size=task.tokenizer.vocab_size,
        max_seq_len=int(config.max_seq_len),
        d_model=int(config.transformer_d_model),
        n_layers=int(config.transformer_layers),
        n_heads=int(config.transformer_heads),
        dropout=float(config.transformer_dropout),
        pad_id=task.tokenizer.pad_id,
        mlp_ratio=float(config.transformer_mlp_ratio),
    )


def _load_state(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()


def _parameter_distance_by_component(
    aligned: DecoderTransformerLM,
    control: DecoderTransformerLM,
) -> list[dict[str, float | str]]:
    aligned_state = aligned.state_dict()
    control_state = control.state_dict()
    groups = {
        "embeddings": ("token_embedding.", "position_embedding."),
        **{f"block_{index + 1}": (f"layers.{index}.",) for index in range(len(aligned.layers))},
        "readout": ("final_norm.", "head."),
    }
    records = []
    for name, prefixes in groups.items():
        keys = [key for key in aligned_state if key.startswith(prefixes)]
        aligned_values = torch.cat([aligned_state[key].detach().float().cpu().flatten() for key in keys])
        control_values = torch.cat([control_state[key].detach().float().cpu().flatten() for key in keys])
        difference = aligned_values - control_values
        records.append(
            {
                "component": name,
                "l2_distance": float(difference.norm()),
                "relative_l2_distance": float(
                    difference.norm() / control_values.norm().clamp_min(1.0e-12)
                ),
                "parameter_cosine": float(
                    torch.dot(aligned_values, control_values)
                    / (aligned_values.norm() * control_values.norm()).clamp_min(1.0e-12)
                ),
            }
        )
    return records


def _layerwise_summary(
    model: DecoderTransformerLM,
    *,
    model_name: str,
    task: NumberNamingTask,
    teacher: FrozenQTeacher,
    supervision: CompiledSupervisionIndex,
    compiled,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    accumulator = AlignmentAccumulator(len(model.layers))
    with torch.no_grad():
        for start in range(0, len(task.eval_examples), int(batch_size)):
            examples = task.eval_examples[start : start + int(batch_size)]
            batch = task.encode_examples(examples, device=device)
            activity, _, _ = _batch_supervision(
                batch,
                supervision,
                compiled,
                tokenizer=task.tokenizer,
                device=device,
            )
            q_trace = teacher(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                activity_targets=activity,
            )
            transformer_trace = model.residual_trace(
                batch.input_ids,
                batch.attention_mask,
            )
            boundaries = transformer_trace.block_boundaries
            updates = torch.stack(
                [after - before for before, after in zip(boundaries, boundaries[1:])],
                dim=-2,
            )
            targets = _pad_teacher_updates(q_trace.depth_updates, batch.input_ids.shape[1])
            accumulator.add(updates, targets, _prediction_mask(batch.labels))
            logging.info(
                "layerwise progress model=%s batch=%s/%s examples=%s/%s",
                model_name,
                start // int(batch_size) + 1,
                (len(task.eval_examples) + int(batch_size) - 1) // int(batch_size),
                min(start + int(batch_size), len(task.eval_examples)),
                len(task.eval_examples),
            )
    return accumulator.summary()


def _write_tables(output_dir: Path, summaries: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(_jsonable(summaries), handle, indent=2)
        handle.write("\n")

    depth_rows = []
    cross_rows = []
    for model_name, layerwise in summaries["layerwise"].items():
        for record in layerwise["per_depth"]:
            depth_rows.append({"model": model_name, **record})
        for block, row in enumerate(layerwise["cross_depth_mse"], start=1):
            for q_depth, mse in enumerate(row, start=1):
                cross_rows.append(
                    {
                        "model": model_name,
                        "transformer_block": block,
                        "q_depth": q_depth,
                        "mse": mse,
                        "cosine": layerwise["cross_depth_cosine"][block - 1][q_depth - 1],
                    }
                )
    for filename, rows in (("layerwise.csv", depth_rows), ("cross_depth.csv", cross_rows)):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    ablation_rows = []
    for model_name, records in summaries["block_ablation"].items():
        for record in records:
            ablation_rows.append({"model": model_name, **record})
    with (output_dir / "block_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    if summaries.get("parameter_distance_aligned_vs_lambda0"):
        rows = summaries["parameter_distance_aligned_vs_lambda0"]
        with (output_dir / "parameter_distance.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--aligned-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device(args.device or get_device(None))
    aligned_config = _config_from_json(args.aligned_run / "config.json", QuantaSteeringConfig)
    aligned_config.save_dir = str(args.aligned_run)
    baseline_config = _config_from_json(args.baseline_run / "config.json", PosetsProbingConfig)
    set_seeds(int(aligned_config.seed))
    task = NumberNamingTask(aligned_config)

    models: dict[str, DecoderTransformerLM] = {
        "normal_baseline": _transformer_from_baseline_config(baseline_config, task),
        "aligned": _transformer_from_alignment_config(aligned_config, task),
    }
    _load_state(models["normal_baseline"], args.baseline_run / "model.pt", device)
    _load_state(models["aligned"], args.aligned_run / "model.pt", device)
    if args.control_run is not None:
        control_config = _config_from_json(args.control_run / "config.json", QuantaSteeringConfig)
        models["lambda0_control"] = _transformer_from_alignment_config(control_config, task)
        _load_state(models["lambda0_control"], args.control_run / "model.pt", device)

    teacher_checkpoint, teacher_run_dir, teacher_config, compiled = _load_teacher_run(
        aligned_config.q_checkpoint
    )
    supervision = CompiledSupervisionIndex(compiled)
    q_model = _build_q_computer(teacher_config, compiled, task)
    metadata = json.loads((Path(teacher_run_dir) / "q_core_metadata.json").read_text())
    _validate_q_teacher_metadata(metadata, compiled, q_model)
    q_model.load_state_dict(torch.load(teacher_checkpoint, map_location="cpu", weights_only=True))
    teacher = FrozenQTeacher(q_model).to(device)

    summaries: dict[str, Any] = {
        "device": str(device),
        "eval_examples": len(task.eval_examples),
        "teacher_checkpoint": teacher_checkpoint,
        "models": {},
        "layerwise": {},
        "block_ablation": {},
    }
    if "lambda0_control" in models:
        summaries["parameter_distance_aligned_vs_lambda0"] = _parameter_distance_by_component(
            models["aligned"],
            models["lambda0_control"],
        )
    for name, model in models.items():
        logging.info("downstream evaluation starting model=%s", name)
        summaries["models"][name] = _evaluate_transformer(model, task, device=device)
        logging.info(
            "downstream evaluation complete model=%s sequence_acc=%.4f",
            name,
            summaries["models"][name]["transformer_exact_sequence_accuracy"],
        )
        summaries["layerwise"][name] = _layerwise_summary(
            model,
            model_name=name,
            task=task,
            teacher=teacher,
            supervision=supervision,
            compiled=compiled,
            device=device,
            batch_size=int(args.batch_size),
        )

        baseline_accuracy = summaries["models"][name]["transformer_exact_sequence_accuracy"]
        records = []
        for block in range(len(model.layers)):
            logging.info("block ablation starting model=%s block=%s", name, block + 1)
            metrics = _evaluate_transformer(
                BlockAblatedTransformer(model, {block}),
                task,
                device=device,
            )
            accuracy = metrics["transformer_exact_sequence_accuracy"]
            records.append(
                {
                    "skipped_block": block + 1,
                    "exact_sequence_accuracy": accuracy,
                    "exact_sequence_accuracy_drop": baseline_accuracy - accuracy,
                    "token_accuracy": metrics["transformer_token_accuracy"],
                    "normalized_edit_distance": metrics["transformer_normalized_edit_distance"],
                }
            )
        summaries["block_ablation"][name] = records

    output_dir = args.output or args.aligned_run / "analysis"
    _write_tables(output_dir, summaries)
    logging.info("analysis complete output=%s", output_dir)


if __name__ == "__main__":
    main()
