from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import logging
import math
import os
import pickle
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
import wandb

from quanta.config import PlotConfig, QuantaNetConfig, QuantaSteeringConfig
from quanta.experiments.base import Experiment
from quanta.experiments.common import (
    append_jsonl,
    build_adam_optimizer,
    log_event,
    save_checkpoint,
    save_config,
    save_results,
)
from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.number_naming.task import NumberNamingTask
from quanta.experiments.scaling_laws.training.trainer_curriculum import scheduled_learning_rate
from quanta.qprogram import CompiledQProgram, CompiledSupervisionIndex, PredictiveState
from quanta.utils import _jsonable, get_device, set_seeds

from .quantanet import (
    QComputer,
    SemanticClassifierSuite,
    parent_necessity_audit,
    qcore_training_loss,
)
from .steering import (
    FrozenQTeacher,
    LayerwiseQAlignment,
    accumulate_depth_scale_statistics,
    alignment_objective,
    rms_depth_scales,
)


class _EpochExampleSampler:
    """Cycle through deterministic shuffled epochs without replacement."""

    def __init__(self, examples, *, seed: int) -> None:
        self.examples = tuple(examples)
        if not self.examples:
            raise ValueError("Q training requires at least one training example.")
        self.rng = random.Random(int(seed))
        self.order: list[int] = []
        self.cursor = 0
        self._start_epoch()

    def take(self, batch_size: int) -> list[Any]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        result = []
        while len(result) < int(batch_size):
            remaining = len(self.order) - self.cursor
            count = min(int(batch_size) - len(result), remaining)
            result.extend(
                self.examples[index]
                for index in self.order[self.cursor : self.cursor + count]
            )
            self.cursor += count
            if self.cursor == len(self.order):
                self._start_epoch()
        return result

    def _start_epoch(self) -> None:
        self.order = list(range(len(self.examples)))
        self.rng.shuffle(self.order)
        self.cursor = 0


def _scheduled_full_evaluation_steps(total_steps: int, interval: int | None) -> set[int]:
    final_step = int(total_steps)
    if final_step <= 0:
        return set()
    if interval is None:
        return {final_step}
    return set(range(int(interval), final_step + 1, int(interval))) | {final_step}


def _save_named_checkpoint(
    model,
    optimizer,
    save_dir: str,
    name: str,
    *,
    semantic_classifiers: SemanticClassifierSuite | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f"{name}_optimizer.pt"))
    if semantic_classifiers is not None:
        torch.save(
            semantic_classifiers.state_dict(),
            os.path.join(save_dir, f"{name}_semantic_classifiers.pt"),
        )


def _validate_q_teacher_metadata(
    metadata: dict[str, Any],
    compiled: CompiledQProgram,
    model: QComputer,
) -> None:
    expected = {
        "compiled_program_fingerprint": compiled.metadata.program_fingerprint,
        "nodes": list(model.core.nodes),
        "depth": model.core.depth,
        "d_source": model.core.d_source,
        "d_quantum": model.core.d_quantum,
        "activation": model.core.activation,
        "add_initial_state": model.core.add_initial_state,
        "all_quanta_output": model.core.all_quanta_output,
    }
    observed = {**metadata, "all_quanta_output": metadata.get("all_quanta_output", True)}
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise ValueError(
            "Q checkpoint metadata does not match the steering Q-core configuration: "
            + ", ".join(mismatches)
        )


def _log_q_evaluation(
    record: dict[str, Any],
    *,
    best_token_accuracy: float | None = None,
    best_sequence_accuracy: float | None = None,
) -> None:
    fields: dict[str, Any] = {"step": int(record["step"])}
    if "predicted_token_accuracy" in record:
        token_accuracy = float(record["predicted_token_accuracy"])
        token_best = token_accuracy if best_token_accuracy is None else best_token_accuracy
        fields["token_accuracy"] = token_accuracy
        fields["best_token_accuracy"] = token_best
    if "predicted_exact_sequence_accuracy" in record:
        sequence_accuracy = float(record["predicted_exact_sequence_accuracy"])
        sequence_best = (
            sequence_accuracy
            if best_sequence_accuracy is None
            else best_sequence_accuracy
        )
        fields["sequence_accuracy"] = sequence_accuracy
        fields["best_sequence_accuracy"] = sequence_best
    for key in ("train_loss", "train_task", "train_gate", "train_message"):
        if key in record:
            fields[key] = float(record[key])
    log_event("quanta_net", "evaluation", **fields)


def _log_alignment_evaluation(
    record: dict[str, Any],
    *,
    total_steps: int,
    best_sequence_accuracy: float,
) -> None:
    log_event(
        "quanta_steering",
        "evaluation",
        step=int(record["step"]),
        total_steps=int(total_steps),
        elapsed_seconds=float(record["elapsed_seconds"]),
        evaluation_seconds=float(record["evaluation_seconds"]),
        learning_rate=float(record["lr"]),
        train_loss=float(record["total_loss"]),
        task_loss=float(record["task_loss"]),
        alignment_loss=float(record["alignment_loss"]),
        cumulative_state_mse=float(record["cumulative_state_mse"]),
        token_accuracy=float(record["transformer_token_accuracy"]),
        sequence_accuracy=float(record["transformer_exact_sequence_accuracy"]),
        best_sequence_accuracy=float(best_sequence_accuracy),
        normalized_edit_distance=float(record["transformer_normalized_edit_distance"]),
    )


def _start_wandb(config, *, experiment: str, extra: dict[str, Any] | None = None) -> None:
    if not config.wandb_project or wandb.run is not None:
        return
    wandb.init(
        project=config.wandb_project,
        mode=config.wandb_mode,
        config={**_jsonable(asdict(config)), **(extra or {})},
        job_type=experiment,
    )


def _log_q_wandb(record: dict[str, Any], *, experiment: str) -> None:
    if wandb.run is None:
        return
    if experiment == "quanta_net":
        mapping = {
            "train_loss": "loss/total",
            "train_task": "loss/task",
            "train_gate": "loss/gate",
            "train_message": "loss/message",
            "predicted_token_accuracy": "accuracy/predicted_token",
            "predicted_exact_sequence_accuracy": "accuracy/predicted_sequence",
            "oracle_token_accuracy": "accuracy/oracle_token",
            "oracle_exact_sequence_accuracy": "accuracy/oracle_sequence",
            "predicted_soft_trace_mae": "routing/soft_trace_mae",
        }
    else:
        mapping = {
            "lr": "optimization/learning_rate",
            "total_loss": "loss/total",
            "task_loss": "loss/task",
            "alignment_loss": "loss/alignment",
            "cumulative_state_mse": "alignment/cumulative_state_mse",
            "transformer_token_accuracy": "accuracy/token",
            "transformer_exact_sequence_accuracy": "accuracy/sequence",
        }
    payload = {
        destination: float(record[source])
        for source, destination in mapping.items()
        if source in record and isinstance(record[source], (int, float))
    }
    wandb.log(payload, step=int(record["step"]))


def _write_run_status(save_dir: str, **fields: Any) -> None:
    path = os.path.join(save_dir, "status.json")
    temporary = f"{path}.tmp"
    payload = {"updated_at_unix": time.time(), **fields}
    with open(temporary, "w") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _default_q_save_dir(experiment_name: str, config: QuantaNetConfig, compiled: CompiledQProgram) -> str:
    config_payload = asdict(config)
    config_payload.pop("save_dir", None)
    identity = {
        "experiment": experiment_name,
        "config": _jsonable(config_payload),
        "program_id": compiled.id,
        "compiler_version": compiled.metadata.compiler_version,
        "program_fingerprint": compiled.metadata.program_fingerprint,
        "primitive_cost_fingerprint": compiled.metadata.primitive_cost_fingerprint,
        "validation_domain_fingerprint": compiled.metadata.validation_domain_fingerprint,
        "training_distribution_fingerprint": compiled.metadata.training_distribution_fingerprint,
        "evaluation_distribution_fingerprint": compiled.metadata.evaluation_distribution_fingerprint,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    program = _path_slug(compiled.id)[:32]
    readable = (
        f"{program}-{compiled.metadata.program_fingerprint[:8]}"
        f"-ds{int(config.d_source)}-dq{int(config.d_quantum)}"
        f"-lr{_path_slug(f'{float(config.lr):g}')}-seed{int(config.seed)}"
    )
    return os.path.join(".experiments", experiment_name, config.data_splits, f"{readable}-{digest}")


_ALIGNMENT_INHERITED_Q_FIELDS = {
    "compiled_program_path",
    "d_source",
    "d_quantum",
    "read_heads",
    "quantum_mlp_ratio",
    "activation",
    "add_initial_state",
    "all_quanta_output",
    "parent_audit_size",
    "full_eval_steps",
    "activity_weight",
    "semantic_weight",
    "max_supervised_classes",
}


def _alignment_config_payload(config: QuantaSteeringConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(config).items()
        if key not in _ALIGNMENT_INHERITED_Q_FIELDS
    }


def _default_alignment_save_dir(
    config: QuantaSteeringConfig,
    compiled: CompiledQProgram,
) -> str:
    config_payload = _alignment_config_payload(config)
    config_payload.pop("save_dir", None)
    identity = {
        "experiment": "quanta_steering",
        "config": _jsonable(config_payload),
        "compiler_version": compiled.metadata.compiler_version,
        "program_fingerprint": compiled.metadata.program_fingerprint,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    readable = (
        f"{_path_slug(compiled.id)[:32]}-{compiled.metadata.program_fingerprint[:8]}"
        f"-layers{int(config.transformer_layers)}-d{int(config.transformer_d_model)}"
        f"-heads{int(config.transformer_heads)}-seed{int(config.seed)}"
    )
    return os.path.join(
        ".experiments",
        "quanta_steering",
        config.data_splits,
        f"{readable}-{digest}",
    )


def _path_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in str(value)).strip("-") or "run"


class QuantaNetExperiment(Experiment):
    def __init__(self, config: QuantaNetConfig, plot_config: PlotConfig | None = None):
        self.config = config
        self.plot_config = plot_config
        self._compiled: CompiledQProgram | None = None
        if self.config.save_dir is None:
            self._compiled = CompiledQProgram.read(self.config.compiled_program_path)
            self.config.save_dir = self._default_save_dir(self._compiled)

    def run(self) -> str:
        set_seeds(int(self.config.seed))
        device = torch.device(get_device(self.config.device))
        compiled = self._compiled or CompiledQProgram.read(self.config.compiled_program_path)
        supervision = CompiledSupervisionIndex(compiled)
        task = NumberNamingTask(self.config)
        _start_wandb(
            self.config,
            experiment="quanta_net",
            extra={"compiled_program_fingerprint": compiled.metadata.program_fingerprint},
        )
        train_sampler = _EpochExampleSampler(task.train, seed=int(self.config.seed))
        self._resolve_steps_from_epochs(len(task.train))
        model = _build_q_computer(self.config, compiled, task).to(device)
        semantic_classifiers = SemanticClassifierSuite(
            compiled=compiled,
            d_quantum=int(self.config.d_quantum),
            max_supervised_classes=int(self.config.max_supervised_classes),
        ).to(device)
        training_modules = torch.nn.ModuleList([model, semantic_classifiers])
        optimizer = build_adam_optimizer(
            training_modules,
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_config(self.config, self.config.save_dir)
        audit = model.parameter_audit()
        log_event(
            "quanta_net",
            "run_start",
            task="number_naming",
            split=self.config.data_splits,
            train_examples=len(task.train),
            eval_examples=len(task.eval_examples),
            quanta=len(compiled.nodes),
            depth=compiled.structure.depth,
            parameters=audit.total_parameters,
            shared_parameters=audit.shared_parameters,
            per_quantum_parameters=audit.per_node_parameter_count,
            training_only_parameters=_parameter_count(semantic_classifiers),
            steps=self.config.steps,
            batch_size=self.config.batch_size,
            device=device,
            save_dir=self.config.save_dir,
        )
        logging.debug(
            "Semantic message supervision nodes=%s excluded_by_cardinality=%s constant_nodes=%s",
            list(semantic_classifiers.supervised_nodes),
            semantic_classifiers.excluded_by_cardinality,
            list(semantic_classifiers.constant_nodes),
        )
        with open(os.path.join(self.config.save_dir, "semantic_classifier_catalog.json"), "w") as handle:
            json.dump(semantic_classifiers.catalog(), handle, indent=2)
        with open(os.path.join(self.config.save_dir, "q_core_metadata.json"), "w") as handle:
            json.dump(
                {
                    "compiled_program_fingerprint": compiled.metadata.program_fingerprint,
                    "nodes": list(compiled.nodes),
                    "depth": model.core.depth,
                    "d_source": model.core.d_source,
                    "d_quantum": model.core.d_quantum,
                    "activation": model.core.activation,
                    "add_initial_state": model.core.add_initial_state,
                    "all_quanta_output": model.core.all_quanta_output,
                    "predicted_routing": "minimum_fuzzy_and",
                    "semantic_supervision": {
                        "message_tensor": "raw_delta",
                        "training_only": True,
                        "weight": float(self.config.semantic_weight),
                        "max_supervised_classes": int(self.config.max_supervised_classes),
                        "supervised_nodes": list(semantic_classifiers.supervised_nodes),
                    },
                },
                handle,
                indent=2,
            )

        history = []
        best_oracle = (-math.inf, -math.inf)
        best_oracle_step: int | None = None
        best_score = (-math.inf, -math.inf, -math.inf)
        best_step: int | None = None
        best_token_accuracy = -math.inf
        best_sequence_accuracy = -math.inf
        start = time.monotonic()
        eval_steps = self._evaluation_steps(len(task.train))
        full_eval_steps = _scheduled_full_evaluation_steps(
            int(self.config.steps),
            self.config.full_eval_steps,
        )
        metrics_path = os.path.join(self.config.save_dir, "metrics.jsonl")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        for step in range(1, int(self.config.steps) + 1):
            lr = scheduled_learning_rate(
                float(self.config.lr),
                step,
                int(self.config.steps),
                self.config.scheduler,
                warmup_phase=float(self.config.warmup_phase),
                plateau_phase=float(self.config.plateau_phase),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            semantic_classifiers.train()
            batch = task.encode_examples(
                train_sampler.take(int(self.config.batch_size)),
                device=device,
            )
            activity, eligibility, semantics = _batch_supervision(
                batch,
                supervision,
                compiled,
                tokenizer=task.tokenizer,
                device=device,
            )
            semantic_targets = semantic_classifiers.encode_targets(
                semantics,
                activity,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = qcore_training_loss(
                model,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                activity_targets=activity,
                eligibility_mask=eligibility,
                activity_weight=float(self.config.activity_weight),
                semantic_weight=float(self.config.semantic_weight),
                semantic_classifiers=semantic_classifiers,
                semantic_targets=semantic_targets,
            )
            loss = losses["loss"]
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            optimizer.step()
            if step in eval_steps:
                metrics = _evaluate_q(
                    model,
                    task,
                    supervision,
                    compiled,
                    device=device,
                    include_full_decode=step in full_eval_steps,
                    parent_audit_size=int(self.config.parent_audit_size),
                )
                record = {
                    "step": step,
                    "full_evaluation": step in full_eval_steps,
                    "train_loss": float(loss.detach().cpu()),
                    "train_task": float(losses["ce_loss"].detach().cpu()),
                    "train_gate": float(losses["activity_loss"].detach().cpu()),
                    "train_message": float(losses["semantic_loss"].detach().cpu()),
                    **metrics,
                }
                best_token_accuracy = max(
                    best_token_accuracy,
                    float(metrics["predicted_token_accuracy"]),
                )
                if "predicted_exact_sequence_accuracy" in metrics:
                    best_sequence_accuracy = max(
                        best_sequence_accuracy,
                        float(metrics["predicted_exact_sequence_accuracy"]),
                    )
                history.append(record)
                append_jsonl(metrics_path, record)
                _log_q_wandb(record, experiment="quanta_net")
                _log_q_evaluation(
                    record,
                    best_token_accuracy=best_token_accuracy,
                    best_sequence_accuracy=(
                        best_sequence_accuracy
                        if best_sequence_accuracy > -math.inf
                        else None
                    ),
                )
                if "predicted_exact_sequence_accuracy" in metrics:
                    oracle_score = (
                        float(metrics["oracle_exact_sequence_accuracy"]),
                        float(metrics["oracle_token_accuracy"]),
                    )
                    if best_oracle_step is None or oracle_score > best_oracle:
                        best_oracle = oracle_score
                        best_oracle_step = step
                        _save_named_checkpoint(
                            model,
                            optimizer,
                            self.config.save_dir,
                            "best_oracle",
                            semantic_classifiers=semantic_classifiers,
                        )
                    predicted_score = (
                        float(metrics["predicted_exact_sequence_accuracy"]),
                        float(metrics["predicted_token_accuracy"]),
                        -float(metrics["predicted_soft_trace_mae"]),
                    )
                    if best_step is None or predicted_score > best_score:
                        best_score = predicted_score
                        best_step = step
                        _save_named_checkpoint(
                            model,
                            optimizer,
                            self.config.save_dir,
                            "best",
                            semantic_classifiers=semantic_classifiers,
                        )
                        # Keep the conventional checkpoint as an alias of the best deployable model.
                        save_checkpoint(model, optimizer, self.config.save_dir)
                        torch.save(
                            semantic_classifiers.state_dict(),
                            os.path.join(self.config.save_dir, "semantic_classifiers.pt"),
                        )
        _save_named_checkpoint(
            model,
            optimizer,
            self.config.save_dir,
            "last",
            semantic_classifiers=semantic_classifiers,
        )
        if best_step is None:
            raise RuntimeError("Q training completed without evaluating and saving a best checkpoint.")
        results = {
            "experiment": "quanta_net",
            "task": "number_naming",
            "config": _jsonable(asdict(self.config)),
            "metrics": history,
            "runtime_seconds": time.monotonic() - start,
            "q_parameters": {
                "total": audit.total_parameters,
                "shared": audit.shared_parameters,
                "per_quantum": audit.per_node_parameter_count,
                "quantum_total": audit.quantum_parameters,
                "semantic_classifiers_training_only": _parameter_count(semantic_classifiers),
            },
            "compiled_program_fingerprint": compiled.metadata.program_fingerprint,
            "predicted_routing": "minimum_fuzzy_and",
            "checkpoints": {
                "default": "model.pt",
                "best": "best_model.pt",
                "best_oracle": "best_oracle_model.pt",
                "last": "last_model.pt",
                "best_step": best_step,
                "best_score": best_score,
                "best_oracle_step": best_oracle_step,
                "best_oracle_score": best_oracle,
            },
        }
        save_results(results, self.config.save_dir)
        log_event(
            "quanta_net",
            "run_complete",
            steps=self.config.steps,
            best_step=best_step,
            best_sequence_accuracy=best_score[0],
            runtime_seconds=results["runtime_seconds"],
            save_dir=self.config.save_dir,
        )
        if wandb.run is not None:
            wandb.finish()
        return self.config.save_dir

    def _resolve_steps_from_epochs(self, train_size: int) -> None:
        if self.config.steps is None:
            self.config.steps = max(1, math.ceil(float(self.config.epochs) * train_size / self.config.batch_size))

    def _evaluation_steps(self, train_size: int) -> set[int]:
        if self.config.eval_steps is not None:
            return set(range(int(self.config.eval_steps), int(self.config.steps) + 1, int(self.config.eval_steps))) | {int(self.config.steps)}
        count = max(1, int(self.config.evals_per_epoch or 1))
        interval = max(1, math.ceil(train_size / self.config.batch_size / count))
        return set(range(interval, int(self.config.steps) + 1, interval)) | {int(self.config.steps)}

    def _default_save_dir(self, compiled: CompiledQProgram) -> str:
        return _default_q_save_dir("quanta_net", self.config, compiled)


class QuantaSteeringExperiment(QuantaNetExperiment):
    config: QuantaSteeringConfig

    def __init__(self, config: QuantaSteeringConfig, plot_config: PlotConfig | None = None):
        self.config = config
        self.plot_config = plot_config
        (
            self._teacher_checkpoint,
            self._teacher_run_dir,
            self._teacher_config,
            self._compiled,
        ) = _load_teacher_run(config.q_checkpoint)
        _validate_alignment_task_identity(config, self._teacher_config)
        if int(config.transformer_d_model) < int(self._teacher_config.d_quantum):
            raise ValueError(
                "transformer_d_model must be at least the saved Q teacher d_quantum: "
                f"transformer_d_model={config.transformer_d_model}, "
                f"d_quantum={self._teacher_config.d_quantum}."
            )
        if self.config.save_dir is None:
            self.config.save_dir = self._default_save_dir(self._compiled)

    def run(self) -> str:
        try:
            return self._run_alignment()
        except Exception as error:
            logging.exception(
                "Q alignment failed: save_dir=%s error=%s",
                self.config.save_dir,
                error,
            )
            if self.config.save_dir is not None:
                os.makedirs(self.config.save_dir, exist_ok=True)
                _write_run_status(
                    self.config.save_dir,
                    status="failed",
                    error_type=type(error).__name__,
                    error=str(error),
                )
            raise
        finally:
            if wandb.run is not None:
                wandb.finish()

    def _run_alignment(self) -> str:
        set_seeds(int(self.config.seed))
        device = torch.device(get_device(self.config.device))
        compiled = self._compiled
        logging.debug(
            "Q alignment setup: device=%s teacher=%s program=%s fingerprint=%s "
            "q_depth=%s transformer_layers=%s d_quantum=%s d_transformer=%s",
            device,
            self._teacher_checkpoint,
            compiled.id,
            compiled.metadata.program_fingerprint,
            compiled.structure.depth,
            self.config.transformer_layers,
            self._teacher_config.d_quantum,
            self.config.transformer_d_model,
        )
        if int(self.config.transformer_layers) != int(compiled.structure.depth):
            raise ValueError(
                "layerwise Q alignment requires transformer_layers to equal compiled Q depth: "
                f"transformer_layers={self.config.transformer_layers}, q_depth={compiled.structure.depth}."
            )
        if self.config.alignment_scales is not None and len(self.config.alignment_scales) != int(
            compiled.structure.depth
        ):
            raise ValueError("alignment_scales must contain exactly one value per compiled Q depth.")
        supervision = CompiledSupervisionIndex(compiled)
        task = NumberNamingTask(self.config)
        _start_wandb(
            self.config,
            experiment="quanta_steering",
            extra={
                "compiled_program_fingerprint": compiled.metadata.program_fingerprint,
                "teacher_checkpoint": self._teacher_checkpoint,
            },
        )
        logging.debug(
            "Q alignment data: split=%s train=%s eval=%s eval_strategy=%s "
            "batch_size=%s eval_batch_size=%s",
            self.config.data_splits,
            len(task.train),
            len(task.eval_examples),
            self.config.eval_strategy,
            self.config.batch_size,
            self.config.eval_batch_size,
        )
        train_sampler = _EpochExampleSampler(task.train, seed=int(self.config.seed))
        self._resolve_steps_from_epochs(len(task.train))
        q_model = _build_q_computer(self._teacher_config, compiled, task)
        checkpoint = self._teacher_checkpoint
        metadata_path = os.path.join(self._teacher_run_dir, "q_core_metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError("steering Q checkpoint is missing q_core_metadata.json.")
        with open(metadata_path) as handle:
            teacher_metadata = json.load(handle)
        _validate_q_teacher_metadata(teacher_metadata, compiled, q_model)
        _validate_teacher_preconditions(self._teacher_run_dir, self.config)
        q_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        teacher = FrozenQTeacher(q_model).to(device)
        logging.debug(
            "Q alignment teacher ready: run_dir=%s parameters=%s oracle_preconditions=passed",
            self._teacher_run_dir,
            _parameter_count(teacher),
        )
        transformer = DecoderTransformerLM(
            vocab_size=task.tokenizer.vocab_size,
            max_seq_len=int(self.config.max_seq_len),
            d_model=int(self.config.transformer_d_model),
            n_layers=int(self.config.transformer_layers),
            n_heads=int(self.config.transformer_heads),
            dropout=float(self.config.transformer_dropout),
            pad_id=task.tokenizer.pad_id,
            mlp_ratio=float(self.config.transformer_mlp_ratio),
        )
        depth_scales = _alignment_depth_scales(
            self.config,
            teacher=teacher,
            task=task,
            supervision=supervision,
            compiled=compiled,
            device=device,
        )
        model = LayerwiseQAlignment(
            transformer=transformer,
            teacher=teacher,
            interface_seed=int(self.config.interface_seed),
            depth_scales=depth_scales,
        ).to(device)
        optimizer = build_adam_optimizer(
            model.transformer,
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_config(_alignment_config_payload(self.config), self.config.save_dir)
        _write_run_status(
            self.config.save_dir,
            status="initializing",
            step=0,
            total_steps=int(self.config.steps),
            output_dir=self.config.save_dir,
        )
        interface_payload = {
            "U": model.interface.matrix.detach().cpu(),
            "depth_scales": model.depth_scales.detach().cpu(),
            "q_depth": model.depth,
            "d_quantum": int(self._teacher_config.d_quantum),
            "d_transformer": int(self.config.transformer_d_model),
            "identity": model.interface.is_identity,
            "seed": int(self.config.interface_seed),
            "scale_mode": self.config.alignment_scale_mode,
            "target_control": self.config.alignment_target_control,
        }
        torch.save(interface_payload, os.path.join(self.config.save_dir, "alignment_interface.pt"))
        interface_metadata = {
            **interface_payload,
            "U": interface_payload["U"].tolist(),
            "depth_scales": interface_payload["depth_scales"].tolist(),
        }
        with open(os.path.join(self.config.save_dir, "alignment_metadata.json"), "w") as handle:
            json.dump(interface_metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        history = []
        eval_steps = self._evaluation_steps(len(task.train))
        metrics_path = os.path.join(self.config.save_dir, "metrics.jsonl")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        log_event(
            "quanta_steering",
            "run_start",
            task="number_naming",
            split=self.config.data_splits,
            train_examples=len(task.train),
            eval_examples=len(task.eval_examples),
            depth=model.depth,
            transformer_width=self.config.transformer_d_model,
            deployment_parameters=_parameter_count(model.deployment_model()),
            steps=self.config.steps,
            evaluations=len(eval_steps),
            batch_size=self.config.batch_size,
            lambda_align=self.config.lambda_align,
            target_control=self.config.alignment_target_control,
            device=device,
            save_dir=self.config.save_dir,
        )
        start = time.monotonic()
        best_sequence_accuracy = -math.inf
        _write_run_status(
            self.config.save_dir,
            status="training",
            step=0,
            total_steps=int(self.config.steps),
            elapsed_seconds=0.0,
        )
        for step in range(1, int(self.config.steps) + 1):
            model.train()
            batch = task.encode_examples(
                train_sampler.take(int(self.config.batch_size)),
                device=device,
            )
            activity, _, _ = _batch_supervision(
                batch,
                supervision,
                compiled,
                tokenizer=task.tokenizer,
                device=device,
            )
            lr = scheduled_learning_rate(
                float(self.config.lr),
                step,
                int(self.config.steps),
                self.config.scheduler,
                warmup_phase=float(self.config.warmup_phase),
                plateau_phase=float(self.config.plateau_phase),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            trace = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                activity_targets=activity,
                valid_prediction_mask=_prediction_mask(batch.labels),
                target_control=self.config.alignment_target_control,
                control_seed=int(self.config.control_seed),
            )
            losses = alignment_objective(
                trace,
                labels=batch.labels,
                lambda_align=float(self.config.lambda_align),
            )
            losses["loss"].backward()
            if any(parameter.grad is not None for parameter in model.teacher.parameters()):
                raise RuntimeError("frozen Q teacher received gradients during steering.")
            optimizer.step()
            if step in eval_steps:
                elapsed_before_eval = time.monotonic() - start
                logging.debug(
                    "Q alignment evaluation starting: step=%s/%s elapsed=%.1fs examples=%s",
                    step,
                    self.config.steps,
                    elapsed_before_eval,
                    len(task.eval_examples),
                )
                _write_run_status(
                    self.config.save_dir,
                    status="evaluating",
                    step=step,
                    total_steps=int(self.config.steps),
                    elapsed_seconds=elapsed_before_eval,
                    eval_examples=len(task.eval_examples),
                )
                evaluation_start = time.monotonic()
                metrics = _evaluate_transformer(model.transformer, task, device=device)
                evaluation_seconds = time.monotonic() - evaluation_start
                record = {
                    "step": step,
                    "epoch": float(step * int(self.config.batch_size) / max(len(task.train), 1)),
                    "elapsed_seconds": time.monotonic() - start,
                    "evaluation_seconds": evaluation_seconds,
                    "lr": lr,
                    "total_loss": float(losses["loss"].detach().cpu()),
                    "task_loss": float(losses["task_loss"].detach().cpu()),
                    "alignment_loss": float(losses["alignment_loss"].detach().cpu()),
                    "alignment_loss_by_depth": [
                        float(value)
                        for value in losses["alignment_loss_by_depth"].detach().cpu()
                    ],
                    "weighted_alignment_loss": float(
                        float(self.config.lambda_align) * losses["alignment_loss"].detach().cpu()
                    ),
                    "cumulative_state_mse": float(losses["cumulative_state_mse"].detach().cpu()),
                    "cumulative_state_mse_by_depth": [
                        float(value)
                        for value in losses["cumulative_state_mse_by_depth"].detach().cpu()
                    ],
                    **metrics,
                }
                history.append(record)
                append_jsonl(metrics_path, record)
                _log_q_wandb(record, experiment="quanta_steering")
                best_sequence_accuracy = max(
                    best_sequence_accuracy,
                    float(record["transformer_exact_sequence_accuracy"]),
                )
                _log_alignment_evaluation(
                    record,
                    total_steps=int(self.config.steps),
                    best_sequence_accuracy=best_sequence_accuracy,
                )
                _write_run_status(
                    self.config.save_dir,
                    status="training",
                    step=step,
                    total_steps=int(self.config.steps),
                    elapsed_seconds=float(record["elapsed_seconds"]),
                    latest_metrics=record,
                    best_sequence_accuracy=best_sequence_accuracy,
                )
        logging.debug("Q alignment checkpoint saving: save_dir=%s", self.config.save_dir)
        _write_run_status(
            self.config.save_dir,
            status="saving",
            step=int(self.config.steps),
            total_steps=int(self.config.steps),
            elapsed_seconds=time.monotonic() - start,
            best_sequence_accuracy=best_sequence_accuracy,
        )
        save_checkpoint(model.deployment_model(), optimizer, self.config.save_dir)
        results = {
            "experiment": "quanta_steering",
            "task": "number_naming",
            "config": _jsonable(_alignment_config_payload(self.config)),
            "metrics": history,
            "runtime_seconds": time.monotonic() - start,
            "deployment_parameters": _parameter_count(model.deployment_model()),
            "teacher_parameters_training_only": _parameter_count(model.teacher),
            "q_depth": model.depth,
            "transformer_blocks": len(model.transformer.layers),
            "alignment_interface": "alignment_interface.pt",
            "alignment_metadata": "alignment_metadata.json",
            "final_inference_uses_q": False,
            "assisted_transformer_pass": False,
            "q_message_injection": False,
        }
        save_results(results, self.config.save_dir)
        with open(os.path.join(self.config.save_dir, "summary.json"), "w") as handle:
            json.dump(_jsonable(results), handle, indent=2)
            handle.write("\n")
        _write_run_status(
            self.config.save_dir,
            status="complete",
            step=int(self.config.steps),
            total_steps=int(self.config.steps),
            runtime_seconds=float(results["runtime_seconds"]),
            best_sequence_accuracy=best_sequence_accuracy,
            model_path=os.path.join(self.config.save_dir, "model.pt"),
            results_path=os.path.join(self.config.save_dir, "results.pkl"),
        )
        log_event(
            "quanta_steering",
            "run_complete",
            steps=self.config.steps,
            best_sequence_accuracy=best_sequence_accuracy,
            runtime_seconds=results["runtime_seconds"],
            save_dir=self.config.save_dir,
        )
        return self.config.save_dir

    def _default_save_dir(self, compiled: CompiledQProgram) -> str:
        return _default_alignment_save_dir(self.config, compiled)


def _load_teacher_run(
    checkpoint_path: str | None,
) -> tuple[str, str, QuantaNetConfig, CompiledQProgram]:
    if checkpoint_path is None:
        raise ValueError("quanta_steering requires q_checkpoint.")
    checkpoint = os.path.normpath(str(checkpoint_path))
    if os.path.isdir(checkpoint):
        run_dir = checkpoint
        checkpoint = os.path.join(run_dir, "model.pt")
    else:
        run_dir = os.path.dirname(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Q teacher checkpoint does not exist: {checkpoint}")
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError("Q teacher run is missing config.json.")
    with open(config_path) as handle:
        payload = json.load(handle)
    init_fields = {item.name for item in fields(QuantaNetConfig) if item.init}
    teacher_config = QuantaNetConfig(
        **{key: value for key, value in payload.items() if key in init_fields}
    )
    if not teacher_config.compiled_program_path:
        raise ValueError(
            "Q teacher predates compiled Q-program checkpoints and cannot provide layerwise targets. "
            "Choose a completed compiler-backed Q run with compiled_program_path in config.json."
        )
    compiled = CompiledQProgram.read(teacher_config.compiled_program_path)
    return checkpoint, run_dir, teacher_config, compiled


def _validate_alignment_task_identity(
    config: QuantaSteeringConfig,
    teacher_config: QuantaNetConfig,
) -> None:
    fields_to_match = (
        "task",
        "language",
        "data_splits",
        "training_size",
        "eval_size",
        "eval_strategy",
        "split_seed",
        "max_number",
        "max_seq_len",
    )
    mismatches = [
        name
        for name in fields_to_match
        if getattr(config, name) != getattr(teacher_config, name)
    ]
    if mismatches:
        raise ValueError(
            "steering task config does not match the saved Q teacher run: "
            + ", ".join(mismatches)
        )


def _alignment_depth_scales(
    config: QuantaSteeringConfig,
    *,
    teacher: FrozenQTeacher,
    task: NumberNamingTask,
    supervision: CompiledSupervisionIndex,
    compiled: CompiledQProgram,
    device: torch.device,
) -> torch.Tensor:
    depth = len(compiled.structure.levels)
    if config.alignment_scales is not None:
        scales = torch.tensor(config.alignment_scales, dtype=torch.float32, device=device)
        if tuple(scales.shape) != (depth,):
            raise ValueError("alignment_scales must contain exactly one value per compiled Q depth.")
        return scales
    if config.alignment_scale_mode == "unit":
        return torch.ones(depth, dtype=torch.float32, device=device)

    squared_norm_sums = torch.zeros(depth, dtype=torch.float64, device=device)
    event_counts = torch.zeros(depth, dtype=torch.float64, device=device)
    scale_batch_size = min(max(int(config.batch_size), 1), len(task.train))
    teacher.eval()
    with torch.no_grad():
        for start in range(0, len(task.train), scale_batch_size):
            batch = task.encode_examples(task.train[start : start + scale_batch_size], device=device)
            activity, _, _ = _batch_supervision(
                batch,
                supervision,
                compiled,
                tokenizer=task.tokenizer,
                device=device,
            )
            trace = teacher(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                activity_targets=activity,
            )
            valid = _prediction_mask(batch.labels)[:, : trace.depth_updates.shape[1]]
            sums, counts = accumulate_depth_scale_statistics(trace.depth_updates, valid)
            squared_norm_sums += sums.to(torch.float64)
            event_counts += counts.to(torch.float64)
    return rms_depth_scales(
        squared_norm_sums,
        event_counts,
        epsilon=float(config.alignment_scale_epsilon),
    ).to(torch.float32)


def _build_q_computer(config: QuantaNetConfig, compiled: CompiledQProgram, task: NumberNamingTask) -> QComputer:
    return QComputer(
        compiled=compiled,
        vocab_size=task.tokenizer.vocab_size,
        max_seq_len=int(config.max_seq_len),
        d_source=int(config.d_source),
        d_quantum=int(config.d_quantum),
        pad_id=task.tokenizer.pad_id,
        sep_id=task.tokenizer.sep_id,
        read_heads=int(config.read_heads),
        mlp_ratio=float(config.quantum_mlp_ratio),
        activation=config.activation,
        add_initial_state=bool(config.add_initial_state),
        all_quanta_output=bool(config.all_quanta_output),
    )


def _batch_supervision(
    batch,
    index: CompiledSupervisionIndex,
    compiled: CompiledQProgram,
    *,
    tokenizer,
    device,
):
    batch_size, query_length = batch.labels[:, 1:].shape
    activity = torch.zeros(batch_size, query_length, len(compiled.nodes), device=device)
    eligibility = torch.zeros_like(activity)
    semantics: list[list[tuple[object | None, ...] | None]] = [
        [None for _ in range(query_length)] for _ in range(batch_size)
    ]
    for row, (number, text) in enumerate(zip(batch.numbers, batch.texts)):
        positions = torch.nonzero(batch.labels[row, 1:] != -100, as_tuple=False).flatten().tolist()
        words = tuple(str(text).split())
        for target_index, position in enumerate(positions):
            state = PredictiveState(
                digits=tuple(int(digit) for digit in str(int(number))),
                prefix=words[:target_index],
                position=target_index,
            )
            supervision = index.lookup(state)
            target_id = int(batch.labels[row, position + 1].item())
            expected_target = tokenizer.id_to_token[target_id]
            if supervision.target != expected_target:
                raise ValueError(
                    "compiled Q-program target disagrees with the NumberNaming batch: "
                    f"number={number}, prefix={state.prefix!r}, "
                    f"compiled={supervision.target!r}, dataset={expected_target!r}."
                )
            activity[row, position] = torch.tensor(supervision.activity, dtype=torch.float32, device=device)
            eligibility[row, position] = torch.tensor(supervision.eligibility, dtype=torch.float32, device=device)
            semantics[row][position] = supervision.semantics
    return activity, eligibility, tuple(tuple(row) for row in semantics)


def _evaluate_q(
    model,
    task,
    supervision,
    compiled,
    *,
    device,
    include_full_decode: bool,
    parent_audit_size: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "oracle_correct": 0,
        "predicted_correct": 0,
        "tokens": 0,
        "soft_trace_abs_error": 0.0,
        "soft_trace_count": 0,
    }
    node_stats = {node: [0.0, 0.0, 0] for node in compiled.nodes}
    semantic_outcome_stats: dict[str, list[float | int]] = {}
    functional_class_stats: dict[str, list[int]] = {}
    target_token_stats: dict[str, list[int]] = {}
    parent_audit_metrics = _evaluate_parent_necessity(
        model,
        task,
        supervision,
        compiled,
        device=device,
        sample_size=parent_audit_size,
    )
    with torch.no_grad():
        for start in range(0, len(task.eval_examples), int(task.config.eval_batch_size)):
            batch = task.encode_examples(task.eval_examples[start : start + int(task.config.eval_batch_size)], device=device)
            activity, eligibility, semantics = _batch_supervision(
                batch,
                supervision,
                compiled,
                tokenizer=task.tokenizer,
                device=device,
            )
            oracle = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, routing="oracle", activity_targets=activity)
            predicted = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, routing="predicted")
            labels = batch.labels[:, 1:]
            valid = labels != -100
            oracle_predictions = oracle.logits.argmax(-1)
            predicted_predictions = predicted.logits.argmax(-1)
            totals["oracle_correct"] += int(((oracle_predictions == labels) & valid).sum())
            totals["predicted_correct"] += int(((predicted_predictions == labels) & valid).sum())
            totals["tokens"] += int(valid.sum())
            trace_error = (predicted.effective_activity - activity.to(predicted.effective_activity.dtype)).abs().mean(-1)
            totals["soft_trace_abs_error"] += float(trace_error[valid].sum())
            totals["soft_trace_count"] += int(valid.sum())
            for row in range(labels.shape[0]):
                for position in torch.nonzero(valid[row], as_tuple=False).flatten().tolist():
                    oracle_correct = int(
                        oracle_predictions[row, position] == labels[row, position]
                    )
                    predicted_correct = int(
                        predicted_predictions[row, position] == labels[row, position]
                    )
                    signature = "|".join(
                        node for index, node in enumerate(compiled.nodes) if bool(activity[row, position, index])
                    )
                    class_values = functional_class_stats.setdefault(signature, [0, 0, 0])
                    class_values[0] += oracle_correct
                    class_values[1] += predicted_correct
                    class_values[2] += 1
                    target = task.tokenizer.id_to_token[int(labels[row, position])]
                    token_values = target_token_stats.setdefault(target, [0, 0, 0])
                    token_values[0] += oracle_correct
                    token_values[1] += predicted_correct
                    token_values[2] += 1
                    semantic_row = semantics[row][position]
                    if semantic_row is None:
                        continue
                    for index, value in enumerate(semantic_row):
                        if value is None:
                            continue
                        key = f"{compiled.nodes[index]}:{json.dumps(value, sort_keys=True)}"
                        values = semantic_outcome_stats.setdefault(key, [0, 0, 0.0, 0])
                        values[0] += oracle_correct
                        values[1] += predicted_correct
                        values[2] += float(trace_error[row, position])
                        values[3] += 1
            for index, node in enumerate(compiled.nodes):
                observed = eligibility[..., index].to(torch.bool) & valid
                probability = predicted.local_gate_probabilities[..., index]
                gold = activity[..., index].to(probability.dtype)
                error = probability - gold
                node_stats[node][0] += float(error[observed].abs().sum())
                node_stats[node][1] += float(error[observed].square().sum())
                node_stats[node][2] += int(observed.sum())
    metrics = {
        "oracle_token_accuracy": totals["oracle_correct"] / max(totals["tokens"], 1),
        "predicted_token_accuracy": totals["predicted_correct"] / max(totals["tokens"], 1),
        "predicted_soft_trace_mae": totals["soft_trace_abs_error"] / max(totals["soft_trace_count"], 1),
    }
    metrics.update(parent_audit_metrics)
    for node, (absolute_error, squared_error, count) in node_stats.items():
        metrics[f"gate/{node}/probability_mae"] = absolute_error / max(count, 1)
        metrics[f"gate/{node}/brier"] = squared_error / max(count, 1)
    for signature, (oracle_correct, predicted_correct, count) in functional_class_stats.items():
        metrics[f"functional_class/{signature}/oracle_accuracy"] = oracle_correct / count
        metrics[f"functional_class/{signature}/predicted_accuracy"] = predicted_correct / count
        metrics[f"functional_class/{signature}/count"] = float(count)
    for token, (oracle_correct, predicted_correct, count) in target_token_stats.items():
        metrics[f"target_token/{token}/oracle_accuracy"] = oracle_correct / count
        metrics[f"target_token/{token}/predicted_accuracy"] = predicted_correct / count
        metrics[f"target_token/{token}/count"] = float(count)
    for outcome, (oracle_correct, predicted_correct, trace_error, count) in semantic_outcome_stats.items():
        metrics[f"semantic_outcome/{outcome}/oracle_accuracy"] = oracle_correct / count
        metrics[f"semantic_outcome/{outcome}/predicted_accuracy"] = predicted_correct / count
        metrics[f"semantic_outcome/{outcome}/soft_trace_mae"] = trace_error / count
        metrics[f"semantic_outcome/{outcome}/count"] = float(count)
    if not include_full_decode:
        return metrics

    decode_examples = task.eval_examples
    for routing in ("oracle", "predicted"):
        decoded = _q_greedy_decode(
            model,
            task,
            [example.number for example in decode_examples],
            supervision=supervision,
            routing=routing,
            device=device,
        )
        exact = exact_without_eos = 0
        by_digit: dict[int, list[int]] = {}
        for (words, reached_eos), example in zip(decoded, decode_examples):
            target = example.text.split()
            correct_words = words == target
            exact += int(correct_words and reached_eos)
            exact_without_eos += int(correct_words)
            digit = len(str(int(example.number)))
            values = by_digit.setdefault(digit, [0, 0])
            values[0] += int(correct_words and reached_eos)
            values[1] += 1
        metrics[f"{routing}_exact_sequence_accuracy"] = exact / max(len(decode_examples), 1)
        metrics[f"{routing}_exact_accuracy_without_eos"] = exact_without_eos / max(len(decode_examples), 1)
        for digit, (correct_digit, count_digit) in by_digit.items():
            metrics[f"{routing}_exact_sequence_accuracy_digits_{digit}"] = correct_digit / count_digit
    return metrics


def _q_greedy_decode(model, task, numbers, *, supervision, routing: str, device):
    if routing not in {"oracle", "predicted"}:
        raise ValueError("routing must be 'oracle' or 'predicted'.")
    prompts = []
    for number in numbers:
        prompt = [task.tokenizer.bos_id]
        prompt.extend(task.tokenizer.token_to_id[f"<D{digit}>"] for digit in task.input_digit_string(number))
        prompt.append(task.tokenizer.sep_id)
        prompts.append(prompt)
    generated: list[list[int]] = [[] for _ in numbers]
    reached_eos = [False for _ in numbers]
    stopped = [False for _ in numbers]
    model.eval()
    with torch.no_grad():
        while True:
            active_indices = [
                index
                for index, prompt in enumerate(prompts)
                if not stopped[index]
                and not reached_eos[index]
                and len(prompt) + len(generated[index]) < int(task.config.max_seq_len)
            ]
            if not active_indices:
                break

            oracle_rows = []
            runnable_indices = []
            for index in active_indices:
                if routing == "oracle":
                    prefix = tuple(task.tokenizer.decode_words(generated[index]))
                    try:
                        oracle_rows.append(
                            supervision.lookup(
                                PredictiveState(
                                    digits=tuple(int(digit) for digit in str(int(numbers[index]))),
                                    prefix=prefix,
                                    position=len(prefix),
                                )
                            )
                        )
                    except KeyError:
                        stopped[index] = True
                        continue
                runnable_indices.append(index)
            if not runnable_indices:
                continue

            sequences = [prompts[index] + generated[index] for index in runnable_indices]
            memory_length = max(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(sequences), memory_length),
                task.tokenizer.pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            query_positions = torch.empty((len(sequences), 1), dtype=torch.long, device=device)
            for row, sequence in enumerate(sequences):
                input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
                attention_mask[row, : len(sequence)] = 1
                query_positions[row, 0] = len(sequence) - 1
            activity = None
            if routing == "oracle":
                activity = torch.tensor(
                    [[row.activity] for row in oracle_rows],
                    dtype=torch.float32,
                    device=device,
                )
            trace = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                query_positions=query_positions,
                routing=routing,
                activity_targets=activity,
            )
            next_tokens = trace.logits[:, 0].argmax(-1).detach().cpu().tolist()
            for index, token in zip(runnable_indices, next_tokens):
                generated[index].append(int(token))
                if int(token) == task.tokenizer.eos_id:
                    reached_eos[index] = True
    return [
        (task.tokenizer.decode_words(tokens), reached_eos[index])
        for index, tokens in enumerate(generated)
    ]


def _evaluate_parent_necessity(
    model,
    task,
    supervision,
    compiled,
    *,
    device,
    sample_size: int,
) -> dict[str, float]:
    if sample_size <= 0 or not task.eval_examples:
        return {}
    sample_size = min(int(sample_size), len(task.eval_examples))
    if sample_size == len(task.eval_examples):
        examples = task.eval_examples
    elif sample_size == 1:
        examples = [task.eval_examples[0]]
    else:
        last = len(task.eval_examples) - 1
        indices = [round(index * last / (sample_size - 1)) for index in range(sample_size)]
        examples = [task.eval_examples[index] for index in indices]
    batch = task.encode_examples(examples, device=device)
    activity, _, _ = _batch_supervision(
        batch,
        supervision,
        compiled,
        tokenizer=task.tokenizer,
        device=device,
    )
    audit = parent_necessity_audit(
        model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        activity_targets=activity,
    )
    baseline = audit["baseline_logits"]
    labels = batch.labels[:, 1:]
    valid = labels != -100
    baseline_accuracy = float(((baseline.argmax(-1) == labels) & valid).sum().cpu() / valid.sum().clamp_min(1).cpu())
    baseline_ce = float(F.cross_entropy(baseline[valid], labels[valid]).cpu())
    metrics = {
        "parent_audit/baseline/token_accuracy": baseline_accuracy,
        "parent_audit/baseline/cross_entropy": baseline_ce,
    }
    for name, delta in audit.items():
        if name == "baseline_logits":
            continue
        changed = baseline + delta
        accuracy = float(((changed.argmax(-1) == labels) & valid).sum().cpu() / valid.sum().clamp_min(1).cpu())
        cross_entropy = float(F.cross_entropy(changed[valid], labels[valid]).cpu())
        metrics[f"parent_audit/{name}/mean_abs_logit_delta"] = float(delta.abs().mean().cpu())
        metrics[f"parent_audit/{name}/token_accuracy_delta"] = accuracy - baseline_accuracy
        metrics[f"parent_audit/{name}/cross_entropy_delta"] = cross_entropy - baseline_ce
    return metrics


def _evaluate_transformer(model, task, *, device) -> dict[str, float]:
    model.eval()
    correct = count = 0
    with torch.no_grad():
        for start in range(0, len(task.eval_examples), int(task.config.eval_batch_size)):
            batch = task.encode_examples(task.eval_examples[start : start + int(task.config.eval_batch_size)], device=device)
            logits = model(**batch.model_inputs)[:, :-1]
            labels = batch.labels[:, 1:]
            valid = labels != -100
            correct += int(((logits.argmax(-1) == labels) & valid).sum())
            count += int(valid.sum())
    sequence = task.decode_metrics(model, task.eval_examples, device=device)
    return {
        "transformer_token_accuracy": correct / max(count, 1),
        "transformer_exact_sequence_accuracy": sequence["exact_accuracy"],
        "transformer_exact_accuracy_without_eos": sequence["exact_accuracy_wo_eos"],
        "transformer_normalized_edit_distance": sequence["normalized_edit_distance"],
        **{
            f"transformer_{name}": value
            for name, value in sequence.items()
            if name.startswith("digit_length_")
        },
    }


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _prediction_mask(labels: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:] != -100
    return mask


def _validate_teacher_preconditions(run_dir: str, config: QuantaSteeringConfig) -> None:
    results_path = os.path.join(run_dir, "results.pkl")
    if not os.path.exists(results_path):
        raise ValueError("steering requires standalone Q results.pkl for precondition validation.")
    with open(results_path, "rb") as handle:
        results = pickle.load(handle)
    records = results.get("metrics") or []
    if not records:
        raise ValueError("standalone Q results contain no evaluation metrics.")
    oracle_token = max(float(record.get("oracle_token_accuracy", 0.0)) for record in records)
    oracle_exact = max(float(record.get("oracle_exact_sequence_accuracy", 0.0)) for record in records)
    required = {
        "oracle_token_accuracy": (oracle_token, config.minimum_oracle_token_accuracy),
        "oracle_exact_sequence_accuracy": (oracle_exact, config.minimum_oracle_exact_accuracy),
    }
    failed = {name: values for name, values in required.items() if values[0] < values[1]}
    if failed:
        details = ", ".join(f"{name}={actual:.6g} < {threshold:.6g}" for name, (actual, threshold) in failed.items())
        raise ValueError("Q checkpoint does not satisfy steering preconditions: " + details)
