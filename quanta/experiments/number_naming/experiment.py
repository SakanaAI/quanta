from __future__ import annotations

import csv
import json
import logging
import math
import os
import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import wandb

from quanta.config import PosetsProbingConfig, PlotConfig
from quanta.experiments.base import Experiment
from quanta.experiments.common import (
    build_adam_optimizer,
    log_event,
    save_checkpoint,
    save_config,
    save_results,
    set_optimizer_learning_rate,
)
from quanta.experiments.scaling_laws.training.trainer_curriculum import scheduled_learning_rate
from quanta.metrics import compute_poset_dynamics_metrics, load_metric_config
from quanta.metrics.utils import learned_threshold_bits
from quanta.utils import _jsonable, set_seeds

from .model import DecoderTransformerLM
from .data import NumberNamingExample
from .names import english_number_name
from .probes import (
    FactorizedExampleTree,
    FactorizedNodeOccurrence,
    ProbeSuite,
    _contextual_probe_ids_for_occurrence,
    nats_to_bits,
    _factorized_operational_probes_for_example,
    _factorized_tree_for_example,
    _functional_operational_probes_for_example,
    _scale_id_for_role,
    _surface_token_role,
    _virtual_token_roles,
)
from .task import NumberNamingTask, _value_position_roles
from .tokenizer import DIGIT_TOKENS, SPECIAL_TOKENS

METRIC_BLUE = "#08306b"
METRIC_YELLOW = "#ffd84d"
ERROR_HISTOGRAM_LIMIT = 12


class PosetsProbingExperiment(Experiment):
    def __init__(self, config: PosetsProbingConfig, plot_config: PlotConfig | None = None):
        self.config = config
        self.plot_config = plot_config
        self.config.save_dir = self.config.save_dir or self._default_save_dir()

    def run(self) -> str:
        set_seeds(int(self.config.seed))
        device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        task = NumberNamingTask(self.config)
        self._resolve_steps_from_epochs(len(task.train))
        if self.config.save_dir is None:
            self.config.save_dir = self._default_save_dir()
        self._start_wandb(task)
        model = DecoderTransformerLM(
            vocab_size=task.tokenizer.vocab_size,
            max_seq_len=int(self.config.max_seq_len),
            d_model=int(self.config.d_model),
            n_layers=int(self.config.n_layers),
            n_heads=int(self.config.n_heads),
            dropout=float(self.config.dropout),
            pad_id=task.tokenizer.pad_id,
            mlp_ratio=float(self.config.mlp_ratio),
        ).to(device)
        model = _maybe_data_parallel(model, self.config, device)
        optimizer = build_adam_optimizer(
            model,
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        resume_state = self._select_resume_state()
        start_step = 0
        history: list[dict[str, Any]] = []
        if resume_state is not None:
            start_step = int(resume_state["steps_run"])
            history = [dict(record) for record in resume_state["history"]]
            _unwrap_model(model).load_state_dict(torch.load(resume_state["model_path"], map_location=device))
            optimizer.load_state_dict(torch.load(resume_state["optimizer_path"], map_location=device))
            _optimizer_to_device(optimizer, device)
            _restore_number_naming_trainer_state(resume_state.get("trainer_state_path"), task)
            log_event(
                "posets_probing",
                "resuming",
                run_dir=resume_state["run_dir"],
                step=start_step,
                total_steps=self.config.steps,
                prior_evaluations=len(history),
            )
        effective_sample_counts = _effective_sample_counts_from_history(history)
        prior_best_record = _best_number_naming_record(history)
        best_exact_accuracy = (
            -math.inf
            if prior_best_record is None
            else float(prior_best_record.get("exact_accuracy", -math.inf))
        )
        best_exact_step = None if prior_best_record is None else int(prior_best_record["step"])

        os.makedirs(self.config.save_dir, exist_ok=True)
        save_config(self.config, self.config.save_dir)
        self._save_vocab(task)
        total_params, trainable_params = _parameter_counts(_unwrap_model(model))
        log_event(
            "posets_probing",
            "run_start",
            task="number_naming",
            split=self.config.data_splits,
            eval_strategy=self.config.eval_strategy,
            train_examples=len(task.train),
            eval_examples=len(task.eval_examples),
            model="decoder_transformer",
            layers=self.config.n_layers,
            width=self.config.d_model,
            heads=self.config.n_heads,
            parameters=total_params,
            trainable_parameters=trainable_params,
            steps=self.config.steps,
            batch_size=self.config.batch_size,
            device=device,
            probes=[suite.id for suite in task.probe_suites],
            audits=self.config.audit_quanta_poset or [],
            save_dir=self.config.save_dir,
        )

        start_time = time.monotonic()
        eval_steps = self._evaluation_steps(len(task.train))
        for step in range(start_step + 1, int(self.config.steps) + 1):
            current_lr = scheduled_learning_rate(
                float(self.config.lr),
                int(step),
                int(self.config.steps),
                self.config.scheduler,
                warmup_phase=float(self.config.warmup_phase),
                plateau_phase=float(self.config.plateau_phase),
            )
            set_optimizer_learning_rate(optimizer, current_lr)
            model.train()
            batch = task.make_batch(split="train", batch_size=int(self.config.batch_size), device=device)
            _update_number_naming_effective_samples(effective_sample_counts, batch, task.probe_suites)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = self._training_loss(model, task, batch, step=step)
            loss.backward()
            optimizer.step()

            if step in eval_steps:
                self._record_evaluation(
                    model,
                    task,
                    history=history,
                    step=step,
                    train_loss=nats_to_bits(float(loss.detach().cpu().item())),
                    train_metrics={**train_metrics, "lr": float(current_lr)},
                    effective_samples=_snapshot_effective_sample_counts(effective_sample_counts),
                    device=device,
                )
                record = history[-1]
                exact_accuracy = float(record["exact_accuracy"])
                if exact_accuracy > best_exact_accuracy:
                    best_exact_accuracy = exact_accuracy
                    best_exact_step = int(step)
                    save_checkpoint(_unwrap_model(model), optimizer, self.config.save_dir)
                    _save_number_naming_trainer_state(self.config.save_dir, task, step)
                    log_event(
                        "posets_probing",
                        "checkpoint",
                        kind="best",
                        step=best_exact_step,
                        sequence_accuracy=best_exact_accuracy,
                        path=os.path.join(self.config.save_dir, "model.pt"),
                    )
                if self.config.save_steps is not None and step % int(self.config.save_steps) == 0:
                    _save_number_naming_last_checkpoint(_unwrap_model(model), optimizer, self.config.save_dir)
                    _save_number_naming_trainer_state(
                        self.config.save_dir,
                        task,
                        step,
                        filename="last_trainer_state.pt",
                    )

        if best_exact_step is None:
            save_checkpoint(_unwrap_model(model), optimizer, self.config.save_dir)
            _save_number_naming_trainer_state(self.config.save_dir, task, int(self.config.steps))
        elif best_exact_step != int(self.config.steps):
            _save_number_naming_last_checkpoint(_unwrap_model(model), optimizer, self.config.save_dir)
            _save_number_naming_trainer_state(
                self.config.save_dir,
                task,
                int(self.config.steps),
                filename="last_trainer_state.pt",
            )
        results = self._build_results(task, history, start_time, resume_state=resume_state)
        results["audit_quanta_poset"] = self._run_audits(model, task, device=device)
        logging.debug("number_naming probe_learning_order=%s", results["probe_learning_order"])
        self._write_final_artifacts(history, results, task)
        if wandb.run is not None:
            metrics_path = os.path.join(
                self.config.save_dir,
                _number_naming_metrics_filename(self.plot_config),
            )
            wandb.log({"figures/metrics": wandb.Image(metrics_path)})
            wandb.finish()
        log_event(
            "posets_probing",
            "run_complete",
            steps=self.config.steps,
            best_sequence_accuracy=results["best_exact_accuracy"],
            best_step=results["best_exact_step"],
            runtime_seconds=results["runtime_seconds"],
            save_dir=self.config.save_dir,
        )
        return self.config.save_dir

    def _select_resume_state(self) -> dict[str, Any] | None:
        current_steps = int(self.config.steps or 0)
        best: dict[str, Any] | None = None
        for candidate_dir in _number_naming_resume_candidate_dirs(self.config):
            try:
                state = _number_naming_resume_state(candidate_dir, self.config)
            except Exception as error:
                logging.warning("ignoring number_naming resume candidate %s: %r", candidate_dir, error)
                continue
            if state is None:
                continue
            steps_run = int(state["steps_run"])
            if steps_run >= current_steps:
                log_event(
                    "posets_probing",
                    "run_reused",
                    run_dir=candidate_dir,
                    steps=steps_run,
                    total_steps=current_steps,
                )
                continue
            if best is None or steps_run > int(best["steps_run"]):
                best = state
        return best

    def _evaluate(self, model, task: NumberNamingTask, *, device) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            metrics = self._teacher_forced_eval_metrics(model, task, device=device)
            decode_count = min(int(self.config.eval_batch_size), len(task.eval_examples))
            decode_examples = _even_sample(task.eval_examples, decode_count)
            metrics.update(task.decode_metrics(model, decode_examples, device=device))
            metrics.update(self._probe_metrics(model, task, device=device))
        return metrics

    def _training_loss(self, model, task: NumberNamingTask, batch, *, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        logits = model(**batch.model_inputs)
        return task.compute_loss(logits, batch), {}

    def _teacher_forced_eval_metrics(self, model, task: NumberNamingTask, *, device) -> dict[str, float]:
        loss_sum = 0.0
        token_count = 0
        correct_count = 0
        role_sums: dict[str, float] = {}
        role_counts: dict[str, int] = {}
        token_sums: dict[str, float] = {}
        token_counts: dict[str, int] = {}
        digit_correct_counts: dict[int, int] = {}
        digit_token_counts: dict[int, int] = {}
        digit_loss_sums: dict[int, float] = {}
        digit_loss_counts: dict[int, int] = {}
        wrong_target_tokens: dict[str, int] = {}
        wrong_number_tokens: dict[str, int] = {}
        ignored_tokens = set(task.tokenizer.token_to_id[token] for token in SPECIAL_TOKENS + DIGIT_TOKENS)
        for examples in _chunks(task.eval_examples, max(1, int(self.config.eval_batch_size))):
            batch = task.encode_examples(examples, device=device)
            logits = model(**batch.model_inputs)
            shift_logits = logits[:, :-1, :]
            shift_labels = batch.labels[:, 1:]
            mask = shift_labels != -100
            losses = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(shift_labels)
            losses = losses * (1.0 / math.log(2.0))
            predictions = shift_logits.argmax(dim=-1)
            loss_sum += float(losses[mask].sum().detach().cpu().item())
            token_count += int(mask.sum().item())
            correct_count += int(((predictions == shift_labels) & mask).sum().item())
            for row, (number, text) in enumerate(zip(batch.numbers, batch.texts)):
                row_losses = losses[row][mask[row]]
                row_predictions = predictions[row][mask[row]]
                row_labels = shift_labels[row][mask[row]]
                row_correct = row_predictions == row_labels
                digit_length = len(str(int(number)))
                digit_loss_sums[digit_length] = digit_loss_sums.get(digit_length, 0.0) + float(row_losses.mean().detach().cpu().item())
                digit_loss_counts[digit_length] = digit_loss_counts.get(digit_length, 0) + 1
                digit_correct_counts[digit_length] = digit_correct_counts.get(digit_length, 0) + int(row_correct.sum().item())
                digit_token_counts[digit_length] = digit_token_counts.get(digit_length, 0) + int(row_labels.numel())
                row_wrong = int((~row_correct).sum().item())
                if row_wrong:
                    wrong_number_tokens[str(int(number))] = wrong_number_tokens.get(str(int(number)), 0) + row_wrong
                    for token_id in row_labels[~row_correct].detach().cpu().tolist():
                        token = _error_token_label(task.tokenizer.id_to_token[int(token_id)])
                        wrong_target_tokens[token] = wrong_target_tokens.get(token, 0) + 1
                roles = _value_position_roles(number, text.split())
                for role, value in zip(roles, row_losses):
                    role_sums[role] = role_sums.get(role, 0.0) + float(value.detach().cpu().item())
                    role_counts[role] = role_counts.get(role, 0) + 1
            for token_id in sorted(set(int(value.item()) for value in shift_labels[mask])):
                if token_id in ignored_tokens:
                    continue
                token = task.tokenizer.id_to_token[token_id]
                values = losses[shift_labels == token_id]
                token_sums[token] = token_sums.get(token, 0.0) + float(values.sum().detach().cpu().item())
                token_counts[token] = token_counts.get(token, 0) + int(values.numel())

        metrics = {
            "eval_loss": loss_sum / max(token_count, 1),
            "token_accuracy": correct_count / max(token_count, 1),
        }
        role_order = [
            "units",
            "tens",
            "hundreds",
            "thousands",
            "ten_thousands",
            "hundred_thousands",
            "millions",
            "ten_millions",
            "hundred_millions",
            "billions",
            "ten_billions",
            "hundred_billions",
        ]
        for role in role_order:
            if role_counts.get(role):
                metrics[f"value_position_loss_{role}"] = role_sums[role] / role_counts[role]
        for digit_length in sorted(digit_loss_counts):
            metrics[f"digit_length_loss_{digit_length}"] = digit_loss_sums[digit_length] / digit_loss_counts[digit_length]
        for digit_length in sorted(digit_token_counts):
            metrics[f"digit_length_token_accuracy_{digit_length}"] = digit_correct_counts[digit_length] / max(digit_token_counts[digit_length], 1)
        for token in sorted(token_sums):
            metrics[f"token_loss_{token}"] = token_sums[token] / token_counts[token]
        metrics.update(_top_error_histogram_metrics(wrong_target_tokens, wrong_number_tokens))
        return metrics

    def _digit_length_metrics(self, model, task: NumberNamingTask, *, device) -> dict[str, float]:
        sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        correct_counts: dict[int, int] = {}
        token_counts: dict[int, int] = {}
        for examples in _chunks(task.eval_examples, max(1, int(self.config.eval_batch_size))):
            batch = task.encode_examples(examples, device=device)
            logits = model(**batch.model_inputs)
            shift_logits = logits[:, :-1, :]
            shift_labels = batch.labels[:, 1:]
            mask = shift_labels != -100
            losses = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(shift_labels)
            losses = losses * (1.0 / math.log(2.0))
            predictions = shift_logits.argmax(dim=-1)
            for row, number in enumerate(batch.numbers):
                if not bool(mask[row].any()):
                    continue
                digit_length = len(str(int(number)))
                sums[digit_length] = sums.get(digit_length, 0.0) + float(losses[row][mask[row]].mean().detach().cpu().item())
                counts[digit_length] = counts.get(digit_length, 0) + 1
                row_mask = mask[row]
                correct_counts[digit_length] = correct_counts.get(digit_length, 0) + int(((predictions[row] == shift_labels[row]) & row_mask).sum().item())
                token_counts[digit_length] = token_counts.get(digit_length, 0) + int(row_mask.sum().item())
        metrics = {
            f"digit_length_loss_{digit_length}": sums[digit_length] / counts[digit_length]
            for digit_length in sorted(sums)
        }
        for digit_length in sorted(token_counts):
            metrics[f"digit_length_token_accuracy_{digit_length}"] = correct_counts[digit_length] / max(token_counts[digit_length], 1)
        return metrics

    def _probe_metrics(self, model, task: NumberNamingTask, *, device) -> dict[str, float]:
        if not task.probe_suites:
            return {}
        metrics = {}
        single_suite = len(task.probe_suites) == 1
        for suite in task.probe_suites:
            metrics.update(
                self._probe_suite_metrics(
                    model,
                    task,
                    suite,
                    device=device,
                    metric_prefix=None if single_suite else suite.id,
                )
            )
        return metrics

    def _probe_suite_metrics(
        self,
        model,
        task: NumberNamingTask,
        suite: ProbeSuite,
        *,
        device,
        metric_prefix: str | None,
    ) -> dict[str, float]:
        if _is_structured_schema_suite(suite):
            return _factorized_probe_metrics(model, task, suite, device=device, metric_prefix=metric_prefix)
        if suite.id == "eng_contextual_tokens":
            return _contextual_probe_metrics(model, task, suite, device=device, metric_prefix=metric_prefix)
        sums = {probe.id: 0.0 for probe in suite.probes}
        counts = {probe.id: 0 for probe in suite.probes}
        batch_size = max(1, int(self.config.eval_batch_size))
        for tagged_chunk in _chunks(suite.tagged_examples, batch_size):
            examples = [
                NumberNamingExample(number=example.number, text=example.text)
                for example in tagged_chunk
            ]
            batch = task.encode_examples(examples, device=device)
            logits = model(**batch.model_inputs)
            shift_logits = logits[:, :-1, :]
            shift_labels = batch.labels[:, 1:]
            mask = shift_labels != -100
            losses = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(shift_labels)
            losses = losses * (1.0 / math.log(2.0))
            for row, tagged in enumerate(tagged_chunk):
                row_losses = losses[row][mask[row]]
                for tags, value in zip(tagged.tags, row_losses):
                    loss_value = float(value.detach().cpu().item())
                    for tag in _iter_probe_tags(tags):
                        sums[tag] += loss_value
                        counts[tag] += 1

        metrics = {}
        role_losses = []
        for probe in suite.probes:
            if counts[probe.id] == 0:
                continue
            loss_bits = sums[probe.id] / counts[probe.id]
            role_losses.append(loss_bits)
            metrics[_probe_metric_key(probe.id, metric_prefix)] = loss_bits
        if role_losses:
            metrics[_probe_metric_key(suite.id, metric_prefix)] = float(sum(role_losses) / len(role_losses))
        return metrics

    def _evaluation_steps(self, train_size: int) -> set[int]:
        steps = int(self.config.steps)
        if self.config.evals_per_epoch is not None:
            eval_steps = {steps}
            total_evals = int(math.ceil(float(self.config.epochs or 0.0) * int(self.config.evals_per_epoch)))
            for index in range(1, total_evals + 1):
                step = int(math.ceil(index * train_size / (int(self.config.evals_per_epoch) * int(self.config.batch_size))))
                if 0 < step <= steps:
                    eval_steps.add(step)
            return eval_steps

        eval_interval = int(self.config.eval_steps)
        return {step for step in range(1, steps + 1) if step % eval_interval == 0 or step == steps}

    def _record_evaluation(
        self,
        model,
        task: NumberNamingTask,
        *,
        history: list[dict[str, Any]],
        step: int,
        train_loss: float | None,
        train_metrics: dict[str, float] | None = None,
        effective_samples: dict[str, Any],
        device,
    ) -> None:
        metrics = self._evaluate(model, task, device=device)
        record = {
            "step": int(step),
            "epoch": float(int(step) * int(self.config.batch_size) / max(len(task.train), 1)),
            "train_loss": None if train_loss is None else float(train_loss),
            "effective_samples": effective_samples,
            **metrics,
        }
        if train_metrics:
            record.update(train_metrics)
        history.append(record)
        if wandb.run is not None:
            wandb.log(_wandb_scalar_metrics(record), step=step)
            for key, value in _wandb_summary_metrics(record).items():
                wandb.run.summary[key] = value
        log_event(
            "posets_probing",
            "evaluation",
            step=step,
            total_steps=self.config.steps,
            epoch=record["epoch"],
            train_loss=math.nan if record["train_loss"] is None else record["train_loss"],
            eval_loss=record["eval_loss"],
            token_accuracy=record["token_accuracy"],
            sequence_accuracy=record["exact_accuracy"],
            normalized_edit_distance=record["normalized_edit_distance"],
        )

    def _qualitative_samples_table(self, model, task: NumberNamingTask, *, step: int, device):
        rows = []
        rng = random.Random((int(self.config.seed) + 1) * 1_000_003 + int(step))
        by_digits: dict[int, list] = {}
        for example in task.eval_examples:
            by_digits.setdefault(len(str(int(example.number))), []).append(example)
        selected = []
        for digits in sorted(by_digits):
            selected.append(rng.choice(by_digits[digits]))
        predictions = task.greedy_decode(model, [example.number for example in selected], device=device)
        for example, prediction_tokens in zip(selected, predictions):
            prediction = " ".join(prediction_tokens)
            rows.append(
                [
                    int(step),
                    len(str(int(example.number))),
                    int(example.number),
                    example.text,
                    prediction,
                    prediction == example.text,
                ]
            )
        return wandb.Table(
            columns=["step", "digits", "number", "target", "prediction", "exact"],
            data=rows,
        )

    def _build_results(
        self,
        task: NumberNamingTask,
        history: list[dict[str, Any]],
        start_time: float,
        resume_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_seconds = float(time.monotonic() - start_time)
        if resume_state is not None:
            runtime_seconds += float(resume_state.get("runtime_seconds", 0.0))
        best_record = _best_number_naming_record(history)
        return {
            "experiment": "posets_probing",
            "task": "number_naming",
            "config": _jsonable(asdict(self.config)),
            "vocab": task.tokenizer.token_to_id,
            "n_train": len(task.train),
            "n_eval": len(task.eval),
            "metrics": history,
            "steps_run": int(history[-1]["step"]) if history else 0,
            "target_steps": int(self.config.steps),
            "epochs_run": float(history[-1]["epoch"]) if history else 0.0,
            "target_epochs": float(self.config.epochs or 0.0),
            "runtime_seconds": runtime_seconds,
            "final_metrics": history[-1] if history else {},
            "best_exact_accuracy": None if best_record is None else float(best_record["exact_accuracy"]),
            "best_exact_step": None if best_record is None else int(best_record["step"]),
            "best_metrics": {} if best_record is None else dict(best_record),
            "probe_learning_order": _probe_learning_order(history, task.probe_suites),
            "probe_quanta_poset": _probe_suites_summary(task.probe_suites),
            "resumed_from": None if resume_state is None else resume_state["run_dir"],
        }

    def _write_final_artifacts(
        self,
        history: list[dict[str, Any]],
        results: dict[str, Any],
        task: NumberNamingTask,
    ) -> None:
        self._write_metrics_jsonl(history)
        self._write_metrics_csv(history)
        save_results(results, self.config.save_dir)
        self._plot_history(history, task)
        self._write_probe_poset_artifacts(task)
        self._plot_probe_history(history, task)
        self._write_factorized_probe_artifacts(history, task)
        with open(os.path.join(self.config.save_dir, "summary.json"), "w") as handle:
            json.dump(_jsonable(results), handle, indent=4)

    def _write_probe_poset_artifacts(self, task: NumberNamingTask) -> None:
        if not task.probe_suites:
            return
        for probe_suite in task.probe_suites:
            suite_dir = _suite_output_dir(self.config.save_dir, probe_suite.id)
            os.makedirs(suite_dir, exist_ok=True)
            mermaid_path = os.path.join(suite_dir, "quanta_poset.mmd")
            png_path = os.path.join(suite_dir, "quanta_poset.png")
            with open(mermaid_path, "w") as handle:
                handle.write(probe_suite.mermaid)
            _write_mermaid_png(probe_suite.id, probe_suite.mermaid, png_path)

    def _write_metrics_jsonl(self, history: list[dict[str, Any]]) -> None:
        if not history:
            return
        path = os.path.join(self.config.save_dir, "metrics.jsonl")
        with open(path, "w") as handle:
            for record in history:
                handle.write(json.dumps(_jsonable(record)) + "\n")

    def _save_vocab(self, task: NumberNamingTask) -> None:
        with open(os.path.join(self.config.save_dir, "vocab.json"), "w") as handle:
            json.dump(task.tokenizer.token_to_id, handle, indent=4)

    def _write_metrics_csv(self, history: list[dict[str, Any]]) -> None:
        if not history:
            return
        path = os.path.join(self.config.save_dir, "metrics.csv")
        fieldnames = []
        for record in history:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    def _plot_history(self, history: list[dict[str, Any]], task: NumberNamingTask) -> None:
        if not history:
            return
        plot_config = self.plot_config or PlotConfig()
        _plot_metrics_panel(
            history,
            os.path.join(self.config.save_dir, _number_naming_metrics_filename(self.plot_config)),
            config=self.config,
            plot_config=plot_config,
            task=task,
        )

    def _plot_probe_history(self, history: list[dict[str, Any]], task: NumberNamingTask) -> None:
        if not history or not task.probe_suites:
            return
        single_suite = len(task.probe_suites) == 1
        for probe_suite in task.probe_suites:
            _plot_probe_losses(
                history,
                probe_suite,
                os.path.join(_suite_output_dir(self.config.save_dir, probe_suite.id), "probe_quanta_losses.png"),
                config=self.config,
                plot_config=self.plot_config,
                train_examples=task.train,
                metric_prefix=None if single_suite else probe_suite.id,
            )

    def _write_factorized_probe_artifacts(self, history: list[dict[str, Any]], task: NumberNamingTask) -> None:
        for suite in task.probe_suites:
            if not _is_structured_schema_suite(suite):
                continue
            _write_factorized_artifacts(
                history,
                suite,
                save_dir=self.config.save_dir,
                config=self.config,
                plot_config=self.plot_config or PlotConfig(),
            )

    def _resolve_steps_from_epochs(self, train_size: int) -> None:
        if self.config.steps is None:
            self.config.steps = int(math.ceil(float(self.config.epochs) * train_size / int(self.config.batch_size)))
        if self.config.epochs is None:
            self.config.epochs = float(int(self.config.steps) * int(self.config.batch_size) / max(train_size, 1))

    def _start_wandb(self, task: NumberNamingTask) -> None:
        if not self.config.wandb_project or wandb.run is not None:
            return
        wandb.init(
            project=self.config.wandb_project,
            mode=self.config.wandb_mode,
            config={
                **_jsonable(asdict(self.config)),
                "n_train": len(task.train),
                "n_eval": len(task.eval),
                "vocab_size": task.tokenizer.vocab_size,
            },
        )

    def _run_audits(self, model, task: NumberNamingTask, *, device) -> dict[str, Any]:
        requested = set(self.config.audit_quanta_poset or [])
        if not requested:
            return {}
        results: dict[str, Any] = {}
        if "patching_compatibility" in requested:
            results["patching_compatibility"] = _patching_compatibility_audit(
                _unwrap_model(model),
                task,
                device=device,
                output_dir=self.config.save_dir,
            )
        if "held_out_transfer" in requested:
            results["held_out_transfer"] = _held_out_transfer_audit(
                _unwrap_model(model),
                task,
                held_out_token_roles=self.config.held_out_token_roles or [],
                device=device,
                output_dir=os.path.join(self.config.save_dir, "held_out_transfer"),
            )
        return results

    def _default_save_dir(self) -> str:
        budget = f"steps{self.config.steps}" if self.config.steps is not None else f"epochs{self.config.epochs:g}"
        return os.path.join(
            ".experiments",
            "posets_probing",
            "number_naming",
            self.config.language,
            self.config.data_splits,
            f"{self.config.eval_strategy}_eval",
            (
                f"layers{self.config.n_layers}-d{self.config.d_model}-heads{self.config.n_heads}"
                f"-{budget}-seed{self.config.seed}"
            ),
        )


def _number_naming_resume_candidate_dirs(config: PosetsProbingConfig) -> list[Path]:
    save_dir = Path(str(config.save_dir))
    parent = save_dir.parent
    candidates = []
    if save_dir.is_dir():
        candidates.append(save_dir)
    if parent.is_dir():
        candidates.extend(path for path in parent.iterdir() if path.is_dir() and path.name.startswith("layers"))
    return sorted(set(candidates))


def _number_naming_resume_state(candidate_dir: Path, config: PosetsProbingConfig) -> dict[str, Any] | None:
    model_path = candidate_dir / "model.pt"
    optimizer_path = candidate_dir / "optimizer.pt"
    results_path = candidate_dir / "results.pkl"
    if not (model_path.is_file() and optimizer_path.is_file() and results_path.is_file()):
        return None
    with results_path.open("rb") as handle:
        results = pickle.load(handle)
    saved_config = _number_naming_saved_config(candidate_dir, results)
    current_config = _number_naming_current_config(config)
    if not _number_naming_configs_match_for_resume(saved_config, current_config):
        return None
    history = _number_naming_resume_history(candidate_dir, results)
    steps_run = int(results.get("steps_run") or (history[-1]["step"] if history else 0))
    if steps_run <= 0:
        return None
    resume_model_path = candidate_dir / "last_model.pt"
    resume_optimizer_path = candidate_dir / "last_optimizer.pt"
    resume_trainer_state_path = candidate_dir / "last_trainer_state.pt"
    if not (resume_model_path.is_file() and resume_optimizer_path.is_file()):
        resume_model_path = model_path
        resume_optimizer_path = optimizer_path
        resume_trainer_state_path = candidate_dir / "trainer_state.pt"
    return {
        "run_dir": str(candidate_dir),
        "model_path": str(resume_model_path),
        "optimizer_path": str(resume_optimizer_path),
        "trainer_state_path": str(resume_trainer_state_path)
        if resume_trainer_state_path.is_file()
        else None,
        "history": [record for record in history if int(record.get("step", 0)) <= steps_run],
        "steps_run": steps_run,
        "runtime_seconds": float(results.get("runtime_seconds", 0.0)),
        "results": results,
    }


def _number_naming_saved_config(candidate_dir: Path, results: dict[str, Any]) -> dict[str, Any]:
    config_path = candidate_dir / "config.json"
    if config_path.is_file():
        with config_path.open("r") as handle:
            return json.load(handle)
    return dict(results.get("config") or {})


def _number_naming_current_config(config: PosetsProbingConfig) -> dict[str, Any]:
    return _jsonable(asdict(config))


def _number_naming_configs_match_for_resume(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    ignored = {"steps", "epochs", "save_dir"}
    return _normalized_resume_config(saved, ignored=ignored) == _normalized_resume_config(current, ignored=ignored)


def _normalized_resume_config(config: dict[str, Any], *, ignored: set[str]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("eval_strategy", "digits_wise")
    normalized.setdefault("eval_samples", None)
    normalized.setdefault("wandb_mode", None)
    normalized.setdefault("device", None)
    normalized.setdefault("width", None)
    normalized.setdefault("save_steps", None)
    normalized.setdefault("audit_quanta_poset", None)
    normalized.setdefault("held_out_token_roles", None)
    return {
        key: _jsonable(value)
        for key, value in sorted(normalized.items())
        if key not in ignored
    }


def _number_naming_resume_history(candidate_dir: Path, results: dict[str, Any]) -> list[dict[str, Any]]:
    if results.get("metrics"):
        return [dict(record) for record in results["metrics"]]
    metrics_jsonl = candidate_dir / "metrics.jsonl"
    if metrics_jsonl.is_file():
        history = []
        with metrics_jsonl.open("r") as handle:
            for line in handle:
                if line.strip():
                    history.append(json.loads(line))
        if history:
            return history
    metrics_csv = candidate_dir / "metrics.csv"
    if metrics_csv.is_file():
        with metrics_csv.open("r", newline="") as handle:
            return [
                {key: _parse_metric_value(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    return []


def _best_number_naming_record(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in history if "exact_accuracy" in record]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(record["exact_accuracy"]))


def _parse_metric_value(value: str):
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    if math.isfinite(parsed) and parsed.is_integer():
        return int(parsed)
    return parsed


def _save_number_naming_last_checkpoint(model: torch.nn.Module, optimizer, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "last_optimizer.pt"))


def _optimizer_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _iter_probe_tags(tags: Any):
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (tags,)
    return tuple(str(tag) for tag in tags if tag is not None)


def _save_number_naming_trainer_state(
    save_dir: str,
    task: NumberNamingTask,
    step: int,
    *,
    filename: str = "trainer_state.pt",
) -> None:
    payload = {
        "step": int(step),
        "task_rng_state": task.rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, os.path.join(save_dir, filename))


def _restore_number_naming_trainer_state(path: str | None, task: NumberNamingTask) -> None:
    if path is None or not os.path.isfile(path):
        return
    state = torch.load(path, map_location="cpu")
    if state.get("task_rng_state") is not None:
        task.rng.setstate(state["task_rng_state"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state_all = state.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def _suite_output_dir(save_dir: str, suite_id: str) -> str:
    return os.path.join(save_dir, suite_id)


def _write_mermaid_png(graph_id: str, mermaid_source: str, output_path: str) -> None:
    _ensure_output_parent(output_path)
    try:
        from mermaid import Graph, Mermaid

        Mermaid(Graph(graph_id, mermaid_source)).to_png(output_path)
    except Exception as error:
        logging.warning("failed to render Mermaid graph %s to %s: %s", graph_id, output_path, error)


def _wandb_scalar_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    direct = {
        "epoch": "progress/epoch",
        "train_loss": "loss/train",
        "eval_loss": "loss/eval",
        "lr": "optimization/learning_rate",
        "token_accuracy": "accuracy/token",
        "exact_accuracy": "accuracy/sequence",
        "normalized_edit_distance": "error/normalized_edit_distance",
    }
    for key, destination in direct.items():
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[destination] = value

    for key, value in record.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if key.startswith("digit_length_loss_"):
            payload[f"loss_by_digits/{key.removeprefix('digit_length_loss_')}"] = value
        elif key.startswith("digit_length_exact_accuracy_") and "without_eos" not in key:
            payload[f"sequence_accuracy_by_digits/{key.removeprefix('digit_length_exact_accuracy_')}"] = value
        elif key.startswith("probe_loss/"):
            parts = key.split("/")
            if len(parts) == 2 or (len(parts) == 3 and parts[1] == parts[2]):
                payload[f"probe_loss/{parts[1]}"] = value
    return payload


def _wandb_summary_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    headline_keys = {
        "progress/epoch",
        "loss/train",
        "loss/eval",
        "accuracy/token",
        "accuracy/sequence",
        "error/normalized_edit_distance",
    }
    return {
        key: value
        for key, value in _wandb_scalar_metrics(record).items()
        if key in headline_keys
    }


def _new_effective_sample_counts() -> dict[str, Any]:
    return {"total": 0, "metrics": {}, "probes": {}, "gradient_exposure": {"total": 0.0, "metrics": {}, "probes": {}}}


def _number_naming_metrics_filename(plot_config: PlotConfig | None) -> str:
    if plot_config is None or plot_config.output_filename == PlotConfig().output_filename:
        return "metrics.png"
    return plot_config.output_filename


def _error_token_label(token: str) -> str:
    if token == "[EOS]":
        return "EOS"
    return str(token)


def _error_metric_suffix(value: str | int) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _top_error_histogram_metrics(
    wrong_target_tokens: dict[str, int],
    wrong_number_tokens: dict[str, int],
    *,
    limit: int = ERROR_HISTOGRAM_LIMIT,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for token, count in _top_error_items(wrong_target_tokens, limit=limit):
        metrics[f"wrong_token_count_{_error_metric_suffix(token)}"] = float(count)
    for number, count in _top_error_items(wrong_number_tokens, limit=limit):
        metrics[f"wrong_number_error_tokens_{_error_metric_suffix(number)}"] = float(count)
    return metrics


def _top_error_items(values: dict[str, int], *, limit: int) -> list[tuple[str, int]]:
    return sorted(
        ((str(key), int(value)) for key, value in values.items() if int(value) > 0),
        key=lambda item: (-item[1], item[0]),
    )[: max(0, int(limit))]


def _effective_sample_counts_from_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    for record in reversed(history):
        effective_samples = record.get("effective_samples")
        if isinstance(effective_samples, dict):
            return _copy_effective_sample_counts(effective_samples)
    return _new_effective_sample_counts()


def _copy_effective_sample_counts(counts: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(counts))


def _snapshot_effective_sample_counts(counts: dict[str, Any]) -> dict[str, Any]:
    return _copy_effective_sample_counts(counts)


def _increment_count(counts: dict[str, int], key: str, value: int = 1) -> None:
    counts[key] = int(counts.get(key, 0)) + int(value)


def _increment_float(counts: dict[str, float], key: str, value: float) -> None:
    counts[key] = float(counts.get(key, 0.0)) + float(value)


def _nested_probe_counts(counts: dict[str, Any], suite_id: str) -> dict[str, Any]:
    probes = counts.setdefault("probes", {})
    return probes.setdefault(suite_id, {})


def _gradient_exposure_counts(counts: dict[str, Any]) -> dict[str, Any]:
    return counts.setdefault("gradient_exposure", {"total": 0.0, "metrics": {}, "probes": {}})


def _nested_gradient_probe_counts(counts: dict[str, Any], suite_id: str) -> dict[str, float]:
    exposure = _gradient_exposure_counts(counts)
    probes = exposure.setdefault("probes", {})
    return probes.setdefault(suite_id, {})


def _update_number_naming_effective_samples(
    counts: dict[str, Any],
    batch,
    probe_suites: list[ProbeSuite],
) -> None:
    metrics = counts.setdefault("metrics", {})
    counts["total"] = int(counts.get("total", 0)) + len(batch.numbers)
    supervised_units_by_example = _batch_supervised_units_by_example(batch)
    total_supervised_units = max(1, sum(supervised_units_by_example))
    gradient_exposure = _gradient_exposure_counts(counts)
    gradient_exposure["total"] = float(gradient_exposure.get("total", 0.0)) + 1.0
    gradient_metrics = gradient_exposure.setdefault("metrics", {})
    for index, (number, text) in enumerate(zip(batch.numbers, batch.texts)):
        number = int(number)
        words = str(text).split()
        example_supervised_units = supervised_units_by_example[index] if index < len(supervised_units_by_example) else len(words)
        _increment_count(metrics, "eval_loss", len(words))
        _increment_float(gradient_metrics, "eval_loss", example_supervised_units / total_supervised_units)
        _increment_count(metrics, f"digit_length_loss_{len(str(number))}")
        _increment_count(metrics, f"digit_length_token_accuracy_{len(str(number))}")
        _increment_count(metrics, f"digit_length_exact_accuracy_{len(str(number))}")
        _increment_count(metrics, f"digit_length_exact_accuracy_wo_eos_{len(str(number))}")
        _increment_float(
            gradient_metrics,
            f"digit_length_loss_{len(str(number))}",
            example_supervised_units / total_supervised_units,
        )
        _increment_float(
            gradient_metrics,
            f"digit_length_token_accuracy_{len(str(number))}",
            example_supervised_units / total_supervised_units,
        )
        _increment_float(
            gradient_metrics,
            f"digit_length_exact_accuracy_{len(str(number))}",
            example_supervised_units / total_supervised_units,
        )
        _increment_float(
            gradient_metrics,
            f"digit_length_exact_accuracy_wo_eos_{len(str(number))}",
            example_supervised_units / total_supervised_units,
        )
        for role in _value_position_roles(number, words):
            _increment_count(metrics, f"value_position_loss_{role}")
            _increment_float(gradient_metrics, f"value_position_loss_{role}", 1.0 / total_supervised_units)
        for token in words:
            _increment_count(metrics, f"token_loss_{token}")
            _increment_float(gradient_metrics, f"token_loss_{token}", 1.0 / total_supervised_units)
        for suite in probe_suites:
            probe_counts = _probe_exposure_counts_for_example(suite, number, str(text), words)
            suite_counts = _nested_probe_counts(counts, suite.id)
            gradient_suite_counts = _nested_gradient_probe_counts(counts, suite.id)
            for probe_id, value in probe_counts.items():
                _increment_count(suite_counts, probe_id, value)
                _increment_float(gradient_suite_counts, probe_id, float(value) / total_supervised_units)


def _batch_supervised_units_by_example(batch) -> list[int]:
    labels = getattr(batch, "labels", None)
    if labels is not None:
        shifted_labels = labels[:, 1:] if getattr(labels, "ndim", 0) >= 2 else labels
        mask = shifted_labels != -100
        return [int(value) for value in mask.sum(dim=1).detach().cpu().tolist()]
    return [len(str(text).split()) for text in getattr(batch, "texts", [])]


def _update_probe_effective_samples(
    counts: dict[str, Any],
    suite: ProbeSuite,
    number: int,
    text: str,
    words: list[str],
) -> None:
    suite_counts = _nested_probe_counts(counts, suite.id)
    for probe_id, value in _probe_exposure_counts_for_example(suite, number, text, words).items():
        _increment_count(suite_counts, probe_id, value)


def _probe_exposure_counts_for_example(
    suite: ProbeSuite,
    number: int,
    text: str,
    words: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if suite.id in {"eng_virtual_tokens", "eng_contextual_tokens"}:
        for word, role in zip(words, _virtual_token_roles(number, words)):
            if suite.id == "eng_contextual_tokens":
                for probe_id in _contextual_probe_ids_for_occurrence(word, role):
                    _increment_count(counts, probe_id)
            else:
                _increment_count(counts, f"{word}__{role}".replace(" ", "_"))
        if suite.id == "eng_contextual_tokens":
            for index, role in enumerate(_virtual_token_roles(number, words)):
                scale_id = _scale_id_for_role(role)
                if scale_id is not None and index + 1 < len(words):
                    _increment_count(counts, scale_id)
        return counts
    if suite.id == "eng_tokens":
        for word in words:
            _increment_count(counts, word.replace(" ", "_"))
        return counts
    if suite.id == "eng_factorized":
        known_probe_ids = {probe.id for probe in suite.probes}
        for probe in _factorized_operational_probes_for_example(NumberNamingExample(number=number, text=text)):
            if probe.role in known_probe_ids:
                _increment_count(counts, probe.role)
        return counts
    if suite.id == "eng_functional":
        known_probe_ids = {probe.id for probe in suite.probes}
        for probe in _functional_operational_probes_for_example(NumberNamingExample(number=number, text=text)):
            if probe.role in known_probe_ids:
                _increment_count(counts, probe.role)
        return counts
    return counts


def _shared_x_values(history: list[dict[str, Any]], config: PosetsProbingConfig, plot_config: PlotConfig) -> list[float]:
    steps = [float(record["step"]) for record in history]
    if plot_config.x_axis == "steps":
        return steps
    if plot_config.x_axis == "gradient_exposure":
        exact = _stored_total_gradient_exposure_axis(history)
        return exact if exact is not None else steps
    exact = _stored_total_sample_axis(history)
    if exact is not None:
        return exact
    return _expected_sample_axis(history, config, rate_per_example=1.0)


def _eval_loss_x_values(
    history: list[dict[str, Any]],
    config: PosetsProbingConfig,
    plot_config: PlotConfig,
    fallback_x_values: list[float],
) -> list[float]:
    if plot_config.x_axis == "gradient_exposure":
        exact = _stored_eval_loss_gradient_exposure_axis(history)
        return exact if exact is not None else fallback_x_values
    if plot_config.x_axis != "effective_samples":
        return fallback_x_values
    exact = _stored_eval_loss_effective_axis(history)
    if exact is not None:
        return exact
    return fallback_x_values


def _stored_eval_loss_gradient_exposure_axis(history: list[dict[str, Any]]) -> list[float] | None:
    values: list[float] = []
    for record in history:
        gradient_exposure = _record_gradient_exposure(record)
        metrics = gradient_exposure.get("metrics") if isinstance(gradient_exposure, dict) else None
        if not isinstance(metrics, dict) or metrics.get("eval_loss") is None:
            return None
        values.append(float(metrics["eval_loss"]))
    return values


def _stored_eval_loss_effective_axis(history: list[dict[str, Any]]) -> list[float] | None:
    values: list[float] = []
    for record in history:
        effective_samples = record.get("effective_samples")
        metrics = effective_samples.get("metrics") if isinstance(effective_samples, dict) else None
        if not isinstance(metrics, dict):
            return None
        if metrics.get("eval_loss") is not None:
            values.append(float(metrics["eval_loss"]))
            continue
        token_total = sum(
            float(value)
            for key, value in metrics.items()
            if str(key).startswith("token_loss_") and value is not None
        )
        if token_total <= 0.0:
            return None
        values.append(token_total)
    return values


def _stored_total_sample_axis(history: list[dict[str, Any]]) -> list[float] | None:
    values: list[float] = []
    for record in history:
        effective_samples = record.get("effective_samples")
        if not isinstance(effective_samples, dict) or effective_samples.get("total") is None:
            return None
        values.append(float(effective_samples["total"]))
    return values


def _stored_total_gradient_exposure_axis(history: list[dict[str, Any]]) -> list[float] | None:
    values: list[float] = []
    for record in history:
        gradient_exposure = _record_gradient_exposure(record)
        if not isinstance(gradient_exposure, dict) or gradient_exposure.get("total") is None:
            return None
        values.append(float(gradient_exposure["total"]))
    return values


def _record_gradient_exposure(record: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(record.get("gradient_exposure"), dict):
        return record["gradient_exposure"]
    effective_samples = record.get("effective_samples")
    if isinstance(effective_samples, dict) and isinstance(effective_samples.get("gradient_exposure"), dict):
        return effective_samples["gradient_exposure"]
    return None


def _expected_sample_axis(
    history: list[dict[str, Any]],
    config: PosetsProbingConfig,
    *,
    rate_per_example: float,
) -> list[float]:
    return [
        float(record["step"]) * float(config.batch_size) * float(rate_per_example)
        for record in history
    ]


def _x_axis_label(x_axis: str) -> str:
    if x_axis == "steps":
        return "step"
    if x_axis == "samples":
        return "training samples"
    if x_axis == "gradient_exposure":
        return "gradient exposure"
    return "effective training samples"


def _set_xscale(axis, x_scale: str) -> None:
    if hasattr(axis, "set_xscale"):
        axis.set_xscale(x_scale)


def _apply_x_axis_limits(
    axis,
    plot_config: PlotConfig,
    *,
    x_value_sets: list[list[float]],
    auto_right: float | None = None,
) -> None:
    finite_values = [
        float(value)
        for values in x_value_sets
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    left = float(plot_config.x_start) if plot_config.x_start is not None else None
    right = auto_right if plot_config.x_lim is None else float(plot_config.x_lim)
    if right is None and finite_values:
        max_x = max(finite_values)
        min_x = min(finite_values)
        if plot_config.x_scale == "log" and max_x > 0:
            right = max_x * 1.08
        elif max_x > min_x:
            right = max_x + (max_x - min_x) * 0.04
        else:
            right = max_x * 1.04 if max_x != 0.0 else 1.0
    if left is None and plot_config.x_scale == "log":
        positive_values = [value for value in finite_values if value > 0]
        if positive_values:
            left = min(positive_values)
    if left is not None and right is not None and right <= left:
        right = None
    if left is not None or right is not None:
        axis.set_xlim(left=left, right=right)


def _metric_effective_axes(
    history: list[dict[str, Any]],
    config: PosetsProbingConfig,
    plot_config: PlotConfig,
    task: NumberNamingTask | None,
) -> dict[str, list[float]]:
    if plot_config.x_axis == "gradient_exposure":
        exact = _stored_metric_gradient_exposure_axes(history)
        if exact:
            return exact
        if task is None or not task.train:
            return {}
        rates = _training_metric_gradient_exposure_rates(task.train)
        return {
            metric: [float(record["step"]) * rate for record in history]
            for metric, rate in rates.items()
            if rate > 0.0
        }
    if plot_config.x_axis != "effective_samples":
        return {}
    exact = _stored_metric_effective_axes(history)
    if exact:
        return exact
    if task is None or not task.train:
        return {}
    rates = _training_metric_rates(task.train)
    return {
        metric: _expected_sample_axis(history, config, rate_per_example=rate)
        for metric, rate in rates.items()
        if rate > 0.0
    }


def _training_metric_rates(train_examples: list[NumberNamingExample]) -> dict[str, float]:
    n_examples = max(len(train_examples), 1)
    rates: dict[str, float] = {}
    for example in train_examples:
        number = int(example.number)
        words = str(example.text).split()
        digit_length = len(str(number))
        for key in (
            f"digit_length_loss_{digit_length}",
            f"digit_length_token_accuracy_{digit_length}",
            f"digit_length_exact_accuracy_{digit_length}",
            f"digit_length_exact_accuracy_wo_eos_{digit_length}",
        ):
            rates[key] = rates.get(key, 0.0) + 1.0
        for role in _value_position_roles(number, words):
            key = f"value_position_loss_{role}"
            rates[key] = rates.get(key, 0.0) + 1.0
        for token in words:
            key = f"token_loss_{token}"
            rates[key] = rates.get(key, 0.0) + 1.0
    return {key: value / n_examples for key, value in rates.items()}


def _training_metric_gradient_exposure_rates(train_examples: list[NumberNamingExample]) -> dict[str, float]:
    total_units = _training_supervised_unit_count(train_examples)
    if total_units <= 0.0:
        return {}
    counts: dict[str, float] = {"eval_loss": total_units}
    for example in train_examples:
        number = int(example.number)
        words = str(example.text).split()
        example_units = len(words) + 1
        digit_length = len(str(number))
        for key in (
            f"digit_length_loss_{digit_length}",
            f"digit_length_token_accuracy_{digit_length}",
            f"digit_length_exact_accuracy_{digit_length}",
            f"digit_length_exact_accuracy_wo_eos_{digit_length}",
        ):
            counts[key] = counts.get(key, 0.0) + example_units
        for role in _value_position_roles(number, words):
            key = f"value_position_loss_{role}"
            counts[key] = counts.get(key, 0.0) + 1.0
        for token in words:
            key = f"token_loss_{token}"
            counts[key] = counts.get(key, 0.0) + 1.0
    return {key: value / total_units for key, value in counts.items()}


def _probe_effective_axes(
    history: list[dict[str, Any]],
    config: PosetsProbingConfig,
    plot_config: PlotConfig,
    probe_suite: ProbeSuite,
    train_examples: list[NumberNamingExample],
) -> dict[str, list[float]]:
    if plot_config.x_axis == "gradient_exposure":
        exact = _stored_probe_gradient_exposure_axes(history, probe_suite)
        if exact:
            return exact
        if not train_examples:
            return {}
        rates = _training_probe_gradient_exposure_rates(probe_suite, train_examples)
        return {
            probe_id: [float(record["step"]) * rate for record in history]
            for probe_id, rate in rates.items()
            if rate > 0.0
        }
    if plot_config.x_axis != "effective_samples":
        return {}
    exact = _stored_probe_effective_axes(history, probe_suite)
    if exact:
        return exact
    if not train_examples:
        return {}
    rates = _training_probe_rates(probe_suite, train_examples)
    return {
        probe_id: _expected_sample_axis(history, config, rate_per_example=rate)
        for probe_id, rate in rates.items()
        if rate > 0.0
    }


def _training_probe_rates(probe_suite: ProbeSuite, train_examples: list[NumberNamingExample]) -> dict[str, float]:
    n_examples = max(len(train_examples), 1)
    counts = {probe.id: 0.0 for probe in probe_suite.probes}
    for example in train_examples:
        words = str(example.text).split()
        if probe_suite.id in {"eng_virtual_tokens", "eng_contextual_tokens"}:
            roles = _virtual_token_roles(int(example.number), words)
            for word, role in zip(words, roles):
                if probe_suite.id == "eng_contextual_tokens":
                    for probe_id in _contextual_probe_ids_for_occurrence(word, role):
                        if probe_id in counts:
                            counts[probe_id] += 1.0
                else:
                    probe_id = f"{word}__{role}".replace(" ", "_")
                    if probe_id in counts:
                        counts[probe_id] += 1.0
            if probe_suite.id == "eng_contextual_tokens":
                for index, role in enumerate(roles):
                    probe_id = _scale_id_for_role(role)
                    if probe_id in counts and index + 1 < len(words):
                        counts[probe_id] += 1.0
        elif probe_suite.id == "eng_tokens":
            for word in words:
                token_id = word.replace(" ", "_")
                if token_id in counts:
                    counts[token_id] += 1.0
        elif probe_suite.id == "eng_factorized":
            for probe in _factorized_operational_probes_for_example(
                NumberNamingExample(number=int(example.number), text=str(example.text))
            ):
                if probe.role in counts:
                    counts[probe.role] += 1.0
        elif probe_suite.id == "eng_functional":
            for probe in _functional_operational_probes_for_example(
                NumberNamingExample(number=int(example.number), text=str(example.text))
            ):
                if probe.role in counts:
                    counts[probe.role] += 1.0
    if probe_suite.id not in counts:
        counts[probe_suite.id] = sum(counts.values()) / max(len(counts), 1)
    return {probe_id: count / n_examples for probe_id, count in counts.items()}


def _training_probe_gradient_exposure_rates(probe_suite: ProbeSuite, train_examples: list[NumberNamingExample]) -> dict[str, float]:
    total_units = _training_supervised_unit_count(train_examples)
    if total_units <= 0.0:
        return {}
    counts = {probe.id: 0.0 for probe in probe_suite.probes}
    for example in train_examples:
        words = str(example.text).split()
        for probe_id, value in _probe_exposure_counts_for_example(
            probe_suite,
            int(example.number),
            str(example.text),
            words,
        ).items():
            if probe_id in counts:
                counts[probe_id] += float(value)
    if probe_suite.id not in counts:
        counts[probe_suite.id] = sum(counts.values()) / max(len(counts), 1)
    return {probe_id: count / total_units for probe_id, count in counts.items()}


def _training_supervised_unit_count(train_examples: list[NumberNamingExample]) -> float:
    return float(sum(len(str(example.text).split()) + 1 for example in train_examples))


def _stored_metric_effective_axes(history: list[dict[str, Any]]) -> dict[str, list[float]]:
    metrics_by_record: list[dict[str, Any]] = []
    for record in history:
        effective_samples = record.get("effective_samples")
        if not isinstance(effective_samples, dict) or not isinstance(effective_samples.get("metrics"), dict):
            return {}
        metrics_by_record.append(effective_samples["metrics"])
    keys = sorted({key for metrics in metrics_by_record for key in metrics})
    return {
        key: [float(metrics.get(key, 0)) for metrics in metrics_by_record]
        for key in keys
    }


def _stored_metric_gradient_exposure_axes(history: list[dict[str, Any]]) -> dict[str, list[float]]:
    metrics_by_record: list[dict[str, Any]] = []
    for record in history:
        gradient_exposure = _record_gradient_exposure(record)
        if not isinstance(gradient_exposure, dict) or not isinstance(gradient_exposure.get("metrics"), dict):
            return {}
        metrics_by_record.append(gradient_exposure["metrics"])
    keys = sorted({key for metrics in metrics_by_record for key in metrics})
    return {
        key: [float(metrics.get(key, 0.0)) for metrics in metrics_by_record]
        for key in keys
    }


def _stored_probe_effective_axes(history: list[dict[str, Any]], probe_suite: ProbeSuite) -> dict[str, list[float]]:
    suite_counts_by_record: list[dict[str, Any]] = []
    for record in history:
        effective_samples = record.get("effective_samples")
        probes = effective_samples.get("probes") if isinstance(effective_samples, dict) else None
        suite_counts = probes.get(probe_suite.id) if isinstance(probes, dict) else None
        if not isinstance(suite_counts, dict):
            return {}
        suite_counts_by_record.append(suite_counts)
    keys = sorted({key for counts in suite_counts_by_record for key in counts})
    return {
        key: [float(counts.get(key, 0)) for counts in suite_counts_by_record]
        for key in keys
    }


def _stored_probe_gradient_exposure_axes(history: list[dict[str, Any]], probe_suite: ProbeSuite) -> dict[str, list[float]]:
    suite_counts_by_record: list[dict[str, Any]] = []
    for record in history:
        gradient_exposure = _record_gradient_exposure(record)
        probes = gradient_exposure.get("probes") if isinstance(gradient_exposure, dict) else None
        suite_counts = probes.get(probe_suite.id) if isinstance(probes, dict) else None
        if not isinstance(suite_counts, dict):
            return {}
        suite_counts_by_record.append(suite_counts)
    keys = sorted({key for counts in suite_counts_by_record for key in counts})
    return {
        key: [float(counts.get(key, 0.0)) for counts in suite_counts_by_record]
        for key in keys
    }


def _contextual_probe_metrics(
    model,
    task: NumberNamingTask,
    suite: ProbeSuite,
    *,
    device,
    metric_prefix: str | None,
) -> dict[str, float]:
    target_ids_by_probe = _probe_target_token_ids(task, suite, device=device)
    sums = {probe.id: 0.0 for probe in suite.probes}
    counts = {probe.id: 0 for probe in suite.probes}
    batch_size = max(1, int(task.config.eval_batch_size))
    for tagged_chunk in _chunks(suite.tagged_examples, batch_size):
        examples = [
            NumberNamingExample(number=example.number, text=example.text)
            for example in tagged_chunk
        ]
        batch = task.encode_examples(examples, device=device)
        logits = model(**batch.model_inputs)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch.labels[:, 1:]
        mask = shift_labels != -100
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        for row, tagged in enumerate(tagged_chunk):
            row_log_probs = log_probs[row][mask[row]]
            for tags, token_log_probs in zip(tagged.tags, row_log_probs):
                for tag in _iter_probe_tags(tags):
                    target_ids = target_ids_by_probe.get(tag)
                    if target_ids is None or target_ids.numel() == 0:
                        continue
                    loss_bits = -torch.logsumexp(token_log_probs[target_ids], dim=0) * (1.0 / math.log(2.0))
                    sums[tag] += float(loss_bits.detach().cpu().item())
                    counts[tag] += 1

    metrics = {}
    role_losses = []
    for probe in suite.probes:
        if counts[probe.id] == 0:
            continue
        loss_bits = sums[probe.id] / counts[probe.id]
        role_losses.append(loss_bits)
        metrics[_probe_metric_key(probe.id, metric_prefix)] = loss_bits
    if role_losses:
        metrics[_probe_metric_key(suite.id, metric_prefix)] = float(sum(role_losses) / len(role_losses))
    return metrics


def _probe_target_token_ids(task: NumberNamingTask, suite: ProbeSuite, *, device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for probe in suite.probes:
        tokens = tuple(probe.target_tokens) or tuple(sorted({example.target for example in probe.examples}))
        token_ids = [
            int(task.tokenizer.token_to_id[token])
            for token in tokens
            if token in task.tokenizer.token_to_id
        ]
        result[probe.id] = torch.tensor(token_ids, dtype=torch.long, device=device)
    return result


def _factorized_probe_metrics(
    model,
    task: NumberNamingTask,
    suite: ProbeSuite,
    *,
    device,
    metric_prefix: str | None,
) -> dict[str, float]:
    del metric_prefix
    if not suite.probes:
        return {}
    examples_by_key = {
        (int(example.number), english_number_name(int(example.number))): NumberNamingExample(
            number=int(example.number),
            text=english_number_name(int(example.number)),
        )
        for probe in suite.probes
        for example in probe.examples
    }
    log_probs_by_key: dict[tuple[int, str], torch.Tensor] = {}
    batch_size = max(1, int(task.config.eval_batch_size))
    with torch.no_grad():
        for chunk in _chunks(list(examples_by_key.values()), batch_size):
            batch = task.encode_examples(chunk, device=device)
            logits = model(**batch.model_inputs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            for row, (number, text) in enumerate(zip(batch.numbers, batch.texts)):
                log_probs_by_key[(int(number), str(text))] = log_probs[row].detach().cpu()

    metrics: dict[str, float] = {}
    values_by_quantum: dict[str, list[float]] = {}
    values_by_instance: dict[str, list[float]] = {}
    values_by_context: dict[tuple[str, str], list[float]] = {}
    values_by_subtype: dict[tuple[str, str], list[float]] = {}
    for probe in suite.probes:
        for example in probe.examples:
            text = english_number_name(int(example.number))
            log_probs = log_probs_by_key.get((int(example.number), text))
            if log_probs is None:
                continue
            target_id = _factorized_target_token_id(task, example.target)
            if target_id is None:
                continue
            position = _probe_prediction_position(task, example)
            if position >= log_probs.shape[0]:
                continue
            value = float((-log_probs[position, target_id] / math.log(2.0)).item())
            values_by_quantum.setdefault(probe.id, []).append(value)
            if example.instance_id:
                values_by_instance.setdefault(example.instance_id, []).append(value)
            if example.context:
                values_by_context.setdefault((probe.id, example.context), []).append(value)
            if example.subtype:
                values_by_subtype.setdefault((probe.id, example.subtype), []).append(value)
    quantum_values: list[float] = []
    for quantum_id, values in sorted(values_by_quantum.items()):
        metric_value = float(sum(values) / len(values))
        metrics[_factorized_schema_metric_key(quantum_id, suite_id=suite.id)] = metric_value
        quantum_values.append(metric_value)
    for instance_id, values in sorted(values_by_instance.items()):
        metrics[_factorized_instance_metric_key(instance_id, suite_id=suite.id)] = float(sum(values) / len(values))
    for (quantum_id, context), values in sorted(values_by_context.items()):
        metrics[_factorized_context_metric_key(quantum_id, context, suite_id=suite.id)] = float(sum(values) / len(values))
    for (quantum_id, subtype), values in sorted(values_by_subtype.items()):
        metrics[_factorized_subtype_metric_key(quantum_id, subtype, suite_id=suite.id)] = float(sum(values) / len(values))
    if quantum_values:
        metrics[_factorized_schema_metric_key(None, suite_id=suite.id)] = float(sum(quantum_values) / len(quantum_values))
    return metrics


def _factorized_target_token_id(task: NumberNamingTask, target: str) -> int | None:
    if target == "EOS":
        return int(task.tokenizer.eos_id)
    token_id = task.tokenizer.token_to_id.get(str(target))
    return None if token_id is None else int(token_id)


def _factorized_schema_metric_key(key: str | None, *, suite_id: str = "eng_factorized") -> str:
    if key is None:
        return f"probe_loss/{suite_id}_schema"
    return f"probe_loss/{suite_id}_schema/{key}"


def _factorized_instance_metric_key(key: str, *, suite_id: str = "eng_factorized") -> str:
    return f"probe_loss/{suite_id}_instance/{key}"


def _factorized_context_metric_key(quantum_id: str, context: str, *, suite_id: str = "eng_factorized") -> str:
    return f"probe_loss/{suite_id}_context/{quantum_id}/{context}"


def _factorized_subtype_metric_key(quantum_id: str, subtype: str, *, suite_id: str = "eng_factorized") -> str:
    return f"probe_loss/{suite_id}_subtype/{quantum_id}/{subtype}"


def _plot_metrics_panel(
    history: list[dict[str, Any]],
    output_path: str,
    *,
    config: PosetsProbingConfig,
    plot_config: PlotConfig | None = None,
    task: NumberNamingTask | None = None,
) -> None:
    plot_config = plot_config or PlotConfig()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes_flat = list(axes.flat)
    x_values = _shared_x_values(history, config, plot_config)
    eval_x_values = _eval_loss_x_values(history, config, plot_config, x_values)
    x_label = _x_axis_label(plot_config.x_axis)
    metric_axes = _metric_effective_axes(history, config, plot_config, task)

    loss_axis = axes_flat[0]
    loss_items = [
        ("train_loss", "Train"),
        ("eval_loss", "Eval"),
    ]
    for (metric, label), color in zip(loss_items, _metric_line_colors(len(loss_items))):
        values = [math.nan if record.get(metric) is None else float(record[metric]) for record in history]
        metric_x_values = eval_x_values if metric == "eval_loss" else x_values
        loss_axis.plot(metric_x_values, values, color=color, linewidth=2, label=label)
    loss_axis.set_title("Training and eval loss")
    loss_axis.set_xlabel(x_label)
    _set_xscale(loss_axis, plot_config.x_scale)
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend(fontsize=8)

    accuracy_axis = axes_flat[1]
    accuracy_items = [("token_accuracy", "Token")]
    if any("exact_sequence_accuracy" in record for record in history):
        accuracy_items.append(("exact_sequence_accuracy", "Exact sequence"))
    elif any("exact_accuracy" in record for record in history):
        accuracy_items.append(("exact_accuracy", "Exact sequence"))
    for (metric, label), color in zip(accuracy_items, _metric_line_colors(len(accuracy_items))):
        values = [math.nan if record.get(metric) is None else float(record[metric]) for record in history]
        accuracy_axis.plot(x_values, values, color=color, linewidth=2, label=label)
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel(x_label)
    _set_xscale(accuracy_axis, plot_config.x_scale)
    accuracy_axis.set_ylim(-0.02, 1.02)
    accuracy_axis.grid(True, alpha=0.25)
    accuracy_axis.legend(fontsize=8)

    digit_exact_axis = axes_flat[2]
    _plot_digit_length_series(
        digit_exact_axis,
        history,
        metric_axes,
        x_values,
        plot_config,
        prefix="digit_length_exact_accuracy_",
        title="Exact accuracy by digit length",
        ylabel="accuracy",
        ylim=(-0.02, 1.02),
    )

    digit_token_axis = axes_flat[3]
    if _has_metric_prefix(history, "digit_length_token_accuracy_"):
        _plot_digit_length_series(
            digit_token_axis,
            history,
            metric_axes,
            x_values,
            plot_config,
            prefix="digit_length_token_accuracy_",
            title="Token accuracy by digit length",
            ylabel="accuracy",
            ylim=(-0.02, 1.02),
        )
    else:
        _plot_digit_length_series(
            digit_token_axis,
            history,
            metric_axes,
            x_values,
            plot_config,
            prefix="digit_length_loss_",
            title="Digit-length loss",
            ylabel="loss (bits)",
        )
        _plot_eval_loss_reference(digit_token_axis, eval_x_values, history)

    _plot_final_histogram(
        axes_flat[4],
        history,
        prefix="wrong_token_count_",
        title="Most wrong target tokens",
        xlabel="wrong teacher-forced tokens",
    )
    _plot_digit_length_series(
        axes_flat[5],
        history,
        metric_axes,
        x_values,
        plot_config,
        prefix="digit_length_exact_accuracy_wo_eos_",
        title="Exact accuracy by digit length (wo/ EOS)",
        ylabel="accuracy",
        ylim=(-0.02, 1.02),
    )
    x_value_sets = [x_values, eval_x_values, *metric_axes.values()]
    for axis in [axes_flat[index] for index in (0, 1, 2, 3, 5)]:
        _apply_x_axis_limits(axis, plot_config, x_value_sets=x_value_sets)
    fig.suptitle(_number_naming_plot_title(config), fontsize=14)
    plt.tight_layout()
    _ensure_output_parent(output_path)
    plt.savefig(output_path, dpi=500)
    plt.close()


def _has_metric_prefix(history: list[dict[str, Any]], prefix: str) -> bool:
    return any(any(key.startswith(prefix) for key in record) for record in history)


def _plot_digit_length_series(
    axis,
    history: list[dict[str, Any]],
    metric_axes: dict[str, list[float]],
    x_values: list[float],
    plot_config: PlotConfig,
    *,
    prefix: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    metrics = sorted(
        {
            key
            for record in history
            for key in record
            if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
        },
        key=lambda key: int(key.removeprefix(prefix)),
    )
    for metric, color in zip(metrics, _metric_line_colors(len(metrics))):
        digit_length = int(metric.removeprefix(prefix))
        values = [math.nan if record.get(metric) is None else float(record[metric]) for record in history]
        axis.plot(metric_axes.get(metric, x_values), values, color=color, linewidth=1.8, label=f"{digit_length} digit")
    axis.set_title(title)
    axis.set_xlabel(_x_axis_label(plot_config.x_axis))
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    _set_xscale(axis, plot_config.x_scale)
    axis.grid(True, alpha=0.25)
    if metrics:
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "No digit metrics", ha="center", va="center", transform=axis.transAxes, fontsize=9)


def _plot_final_histogram(
    axis,
    history: list[dict[str, Any]],
    *,
    prefix: str,
    title: str,
    xlabel: str,
    limit: int = ERROR_HISTOGRAM_LIMIT,
) -> None:
    latest = history[-1] if history else {}
    items = [
        (key.removeprefix(prefix).replace("_", " "), float(value))
        for key, value in latest.items()
        if key.startswith(prefix) and isinstance(value, (int, float)) and float(value) > 0.0
    ]
    items = sorted(items, key=lambda item: (-item[1], item[0]))[: max(0, int(limit))]
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(True, axis="x", alpha=0.25)
    if not items:
        axis.text(0.5, 0.5, "No errors", ha="center", va="center", transform=axis.transAxes, fontsize=9)
        axis.set_yticks([])
        return
    labels = [label for label, _ in reversed(items)]
    values = [value for _, value in reversed(items)]
    axis.barh(labels, values, color=METRIC_BLUE, alpha=0.85)
    axis.tick_params(axis="y", labelsize=7)


def _number_naming_plot_title(config) -> str:
    if hasattr(config, "encoder_d_model"):
        encoder_depth = getattr(config, "encoder_layers", "cross_attention")
        return (
            "Number naming, QuantaNet "
            f"(encoder_depth={encoder_depth}, encoder_width={config.encoder_d_model}), "
            f"data_splits={config.data_splits}"
        )
    return (
        "Number naming, Transformer "
        f"(depth={config.n_layers}, width={config.d_model}), data_splits={config.data_splits}"
    )


def _value_position_metric_role(metric: str) -> str:
    return metric.removeprefix("value_position_loss_")


def _plot_eval_loss_reference(axis, x_values: list[float], history: list[dict[str, Any]]) -> None:
    values = [math.nan if record.get("eval_loss") is None else float(record["eval_loss"]) for record in history]
    axis.plot(
        x_values,
        values,
        color="#D62728",
        linestyle="--",
        linewidth=1.6,
        label="Eval loss",
    )


def _token_loss_metric_token(metric: str) -> str:
    return metric.removeprefix("token_loss_")


def _token_learning_order(history: list[dict[str, Any]]) -> list[str]:
    learned = []
    threshold = learned_threshold_bits()
    token_keys = sorted(
        {key for record in history for key in record if key.startswith("token_loss_")},
        key=_token_loss_metric_token,
    )
    for key in token_keys:
        for record in history:
            value = record.get(key)
            if value is not None and float(value) <= threshold:
                learned.append((int(record["step"]), _token_loss_metric_token(key)))
                break
    return [token for _, token in sorted(learned, key=lambda item: (item[0], item[1]))]


def _probe_learning_order(history: list[dict[str, Any]], suites: list[ProbeSuite]) -> list[str] | dict[str, list[str]]:
    if not suites:
        return _token_learning_order(history)
    single_suite = len(suites) == 1
    orders = {
        suite.id: _single_probe_suite_learning_order(
            history,
            suite,
            metric_prefix=None if single_suite else suite.id,
        )
        for suite in suites
    }
    if single_suite:
        return next(iter(orders.values()))
    return orders


def _single_probe_suite_learning_order(
    history: list[dict[str, Any]],
    suite: ProbeSuite,
    *,
    metric_prefix: str | None,
) -> list[str]:
    learned = []
    threshold = learned_threshold_bits()
    for probe in sorted(suite.probes, key=lambda item: item.id):
        metric = _probe_metric_key(probe.id, metric_prefix)
        for record in history:
            value = record.get(metric)
            if value is not None and float(value) <= threshold:
                learned.append((int(record["step"]), probe.id))
                break
    return [probe_id for _, probe_id in sorted(learned, key=lambda item: (item[0], item[1]))]


def _probe_suite_summary(suite: ProbeSuite | None) -> dict[str, Any] | None:
    if suite is None:
        return None
    return {
        "id": suite.id,
        "nodes": suite.nodes,
        "edges": suite.edges,
        "mermaid": suite.mermaid,
        "probes": [
            {
                "id": probe.id,
                "label": probe.label,
                "n_examples": len(probe.examples),
                "target_tokens": list(probe.target_tokens),
                "examples": [
                    {
                        "number": example.number,
                        "prefix": list(example.prefix),
                        "target": example.target,
                    }
                    for example in probe.examples[:10]
                ],
            }
            for probe in suite.probes
        ],
        "n_tagged_eval_examples": len(suite.tagged_examples),
        "n_factorized_occurrences": len(suite.factorized_occurrences),
        "factorized_trees": [
            {
                "number": tree.number,
                "root_node_id": tree.root_node_id,
                "n_nodes": len(tree.occurrences),
            }
            for tree in suite.factorized_trees[:10]
        ],
    }


def _probe_suites_summary(suites: list[ProbeSuite]) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not suites:
        return None
    summaries = [_probe_suite_summary(suite) for suite in suites]
    if len(summaries) == 1:
        return summaries[0]
    return [summary for summary in summaries if summary is not None]


def _metric_line_colors(count: int) -> list:
    if count <= 0:
        return []
    if count == 1:
        return [METRIC_BLUE]
    if count == 2:
        return [METRIC_BLUE, METRIC_YELLOW]
    cmap = LinearSegmentedColormap.from_list("metric_blue_yellow", [METRIC_BLUE, METRIC_YELLOW])
    return [cmap(index / (count - 1)) for index in range(count)]


def _maybe_data_parallel(model: torch.nn.Module, config: PosetsProbingConfig, device: torch.device) -> torch.nn.Module:
    if device.type == "cuda" and getattr(config, "device", None) is None and torch.cuda.device_count() > 1:
        logging.debug(
            "number_naming using single CUDA device; DataParallel disabled for small autoregressive runs "
            "to avoid scatter/gather overhead"
        )
    return model


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _even_sample(items: list, count: int) -> list:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def _chunks(items: list, size: int):
    for start in range(0, len(items), int(size)):
        yield items[start : start + int(size)]


def _plot_probe_losses(
    history: list[dict[str, Any]],
    probe_suite: ProbeSuite,
    output_path: str,
    *,
    config: PosetsProbingConfig | None = None,
    plot_config: PlotConfig | None = None,
    train_examples: list[NumberNamingExample] | None = None,
    metric_prefix: str | None = None,
) -> None:
    config = config or PosetsProbingConfig()
    plot_config = plot_config or PlotConfig()
    x_values = _shared_x_values(history, config, plot_config)
    eval_x_values = _eval_loss_x_values(history, config, plot_config, x_values)
    x_label = _x_axis_label(plot_config.x_axis)
    probe_axes = _probe_effective_axes(history, config, plot_config, probe_suite, train_examples or [])
    probe_series = _probe_plot_series(history, probe_suite, metric_prefix=metric_prefix)
    for series in probe_series:
        if series["id"] in probe_axes:
            series["x_values"] = probe_axes[series["id"]]
    threshold = learned_threshold_bits()
    learned_series = [
        series
        for series in probe_series
        if _curve_reaches_threshold(series["values"], threshold)
    ]
    probe_metrics = _probe_dynamics_metrics(
        probe_suite=probe_suite,
        probe_series=probe_series,
        fallback_x_values=x_values,
        unit=_metric_unit_label(plot_config.x_axis),
    )
    eval_series = {
        "label": "Eval loss",
        "values": [float(record["eval_loss"]) for record in history],
        "x_values": eval_x_values,
        "color": "#D62728",
        "linewidth": 2.2,
        "alpha": 1.0,
        "linestyle": "--",
    }
    probe_x_value_sets = [eval_x_values, x_values, *[series.get("x_values", x_values) for series in probe_series]]
    probe_auto_right = _probe_effective_auto_x_lim(probe_series, plot_config)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    panels = [
        (axes[0], _probe_metrics_title("All probe losses", probe_metrics), probe_series),
        (
            axes[1],
            _probe_metrics_title(f"Learned probe losses (threshold <= {threshold:g} bits)", probe_metrics),
            learned_series,
        ),
    ]
    for axis, title, series_list in panels:
        plotted_series = [eval_series] + series_list
        _plot_probe_series(axis, x_values, plotted_series)
        _apply_x_axis_limits(axis, plot_config, x_value_sets=probe_x_value_sets, auto_right=probe_auto_right)
        _label_probe_curves(axis, plotted_series, fallback_x_values=x_values)
        axis.set_title(title)
        axis.set_ylabel("loss (bits)")
        _set_xscale(axis, plot_config.x_scale)
        axis.grid(True, alpha=0.25)
    axes[1].set_xlabel(x_label)

    fig.tight_layout(rect=(0.0, 0.02, 0.98, 1.0))
    _ensure_output_parent(output_path)
    plt.savefig(output_path, dpi=500)
    plt.close()


def _probe_dynamics_metrics(
    *,
    probe_suite: ProbeSuite,
    probe_series: list[dict[str, Any]],
    fallback_x_values: list[float],
    unit: str,
) -> dict[str, Any] | None:
    loss_curves = {
        series["id"]: np.asarray(series["values"], dtype=float)
        for series in probe_series
        if len(series.get("values", [])) > 0
    }
    if not loss_curves:
        return None
    steps = {
        series["id"]: np.asarray(series.get("x_values", fallback_x_values), dtype=float)
        for series in probe_series
        if series["id"] in loss_curves
    }
    graph: dict[str, list[str]] = {}
    for prerequisite, dependent in probe_suite.edges:
        if prerequisite in loss_curves and dependent in loss_curves:
            graph.setdefault(dependent, []).append(prerequisite)
    unlearned, learned = load_metric_config()
    return compute_poset_dynamics_metrics(
        loss_curves=loss_curves,
        steps=steps,
        graph=graph,
        learned_threshold=learned,
        unlearned_threshold=unlearned,
        unit=unit,
    )


def _probe_metrics_title(label: str, metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return label
    pde = metrics["pde"]
    dte = metrics["dte"]
    return (
        f"{label} ({metrics['unit']}): "
        f"PDE={pde['PDE']:.2f} ({metrics['pde_censored_percent']:.1f}% deps excluded), "
        f"DTE={dte['mean_dte_area']:.2f} ({metrics['dte_excluded_percent']:.1f}% quanta excluded)"
    )


def _metric_unit_label(x_axis: str) -> str:
    if x_axis == "gradient_exposure":
        return "gradient exposure"
    if x_axis == "effective_samples":
        return "effective samples"
    if x_axis == "samples":
        return "training samples"
    return "steps"


def _probe_metric_key(probe_id: str, metric_prefix: str | None = None) -> str:
    if metric_prefix is None:
        return f"probe_loss/{probe_id}"
    return f"probe_loss/{metric_prefix}/{probe_id}"


def _is_structured_schema_suite(probe_suite: ProbeSuite) -> bool:
    return probe_suite.id in {"eng_factorized", "eng_functional"}


def _probe_plot_series(
    history: list[dict[str, Any]],
    probe_suite: ProbeSuite,
    *,
    metric_prefix: str | None = None,
) -> list[dict[str, Any]]:
    if probe_suite.id in {"eng_virtual_tokens", "eng_tokens", "eng_contextual_tokens"}:
        colors = _metric_line_colors(len(probe_suite.probes))
        return [
            {
                "id": probe.id,
                "label": probe.label,
                "values": [
                    math.nan
                    if record.get(_probe_metric_key(probe.id, metric_prefix)) is None
                    else float(record[_probe_metric_key(probe.id, metric_prefix)])
                    for record in history
                ],
                "color": color,
                "linewidth": 1.1,
                "alpha": 0.82,
                "linestyle": "-",
            }
            for probe, color in zip(probe_suite.probes, colors)
        ]
    if _is_structured_schema_suite(probe_suite):
        probes = list(probe_suite.probes)
        colors = _metric_line_colors(len(probes))
        return [
            {
                "id": probe.id,
                "label": probe.label,
                "values": [
                    math.nan
                    if record.get(_factorized_schema_metric_key(probe.id, suite_id=probe_suite.id)) is None
                    else float(record[_factorized_schema_metric_key(probe.id, suite_id=probe_suite.id)])
                    for record in history
                ],
                "color": color,
                "linewidth": 1.1,
                "alpha": 0.82,
                "linestyle": "-",
            }
            for probe, color in zip(probes, colors)
        ]
    return [
        {
            "id": probe_suite.id,
            "label": probe_suite.id,
            "values": [
                _record_mean(record, "probe_loss/", exclude_keys={_probe_metric_key(probe_suite.id, metric_prefix)})
                for record in history
            ],
            "color": METRIC_BLUE,
            "linewidth": 2.0,
            "alpha": 1.0,
            "linestyle": "-",
        }
    ]


def _probe_effective_auto_x_lim(series_list: list[dict[str, Any]], plot_config: PlotConfig) -> float | None:
    if plot_config.x_axis not in {"effective_samples", "gradient_exposure"} or plot_config.x_lim is not None:
        return None
    endpoints = []
    all_endpoints = []
    for series in series_list:
        x_values = list(series.get("x_values") or [])
        values = list(series.get("values") or [])
        finite_x = [
            float(x)
            for x, y in zip(x_values, values)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if not finite_x:
            continue
        endpoint = finite_x[-1]
        all_endpoints.append(endpoint)
        if _is_broad_probe_reference(series):
            continue
        endpoints.append(endpoint)
    if len(endpoints) < 4:
        endpoints = all_endpoints
    if not endpoints:
        return None
    endpoints = sorted(endpoints)
    index = min(len(endpoints) - 1, max(0, int(round(0.9 * (len(endpoints) - 1)))))
    right = endpoints[index]
    left = float(plot_config.x_start) if plot_config.x_start is not None else 0.0
    if right <= left:
        return None
    return right * 1.18 if plot_config.x_scale == "log" else right + (right - left) * 0.18


def _is_broad_probe_reference(series: dict[str, Any]) -> bool:
    series_id = str(series.get("id") or "")
    label = str(series.get("label") or "")
    return series_id.startswith("LAYER_") or label.startswith("Layer ")


def _plot_probe_series(axis, x_values: list[float], series_list: list[dict[str, Any]]) -> list[tuple[Any, str]]:
    plotted = []
    for series in series_list:
        handle = axis.plot(
            series.get("x_values", x_values),
            series["values"],
            color=series["color"],
            linewidth=series["linewidth"],
            alpha=series["alpha"],
            linestyle=series["linestyle"],
            label=series["label"],
        )[0]
        plotted.append((handle, series["label"]))
    return plotted


def _write_factorized_artifacts(
    history: list[dict[str, Any]],
    suite: ProbeSuite,
    *,
    save_dir: str,
    config: PosetsProbingConfig,
    plot_config: PlotConfig,
) -> None:
    if not history:
        return
    suite_dir = _suite_output_dir(save_dir, suite.id)
    os.makedirs(suite_dir, exist_ok=True)
    x_axes = _probe_effective_axes(history, config, plot_config, suite, [])
    learned_steps = _factorized_learned_steps(
        history,
        threshold=float(config.factorized_learned_threshold),
        stability_window=int(config.factorized_stability_window),
        x_axes=x_axes,
        suite_id=suite.id,
    )
    edge_rows = _factorized_edge_violations(suite, learned_steps, tolerance=0.0)
    _write_json(
        os.path.join(suite_dir, "schema_poset.json"),
        {
            "nodes": suite.nodes,
            "edges": [{"parent": parent, "child": child} for parent, child in suite.edges],
        },
    )
    _write_factorized_diagnostic_instances(os.path.join(suite_dir, "diagnostic_instances.jsonl"), suite)
    _write_factorized_loss_csv(
        os.path.join(suite_dir, "probe_loss_schema.csv"),
        history,
        prefix=f"probe_loss/{suite.id}_schema/",
        aggregate_key=_factorized_schema_metric_key(None, suite_id=suite.id),
    )
    _write_factorized_loss_csv(
        os.path.join(suite_dir, "probe_loss_context.csv"),
        history,
        prefix=f"probe_loss/{suite.id}_context/",
    )
    _write_factorized_edge_csv(os.path.join(suite_dir, "edge_violations_schema.csv"), edge_rows)
    _write_json(
        os.path.join(suite_dir, "learned_steps_schema.json"),
        dict(sorted(learned_steps.items())),
    )

    for tree in _select_factorized_trees(suite.factorized_trees):
        tree_path = os.path.join(suite_dir, f"factorized_tree_{tree.number}.txt")
        with open(tree_path, "w") as handle:
            handle.write(_factorized_tree_printout(tree, history, learned_steps, suite_id=suite.id))
        _plot_factorized_tree_losses(
            history,
            tree,
            os.path.join(suite_dir, f"factorized_tree_{tree.number}.png"),
            config=config,
            plot_config=plot_config,
            suite_id=suite.id,
        )


def _factorized_learned_steps(
    history: list[dict[str, Any]],
    *,
    threshold: float = 0.1,
    stability_window: int = 2,
    x_axes: dict[str, list[float]] | None = None,
    suite_id: str = "eng_factorized",
) -> dict[str, float | None]:
    learned: dict[str, float | None] = {}
    node_ids: set[str] = set()
    for record in history:
        for key in record:
            prefix = f"probe_loss/{suite_id}_schema/"
            if not key.startswith(prefix):
                continue
            node_id = key.removeprefix(prefix)
            if node_id:
                node_ids.add(node_id)
    for node_id in node_ids:
        learned[node_id] = _first_stable_learned_step(
            history,
            _factorized_schema_metric_key(node_id, suite_id=suite_id),
            threshold=threshold,
            stability_window=stability_window,
            x_values=None if x_axes is None else x_axes.get(node_id),
        )
    return learned


def _first_stable_learned_step(
    history: list[dict[str, Any]],
    metric: str,
    *,
    threshold: float,
    stability_window: int,
    x_values: list[float] | None = None,
) -> float | None:
    for start in range(len(history)):
        window = history[start : start + int(stability_window)]
        if len(window) < int(stability_window):
            return None
        values = [record.get(metric) for record in window]
        if all(value is not None and float(value) <= threshold for value in values):
            if x_values is not None and start < len(x_values):
                return float(x_values[start])
            return float(history[start]["step"])
    return None


def _factorized_edge_violations(
    suite: ProbeSuite,
    learned_steps: dict[str, float | None],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    for parent, child in sorted(set(suite.edges)):
        parent_step = learned_steps.get(parent)
        child_step = learned_steps.get(child)
        violation = (
            child_step is not None
            and parent_step is not None
            and float(child_step) + float(tolerance) < float(parent_step)
        )
        rows.append(
            {
                "parent_node": parent,
                "parent_learned_step": parent_step,
                "child_node": child,
                "child_learned_step": child_step,
                "violation": violation,
            }
        )
    return rows


def _write_factorized_edge_csv(path: str, rows: list[dict[str, Any]]) -> None:
    _ensure_output_parent(path)
    with open(path, "w", newline="") as handle:
        fieldnames = ["parent_node", "child_node", "parent_learned_step", "child_learned_step", "violation"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_factorized_diagnostic_instances(path: str, suite: ProbeSuite) -> None:
    _ensure_output_parent(path)
    with open(path, "w") as handle:
        for probe in suite.probes:
            for example in probe.examples:
                handle.write(
                    json.dumps(
                        {
                            "quantum_id": probe.id,
                            "instance_id": example.instance_id,
                            "number": example.number,
                            "prefix": list(example.prefix),
                            "target": example.target,
                            "context": example.context,
                            "subtype": example.subtype,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def _write_factorized_loss_csv(
    path: str,
    history: list[dict[str, Any]],
    *,
    prefix: str,
    aggregate_key: str | None = None,
) -> None:
    metric_keys = sorted({key for record in history for key in record if key.startswith(prefix)})
    if aggregate_key is not None:
        metric_keys = [aggregate_key] + [key for key in metric_keys if key != aggregate_key]
    _ensure_output_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "metric", "loss"])
        for record in history:
            step = record.get("step")
            for key in metric_keys:
                if key not in record or record[key] is None:
                    continue
                metric = key if aggregate_key is not None and key == aggregate_key else key.removeprefix(prefix)
                writer.writerow([step, metric, float(record[key])])


def _select_factorized_trees(trees: list[FactorizedExampleTree]) -> list[FactorizedExampleTree]:
    preferred = [225485, 225000, 225005, 485, 485000, 100000, 100005, 405000, 405005]
    by_number = {tree.number: tree for tree in trees}
    selected = [by_number[number] for number in preferred if number in by_number]
    if selected:
        return selected
    return _even_sample(trees, min(3, len(trees)))


def _factorized_tree_printout(
    tree: FactorizedExampleTree,
    history: list[dict[str, Any]],
    learned_steps: dict[str, float | None],
    *,
    suite_id: str = "eng_factorized",
) -> str:
    latest = history[-1] if history else {}
    by_id = {occurrence.node_id: occurrence for occurrence in tree.occurrences}
    lines = [f"{tree.number} -> {tree.text}"]

    def visit(node_id: str, depth: int) -> None:
        occurrence = by_id[node_id]
        tokens = str(occurrence.text).split()
        token_span = " ".join(tokens[position] for position in occurrence.token_positions if position < len(tokens))
        quantum_id = occurrence.quantum_id or occurrence.node_type
        loss = latest.get(_factorized_schema_metric_key(quantum_id, suite_id=suite_id))
        learned = learned_steps.get(quantum_id)
        lines.append(
            f"{'  ' * depth}{occurrence.node_id} quantum={quantum_id} role={occurrence.role} span={token_span!r} "
            f"loss={_format_optional_float(loss)} learned_x={learned}"
        )
        for child in occurrence.children:
            if child in by_id:
                visit(child, depth + 1)

    visit(tree.root_node_id, 0)
    return "\n".join(lines) + "\n"


def _plot_factorized_tree_losses(
    history: list[dict[str, Any]],
    tree: FactorizedExampleTree,
    output_path: str,
    *,
    config: PosetsProbingConfig,
    plot_config: PlotConfig,
    suite_id: str = "eng_factorized",
) -> None:
    x_values = _shared_x_values(history, config, plot_config)
    fig, axis = plt.subplots(figsize=(14, 8))
    occurrences = sorted(tree.occurrences, key=lambda item: (_factorized_depth(item.node_id, tree), item.node_id))
    colors = _metric_line_colors(len(occurrences))
    series_list = []
    for occurrence, color in zip(occurrences, colors):
        quantum_id = occurrence.quantum_id or occurrence.node_type
        values = [
            math.nan
            if record.get(_factorized_schema_metric_key(quantum_id, suite_id=suite_id)) is None
            else float(record[_factorized_schema_metric_key(quantum_id, suite_id=suite_id)])
            for record in history
        ]
        series = {
            "id": occurrence.node_id,
            "label": f"{occurrence.node_id} [{quantum_id}]",
            "values": values,
            "x_values": x_values,
            "color": color,
            "linewidth": 1.2,
            "alpha": 0.85,
            "linestyle": "-",
        }
        series_list.append(series)
    _plot_probe_series(axis, x_values, series_list)
    _label_probe_curves(axis, series_list, max_labels=36)
    axis.set_title(f"Factorized node losses for {tree.number}")
    axis.set_xlabel(_x_axis_label(plot_config.x_axis))
    axis.set_ylabel("loss (bits)")
    _set_xscale(axis, plot_config.x_scale)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    _ensure_output_parent(output_path)
    plt.savefig(output_path, dpi=400)
    plt.close()


def _factorized_depth(node_id: str, tree: FactorizedExampleTree) -> int:
    by_id = {occurrence.node_id: occurrence for occurrence in tree.occurrences}

    def visit(current: str) -> int:
        children = [child for child in by_id.get(current, FactorizedNodeOccurrence(0, "", "", "", None, "", (), ())).children if child in by_id]
        if not children:
            return 0
        return 1 + max(visit(child) for child in children)

    return visit(node_id)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "NA"


def _label_probe_curves(
    axis,
    series_list: list[dict[str, Any]],
    *,
    fallback_x_values: list[float] | None = None,
    max_labels: int = 28,
) -> None:
    candidates = []
    x_min, x_max = axis.get_xlim() if hasattr(axis, "get_xlim") else (math.nan, math.nan)
    for index, series in enumerate(series_list):
        values = list(series.get("values") or [])
        x_values = list(series.get("x_values") or fallback_x_values or [])
        if not x_values:
            continue
        finite = [
            (float(x), float(y))
            for x, y in zip(x_values, values)
            if math.isfinite(float(y))
        ]
        if not finite:
            continue
        visible = [
            (x, y)
            for x, y in finite
            if (
                not math.isfinite(float(x_min))
                or not math.isfinite(float(x_max))
                or (float(x_min) <= x <= float(x_max))
            )
        ]
        if not visible:
            continue
        x, y = visible[-1]
        candidates.append((index, x, y, series))
    if not candidates:
        return
    if hasattr(axis, "margins"):
        axis.margins(x=0.08)
    if len(candidates) > max_labels:
        eval_items = [item for item in candidates if item[3].get("label") == "Eval loss"]
        remaining = [item for item in candidates if item[3].get("label") != "Eval loss"]
        keep = max_labels - len(eval_items)
        sampled = _select_curve_label_candidates(remaining, keep, x_min=x_min, x_max=x_max) if keep > 0 else []
        candidates = eval_items + sampled
    candidates.sort(key=lambda item: item[2])
    y_min, y_max = axis.get_ylim()
    if not math.isfinite(y_min) or not math.isfinite(y_max) or y_min == y_max:
        return
    min_gap = (y_max - y_min) * 0.035
    placed: list[float] = []
    for _, x, y, series in candidates:
        label_y = min(max(y, y_min), y_max)
        if placed:
            label_y = max(label_y, placed[-1] + min_gap)
        if label_y > y_max:
            label_y = y_max
        placed.append(label_y)
        label_on_left = (
            math.isfinite(float(x_min))
            and math.isfinite(float(x_max))
            and float(x_max) > float(x_min)
            and (float(x) - float(x_min)) / (float(x_max) - float(x_min)) > 0.88
        )
        axis.annotate(
            str(series["label"]),
            xy=(x, label_y),
            xytext=(-6 if label_on_left else 6, 0),
            textcoords="offset points",
            ha="right" if label_on_left else "left",
            va="center",
            fontsize=6,
            color=series["color"],
            clip_on=False,
        )


def _select_curve_label_candidates(
    candidates: list[tuple[int, float, float, dict[str, Any]]],
    keep: int,
    *,
    x_min: float,
    x_max: float,
) -> list[tuple[int, float, float, dict[str, Any]]]:
    if keep <= 0:
        return []
    if len(candidates) <= keep:
        return candidates
    if not (math.isfinite(float(x_min)) and math.isfinite(float(x_max)) and float(x_max) > float(x_min)):
        return _even_sample(sorted(candidates, key=lambda item: item[2]), keep)
    x_span = float(x_max) - float(x_min)
    early = [item for item in candidates if (float(item[1]) - float(x_min)) / x_span < 0.18]
    later = [item for item in candidates if item not in early]
    later_keep = min(len(later), max(keep // 2, keep - min(8, len(early))))
    early_keep = keep - later_keep
    selected = []
    if later_keep > 0:
        selected.extend(_even_sample(sorted(later, key=lambda item: item[2]), later_keep))
    if early_keep > 0:
        selected.extend(_even_sample(sorted(early, key=lambda item: item[2]), early_keep))
    return selected


def _patching_compatibility_audit(
    model: DecoderTransformerLM,
    task: NumberNamingTask,
    *,
    device,
    output_dir: str,
) -> dict[str, Any]:
    model.eval()
    results: dict[str, Any] = {}
    with torch.no_grad():
        for suite in task.probe_suites:
            suite_dir = _suite_output_dir(output_dir, suite.id)
            os.makedirs(suite_dir, exist_ok=True)
            all_probes = [probe for probe in suite.probes if probe.examples]
            selected_probes = _select_patching_compatibility_probes(
                suite,
                all_probes,
                max_classes=getattr(task.config, "patching_compatibility_max_classes", 48),
            )
            examples_per_class = int(getattr(task.config, "patching_compatibility_examples_per_class", 8))
            labels = [probe.id for probe in selected_probes]
            matrix: list[list[float]] = []
            counts: list[list[int]] = []
            hidden_cache: dict[tuple[int, tuple[str, ...], str], torch.Tensor] = {}
            baseline_cache: dict[tuple[int, tuple[str, ...], str], float] = {}
            target_examples_by_probe = {
                probe.id: _even_sample(probe.examples, min(examples_per_class, len(probe.examples)))
                for probe in selected_probes
            }
            for target_examples in target_examples_by_probe.values():
                for target_example in target_examples:
                    baseline_cache.setdefault(
                        _probe_example_cache_key(target_example),
                        _probe_example_loss(model, task, target_example, device=device),
                    )
            for source_probe in selected_probes:
                source_examples = _even_sample(source_probe.examples, min(examples_per_class, len(source_probe.examples)))
                row = []
                count_row = []
                source_states = [
                    hidden_cache.setdefault(
                        _probe_example_cache_key(source_example),
                        _probe_example_hidden(model, task, source_example, device=device),
                    )
                    for source_example in source_examples
                ]
                for target_probe in selected_probes:
                    target_examples = target_examples_by_probe[target_probe.id]
                    deltas = []
                    for source_state in source_states:
                        for target_example in target_examples:
                            baseline = baseline_cache[_probe_example_cache_key(target_example)]
                            patched = _probe_example_loss(
                                model,
                                task,
                                target_example,
                                device=device,
                                patched_hidden=source_state,
                            )
                            deltas.append(float(patched - baseline))
                    row.append(float(sum(deltas) / len(deltas)) if deltas else math.nan)
                    count_row.append(len(deltas))
                matrix.append(row)
                counts.append(count_row)
            suite_result = {
                "activation": "final_hidden",
                "class_selection": {
                    "total_classes": len(all_probes),
                    "selected_classes": len(selected_probes),
                    "max_classes": getattr(task.config, "patching_compatibility_max_classes", 48),
                    "examples_per_class": examples_per_class,
                    "selection": "full" if len(selected_probes) == len(all_probes) else "representative_even_sample",
                },
                "source_classes": labels,
                "target_classes": labels,
                "loss_delta_bits": matrix,
                "pair_counts": counts,
            }
            results[suite.id] = suite_result
            _write_json(os.path.join(suite_dir, "patching_compatibility.json"), suite_result)
            _plot_matrix(
                matrix,
                labels,
                labels,
                os.path.join(suite_dir, "patching_compatibility.png"),
                title=f"Patching compatibility: {suite.id}",
                color_label="loss delta (bits)",
            )
    return results


def _select_patching_compatibility_probes(
    suite: ProbeSuite,
    probes: list[Any],
    *,
    max_classes: int | None,
) -> list[Any]:
    probes = sorted(probes, key=lambda probe: probe.id)
    if max_classes is None or len(probes) <= int(max_classes):
        return probes
    if suite.id != "eng_factorized":
        return _even_sample(probes, int(max_classes))

    by_family: dict[str, list[Any]] = {}
    for probe in probes:
        by_family.setdefault(_factorized_probe_family(probe.id), []).append(probe)
    family_order = [
        "UNIT_LEX",
        "TEEN_LEX",
        "TEN",
        "ONE_HUNDRED",
        "ONE_THOUSAND",
        "TENS",
        "HUNDREDS",
        "THOUSANDS",
        "TEEN_THOUSANDS",
        "TEN_THOUSANDS",
        "TENS_THOUSANDS",
        "HUNDRED_THOUSANDS",
        "EOS_DECISION",
        "EMIT",
    ]
    selected: list[Any] = []
    families = [family for family in family_order if family in by_family]
    base_quota = max(1, int(max_classes) // max(len(families), 1))
    for family in families:
        remaining = int(max_classes) - len(selected)
        if remaining <= 0:
            break
        selected.extend(_even_sample(by_family[family], min(base_quota, remaining, len(by_family[family]))))
    if len(selected) < int(max_classes):
        selected_ids = {probe.id for probe in selected}
        remaining_probes = [probe for probe in probes if probe.id not in selected_ids]
        selected.extend(_even_sample(remaining_probes, int(max_classes) - len(selected)))
    return sorted(selected[: int(max_classes)], key=lambda probe: probe.id)


def _factorized_probe_family(probe_id: str) -> str:
    return probe_id.split("(", 1)[0]


def _probe_example_cache_key(example) -> tuple[int, tuple[str, ...], str]:
    return (int(example.number), tuple(example.prefix), str(example.target))


def _probe_example_hidden(model: DecoderTransformerLM, task: NumberNamingTask, example, *, device) -> torch.Tensor:
    batch = task.encode_examples([NumberNamingExample(number=example.number, text=english_number_name(example.number))], device=device)
    position = _probe_prediction_position(task, example)
    hidden = model.hidden_states(**batch.model_inputs)
    return hidden[0, position].detach().clone()


def _probe_example_loss(
    model: DecoderTransformerLM,
    task: NumberNamingTask,
    example,
    *,
    device,
    patched_hidden: torch.Tensor | None = None,
) -> float:
    batch = task.encode_examples([NumberNamingExample(number=example.number, text=english_number_name(example.number))], device=device)
    position = _probe_prediction_position(task, example)
    hidden = model.hidden_states(**batch.model_inputs)
    if patched_hidden is not None:
        hidden = hidden.clone()
        hidden[0, position] = patched_hidden.to(hidden.device)
    logits = model.head(hidden)
    target = batch.labels[0, position + 1].view(1)
    loss = torch.nn.functional.cross_entropy(logits[0, position].view(1, -1), target)
    return nats_to_bits(float(loss.detach().cpu().item()))


def _probe_prediction_position(task: NumberNamingTask, example) -> int:
    sep_position = 1 + len(task.input_digit_string(example.number))
    return sep_position + len(example.prefix)


def _held_out_transfer_audit(
    model: DecoderTransformerLM,
    task: NumberNamingTask,
    *,
    held_out_token_roles: list[dict[str, str]],
    device,
    output_dir: str,
) -> dict[str, Any]:
    requested = {(str(item["token"]).lower(), str(item["role"])) for item in held_out_token_roles}
    os.makedirs(output_dir, exist_ok=True)
    if not requested:
        result = {"held_out_token_roles": [], "groups": {}}
        _write_json(os.path.join(output_dir, "held_out_transfer.json"), result)
        return result
    held_out = _token_role_metrics(model, task, requested, match_requested=True, device=device)
    seen = _token_role_metrics(model, task, requested, match_requested=False, device=device)
    result = {
        "held_out_token_roles": [{"token": token, "role": role} for token, role in sorted(requested)],
        "groups": {
            "held_out": held_out,
            "seen_other_token_roles": seen,
        },
    }
    _write_json(os.path.join(output_dir, "held_out_transfer.json"), result)
    _plot_held_out_transfer(result, os.path.join(output_dir, "held_out_transfer.png"))
    return result


def _token_role_metrics(
    model: DecoderTransformerLM,
    task: NumberNamingTask,
    requested: set[tuple[str, str]],
    *,
    match_requested: bool,
    device,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for examples in _chunks(task.eval_examples, max(1, int(task.config.eval_batch_size))):
            batch = task.encode_examples(examples, device=device)
            logits = model(**batch.model_inputs)
            shift_logits = logits[:, :-1, :]
            shift_labels = batch.labels[:, 1:]
            mask = shift_labels != -100
            losses = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(shift_labels)
            losses = losses * (1.0 / math.log(2.0))
            predictions = shift_logits.argmax(dim=-1)
            for row, (number, text) in enumerate(zip(batch.numbers, batch.texts)):
                roles = _virtual_token_roles(int(number), text.split())
                row_losses = losses[row][mask[row]]
                row_predictions = predictions[row][mask[row]]
                row_labels = shift_labels[row][mask[row]]
                for index, (word, role) in enumerate(zip(text.split(), roles)):
                    is_requested = (word.lower(), role) in requested
                    if is_requested != match_requested or index >= len(row_losses):
                        continue
                    loss_sum += float(row_losses[index].detach().cpu().item())
                    correct += int(row_predictions[index].item() == row_labels[index].item())
                    count += 1
    return {
        "token_count": count,
        "loss_bits": loss_sum / max(count, 1),
        "token_accuracy": correct / max(count, 1),
    }


def _plot_matrix(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    output_path: str,
    *,
    title: str,
    color_label: str,
) -> None:
    if not matrix:
        return
    fig, axis = plt.subplots(figsize=(max(6, len(col_labels) * 0.35), max(5, len(row_labels) * 0.3)))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    axis.set_title(title)
    axis.set_xlabel("target class")
    axis.set_ylabel("source class")
    axis.set_xticks(range(len(col_labels)))
    axis.set_xticklabels(col_labels, rotation=90, fontsize=6)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=6)
    fig.colorbar(image, ax=axis, label=color_label)
    fig.tight_layout()
    _ensure_output_parent(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_held_out_transfer(result: dict[str, Any], output_path: str) -> None:
    groups = result.get("groups") or {}
    labels = list(groups)
    if not labels:
        return
    losses = [float(groups[label].get("loss_bits", math.nan)) for label in labels]
    accuracies = [float(groups[label].get("token_accuracy", math.nan)) for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(labels, losses, color=METRIC_BLUE)
    axes[0].set_title("Held-out transfer loss")
    axes[0].set_ylabel("loss (bits)")
    axes[1].bar(labels, accuracies, color=METRIC_YELLOW)
    axes[1].set_title("Held-out transfer accuracy")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _ensure_output_parent(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _write_json(path: str, payload: Any) -> None:
    _ensure_output_parent(path)
    with open(path, "w") as handle:
        json.dump(_jsonable(payload), handle, indent=4)


def _ensure_output_parent(output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _curve_reaches_threshold(values: list[float], threshold: float) -> bool:
    return any(not math.isnan(value) and value <= threshold for value in values)


def _record_mean(record: dict[str, Any], prefix: str, *, exclude_keys: set[str] | None = None) -> float:
    exclude_keys = exclude_keys or set()
    values = [
        float(value)
        for key, value in record.items()
        if key.startswith(prefix) and key not in exclude_keys and value is not None
    ]
    if not values:
        return math.nan
    return float(sum(values) / len(values))
