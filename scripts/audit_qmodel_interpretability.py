"""Audit Q-model quantum gates and sparse writers against exact task semantics.

The audit deliberately separates three claims:

1. semantic selectivity: a feature hypothesis chosen on one held-out calibration
   panel must retain precision/recall on a disjoint held-out test panel;
2. concept coverage: each pre-specified NumberNaming concept should be recoverable
   by one feature, or by a very small union of features;
3. causal use: sparse writer outputs must materially contribute to Q-model
   behavior rather than merely co-activate beside the direct attention residual.

The generated JSON is the canonical machine-readable artifact.  A compact
Markdown report and per-writer activation examples are emitted alongside it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from quanta.config import load_experiment_config
from quanta.experiments.number_naming.factorized_ar import build_token_labels
from quanta.experiments.number_naming.functional_ar import (
    build_functional_token_labels,
)
from quanta.experiments.number_naming.task import _value_position_roles
from quanta.experiments.quanta_discovery.qgraph import (
    QuantumGraphEdge,
    _per_quantum_attention_components,
    forward_quanta,
)
from quanta.experiments.quanta_discovery.qmodel import (
    CheapCausalAttentionQuantumLayer,
    QuantumReadout,
)
from quanta.experiments.quanta_discovery.trajectory import build_model_and_task
from scripts.train_quanta_model import (
    _evaluate_teacher_forced,
    _event_coordinates,
    _existence_table,
    _load_priority,
)


ALGORITHMIC_PREFIXES = (
    "functional_",
    "factorized_",
    "value_role=",
)
INPUT_PREFIXES = ("digit_length=", "input_digit[")
OUTPUT_PREFIXES = ("target=", "target_position=", "previous_target=")


@dataclass(frozen=True)
class FeatureRef:
    kind: str
    layer: int
    index: int
    owner_quantum: int

    @property
    def name(self) -> str:
        if self.kind == "quantum":
            return f"L{self.layer}.Q{self.index}"
        return f"L{self.layer}.Q{self.owner_quantum}.W{self.index}"


@dataclass(frozen=True)
class TracePanel:
    events: list[dict[str, Any]]
    quantum_scores: tuple[np.ndarray, ...]
    writer_scores: tuple[np.ndarray, ...]
    behavior: dict[str, Any]
    path_energy: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class AttentionReaderPanel:
    """Held-out reader measurements aligned with sparse writer events."""

    events: list[dict[str, Any]]
    quantum_entropy: tuple[np.ndarray, ...]
    quantum_effective_support: tuple[np.ndarray, ...]
    quantum_top1_mass: tuple[np.ndarray, ...]
    quantum_top2_mass: tuple[np.ndarray, ...]
    quantum_top_role: tuple[list[list[str]], ...]
    quantum_top_label: tuple[list[list[str]], ...]
    quantum_top2_roles: tuple[list[list[tuple[str, ...]]], ...]
    quantum_top2_labels: tuple[list[list[tuple[str, ...]]], ...]
    quantum_active: tuple[np.ndarray, ...]
    writer_owner_active: tuple[np.ndarray, ...]
    writer_scores: tuple[np.ndarray, ...]
    writer_local_scores: tuple[np.ndarray, ...]
    writer_context_scores: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a fitted Q-model against exact NumberNaming semantics and "
            "sparse-output counterfactuals."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Source discovery artifact.")
    parser.add_argument("q_model_dir", type=Path, help="Completed fitted-Q artifact.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--priority-dir",
        type=Path,
        help="Optional factorization used by the fitted Q-model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--calibration-examples",
        type=int,
        default=0,
        help="0 uses the complete stratified first half of the eval split.",
    )
    parser.add_argument(
        "--test-examples",
        type=int,
        default=0,
        help="0 uses the complete stratified second half of the eval split.",
    )
    parser.add_argument("--counterfactual-examples", type=int, default=1024)
    parser.add_argument("--causal-writers", type=int, default=16)
    parser.add_argument("--min-concept-positives", type=int, default=20)
    parser.add_argument("--interpretability-f1", type=float, default=0.60)
    parser.add_argument(
        "--compound-top-concepts",
        type=int,
        default=16,
        help=(
            "Calibration-selected atomic concepts considered for two-literal "
            "AND/OR descriptions per feature and concept family."
        ),
    )
    parser.add_argument(
        "--compound-description-penalty",
        type=float,
        default=0.02,
        help="Calibration F1 penalty for each description literal beyond the first.",
    )
    parser.add_argument(
        "--redundancy-causal-examples",
        type=int,
        default=64,
        help=(
            "Held-out test examples used for per-writer target-logit deletion "
            "signatures; 0 disables this bounded counterfactual diagnostic."
        ),
    )
    parser.add_argument(
        "--redundancy-cluster-cosine",
        type=float,
        default=0.90,
        help="Cosine threshold for within-quantum causal-logit signature clusters.",
    )
    parser.add_argument(
        "--attention-reader-examples",
        type=int,
        default=0,
        help=(
            "Held-out test examples for frozen attention-reader metrics and "
            "interventions; 0 disables this diagnostic."
        ),
    )
    parser.add_argument(
        "--attention-top-k-values",
        type=str,
        default="1,2,4",
        help=(
            "Comma-separated top-k values evaluated without retraining in the "
            "reader diagnostic."
        ),
    )
    parser.add_argument(
        "--attention-reader-min-events",
        type=int,
        default=20,
        help=(
            "Minimum calibration events required to retain a frozen top-two "
            "reader semantic route."
        ),
    )
    parser.add_argument("--top-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17_029)
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _stratified_eval_split(
    examples: Sequence[Any], *, seed: int
) -> tuple[list[Any], list[Any]]:
    """Make deterministic disjoint halves within every input digit length."""

    by_digits: dict[int, list[Any]] = {}
    for example in examples:
        by_digits.setdefault(len(str(int(example.number))), []).append(example)
    calibration: list[Any] = []
    test: list[Any] = []
    for digits, bucket in sorted(by_digits.items()):
        shuffled = list(bucket)
        random.Random(int(seed) + 997 * digits).shuffle(shuffled)
        midpoint = len(shuffled) // 2
        calibration.extend(shuffled[:midpoint])
        test.extend(shuffled[midpoint:])
    calibration.sort(key=lambda example: int(example.number))
    test.sort(key=lambda example: int(example.number))
    return calibration, test


def _balanced_limit(examples: Sequence[Any], count: int, *, seed: int) -> list[Any]:
    if int(count) <= 0 or int(count) >= len(examples):
        return list(examples)
    by_digits: dict[int, list[Any]] = {}
    for example in examples:
        by_digits.setdefault(len(str(int(example.number))), []).append(example)
    for digits, bucket in by_digits.items():
        random.Random(int(seed) + 1597 * digits).shuffle(bucket)
    result: list[Any] = []
    digits = sorted(by_digits)
    cursor = {value: 0 for value in digits}
    while len(result) < int(count):
        progressed = False
        for value in digits:
            index = cursor[value]
            if index < len(by_digits[value]):
                result.append(by_digits[value][index])
                cursor[value] += 1
                progressed = True
                if len(result) == int(count):
                    break
        if not progressed:
            break
    result.sort(key=lambda example: int(example.number))
    return result


def _event_descriptions(examples: Sequence[Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    place_names = (
        "units",
        "tens",
        "hundreds",
        "thousands",
        "ten_thousands",
        "hundred_thousands",
    )
    for example in examples:
        number = int(example.number)
        words = str(example.text).split()
        functional = build_functional_token_labels(number)
        factorized = build_token_labels(number)
        roles = _value_position_roles(number, words) + ["eos"]
        if not (len(functional) == len(factorized) == len(roles)):
            raise RuntimeError(f"semantic labels do not align for {number}")
        static: set[str] = {f"digit_length={len(str(number))}"}
        digits_by_place: dict[str, float] = {
            f"input_digit[{place_name}]": float("nan")
            for place_name in place_names
        }
        place_values: dict[str, float] = {
            f"place_value[{place_name}]": float("nan")
            for place_name in place_names
        }
        for place, digit in enumerate(reversed(str(number))):
            static.add(f"input_digit[{place_names[place]}]={digit}")
            digits_by_place[f"input_digit[{place_names[place]}]"] = float(digit)
            place_values[f"place_value[{place_names[place]}]"] = float(
                int(digit) * (10**place)
            )
        for index, (functional_label, factorized_label, role) in enumerate(
            zip(functional, factorized, roles)
        ):
            concepts = set(static)
            concepts.add(f"target={functional_label.target}")
            concepts.add(f"target_position={index}")
            previous = "START" if index == 0 else functional[index - 1].target
            concepts.add(f"previous_target={previous}")
            concepts.add(f"value_role={role}")
            concepts.add(f"functional_owner={functional_label.owner}")
            concepts.add(f"functional_subtype={functional_label.subtype}")
            concepts.add(f"functional_context={functional_label.context}")
            concepts.update(
                f"functional_active={node}" for node in functional_label.active
            )
            concepts.add(
                f"factorized_owner={factorized_label.owner.family.value}"
            )
            concepts.update(
                f"factorized_active={instance.family.value}"
                for instance in factorized_label.active_closure
            )
            events.append(
                {
                    "number": number,
                    "token_index": index,
                    "prefix": list(functional_label.prefix),
                    "target": functional_label.target,
                    "functional_owner": functional_label.owner,
                    "functional_context": functional_label.context,
                    "functional_subtype": functional_label.subtype,
                    "concepts": sorted(concepts),
                    "continuous": {
                        "input_number": float(number),
                        "digit_length": float(len(str(number))),
                        "target_position": float(index),
                        "prefix_length": float(index),
                        "remaining_output_tokens": float(len(functional) - index),
                        **digits_by_place,
                        **place_values,
                    },
                }
            )
    return events


def _behavior_record(
    *,
    labels: torch.Tensor,
    source_logits: torch.Tensor,
    quantum_logits: torch.Tensor,
) -> dict[str, float | int]:
    source_predictions = source_logits.argmax(dim=-1)
    quantum_predictions = quantum_logits.argmax(dim=-1)
    count = int(len(labels))
    return {
        "prediction_events": count,
        "source_correct": int((source_predictions == labels).sum()),
        "quantum_correct": int((quantum_predictions == labels).sum()),
        "source_quantum_agreement": int(
            (source_predictions == quantum_predictions).sum()
        ),
        "source_nll_nats": float(
            F.cross_entropy(source_logits, labels, reduction="sum")
        ),
        "quantum_nll_nats": float(
            F.cross_entropy(quantum_logits, labels, reduction="sum")
        ),
        "source_to_quantum_kl_nats": float(
            F.kl_div(
                F.log_softmax(quantum_logits, dim=-1),
                F.softmax(source_logits, dim=-1),
                reduction="sum",
            )
        ),
    }


def _finish_behavior(total: dict[str, float | int]) -> dict[str, float | int]:
    count = max(int(total["prediction_events"]), 1)
    return {
        "prediction_events": int(total["prediction_events"]),
        "source_accuracy": int(total["source_correct"]) / count,
        "quantum_accuracy": int(total["quantum_correct"]) / count,
        "source_quantum_argmax_agreement": int(
            total["source_quantum_agreement"]
        )
        / count,
        "source_nll_bits": float(total["source_nll_nats"]) / count / math.log(2),
        "quantum_nll_bits": float(total["quantum_nll_nats"])
        / count
        / math.log(2),
        "source_to_quantum_kl_bits": float(total["source_to_quantum_kl_nats"])
        / count
        / math.log(2),
    }


@torch.no_grad()
def _collect_panel(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
) -> TracePanel:
    quantum_parts: list[list[np.ndarray]] = [[] for _ in modules]
    writer_parts: list[list[np.ndarray]] = [[] for _ in modules]
    energy = [
        {
            "events": 0.0,
            "writer_squared_norm": 0.0,
            "attention_squared_norm": 0.0,
            "combined_squared_norm": 0.0,
            "writer_attention_dot": 0.0,
        }
        for _ in modules
    ]
    behavior_total: dict[str, float | int] = {
        "prediction_events": 0,
        "source_correct": 0,
        "quantum_correct": 0,
        "source_quantum_agreement": 0,
        "source_nll_nats": 0.0,
        "quantum_nll_nats": 0.0,
        "source_to_quantum_kl_nats": 0.0,
    }
    for start in range(0, len(examples), int(batch_size)):
        selected = list(examples[start : start + int(batch_size)])
        batch = task.encode_examples(selected, device=device)
        rows, positions = _event_coordinates(batch)
        source_trace = source.residual_trace(**batch.model_inputs)
        expanded_existence = tuple(
            value.expand(len(selected), -1) for value in existences
        )
        trace = forward_quanta(
            source,
            readout,
            modules,
            batch.input_ids,
            batch.attention_mask,
            expanded_existence,
            edges,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        labels = batch.labels[:, 1:][rows, positions]
        record = _behavior_record(
            labels=labels,
            source_logits=source_trace.logits[rows, positions],
            quantum_logits=trace.logits[rows, positions],
        )
        for key, value in record.items():
            behavior_total[key] += value
        for layer, module in enumerate(modules):
            effective = trace.effective_gates[layer][rows, positions]
            existence = expanded_existence[layer][rows]
            quantum_execution = effective * existence
            quantum_probability = torch.sigmoid(
                trace.gate_logits[layer][rows, positions]
            )
            quantum_parts[layer].append(
                (quantum_probability * quantum_execution).float().cpu().numpy()
            )

            owner_execution = quantum_execution.index_select(
                1, module.writer_owner
            )
            preactivation = trace.writer_gate_logits[layer][rows, positions]
            writer_score = (
                trace.writer_gates[layer][rows, positions]
                * preactivation
                * owner_execution
            )
            writer_parts[layer].append(writer_score.float().cpu().numpy())

            decoder = F.normalize(module.output_weight, dim=-1)
            writer_update = torch.einsum("ew,wd->ed", writer_score, decoder)
            if module.attention_output_weight is None:
                attention_update = torch.zeros_like(writer_update)
            else:
                context = trace.attention_contexts[layer][rows, positions]
                attention_by_quantum = torch.einsum(
                    "eqv,qvd->eqd", context, module.attention_output_weight
                )
                attention_update = (
                    quantum_execution[..., None] * attention_by_quantum
                ).sum(dim=1)
            combined = writer_update + attention_update
            energy[layer]["events"] += float(len(rows))
            energy[layer]["writer_squared_norm"] += float(
                writer_update.square().sum()
            )
            energy[layer]["attention_squared_norm"] += float(
                attention_update.square().sum()
            )
            energy[layer]["combined_squared_norm"] += float(combined.square().sum())
            energy[layer]["writer_attention_dot"] += float(
                (writer_update * attention_update).sum()
            )
    descriptions = _event_descriptions(examples)
    behavior = _finish_behavior(behavior_total)
    if len(descriptions) != int(behavior["prediction_events"]):
        raise RuntimeError("trace events and semantic labels do not align")
    path_energy = []
    for layer in energy:
        writer_energy = layer["writer_squared_norm"]
        attention_energy = layer["attention_squared_norm"]
        dot = layer["writer_attention_dot"]
        denominator = math.sqrt(max(writer_energy * attention_energy, 1.0e-24))
        path_energy.append(
            {
                **layer,
                "writer_to_attention_energy_ratio": writer_energy
                / max(attention_energy, 1.0e-24),
                "aggregate_writer_attention_cosine": dot / denominator,
            }
        )
    return TracePanel(
        events=descriptions,
        quantum_scores=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in quantum_parts
        ),
        writer_scores=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in writer_parts
        ),
        behavior=behavior,
        path_energy=tuple(path_energy),
    )


def _attention_key_labels(batch: Any, task: Any) -> tuple[list[list[str]], list[list[str]]]:
    """Describe each causal key by source role and role/value label."""

    place_names = (
        "units",
        "tens",
        "hundreds",
        "thousands",
        "ten_thousands",
        "hundred_thousands",
    )
    roles: list[list[str]] = []
    labels: list[list[str]] = []
    length = int(batch.input_ids.shape[1])
    for row, number in enumerate(batch.numbers):
        digits = str(int(number))
        role_row = ["padding"] * length
        label_row = ["padding"] * length
        role_row[0] = "bos"
        label_row[0] = "bos"
        for offset, digit in enumerate(digits):
            place = place_names[len(digits) - offset - 1]
            role = f"input_digit[{place}]"
            role_row[offset + 1] = role
            label_row[offset + 1] = f"{role}={digit}"
        separator = len(digits) + 1
        role_row[separator] = "separator"
        label_row[separator] = "separator"
        valid = int(batch.attention_mask[row].sum())
        for position in range(separator + 1, valid):
            token = task.tokenizer.id_to_token[int(batch.input_ids[row, position])]
            role_row[position] = "output_prefix"
            label_row[position] = f"output_prefix={token}"
        roles.append(role_row)
        labels.append(label_row)
    return roles, labels


@torch.no_grad()
def _collect_attention_reader_panel(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
) -> AttentionReaderPanel:
    """Collect frozen reader statistics aligned with teacher-forced events."""

    entropy_parts: list[list[np.ndarray]] = [[] for _ in modules]
    support_parts: list[list[np.ndarray]] = [[] for _ in modules]
    top1_parts: list[list[np.ndarray]] = [[] for _ in modules]
    top2_parts: list[list[np.ndarray]] = [[] for _ in modules]
    top_role_parts: list[list[list[str]]] = [[] for _ in modules]
    top_label_parts: list[list[list[str]]] = [[] for _ in modules]
    top2_role_parts: list[list[list[tuple[str, ...]]]] = [[] for _ in modules]
    top2_label_parts: list[list[list[tuple[str, ...]]]] = [[] for _ in modules]
    quantum_active_parts: list[list[np.ndarray]] = [[] for _ in modules]
    owner_active_parts: list[list[np.ndarray]] = [[] for _ in modules]
    writer_parts: list[list[np.ndarray]] = [[] for _ in modules]
    local_parts: list[list[np.ndarray]] = [[] for _ in modules]
    context_parts: list[list[np.ndarray]] = [[] for _ in modules]
    event_parts: list[dict[str, Any]] = []
    for start in range(0, len(examples), int(batch_size)):
        selected = list(examples[start : start + int(batch_size)])
        batch = task.encode_examples(selected, device=device)
        rows, positions = _event_coordinates(batch)
        expanded_existence = tuple(
            value.expand(len(selected), -1) for value in existences
        )
        trace = forward_quanta(
            source,
            readout,
            modules,
            batch.input_ids,
            batch.attention_mask,
            expanded_existence,
            edges,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        key_roles, key_labels = _attention_key_labels(batch, task)
        event_parts.extend(_event_descriptions(selected))
        for layer, module in enumerate(modules):
            normalized = module.norm(trace.routed_inputs[layer])
            _, weights, _ = _per_quantum_attention_components(
                module, normalized, batch.attention_mask
            )
            event_weights = weights[rows, :, positions, :]
            entropy = -(
                event_weights.clamp_min(1.0e-24)
                * event_weights.clamp_min(1.0e-24).log()
            ).sum(dim=-1)
            top = event_weights.topk(min(2, event_weights.shape[-1]), dim=-1).values
            entropy_parts[layer].append(entropy.float().cpu().numpy())
            support_parts[layer].append(entropy.exp().float().cpu().numpy())
            top1_parts[layer].append(top[..., 0].float().cpu().numpy())
            top2_parts[layer].append(top.sum(dim=-1).float().cpu().numpy())
            top_indices = (
                event_weights.topk(
                    min(2, event_weights.shape[-1]), dim=-1
                ).indices.cpu().tolist()
            )
            top_role_parts[layer].extend(
                [
                    [key_roles[int(row)][int(indices[0])] for indices in event]
                    for row, event in zip(rows.tolist(), top_indices)
                ]
            )
            top_label_parts[layer].extend(
                [
                    [key_labels[int(row)][int(indices[0])] for indices in event]
                    for row, event in zip(rows.tolist(), top_indices)
                ]
            )
            top2_role_parts[layer].extend(
                [
                    [
                        tuple(key_roles[int(row)][int(index)] for index in indices)
                        for indices in event
                    ]
                    for row, event in zip(rows.tolist(), top_indices)
                ]
            )
            top2_label_parts[layer].extend(
                [
                    [
                        tuple(key_labels[int(row)][int(index)] for index in indices)
                        for indices in event
                    ]
                    for row, event in zip(rows.tolist(), top_indices)
                ]
            )
            execution = (
                trace.effective_gates[layer][rows, positions]
                * expanded_existence[layer][rows]
            )
            owner_execution = execution.index_select(1, module.writer_owner)
            quantum_active_parts[layer].append(
                execution.float().cpu().numpy()
            )
            owner_active_parts[layer].append(
                owner_execution.float().cpu().numpy()
            )
            writer_score = (
                trace.writer_gates[layer][rows, positions]
                * trace.writer_gate_logits[layer][rows, positions]
                * owner_execution
            )
            writer_parts[layer].append(writer_score.float().cpu().numpy())
            local_parts[layer].append(
                trace.writer_local_scores[layer][rows, positions]
                .float()
                .cpu()
                .numpy()
            )
            context_parts[layer].append(
                trace.writer_context_scores[layer][rows, positions]
                .float()
                .cpu()
                .numpy()
            )
    return AttentionReaderPanel(
        events=event_parts,
        quantum_entropy=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in entropy_parts
        ),
        quantum_effective_support=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in support_parts
        ),
        quantum_top1_mass=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in top1_parts
        ),
        quantum_top2_mass=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in top2_parts
        ),
        quantum_top_role=tuple(top_role_parts),
        quantum_top_label=tuple(top_label_parts),
        quantum_top2_roles=tuple(top2_role_parts),
        quantum_top2_labels=tuple(top2_label_parts),
        quantum_active=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in quantum_active_parts
        ),
        writer_owner_active=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in owner_active_parts
        ),
        writer_scores=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in writer_parts
        ),
        writer_local_scores=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in local_parts
        ),
        writer_context_scores=tuple(
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for parts in context_parts
        ),
    )


def _mode(values: Sequence[Any]) -> Any | None:
    if not values:
        return None
    counts: dict[Any, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return min(counts, key=lambda value: (-counts[value], repr(value)))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    denominator = float(np.asarray(weights, dtype=np.float64).sum())
    if denominator <= 1.0e-24:
        return None
    return float(
        np.dot(
            np.asarray(values, dtype=np.float64).reshape(-1),
            np.asarray(weights, dtype=np.float64).reshape(-1),
        )
        / denominator
    )


def _rich_writer_selected(record: dict[str, Any], threshold: float) -> bool:
    hypothesis = record["conditional_on_owner"]["compound_hypotheses"]["all"]
    return bool(
        hypothesis is not None and hypothesis["calibration"]["f1"] >= threshold
    )


def _signature_text(signature: tuple[str, ...] | None) -> str | None:
    if signature is None:
        return None
    return " -> ".join(signature)


def _weighted_mode(
    values: Sequence[tuple[str, ...]], weights: np.ndarray
) -> tuple[str, ...] | None:
    if not values:
        return None
    totals: dict[tuple[str, ...], float] = defaultdict(float)
    for value, weight in zip(values, np.asarray(weights, dtype=np.float64)):
        totals[value] += float(weight)
    return min(totals, key=lambda value: (-totals[value], repr(value)))


def _signature_hypothesis(
    calibration_signatures: Sequence[tuple[str, ...]],
    test_signatures: Sequence[tuple[str, ...]],
    calibration_active: np.ndarray,
    test_active: np.ndarray,
    calibration_condition: np.ndarray,
    test_condition: np.ndarray,
    *,
    minimum_events: int,
) -> dict[str, Any] | None:
    """Fit one joint reader signature on calibration and freeze it on test."""

    calibration_condition = np.asarray(calibration_condition, dtype=bool)
    test_condition = np.asarray(test_condition, dtype=bool)
    calibration_active = np.asarray(calibration_active, dtype=bool)
    test_active = np.asarray(test_active, dtype=bool)
    candidate_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for signature, allowed in zip(calibration_signatures, calibration_condition):
        if allowed:
            candidate_counts[signature] += 1
    candidates = [
        signature
        for signature, count in candidate_counts.items()
        if count >= int(minimum_events)
    ]
    if not candidates:
        return None

    calibration_target = calibration_active[calibration_condition]
    if (
        int(calibration_target.sum()) < int(minimum_events)
        or calibration_target.all()
    ):
        return None
    scored: list[tuple[float, int, tuple[str, ...], dict[str, Any]]] = []
    for signature in candidates:
        prediction = np.asarray(
            [value == signature for value in calibration_signatures], dtype=bool
        )[calibration_condition]
        metrics = binary_metrics(prediction, calibration_target)
        scored.append((metrics["f1"], candidate_counts[signature], signature, metrics))
    _, _, selected, calibration_metrics = min(
        scored, key=lambda item: (-item[0], -item[1], repr(item[2]))
    )
    test_prediction = np.asarray(
        [value == selected for value in test_signatures], dtype=bool
    )[test_condition]
    return {
        "signature": list(selected),
        "calibration_signature_events": candidate_counts[selected],
        "calibration": calibration_metrics,
        "test": binary_metrics(test_prediction, test_active[test_condition]),
    }


def _conditional_reader_routes(
    calibration_events: Sequence[dict[str, Any]],
    test_events: Sequence[dict[str, Any]],
    calibration_signatures: Sequence[tuple[str, ...]],
    test_signatures: Sequence[tuple[str, ...]],
    calibration_active: np.ndarray,
    test_active: np.ndarray,
    calibration_energy: np.ndarray,
    test_energy: np.ndarray,
    *,
    minimum_events: int,
) -> list[dict[str, Any]]:
    """Freeze top-two reads within an output state, rather than pooling roles."""

    def state(event: dict[str, Any]) -> tuple[str, str, int]:
        return (
            str(event["functional_owner"]),
            str(event["functional_subtype"]),
            int(event["token_index"]),
        )

    calibration_active = np.asarray(calibration_active, dtype=bool)
    test_active = np.asarray(test_active, dtype=bool)
    calibration_energy = np.asarray(calibration_energy, dtype=np.float64)
    test_energy = np.asarray(test_energy, dtype=np.float64)
    calibration_states = [state(event) for event in calibration_events]
    test_states = [state(event) for event in test_events]
    routes: list[dict[str, Any]] = []
    for route_state in sorted(set(calibration_states), key=repr):
        calibration_mask = np.asarray(
            [value == route_state for value in calibration_states], dtype=bool
        ) & calibration_active
        if int(calibration_mask.sum()) < int(minimum_events):
            continue
        role = _weighted_mode(
            [
                signature
                for signature, selected in zip(calibration_signatures, calibration_mask)
                if selected
            ],
            calibration_energy[calibration_mask],
        )
        if role is None:
            continue
        test_mask = np.asarray(
            [value == route_state for value in test_states], dtype=bool
        ) & test_active
        test_values = [
            signature
            for signature, selected in zip(test_signatures, test_mask)
            if selected
        ]
        matches = np.asarray([value == role for value in test_values], dtype=float)
        routes.append(
            {
                "functional_owner": route_state[0],
                "functional_subtype": route_state[1],
                "target_position": route_state[2],
                "calibration_active_events": int(calibration_mask.sum()),
                "calibration_active_energy": float(calibration_energy[calibration_mask].sum()),
                "frozen_ordered_top2": list(role),
                "test_active_events": int(test_mask.sum()),
                "test_ordered_top2_stability": (
                    None if not len(matches) else float(matches.mean())
                ),
                "test_energy_weighted_ordered_top2_stability": _weighted_mean(
                    matches, test_energy[test_mask]
                ),
            }
        )
    return sorted(
        routes,
        key=lambda row: (-row["calibration_active_energy"], -row["calibration_active_events"]),
    )


def _attention_reader_summary(
    calibration: AttentionReaderPanel,
    test: AttentionReaderPanel,
    writer_records: Sequence[dict[str, Any]],
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    *,
    threshold: float,
    minimum_events: int = 20,
) -> dict[str, Any]:
    """Freeze joint top-two reader semantics and score them on test."""

    quantum_rows: list[dict[str, Any]] = []
    for layer, module in enumerate(modules):
        for quantum in range(module.quantum_count):
            calibration_roles = [
                row[quantum] for row in calibration.quantum_top2_roles[layer]
            ]
            calibration_labels = [
                row[quantum] for row in calibration.quantum_top2_labels[layer]
            ]
            test_roles = [row[quantum] for row in test.quantum_top2_roles[layer]]
            test_labels = [row[quantum] for row in test.quantum_top2_labels[layer]]
            calibration_active = calibration.quantum_active[layer][:, quantum] > 0
            test_active = test.quantum_active[layer][:, quantum] > 0
            role = _weighted_mode(
                [value for value, active in zip(calibration_roles, calibration_active) if active],
                calibration.quantum_active[layer][calibration_active, quantum],
            )
            label = _weighted_mode(
                [value for value, active in zip(calibration_labels, calibration_active) if active],
                calibration.quantum_active[layer][calibration_active, quantum],
            )
            test_role_values = [
                value for value, active in zip(test_roles, test_active) if active
            ]
            test_label_values = [
                value for value, active in zip(test_labels, test_active) if active
            ]
            quantum_rows.append(
                {
                    "name": f"L{layer}.Q{quantum}",
                    "layer": layer,
                    "quantum": quantum,
                    "calibration_ordered_top2_roles": _signature_text(role),
                    "calibration_ordered_top2_labels": _signature_text(label),
                    "test_ordered_top2_role_stability": None
                    if role is None or not test_role_values
                    else float(np.mean([value == role for value in test_role_values])),
                    "test_ordered_top2_label_stability": None
                    if label is None or not test_label_values
                    else float(np.mean([value == label for value in test_label_values])),
                    "conditional_role_routes": _conditional_reader_routes(
                        calibration.events,
                        test.events,
                        calibration_roles,
                        test_roles,
                        calibration_active,
                        test_active,
                        calibration.quantum_active[layer][:, quantum],
                        test.quantum_active[layer][:, quantum],
                        minimum_events=minimum_events,
                    ),
                    "test_attention_entropy_nats": float(
                        test.quantum_entropy[layer][:, quantum].mean()
                    ),
                    "test_effective_support": float(
                        test.quantum_effective_support[layer][:, quantum].mean()
                    ),
                    "test_top1_mass": float(
                        test.quantum_top1_mass[layer][:, quantum].mean()
                    ),
                    "test_top2_mass": float(
                        test.quantum_top2_mass[layer][:, quantum].mean()
                    ),
                }
            )

    records_by_location = {
        (int(record["layer"]), int(record["index"])): record
        for record in writer_records
    }
    writer_rows: list[dict[str, Any]] = []
    for layer, module in enumerate(modules):
        for writer in range(module.writer_count):
            record = records_by_location[(layer, writer)]
            owner = int(module.writer_owner[writer])
            calibration_active = calibration.writer_scores[layer][:, writer] > 0
            test_active = test.writer_scores[layer][:, writer] > 0
            calibration_roles = [row[owner] for row in calibration.quantum_top2_roles[layer]]
            calibration_labels = [row[owner] for row in calibration.quantum_top2_labels[layer]]
            test_roles = [row[owner] for row in test.quantum_top2_roles[layer]]
            test_labels = [row[owner] for row in test.quantum_top2_labels[layer]]
            calibration_owner_active = calibration.writer_owner_active[layer][:, writer] > 0
            test_owner_active = test.writer_owner_active[layer][:, writer] > 0
            role = _weighted_mode(
                [value for value, active in zip(calibration_roles, calibration_active) if active],
                calibration.writer_scores[layer][calibration_active, writer] ** 2,
            )
            label = _weighted_mode(
                [value for value, active in zip(calibration_labels, calibration_active) if active],
                calibration.writer_scores[layer][calibration_active, writer] ** 2,
            )
            test_role_values = [
                value for value, active in zip(test_roles, test_active) if active
            ]
            test_label_values = [
                value for value, active in zip(test_labels, test_active) if active
            ]
            energy = test.writer_scores[layer][:, writer].astype(np.float64) ** 2
            local = test.writer_local_scores[layer][:, writer]
            context = test.writer_context_scores[layer][:, writer]
            context_fraction = np.abs(context) / (
                np.abs(local) + np.abs(context) + 1.0e-24
            )
            writer_rows.append(
                {
                    "name": record["name"],
                    "layer": layer,
                    "writer": writer,
                    "owner_quantum": owner,
                    "rich_selected_on_calibration": _rich_writer_selected(
                        record, threshold
                    ),
                    "calibration_ordered_top2_roles": _signature_text(role),
                    "calibration_ordered_top2_labels": _signature_text(label),
                    "test_ordered_top2_role_stability": None
                    if role is None or not test_role_values
                    else float(np.mean([value == role for value in test_role_values])),
                    "test_ordered_top2_label_stability": None
                    if label is None or not test_label_values
                    else float(np.mean([value == label for value in test_label_values])),
                    "ordered_top2_role_activation_rule": _signature_hypothesis(
                        calibration_roles,
                        test_roles,
                        calibration_active,
                        test_active,
                        calibration_owner_active,
                        test_owner_active,
                        minimum_events=minimum_events,
                    ),
                    "ordered_top2_label_activation_rule": _signature_hypothesis(
                        calibration_labels,
                        test_labels,
                        calibration_active,
                        test_active,
                        calibration_owner_active,
                        test_owner_active,
                        minimum_events=minimum_events,
                    ),
                    "conditional_ordered_top2_role_routes": _conditional_reader_routes(
                        calibration.events,
                        test.events,
                        calibration_roles,
                        test_roles,
                        calibration_active,
                        test_active,
                        calibration.writer_scores[layer][:, writer] ** 2,
                        energy,
                        minimum_events=minimum_events,
                    ),
                    "conditional_ordered_top2_label_routes": _conditional_reader_routes(
                        calibration.events,
                        test.events,
                        calibration_labels,
                        test_labels,
                        calibration_active,
                        test_active,
                        calibration.writer_scores[layer][:, writer] ** 2,
                        energy,
                        minimum_events=minimum_events,
                    ),
                    "test_attention_entropy_nats": _weighted_mean(
                        test.quantum_entropy[layer][:, owner], energy
                    ),
                    "test_effective_support": _weighted_mean(
                        test.quantum_effective_support[layer][:, owner], energy
                    ),
                    "test_top1_mass": _weighted_mean(
                        test.quantum_top1_mass[layer][:, owner], energy
                    ),
                    "test_top2_mass": _weighted_mean(
                        test.quantum_top2_mass[layer][:, owner], energy
                    ),
                    "test_context_fraction_of_scalar_magnitude": _weighted_mean(
                        context_fraction, energy
                    ),
                    "test_active_event_count": int(test_active.sum()),
                    "test_writer_energy": float(energy.sum()),
                }
            )
    selected = [row for row in writer_rows if row["rich_selected_on_calibration"]]
    unselected = [row for row in writer_rows if not row["rich_selected_on_calibration"]]

    def group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        def median(key: str) -> float | None:
            values = [row[key] for row in rows if row[key] is not None]
            return None if not values else float(np.median(values))

        return {
            "writer_count": len(rows),
            "median_test_attention_entropy_nats": median("test_attention_entropy_nats"),
            "median_test_effective_support": median("test_effective_support"),
            "median_test_top1_mass": median("test_top1_mass"),
            "median_test_top2_mass": median("test_top2_mass"),
            "median_test_ordered_top2_role_stability": median(
                "test_ordered_top2_role_stability"
            ),
            "median_test_ordered_top2_label_stability": median(
                "test_ordered_top2_label_stability"
            ),
            "median_test_ordered_top2_role_activation_f1": float(np.median([
                row["ordered_top2_role_activation_rule"]["test"]["f1"]
                for row in rows
                if row["ordered_top2_role_activation_rule"] is not None
            ])) if any(row["ordered_top2_role_activation_rule"] is not None for row in rows) else None,
            "median_test_ordered_top2_label_activation_f1": float(np.median([
                row["ordered_top2_label_activation_rule"]["test"]["f1"]
                for row in rows
                if row["ordered_top2_label_activation_rule"] is not None
            ])) if any(row["ordered_top2_label_activation_rule"] is not None for row in rows) else None,
            "median_test_context_fraction": median(
                "test_context_fraction_of_scalar_magnitude"
            ),
        }

    return {
        "selection": (
            "Ordered top-two role and role/value signatures are selected jointly from "
            "calibration writer events, then frozen and scored on the disjoint test "
            "panel. Activation-rule F1 is conditioned on the owner quantum being active; "
            "state routes are conditioned on functional owner, subtype, and target position."
        ),
        "minimum_calibration_events": int(minimum_events),
        "quantum_reads": quantum_rows,
        "writers": writer_rows,
        "writer_groups": {
            "rich_selected": group_summary(selected),
            "rich_unselected": group_summary(unselected),
        },
    }


def _causal_prefix_permutation(
    attention_mask: torch.Tensor,
    *,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Permute each query's valid causal source positions without future leakage."""

    batch, length = attention_mask.shape
    result = torch.arange(length, device=attention_mask.device).repeat(
        batch, length, 1
    )
    for row in range(batch):
        valid_length = int(attention_mask[row].sum())
        for target in range(valid_length):
            indices = np.arange(target + 1)
            result[row, target, : target + 1] = torch.as_tensor(
                rng.permutation(indices), device=attention_mask.device
            )
    return result


def _matched_example_permutation(
    examples: Sequence[Any], *, rng: np.random.Generator, device: torch.device
) -> torch.Tensor:
    """Swap whole value streams only among examples with matching output shape."""

    result = np.arange(len(examples), dtype=np.int64)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[(len(str(int(example.number))), len(str(example.text).split()))].append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        order = np.asarray(indices, dtype=np.int64)
        rng.shuffle(order)
        result[order] = np.roll(order, -1)
    return torch.as_tensor(result, device=device)


@torch.no_grad()
def _attention_intervention_behavior(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
    mode: str,
    seed: int,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate frozen reader interventions without changing fitted parameters."""

    if mode not in {
        "softmax",
        "top_k",
        "position_shuffle",
        "value_swap",
        "local_swap",
    }:
        raise ValueError(f"unsupported reader intervention {mode!r}")
    total: dict[str, float | int] = {
        "prediction_events": 0,
        "source_correct": 0,
        "quantum_correct": 0,
        "source_quantum_agreement": 0,
        "source_nll_nats": 0.0,
        "quantum_nll_nats": 0.0,
        "source_to_quantum_kl_nats": 0.0,
    }
    swap_total = swap_count = same_target = changed_target = 0
    donor_logit_gain = receiver_logit_change = 0.0
    for start in range(0, len(examples), int(batch_size)):
        selected = list(examples[start : start + int(batch_size)])
        batch = task.encode_examples(selected, device=device)
        rows, positions = _event_coordinates(batch)
        expanded_existence = tuple(
            value.expand(len(selected), -1) for value in existences
        )
        factual = forward_quanta(
            source,
            readout,
            modules,
            batch.input_ids,
            batch.attention_mask,
            expanded_existence,
            edges,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        intervention: dict[str, Any] = {}
        permutation: torch.Tensor | None = None
        if mode == "softmax":
            trace = factual
        elif mode == "top_k":
            intervention["attention_top_k"] = top_k
            trace = forward_quanta(
                source,
                readout,
                modules,
                batch.input_ids,
                batch.attention_mask,
                expanded_existence,
                edges,
                hard_gates=True,
                routing_mode=routing_mode,
                enforce_parent_gate_closure=enforce_parent_gate_closure,
                **intervention,
            )
        else:
            intervention.update(
                {
                    "local_gate_overrides": factual.local_gates,
                    "writer_gate_overrides": factual.writer_gates,
                }
            )
            rng = np.random.default_rng(int(seed) + int(start))
            if mode == "position_shuffle":
                intervention["attention_position_permutation"] = _causal_prefix_permutation(
                    batch.attention_mask, rng=rng
                )
                intervention["writer_local_score_overrides"] = factual.writer_local_scores
            else:
                permutation = _matched_example_permutation(
                    selected, rng=rng, device=device
                )
                if mode == "value_swap":
                    intervention["attention_value_permutation"] = permutation
                    intervention["writer_local_score_overrides"] = factual.writer_local_scores
                else:
                    intervention["writer_local_score_overrides"] = tuple(
                        value.index_select(0, permutation)
                        for value in factual.writer_local_scores
                    )
            trace = forward_quanta(
                source,
                readout,
                modules,
                batch.input_ids,
                batch.attention_mask,
                expanded_existence,
                edges,
                hard_gates=True,
                routing_mode=routing_mode,
                enforce_parent_gate_closure=enforce_parent_gate_closure,
                **intervention,
            )
        source_logits = source.residual_trace(**batch.model_inputs).logits[
            rows, positions
        ]
        labels = batch.labels[:, 1:][rows, positions]
        record = _behavior_record(
            labels=labels,
            source_logits=source_logits,
            quantum_logits=trace.logits[rows, positions],
        )
        for key, value in record.items():
            total[key] += value
        if permutation is not None:
            donor_labels = batch.labels[permutation, 1:][rows, positions]
            valid = donor_labels != -100
            if bool(valid.any()):
                factual_logits = factual.logits[rows, positions]
                swapped_logits = trace.logits[rows, positions]
                donor_logit_gain += float(
                    (
                        swapped_logits.gather(1, donor_labels[:, None])[:, 0]
                        - factual_logits.gather(1, donor_labels[:, None])[:, 0]
                    )[valid].sum()
                )
                receiver_logit_change += float(
                    (
                        swapped_logits.gather(1, labels[:, None])[:, 0]
                        - factual_logits.gather(1, labels[:, None])[:, 0]
                    )[valid].sum()
                )
                swap_total += int(valid.sum())
                same_target += int((donor_labels[valid] == labels[valid]).sum())
                changed_target += int((donor_labels[valid] != labels[valid]).sum())
                swap_count += 1
    result = _finish_behavior(total)
    if swap_total:
        result["matched_swap"] = {
            "events": swap_total,
            "batches": swap_count,
            "same_target_events": same_target,
            "different_target_events": changed_target,
            "mean_donor_target_logit_change": donor_logit_gain / swap_total,
            "mean_receiver_target_logit_change": receiver_logit_change / swap_total,
        }
    return result


def _parse_top_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if any(value <= 0 for value in values):
        raise ValueError("attention-top-k-values must contain positive integers")
    return values


def _attach_reader_causal_effects(
    reader: dict[str, Any], causal: dict[str, Any]
) -> None:
    """Attach bounded individual deletion effects to the reader diagnostics."""

    effects = {
        str(record["name"]): record
        for record in causal["individual_high_energy_writer_ablations"]
    }
    for writer in reader["writers"]:
        effect = effects.get(str(writer["name"]))
        if effect is None:
            writer["causal_ablation"] = None
        else:
            writer["causal_ablation"] = {
                "delta_quantum_accuracy": effect["delta_quantum_accuracy"],
                "delta_quantum_nll_bits": effect["delta_quantum_nll_bits"],
            }


def _concept_matrices(
    calibration_events: Sequence[dict[str, Any]],
    test_events: Sequence[dict[str, Any]],
    *,
    minimum_positives: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = sorted(
        {
            concept
            for event in (*calibration_events, *test_events)
            for concept in event["concepts"]
        }
    )
    lookup = {name: index for index, name in enumerate(names)}

    def encode(events: Sequence[dict[str, Any]]) -> np.ndarray:
        matrix = np.zeros((len(events), len(names)), dtype=bool)
        for row, event in enumerate(events):
            matrix[row, [lookup[name] for name in event["concepts"]]] = True
        return matrix

    calibration = encode(calibration_events)
    test = encode(test_events)
    keep = (
        (calibration.sum(axis=0) >= int(minimum_positives))
        & ((~calibration).sum(axis=0) >= int(minimum_positives))
        & (test.sum(axis=0) >= int(minimum_positives))
        & ((~test).sum(axis=0) >= int(minimum_positives))
    )
    return (
        [name for name, selected in zip(names, keep) if bool(selected)],
        calibration[:, keep],
        test[:, keep],
    )


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shapes")
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "count": int(len(target)),
        "target_prevalence": float(target.mean()) if len(target) else 0.0,
        "activation_prevalence": float(prediction.mean()) if len(prediction) else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    """Tie-aware average precision without a scikit-learn dependency."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    positives = int(target.sum())
    if scores.shape != target.shape:
        raise ValueError("scores and target must have equal shapes")
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_target = target[order]
    boundaries = np.r_[
        np.nonzero(ordered_scores[1:] != ordered_scores[:-1])[0] + 1,
        len(scores),
    ]
    cumulative_true = np.cumsum(ordered_target, dtype=np.int64)
    previous_recall = 0.0
    result = 0.0
    for end in boundaries:
        true_positive = int(cumulative_true[end - 1])
        recall = true_positive / positives
        precision = true_positive / int(end)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def _concept_family_indices(names: Sequence[str]) -> dict[str, np.ndarray]:
    families = {
        "algorithmic": ALGORITHMIC_PREFIXES,
        "input": INPUT_PREFIXES,
        "output": OUTPUT_PREFIXES,
    }
    result = {"all": np.arange(len(names), dtype=np.int64)}
    for family, prefixes in families.items():
        result[family] = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if name.startswith(prefixes)
            ],
            dtype=np.int64,
        )
    return result


def _best_hypothesis(
    active: np.ndarray,
    concepts: np.ndarray,
    indices: np.ndarray,
) -> int | None:
    if not len(indices) or not bool(active.any()):
        return None
    true_positive = concepts[active][:, indices].sum(axis=0, dtype=np.int64)
    predicted = int(active.sum())
    actual = concepts[:, indices].sum(axis=0, dtype=np.int64)
    f1 = 2.0 * true_positive / np.maximum(predicted + actual, 1)
    return int(indices[int(np.argmax(f1))])


def _fit_hypotheses(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_concepts: np.ndarray,
    test_concepts: np.ndarray,
    concept_names: Sequence[str],
    feature_refs: Sequence[FeatureRef],
) -> list[dict[str, Any]]:
    if calibration_scores.shape[1] != len(feature_refs):
        raise ValueError("feature metadata does not match score columns")
    families = _concept_family_indices(concept_names)
    records: list[dict[str, Any]] = []
    for feature, reference in enumerate(feature_refs):
        calibration_score = calibration_scores[:, feature]
        test_score = test_scores[:, feature]
        calibration_active = calibration_score > 0
        test_active = test_score > 0
        hypotheses: dict[str, Any] = {}
        for family, indices in families.items():
            selected = _best_hypothesis(
                calibration_active, calibration_concepts, indices
            )
            if selected is None:
                hypotheses[family] = None
                continue
            calibration_metrics = binary_metrics(
                calibration_active, calibration_concepts[:, selected]
            )
            test_metrics = binary_metrics(test_active, test_concepts[:, selected])
            hypotheses[family] = {
                "concept": concept_names[selected],
                "calibration": calibration_metrics,
                "test": {
                    **test_metrics,
                    "average_precision": average_precision(
                        test_score, test_concepts[:, selected]
                    ),
                },
            }
        records.append(
            {
                "name": reference.name,
                "kind": reference.kind,
                "layer": reference.layer,
                "index": reference.index,
                "owner_quantum": reference.owner_quantum,
                "calibration_activation_count": int(calibration_active.sum()),
                "test_activation_count": int(test_active.sum()),
                "calibration_activation_rate": float(calibration_active.mean()),
                "test_activation_rate": float(test_active.mean()),
                "calibration_mean_squared_magnitude": float(
                    np.square(calibration_score).mean()
                ),
                "test_mean_squared_magnitude": float(np.square(test_score).mean()),
                "hypotheses": hypotheses,
            }
        )
    return records


def _top_atomic_indices(
    active: np.ndarray,
    concepts: np.ndarray,
    indices: np.ndarray,
    *,
    limit: int,
) -> np.ndarray:
    """Return calibration-ranked atomic descriptions for compound expansion."""

    if not len(indices) or not bool(active.any()) or int(limit) <= 0:
        return np.empty(0, dtype=np.int64)
    true_positive = concepts[active][:, indices].sum(axis=0, dtype=np.int64)
    f1 = 2.0 * true_positive / np.maximum(
        int(active.sum()) + concepts[:, indices].sum(axis=0), 1
    )
    order = np.argsort(-f1, kind="mergesort")
    return indices[order[: min(int(limit), len(order))]]


def _best_compound_hypothesis(
    active: np.ndarray,
    concepts: np.ndarray,
    indices: np.ndarray,
    concept_names: Sequence[str],
    *,
    top_concepts: int,
    description_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Select a penalized atomic or two-literal Boolean description.

    Candidate literals are calibration-ranked atomic concepts.  This bounded
    expansion makes the description language richer without silently searching
    the full quadratic concept dictionary for every feature.
    """

    if not len(indices) or not bool(active.any()):
        return None
    if float(description_penalty) < 0.0:
        raise ValueError("description_penalty must be non-negative")
    candidates = _top_atomic_indices(
        active, concepts, indices, limit=top_concepts
    )
    if not len(candidates):
        return None
    best_target: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    best_score = -float("inf")

    def consider(
        target: np.ndarray,
        *,
        operator: str,
        terms: Sequence[int],
    ) -> None:
        nonlocal best_target, best_record, best_score
        metrics = binary_metrics(active, target)
        description_length = len(terms)
        selection_score = float(metrics["f1"]) - float(description_penalty) * (
            description_length - 1
        )
        if selection_score <= best_score:
            return
        best_score = selection_score
        best_target = target
        best_record = {
            "concept": (
                concept_names[terms[0]]
                if description_length == 1
                else f"({concept_names[terms[0]]}) {operator} ({concept_names[terms[1]]})"
            ),
            "operator": operator,
            "terms": [concept_names[index] for index in terms],
            "description_length": description_length,
            "selection_score": selection_score,
            "calibration": metrics,
        }

    for index in candidates:
        consider(concepts[:, index], operator="ATOM", terms=(int(index),))
    for first, second in combinations(candidates.tolist(), 2):
        consider(
            np.logical_and(concepts[:, first], concepts[:, second]),
            operator="AND",
            terms=(int(first), int(second)),
        )
        consider(
            np.logical_or(concepts[:, first], concepts[:, second]),
            operator="OR",
            terms=(int(first), int(second)),
        )
    if best_target is None or best_record is None:
        return None
    return best_target, best_record


def _fit_compound_hypotheses(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_concepts: np.ndarray,
    test_concepts: np.ndarray,
    concept_names: Sequence[str],
    feature_refs: Sequence[FeatureRef],
    *,
    top_concepts: int,
    description_penalty: float,
) -> list[dict[str, Any]]:
    """Fit bounded Boolean descriptions on calibration and freeze them on test."""

    if calibration_scores.shape[1] != len(feature_refs):
        raise ValueError("feature metadata does not match score columns")
    families = _concept_family_indices(concept_names)
    records: list[dict[str, Any]] = []
    for feature, reference in enumerate(feature_refs):
        calibration_score = calibration_scores[:, feature]
        test_score = test_scores[:, feature]
        calibration_active = calibration_score > 0
        test_active = test_score > 0
        hypotheses: dict[str, Any] = {}
        for family, indices in families.items():
            selected = _best_compound_hypothesis(
                calibration_active,
                calibration_concepts,
                indices,
                concept_names,
                top_concepts=top_concepts,
                description_penalty=description_penalty,
            )
            if selected is None:
                hypotheses[family] = None
                continue
            _, record = selected
            terms = record["terms"]
            lookup = {name: index for index, name in enumerate(concept_names)}
            term_indices = [lookup[name] for name in terms]
            if record["operator"] == "ATOM":
                test_target = test_concepts[:, term_indices[0]]
            elif record["operator"] == "AND":
                test_target = np.logical_and(
                    test_concepts[:, term_indices[0]], test_concepts[:, term_indices[1]]
                )
            else:
                test_target = np.logical_or(
                    test_concepts[:, term_indices[0]], test_concepts[:, term_indices[1]]
                )
            hypotheses[family] = {
                **record,
                "test": {
                    **binary_metrics(test_active, test_target),
                    "average_precision": average_precision(test_score, test_target),
                },
            }
        records.append(
            {
                "name": reference.name,
                "kind": reference.kind,
                "layer": reference.layer,
                "index": reference.index,
                "owner_quantum": reference.owner_quantum,
                "hypotheses": hypotheses,
            }
        )
    return records


def _continuous_matrices(
    calibration_events: Sequence[dict[str, Any]],
    test_events: Sequence[dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = sorted(
        set.intersection(
            *(set(event["continuous"]) for event in calibration_events),
            *(set(event["continuous"]) for event in test_events),
        )
    )
    if not names:
        return [], np.empty((len(calibration_events), 0)), np.empty((len(test_events), 0))

    def encode(events: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [[event["continuous"][name] for name in names] for event in events],
            dtype=np.float64,
        )

    return names, encode(calibration_events), encode(test_events)


def _amplitude_metrics(scores: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(scores) & np.isfinite(values)
    scores = np.asarray(scores, dtype=np.float64)[finite]
    values = np.asarray(values, dtype=np.float64)[finite]
    if len(scores) < 2 or np.std(scores) <= 1.0e-12 or np.std(values) <= 1.0e-12:
        return {
            "count": int(len(scores)),
            "pearson_r": 0.0,
            "r_squared": 0.0,
            "slope": 0.0,
            "intercept": float(scores.mean()) if len(scores) else 0.0,
        }
    design = np.column_stack((values, np.ones_like(values)))
    slope, intercept = np.linalg.lstsq(design, scores, rcond=None)[0]
    prediction = slope * values + intercept
    residual = float(np.square(scores - prediction).sum())
    total = float(np.square(scores - scores.mean()).sum())
    correlation = float(np.corrcoef(scores, values)[0, 1])
    return {
        "count": int(len(scores)),
        "pearson_r": correlation,
        "r_squared": 1.0 - residual / max(total, 1.0e-24),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def _fit_amplitude_hypotheses(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_values: np.ndarray,
    test_values: np.ndarray,
    names: Sequence[str],
    feature_refs: Sequence[FeatureRef],
) -> list[dict[str, Any]]:
    """Choose a scalar amplitude relation on calibration and freeze it on test."""

    records: list[dict[str, Any]] = []
    for feature, reference in enumerate(feature_refs):
        candidates = [
            _amplitude_metrics(calibration_scores[:, feature], calibration_values[:, index])
            for index in range(len(names))
        ]
        if not candidates:
            hypothesis = None
        else:
            selected = int(
                np.argmax([abs(float(candidate["pearson_r"])) for candidate in candidates])
            )
            calibration = candidates[selected]
            hypothesis = {
                "variable": names[selected],
                "selection_abs_pearson_r": abs(float(calibration["pearson_r"])),
                "calibration": calibration,
                "test": _amplitude_metrics(
                    test_scores[:, feature], test_values[:, selected]
                ),
            }
        records.append(
            {
                "name": reference.name,
                "kind": reference.kind,
                "layer": reference.layer,
                "index": reference.index,
                "owner_quantum": reference.owner_quantum,
                "hypothesis": hypothesis,
            }
        )
    return records


def _add_conditional_writer_hypotheses(
    records: list[dict[str, Any]],
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_quantum_scores: Sequence[np.ndarray],
    test_quantum_scores: Sequence[np.ndarray],
    calibration_concepts: np.ndarray,
    test_concepts: np.ndarray,
    concept_names: Sequence[str],
) -> None:
    """Score the finer writer distinction only where its owner Q executes."""

    families = _concept_family_indices(concept_names)
    for feature, record in enumerate(records):
        layer = int(record["layer"])
        owner = int(record["owner_quantum"])
        calibration_condition = calibration_quantum_scores[layer][:, owner] > 0
        test_condition = test_quantum_scores[layer][:, owner] > 0
        calibration_score = calibration_scores[:, feature][calibration_condition]
        test_score = test_scores[:, feature][test_condition]
        calibration_labels = calibration_concepts[calibration_condition]
        test_labels = test_concepts[test_condition]
        calibration_active = calibration_score > 0
        test_active = test_score > 0
        conditional: dict[str, Any] = {}
        for family, indices in families.items():
            selected = _best_hypothesis(
                calibration_active, calibration_labels, indices
            )
            if selected is None or not len(test_labels):
                conditional[family] = None
                continue
            conditional[family] = {
                "concept": concept_names[selected],
                "calibration": binary_metrics(
                    calibration_active, calibration_labels[:, selected]
                ),
                "test": {
                    **binary_metrics(test_active, test_labels[:, selected]),
                    "average_precision": average_precision(
                        test_score, test_labels[:, selected]
                    ),
                },
            }
        record["conditional_on_owner"] = {
            "calibration_owner_events": int(calibration_condition.sum()),
            "test_owner_events": int(test_condition.sum()),
            "hypotheses": conditional,
        }


def _add_conditional_writer_compound_hypotheses(
    records: list[dict[str, Any]],
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_quantum_scores: Sequence[np.ndarray],
    test_quantum_scores: Sequence[np.ndarray],
    calibration_concepts: np.ndarray,
    test_concepts: np.ndarray,
    concept_names: Sequence[str],
    *,
    top_concepts: int,
    description_penalty: float,
) -> None:
    """Add calibration-selected Boolean descriptions within owner-Q support."""

    families = _concept_family_indices(concept_names)
    lookup = {name: index for index, name in enumerate(concept_names)}
    for feature, record in enumerate(records):
        layer = int(record["layer"])
        owner = int(record["owner_quantum"])
        calibration_condition = calibration_quantum_scores[layer][:, owner] > 0
        test_condition = test_quantum_scores[layer][:, owner] > 0
        calibration_score = calibration_scores[:, feature][calibration_condition]
        test_score = test_scores[:, feature][test_condition]
        calibration_labels = calibration_concepts[calibration_condition]
        test_labels = test_concepts[test_condition]
        calibration_active = calibration_score > 0
        test_active = test_score > 0
        conditional: dict[str, Any] = {}
        for family, indices in families.items():
            selected = _best_compound_hypothesis(
                calibration_active,
                calibration_labels,
                indices,
                concept_names,
                top_concepts=top_concepts,
                description_penalty=description_penalty,
            )
            if selected is None or not len(test_labels):
                conditional[family] = None
                continue
            _, hypothesis = selected
            term_indices = [lookup[name] for name in hypothesis["terms"]]
            if hypothesis["operator"] == "ATOM":
                test_target = test_labels[:, term_indices[0]]
            elif hypothesis["operator"] == "AND":
                test_target = np.logical_and(
                    test_labels[:, term_indices[0]], test_labels[:, term_indices[1]]
                )
            else:
                test_target = np.logical_or(
                    test_labels[:, term_indices[0]], test_labels[:, term_indices[1]]
                )
            conditional[family] = {
                **hypothesis,
                "test": {
                    **binary_metrics(test_active, test_target),
                    "average_precision": average_precision(test_score, test_target),
                },
            }
        record["conditional_on_owner"]["compound_hypotheses"] = conditional


def _add_conditional_writer_amplitude_hypotheses(
    records: list[dict[str, Any]],
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_quantum_scores: Sequence[np.ndarray],
    test_quantum_scores: Sequence[np.ndarray],
    calibration_values: np.ndarray,
    test_values: np.ndarray,
    names: Sequence[str],
) -> None:
    """Add owner-conditional scalar amplitude descriptions without test fitting."""

    for feature, record in enumerate(records):
        layer = int(record["layer"])
        owner = int(record["owner_quantum"])
        calibration_condition = calibration_quantum_scores[layer][:, owner] > 0
        test_condition = test_quantum_scores[layer][:, owner] > 0
        candidates = [
            _amplitude_metrics(
                calibration_scores[:, feature][calibration_condition],
                calibration_values[:, index][calibration_condition],
            )
            for index in range(len(names))
        ]
        if not candidates:
            record["conditional_on_owner"]["amplitude_hypothesis"] = None
            continue
        selected = int(
            np.argmax([abs(float(candidate["pearson_r"])) for candidate in candidates])
        )
        calibration = candidates[selected]
        record["conditional_on_owner"]["amplitude_hypothesis"] = {
            "variable": names[selected],
            "selection_abs_pearson_r": abs(float(calibration["pearson_r"])),
            "calibration": calibration,
            "test": _amplitude_metrics(
                test_scores[:, feature][test_condition],
                test_values[:, selected][test_condition],
            ),
        }


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"q25": None, "median": None, "q75": None, "mean": None}
    return {
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "mean": float(array.mean()),
    }


def _feature_summary(
    records: Sequence[dict[str, Any]],
    *,
    family: str,
    threshold: float,
    hypothesis_key: str = "hypotheses",
) -> dict[str, Any]:
    def hypotheses(record: dict[str, Any]) -> dict[str, Any]:
        if hypothesis_key == "hypotheses":
            return record["hypotheses"]
        return record[hypothesis_key]["hypotheses"]

    active = [record for record in records if hypotheses(record)[family] is not None]
    f1_values = [hypotheses(record)[family]["test"]["f1"] for record in active]
    precision = [
        hypotheses(record)[family]["test"]["precision"] for record in active
    ]
    recall = [hypotheses(record)[family]["test"]["recall"] for record in active]
    ap = [
        hypotheses(record)[family]["test"]["average_precision"]
        for record in active
    ]
    weights = np.asarray(
        [record["test_mean_squared_magnitude"] for record in active],
        dtype=np.float64,
    )
    validated = np.asarray([value >= float(threshold) for value in f1_values])
    return {
        "total_features": len(records),
        "active_features": len(active),
        "dead_features": len(records) - len(active),
        "heldout_f1": _quantiles(f1_values),
        "heldout_precision": _quantiles(precision),
        "heldout_recall": _quantiles(recall),
        "heldout_average_precision": _quantiles(ap),
        "fraction_f1_at_least_0p5": float(
            np.mean(np.asarray(f1_values) >= 0.5)
        )
        if f1_values
        else 0.0,
        "fraction_f1_at_least_0p6": float(
            np.mean(np.asarray(f1_values) >= 0.6)
        )
        if f1_values
        else 0.0,
        "fraction_f1_at_least_0p7": float(
            np.mean(np.asarray(f1_values) >= 0.7)
        )
        if f1_values
        else 0.0,
        "fraction_f1_at_least_0p8": float(
            np.mean(np.asarray(f1_values) >= 0.8)
        )
        if f1_values
        else 0.0,
        "interpretability_threshold": float(threshold),
        "fraction_meeting_threshold": float(validated.mean())
        if len(validated)
        else 0.0,
        "activation_energy_fraction_meeting_threshold": float(
            weights[validated].sum() / max(weights.sum(), 1.0e-24)
        )
        if len(weights)
        else 0.0,
    }


def _compound_feature_summary(
    records: Sequence[dict[str, Any]],
    *,
    family: str,
    threshold: float,
    conditional: bool = False,
) -> dict[str, Any]:
    if conditional:
        hypotheses = [
            record["conditional_on_owner"]["compound_hypotheses"][family]
            for record in records
        ]
    else:
        hypotheses = [record["compound_hypotheses"][family] for record in records]
    active = [hypothesis for hypothesis in hypotheses if hypothesis is not None]
    test_f1 = [hypothesis["test"]["f1"] for hypothesis in active]
    return {
        "total_features": len(records),
        "active_features": len(active),
        "heldout_f1": _quantiles(test_f1),
        "fraction_f1_at_least_0p6": float(
            np.mean(np.asarray(test_f1) >= 0.6)
        )
        if test_f1
        else 0.0,
        "fraction_meeting_threshold": float(
            np.mean(np.asarray(test_f1) >= float(threshold))
        )
        if test_f1
        else 0.0,
        "fraction_two_literal_descriptions": float(
            np.mean(
                np.asarray(
                    [hypothesis["description_length"] == 2 for hypothesis in active]
                )
            )
        )
        if active
        else 0.0,
    }


def _amplitude_summary(
    records: Sequence[dict[str, Any]], *, conditional: bool = False
) -> dict[str, Any]:
    hypotheses = [
        (
            record["conditional_on_owner"]["amplitude_hypothesis"]
            if conditional
            else record["amplitude_hypothesis"]
        )
        for record in records
    ]
    active = [hypothesis for hypothesis in hypotheses if hypothesis is not None]
    return {
        "total_features": len(records),
        "described_features": len(active),
        "heldout_abs_pearson_r": _quantiles(
            [abs(float(hypothesis["test"]["pearson_r"])) for hypothesis in active]
        ),
        "heldout_r_squared": _quantiles(
            [float(hypothesis["test"]["r_squared"]) for hypothesis in active]
        ),
    }


def _coverage(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_concepts: np.ndarray,
    test_concepts: np.ndarray,
    concept_names: Sequence[str],
    feature_refs: Sequence[FeatureRef],
) -> dict[str, Any]:
    calibration_active = calibration_scores > 0
    test_active = test_scores > 0
    calibration_tp = (
        calibration_active.astype(np.float32).T
        @ calibration_concepts.astype(np.float32)
    )
    calibration_f1 = 2.0 * calibration_tp / np.maximum(
        calibration_active.sum(axis=0)[:, None]
        + calibration_concepts.sum(axis=0)[None, :],
        1,
    )
    records = []
    for concept, name in enumerate(concept_names):
        order = np.argsort(-calibration_f1[:, concept], kind="mergesort")
        best = int(order[0])
        top = order[: min(4, len(order))]
        single = binary_metrics(
            test_active[:, best], test_concepts[:, concept]
        )
        union = binary_metrics(
            test_active[:, top].any(axis=1), test_concepts[:, concept]
        )
        records.append(
            {
                "concept": name,
                "test_prevalence": float(test_concepts[:, concept].mean()),
                "best_single_feature": feature_refs[best].name,
                "best_single_calibration_f1": float(
                    calibration_f1[best, concept]
                ),
                "best_single_test": single,
                "top4_features": [feature_refs[int(index)].name for index in top],
                "top4_union_test": union,
            }
        )
    single_f1 = [record["best_single_test"]["f1"] for record in records]
    union_f1 = [record["top4_union_test"]["f1"] for record in records]
    return {
        "concepts": records,
        "best_single_heldout_f1": _quantiles(single_f1),
        "top4_union_heldout_f1": _quantiles(union_f1),
        "fraction_single_f1_at_least_0p6": float(
            np.mean(np.asarray(single_f1) >= 0.6)
        ),
        "fraction_top4_union_f1_at_least_0p6": float(
            np.mean(np.asarray(union_f1) >= 0.6)
        ),
    }


def _feature_refs(
    modules: Sequence[CheapCausalAttentionQuantumLayer], *, kind: str
) -> list[FeatureRef]:
    result = []
    for layer, module in enumerate(modules):
        if kind == "quantum":
            result.extend(
                FeatureRef("quantum", layer, quantum, quantum)
                for quantum in range(module.quantum_count)
            )
        elif kind == "writer":
            result.extend(
                FeatureRef(
                    "writer", layer, writer, int(module.writer_owner[writer])
                )
                for writer in range(module.writer_count)
            )
        else:
            raise ValueError(f"unsupported feature kind {kind!r}")
    return result


@torch.no_grad()
def _target_logit_vector(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
) -> np.ndarray:
    """Return target-logit values under the current sparse-output intervention."""

    parts: list[np.ndarray] = []
    for start in range(0, len(examples), int(batch_size)):
        selected = list(examples[start : start + int(batch_size)])
        batch = task.encode_examples(selected, device=device)
        rows, positions = _event_coordinates(batch)
        expanded_existence = tuple(
            value.expand(len(selected), -1) for value in existences
        )
        trace = forward_quanta(
            source,
            readout,
            modules,
            batch.input_ids,
            batch.attention_mask,
            expanded_existence,
            edges,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        labels = batch.labels[:, 1:][rows, positions]
        parts.append(
            trace.logits[rows, positions].gather(1, labels[:, None])[:, 0]
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.float32)


@torch.no_grad()
def _writer_causal_logit_signatures(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    writer_records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
) -> dict[str, np.ndarray]:
    """Measure each writer's held-out target-logit deletion signature.

    The signature for writer j is full_target_logit - target_logit_without_j at
    every event in a bounded held-out panel.  It is deliberately an intervention
    in the fitted Q-model, rather than a decoder-weight proxy.
    """

    if not examples:
        return {}
    all_keep = [np.ones(module.writer_count, dtype=bool) for module in modules]
    baseline = _target_logit_vector(
        source,
        readout,
        modules,
        task,
        examples,
        existences,
        edges,
        device=device,
        batch_size=batch_size,
        routing_mode=routing_mode,
        enforce_parent_gate_closure=enforce_parent_gate_closure,
    )
    signatures: dict[str, np.ndarray] = {}
    for record in writer_records:
        keep = [values.copy() for values in all_keep]
        keep[int(record["layer"])][int(record["index"])] = False
        with _masked_outputs(modules, writer_keep=keep):
            ablated = _target_logit_vector(
                source,
                readout,
                modules,
                task,
                examples,
                existences,
                edges,
                device=device,
                batch_size=batch_size,
                routing_mode=routing_mode,
                enforce_parent_gate_closure=enforce_parent_gate_closure,
            )
        signatures[str(record["name"])] = baseline - ablated
    return signatures


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-24:
        return None
    return float(np.dot(first, second) / denominator)


def _within_quantum_redundancy(
    writer_calibration: np.ndarray,
    writer_test: np.ndarray,
    writer_refs: Sequence[FeatureRef],
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    causal_signatures: dict[str, np.ndarray],
    *,
    cluster_cosine: float,
) -> dict[str, Any]:
    """Report support, direction, and intervention similarity within each bank."""

    if not -1.0 <= float(cluster_cosine) <= 1.0:
        raise ValueError("cluster_cosine must lie in [-1, 1]")
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for column, reference in enumerate(writer_refs):
        grouped[(reference.layer, reference.owner_quantum)].append(column)
    banks: list[dict[str, Any]] = []
    for (layer, owner), columns in sorted(grouped.items()):
        active_calibration = writer_calibration[:, columns] > 0
        active_test = writer_test[:, columns] > 0
        decoder = F.normalize(modules[layer].output_weight, dim=-1).detach().cpu().numpy()
        pairs: list[dict[str, Any]] = []
        cluster_edges: list[tuple[int, int]] = []
        for local_first, local_second in combinations(range(len(columns)), 2):
            first = columns[local_first]
            second = columns[local_second]
            calibration_both = np.logical_and(
                active_calibration[:, local_first], active_calibration[:, local_second]
            )
            test_both = np.logical_and(
                active_test[:, local_first], active_test[:, local_second]
            )
            calibration_union = np.logical_or(
                active_calibration[:, local_first], active_calibration[:, local_second]
            )
            test_union = np.logical_or(
                active_test[:, local_first], active_test[:, local_second]
            )
            signature_cosine = _cosine_similarity(
                causal_signatures.get(writer_refs[first].name, np.empty(0)),
                causal_signatures.get(writer_refs[second].name, np.empty(0)),
            )
            if signature_cosine is not None and signature_cosine >= float(cluster_cosine):
                cluster_edges.append((local_first, local_second))
            pairs.append(
                {
                    "first": writer_refs[first].name,
                    "second": writer_refs[second].name,
                    "calibration_jaccard": float(
                        calibration_both.sum() / max(calibration_union.sum(), 1)
                    ),
                    "test_jaccard": float(test_both.sum() / max(test_union.sum(), 1)),
                    "test_coactivation_rate": float(test_both.mean()),
                    "test_second_given_first": float(
                        test_both.sum() / max(active_test[:, local_first].sum(), 1)
                    ),
                    "test_first_given_second": float(
                        test_both.sum() / max(active_test[:, local_second].sum(), 1)
                    ),
                    "decoder_cosine": _cosine_similarity(
                        decoder[int(writer_refs[first].index)],
                        decoder[int(writer_refs[second].index)],
                    ),
                    "causal_target_logit_signature_cosine": signature_cosine,
                }
            )

        parents = list(range(len(columns)))

        def find(value: int) -> int:
            while parents[value] != value:
                parents[value] = parents[parents[value]]
                value = parents[value]
            return value

        for first, second in cluster_edges:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parents[second_root] = first_root
        clusters: dict[int, list[int]] = defaultdict(list)
        for local_index in range(len(columns)):
            clusters[find(local_index)].append(local_index)
        banks.append(
            {
                "layer": layer,
                "owner_quantum": owner,
                "writer_count": len(columns),
                "pair_count": len(pairs),
                "pairs": pairs,
                "causal_signature_cluster_cosine": float(cluster_cosine),
                "causal_signature_clusters": [
                    [writer_refs[columns[index]].name for index in members]
                    for members in clusters.values()
                    if len(members) > 1
                ],
            }
        )
    return {
        "causal_signature_available": bool(causal_signatures),
        "causal_signature_definition": (
            "per-event full target logit minus target logit after deleting one "
            "writer inside the fitted Q-model"
        ),
        "banks": banks,
    }


def _dashboard_event(event: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "score": float(score),
        "number": event["number"],
        "token_index": event["token_index"],
        "prefix": event["prefix"],
        "target": event["target"],
        "functional_owner": event["functional_owner"],
        "functional_context": event["functional_context"],
        "functional_subtype": event["functional_subtype"],
    }


def _writer_dashboards(
    scores: np.ndarray,
    events: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    top_examples: int,
) -> list[dict[str, Any]]:
    dashboards = []
    for feature, record in enumerate(records):
        values = scores[:, feature]
        active = np.nonzero(values > 0)[0]
        ordered = active[np.argsort(-values[active], kind="mergesort")]
        top = ordered[: int(top_examples)]
        quantile_rows: list[int] = []
        if len(ordered):
            ascending = ordered[::-1]
            for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
                index = int(round(quantile * (len(ascending) - 1)))
                row = int(ascending[index])
                if row not in quantile_rows:
                    quantile_rows.append(row)
        dashboards.append(
            {
                "name": record["name"],
                "layer": record["layer"],
                "writer": record["index"],
                "owner_quantum": record["owner_quantum"],
                "activation_count": int(len(active)),
                "best_hypotheses": record["hypotheses"],
                "compound_hypotheses": record.get("compound_hypotheses"),
                "amplitude_hypothesis": record.get("amplitude_hypothesis"),
                "conditional_compound_hypotheses": record.get(
                    "conditional_on_owner", {}
                ).get("compound_hypotheses"),
                "conditional_amplitude_hypothesis": record.get(
                    "conditional_on_owner", {}
                ).get("amplitude_hypothesis"),
                "top_activations": [
                    _dashboard_event(events[int(row)], values[int(row)])
                    for row in top
                ],
                "activation_quantiles": [
                    _dashboard_event(events[row], values[row])
                    for row in quantile_rows
                ],
            }
        )
    return dashboards


@contextmanager
def _masked_outputs(
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    *,
    writer_keep: Sequence[np.ndarray] | None = None,
    keep_attention_output: bool = True,
) -> Iterator[None]:
    writer_originals = [module.output_weight.detach().clone() for module in modules]
    attention_originals = [
        None
        if module.attention_output_weight is None
        else module.attention_output_weight.detach().clone()
        for module in modules
    ]
    try:
        with torch.no_grad():
            if writer_keep is not None:
                for module, keep in zip(modules, writer_keep):
                    mask = torch.as_tensor(
                        keep,
                        device=module.output_weight.device,
                        dtype=module.output_weight.dtype,
                    )
                    module.output_weight.mul_(mask[:, None])
            if not keep_attention_output:
                for module in modules:
                    if module.attention_output_weight is not None:
                        module.attention_output_weight.zero_()
        yield
    finally:
        with torch.no_grad():
            for module, writer in zip(modules, writer_originals):
                module.output_weight.copy_(writer)
            for module, attention in zip(modules, attention_originals):
                if attention is not None:
                    module.attention_output_weight.copy_(attention)


def _compact_behavior(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "examples",
        "prediction_events",
        "source_accuracy",
        "quantum_accuracy",
        "quantum_source_argmax_agreement",
        "source_nll_bits",
        "quantum_nll_bits",
        "source_to_quantum_kl_bits",
    )
    return {key: record[key] for key in keys if key in record}


def _behavior_delta(
    intervention: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    return {
        "quantum_accuracy_change": float(intervention["quantum_accuracy"])
        - float(baseline["quantum_accuracy"]),
        "quantum_nll_bits_change": float(intervention["quantum_nll_bits"])
        - float(baseline["quantum_nll_bits"]),
        "source_agreement_change": float(
            intervention["quantum_source_argmax_agreement"]
        )
        - float(baseline["quantum_source_argmax_agreement"]),
    }


@torch.no_grad()
def _edge_message_shuffle_behavior(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    shuffled_edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
    seed: int,
) -> dict[str, Any]:
    """Replace selected parent messages with matched-example factual messages."""

    total: dict[str, float | int] = {
        "prediction_events": 0,
        "source_correct": 0,
        "quantum_correct": 0,
        "source_quantum_agreement": 0,
        "source_nll_nats": 0.0,
        "quantum_nll_nats": 0.0,
        "source_to_quantum_kl_nats": 0.0,
    }
    swapped_examples = 0
    for start in range(0, len(examples), int(batch_size)):
        selected = list(examples[start : start + int(batch_size)])
        batch = task.encode_examples(selected, device=device)
        rows, positions = _event_coordinates(batch)
        expanded_existence = tuple(
            value.expand(len(selected), -1) for value in existences
        )
        factual = forward_quanta(
            source,
            readout,
            modules,
            batch.input_ids,
            batch.attention_mask,
            expanded_existence,
            edges,
            hard_gates=True,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        permutation = _matched_example_permutation(
            selected,
            rng=np.random.default_rng(int(seed) + int(start)),
            device=device,
        )
        swapped_examples += int(
            (permutation != torch.arange(len(selected), device=device)).sum()
        )
        replacements = {
            edge: factual.contributions[int(edge[0])][:, :, int(edge[1])]
            .index_select(0, permutation)
            .detach()
            for edge in shuffled_edges
        }
        shuffled = forward_quanta(
            source,
            readout,
            modules,
            batch.input_ids,
            batch.attention_mask,
            expanded_existence,
            edges,
            hard_gates=True,
            edge_replacements=replacements,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        source_logits = source.residual_trace(**batch.model_inputs).logits[
            rows, positions
        ]
        labels = batch.labels[:, 1:][rows, positions]
        record = _behavior_record(
            labels=labels,
            source_logits=source_logits,
            quantum_logits=shuffled.logits[rows, positions],
        )
        for key, value in record.items():
            total[key] += value
    behavior = _finish_behavior(total)
    behavior["quantum_source_argmax_agreement"] = behavior.pop(
        "source_quantum_argmax_agreement"
    )
    return {
        **behavior,
        "examples": len(examples),
        "matched_examples_replaced": swapped_examples,
        "matched_example_replacement_fraction": swapped_examples
        / max(len(examples), 1),
    }


@torch.no_grad()
def _evaluate_edge_composition(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
    seed: int,
) -> dict[str, Any]:
    """Test global channels by zeroing and matched-example message replacement."""

    def zero(selected: Sequence[QuantumGraphEdge]) -> dict[str, Any]:
        return _compact_behavior(
            _evaluate_teacher_forced(
                source,
                readout,
                modules,
                task,
                examples,
                existences,
                edges,
                device=device,
                batch_size=batch_size,
                edge_scales={edge: 0.0 for edge in selected},
                routing_mode=routing_mode,
                enforce_parent_gate_closure=enforce_parent_gate_closure,
            )
        )

    baseline = zero(())
    individual_zero = []
    individual_shuffle = []
    for index, edge in enumerate(edges):
        ablated = zero((edge,))
        shuffled = _edge_message_shuffle_behavior(
            source,
            readout,
            modules,
            task,
            examples,
            existences,
            edges,
            (edge,),
            device=device,
            batch_size=batch_size,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
            seed=int(seed) + index,
        )
        individual_zero.append(
            {"edge": list(edge), "behavior": ablated, **_behavior_delta(ablated, baseline)}
        )
        individual_shuffle.append(
            {"edge": list(edge), "behavior": shuffled, **_behavior_delta(shuffled, baseline)}
        )
    if edges:
        joint_zero = zero(edges)
        joint_shuffle = _edge_message_shuffle_behavior(
            source,
            readout,
            modules,
            task,
            examples,
            existences,
            edges,
            edges,
            device=device,
            batch_size=batch_size,
            routing_mode=routing_mode,
            enforce_parent_gate_closure=enforce_parent_gate_closure,
            seed=int(seed) + len(edges),
        )
    else:
        joint_zero = baseline
        joint_shuffle = {**baseline, "matched_examples_replaced": 0, "matched_example_replacement_fraction": 0.0}
    return {
        "selection": "all frozen globally selected edges",
        "message_shuffle": (
            "Factual parent contributions are replaced by contributions from a "
            "different held-out example matched on digit length and output-token count; "
            "the downstream child is re-executed without freezing its local state."
        ),
        "baseline": baseline,
        "individual_zero": individual_zero,
        "joint_zero": {"behavior": joint_zero, **_behavior_delta(joint_zero, baseline)},
        "individual_message_shuffle": individual_shuffle,
        "joint_message_shuffle": {
            "behavior": joint_shuffle,
            **_behavior_delta(joint_shuffle, baseline),
        },
    }


def _validated_writer_masks(
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    writer_records: Sequence[dict[str, Any]],
    *,
    threshold: float,
    conditional: bool,
    rich: bool = False,
) -> list[np.ndarray]:
    """Select writers by frozen calibration semantics, globally or inside owner Q."""

    keep = [np.zeros(module.writer_count, dtype=bool) for module in modules]
    for record in writer_records:
        if conditional:
            owner_record = record["conditional_on_owner"]
            hypothesis_key = "compound_hypotheses" if rich else "hypotheses"
            hypothesis = owner_record[hypothesis_key]["all"]
        else:
            hypothesis_key = "compound_hypotheses" if rich else "hypotheses"
            hypothesis = record[hypothesis_key]["all"]
        if (
            hypothesis is not None
            and hypothesis["calibration"]["f1"] >= float(threshold)
        ):
            keep[record["layer"]][record["index"]] = True
    return keep


@torch.no_grad()
def _evaluate_counterfactuals(
    source: torch.nn.Module,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    task: Any,
    examples: Sequence[Any],
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    writer_records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
    enforce_parent_gate_closure: bool,
    threshold: float,
    causal_writers: int,
) -> dict[str, Any]:
    def evaluate(
        writer_keep: Sequence[np.ndarray] | None,
        keep_attention_output: bool,
    ) -> dict[str, Any]:
        with _masked_outputs(
            modules,
            writer_keep=writer_keep,
            keep_attention_output=keep_attention_output,
        ):
            return _compact_behavior(
                _evaluate_teacher_forced(
                    source,
                    readout,
                    modules,
                    task,
                    examples,
                    existences,
                    edges,
                    device=device,
                    batch_size=batch_size,
                    routing_mode=routing_mode,
                    enforce_parent_gate_closure=enforce_parent_gate_closure,
                )
            )

    all_keep = [np.ones(module.writer_count, dtype=bool) for module in modules]
    none_keep = [np.zeros(module.writer_count, dtype=bool) for module in modules]
    globally_validated_keep = _validated_writer_masks(
        modules,
        writer_records,
        threshold=threshold,
        conditional=False,
    )
    conditionally_validated_keep = _validated_writer_masks(
        modules,
        writer_records,
        threshold=threshold,
        conditional=True,
    )
    conditionally_rich_validated_keep = _validated_writer_masks(
        modules,
        writer_records,
        threshold=threshold,
        conditional=True,
        rich=True,
    )
    globally_unvalidated_keep = [~values for values in globally_validated_keep]
    conditionally_unvalidated_keep = [
        ~values for values in conditionally_validated_keep
    ]
    conditionally_rich_unvalidated_keep = [
        ~values for values in conditionally_rich_validated_keep
    ]
    scenarios = {
        "full": evaluate(None, True),
        "writers_only": evaluate(all_keep, False),
        "attention_output_only": evaluate(none_keep, True),
        "embedding_only": evaluate(none_keep, False),
        "globally_validated_writers_plus_attention": evaluate(
            globally_validated_keep, True
        ),
        "globally_unvalidated_writers_plus_attention": evaluate(
            globally_unvalidated_keep, True
        ),
        "globally_validated_writers_only": evaluate(
            globally_validated_keep, False
        ),
        "conditionally_validated_writers_plus_attention": evaluate(
            conditionally_validated_keep, True
        ),
        "conditionally_unvalidated_writers_plus_attention": evaluate(
            conditionally_unvalidated_keep, True
        ),
        "conditionally_validated_writers_only": evaluate(
            conditionally_validated_keep, False
        ),
        "conditionally_rich_validated_writers_plus_attention": evaluate(
            conditionally_rich_validated_keep, True
        ),
        "conditionally_rich_unvalidated_writers_plus_attention": evaluate(
            conditionally_rich_unvalidated_keep, True
        ),
        "conditionally_rich_validated_writers_only": evaluate(
            conditionally_rich_validated_keep, False
        ),
        "conditionally_rich_unvalidated_writers_only": evaluate(
            conditionally_rich_unvalidated_keep, False
        ),
    }

    baseline = scenarios["full"]
    ranked = sorted(
        writer_records,
        key=lambda record: record["test_mean_squared_magnitude"],
        reverse=True,
    )[: max(int(causal_writers), 0)]
    individual = []
    for record in ranked:
        keep = [values.copy() for values in all_keep]
        keep[record["layer"]][record["index"]] = False
        ablated = evaluate(keep, True)
        hypothesis = record["hypotheses"]["all"]
        individual.append(
            {
                "name": record["name"],
                "best_concept": None if hypothesis is None else hypothesis["concept"],
                "heldout_f1": None
                if hypothesis is None
                else hypothesis["test"]["f1"],
                "mean_squared_magnitude": record[
                    "test_mean_squared_magnitude"
                ],
                "ablated": ablated,
                "delta_quantum_accuracy": ablated["quantum_accuracy"]
                - baseline["quantum_accuracy"],
                "delta_quantum_nll_bits": ablated["quantum_nll_bits"]
                - baseline["quantum_nll_bits"],
            }
        )
    full_accuracy = scenarios["full"]["quantum_accuracy"]
    embedding_accuracy = scenarios["embedding_only"]["quantum_accuracy"]
    attention_accuracy = scenarios["attention_output_only"]["quantum_accuracy"]
    no_direct_denominator = full_accuracy - embedding_accuracy
    direct_denominator = full_accuracy - attention_accuracy
    return {
        "selection": (
            "atomic and rich owner-conditional writer groups use calibration F1 only; "
            "counterfactual examples are held out from Q-model training"
        ),
        "interpretability_f1_threshold": float(threshold),
        "globally_validated_writer_count": int(
            sum(value.sum() for value in globally_validated_keep)
        ),
        "conditionally_validated_writer_count": int(
            sum(value.sum() for value in conditionally_validated_keep)
        ),
        "conditionally_rich_validated_writer_count": int(
            sum(value.sum() for value in conditionally_rich_validated_keep)
        ),
        "scenarios": scenarios,
        "normalized_recovery": {
            "no_direct_rich_writer_only": (
                None
                if abs(no_direct_denominator) < 1e-12
                else (
                    scenarios["conditionally_rich_validated_writers_only"][
                        "quantum_accuracy"
                    ]
                    - embedding_accuracy
                )
                / no_direct_denominator
            ),
            "direct_rich_writers_plus_attention": (
                None
                if abs(direct_denominator) < 1e-12
                else (
                    scenarios[
                        "conditionally_rich_validated_writers_plus_attention"
                    ]["quantum_accuracy"]
                    - attention_accuracy
                )
                / direct_denominator
            ),
        },
        "individual_high_energy_writer_ablations": individual,
    }


def _load_models(
    run_dir: Path,
    q_model_dir: Path,
    config: Any,
    *,
    device: torch.device,
    priority_dir: Path | None = None,
) -> tuple[
    Any,
    torch.nn.Module,
    QuantumReadout,
    tuple[CheapCausalAttentionQuantumLayer, ...],
    tuple[torch.Tensor, ...],
    tuple[QuantumGraphEdge, ...],
    dict[str, Any],
]:
    task, source = build_model_and_task(config, device=device)
    q_summary = json.loads((q_model_dir / "summary.json").read_text())
    global_edge_discovery = (
        q_summary.get("edge_discovery", {}).get("mode") == "global_static"
    )
    if priority_dir is None:
        recorded_priority_dir = q_summary.get("priority_source", {}).get(
            "directory"
        )
        if recorded_priority_dir is not None:
            recorded_path = Path(recorded_priority_dir)
            priority_dir = (
                None if recorded_path.resolve() == run_dir.resolve() else recorded_path
            )
    supports, temporal_priority, edges, priority_summary = _load_priority(
        run_dir,
        priority_dir,
        raw_stage_b=global_edge_discovery,
    )
    if global_edge_discovery:
        edges = tuple(
            tuple(int(value) for value in edge)
            for edge in q_summary["executed_edges"]
        )
    checkpoint_metadata = json.loads(
        (run_dir / "checkpoint_metadata.json").read_text()
    )
    final_state = torch.load(
        run_dir / checkpoint_metadata["checkpoint_files"][-1],
        map_location=device,
        weights_only=True,
    )
    source.load_state_dict(final_state, strict=True)
    source.eval()

    quantum_counts = tuple(int(value.shape[1]) for value in supports)
    writer_counts = tuple(
        tuple(int(value) for value in layer)
        for layer in q_summary["writer_counts_by_layer"]
    )
    modules = tuple(
        CheapCausalAttentionQuantumLayer(
            int(config.d_model),
            count,
            attention_rank=int(config.q_attention_rank),
            attention_value_dim=int(config.q_attention_value_dim),
            attention_direct_residual=bool(config.q_attention_direct_residual),
            share_attention_projections=bool(config.q_share_attention_projections),
            attention_query_offsets=bool(config.q_attention_query_offsets),
            writer_counts=layer_writer_counts,
            writer_average_k=float(config.q_writer_average_k),
            writer_activation=str(config.q_writer_activation),
            writer_sparsity=str(config.q_writer_sparsity),
            writer_jump_threshold=float(config.q_writer_jump_threshold),
            writer_jump_bandwidth=float(config.q_writer_jump_bandwidth),
        ).to(device)
        for count, layer_writer_counts in zip(quantum_counts, writer_counts)
    )
    for layer, module in enumerate(modules):
        module.load_state_dict(
            torch.load(
                q_model_dir / f"layer_{layer}.pt",
                map_location=device,
                weights_only=True,
            ),
            strict=True,
        )
        module.eval()
    readout = QuantumReadout(
        int(config.d_model), int(final_state["token_embedding.weight"].shape[0])
    ).to(device)
    readout.copy_source_readout(source)
    readout.eval()
    existence_tables = _existence_table(
        temporal_priority,
        edges,
        int(config.steps),
        apply_closure=not global_edge_discovery,
    )
    existences = tuple(
        torch.as_tensor(table[-1], device=device)[None]
        for table in existence_tables
    )
    metadata = {
        "q_summary": q_summary,
        "priority_summary": priority_summary,
        "global_edge_discovery": global_edge_discovery,
        "quantum_counts_by_layer": quantum_counts,
        "writer_counts_by_layer": writer_counts,
    }
    return task, source, readout, modules, existences, edges, metadata


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Q-model interpretability audit",
        "",
        "This report is generated from `audit.json`; hypotheses are selected on a",
        "held-out calibration half and scored unchanged on a disjoint held-out test half.",
        "",
        "## Behavioral fidelity",
        "",
        "| panel | source accuracy | Q accuracy | agreement | Q NLL (bits) | KL (bits) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for panel in ("calibration", "test"):
        row = result["panels"][panel]["behavior"]
        lines.append(
            f"| {panel} | {row['source_accuracy']:.3f} | {row['quantum_accuracy']:.3f} | "
            f"{row['source_quantum_argmax_agreement']:.3f} | {row['quantum_nll_bits']:.3f} | "
            f"{row['source_to_quantum_kl_bits']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out feature semantics",
            "",
            "| feature | concept family | active | median F1 | >=0.6 | energy in >=0.6 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for kind in ("quantum", "writer"):
        for family in ("all", "algorithmic", "input", "output"):
            summary = result["semantics"][kind]["summary"][family]
            median = summary["heldout_f1"]["median"]
            lines.append(
                f"| {kind} | {family} | {summary['active_features']} | "
                f"{0.0 if median is None else median:.3f} | "
                f"{summary['fraction_f1_at_least_0p6']:.3f} | "
                f"{summary['activation_energy_fraction_meeting_threshold']:.3f} |"
            )
    for family in ("all", "algorithmic", "input", "output"):
        summary = result["semantics"]["writer"]["conditional_summary"][family]
        median = summary["heldout_f1"]["median"]
        lines.append(
            f"| writer given owner Q | {family} | {summary['active_features']} | "
            f"{0.0 if median is None else median:.3f} | "
            f"{summary['fraction_f1_at_least_0p6']:.3f} | "
                f"{summary['activation_energy_fraction_meeting_threshold']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Rich held-out descriptions",
            "",
            "Atomic F1 above is retained for comparability. This section additionally",
            "selects a calibration-only atomic or two-literal AND/OR description with an",
            "explicit complexity penalty, then freezes it on test. Amplitude descriptions",
            "select one scalar relation by calibration absolute Pearson correlation.",
            "",
            "| feature | concept family | condition | median F1 | two-literal fraction |",
            "|---|---|---|---:|---:|",
        ]
    )
    for kind in ("quantum", "writer"):
        summary_key = "compound_summary"
        for family in ("all", "algorithmic", "input", "output"):
            summary = result["semantics"][kind][summary_key][family]
            median = summary["heldout_f1"]["median"]
            lines.append(
                f"| {kind} | {family} | global | "
                f"{0.0 if median is None else median:.3f} | "
                f"{summary['fraction_two_literal_descriptions']:.3f} |"
            )
    for family in ("all", "algorithmic", "input", "output"):
        summary = result["semantics"]["writer"]["conditional_compound_summary"][
            family
        ]
        median = summary["heldout_f1"]["median"]
        lines.append(
            f"| writer | {family} | given owner Q | "
            f"{0.0 if median is None else median:.3f} | "
            f"{summary['fraction_two_literal_descriptions']:.3f} |"
        )
    amplitude = result["semantics"]["writer"]["conditional_amplitude_summary"]
    lines.extend(
        [
            "",
            "- Conditional writer amplitude median held-out |Pearson r|: "
            f"{amplitude['heldout_abs_pearson_r']['median']:.3f}",
            "- The complete per-feature Boolean and amplitude records are in `audit.json`.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Sparse-output causal decomposition",
            "",
            "| scenario | Q accuracy | Q NLL (bits) | agreement |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in result["causal"]["scenarios"].items():
        lines.append(
            f"| {name} | {row['quantum_accuracy']:.3f} | "
            f"{row['quantum_nll_bits']:.3f} | "
            f"{row['quantum_source_argmax_agreement']:.3f} |"
        )
    normalized = result["causal"]["normalized_recovery"]
    no_direct_recovery = normalized["no_direct_rich_writer_only"]
    direct_recovery = normalized["direct_rich_writers_plus_attention"]
    lines.extend(
        [
            "",
            "- Rich writer-only recovery over embedding baseline: "
            + ("n/a" if no_direct_recovery is None else f"{no_direct_recovery:.3f}"),
            "- Rich writers-plus-attention recovery over attention baseline: "
            + ("n/a" if direct_recovery is None else f"{direct_recovery:.3f}"),
        ]
    )
    edge_composition = result["edge_composition"]
    edge_baseline = edge_composition["baseline"]
    edge_joint_zero = edge_composition["joint_zero"]
    edge_joint_shuffle = edge_composition["joint_message_shuffle"]
    lines.extend(
        [
            "",
            "## Global-edge composition",
            "",
            f"- Selected frozen edges: {len(edge_composition['individual_zero'])}",
            f"- Baseline Q accuracy: {edge_baseline['quantum_accuracy']:.3f}",
            "- Joint edge-zero accuracy change: "
            f"{edge_joint_zero['quantum_accuracy_change']:.3f}; NLL change: "
            f"{edge_joint_zero['quantum_nll_bits_change']:.3f} bits",
            "- Joint matched-message-shuffle accuracy change: "
            f"{edge_joint_shuffle['quantum_accuracy_change']:.3f}; NLL change: "
            f"{edge_joint_shuffle['quantum_nll_bits_change']:.3f} bits",
            "- Message shuffling tests parent-specific content inside the fitted "
            "Q-model; it does not prove source-mechanism identity.",
        ]
    )
    reader = result.get("attention_reader", {})
    if reader.get("enabled", True) is not False:
        selected = reader["writer_groups"]["rich_selected"]
        unselected = reader["writer_groups"]["rich_unselected"]
        lines.extend(
            [
                "",
                "## Frozen attention-reader audit",
                "",
                "Ordered top-two reader role and role/value signatures are selected "
                "jointly on calibration and evaluated unchanged on the disjoint test "
                "panel. Writer activation-rule F1 is conditional on owner-Q support; "
                "per-state routes avoid pooling different output subtypes and positions.",
                "",
                "| writer group | count | entropy (nats) | effective support | top-1 mass | top-2 mass | ordered role stability | ordered role-rule F1 | context fraction |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in (("rich selected", selected), ("rich unselected", unselected)):
            lines.append(
                f"| {name} | {row['writer_count']} | "
                f"{0.0 if row['median_test_attention_entropy_nats'] is None else row['median_test_attention_entropy_nats']:.3f} | "
                f"{0.0 if row['median_test_effective_support'] is None else row['median_test_effective_support']:.3f} | "
                f"{0.0 if row['median_test_top1_mass'] is None else row['median_test_top1_mass']:.3f} | "
                f"{0.0 if row['median_test_top2_mass'] is None else row['median_test_top2_mass']:.3f} | "
                f"{0.0 if row['median_test_ordered_top2_role_stability'] is None else row['median_test_ordered_top2_role_stability']:.3f} | "
                f"{0.0 if row['median_test_ordered_top2_role_activation_f1'] is None else row['median_test_ordered_top2_role_activation_f1']:.3f} | "
                f"{0.0 if row['median_test_context_fraction'] is None else row['median_test_context_fraction']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### Highest-energy frozen reader routes",
                "",
                "| writer | rich-selected | frozen ordered role pair | role-rule F1 | label-rule F1 | state routes | causal accuracy delta |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            reader["writers"],
            key=lambda item: -float(item["test_writer_energy"]),
        )[:12]:
            role_rule = row["ordered_top2_role_activation_rule"]
            label_rule = row["ordered_top2_label_activation_rule"]
            causal_effect = row.get("causal_ablation")
            role_f1 = "n/a" if role_rule is None else f"{role_rule['test']['f1']:.3f}"
            label_f1 = "n/a" if label_rule is None else f"{label_rule['test']['f1']:.3f}"
            causal_delta = (
                "n/a"
                if causal_effect is None
                else f"{causal_effect['delta_quantum_accuracy']:.3f}"
            )
            lines.append(
                f"| {row['name']} | {str(row['rich_selected_on_calibration']).lower()} | "
                f"{row['calibration_ordered_top2_roles'] or 'n/a'} | "
                f"{role_f1} | {label_f1} | "
                f"{len(row['conditional_ordered_top2_role_routes'])} | "
                f"{causal_delta} |"
            )
        lines.extend(
            [
                "",
                "| frozen reader intervention | Q accuracy | agreement | Q NLL (bits) | donor logit change | receiver logit change |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in reader["interventions"].items():
            swap = row.get("matched_swap", {})
            donor_change = (
                "n/a"
                if not swap
                else f"{swap['mean_donor_target_logit_change']:.3f}"
            )
            receiver_change = (
                "n/a"
                if not swap
                else f"{swap['mean_receiver_target_logit_change']:.3f}"
            )
            lines.append(
                f"| {name} | {row['quantum_accuracy']:.3f} | "
                f"{row['source_quantum_argmax_agreement']:.3f} | "
                f"{row['quantum_nll_bits']:.3f} | "
                f"{donor_change} | {receiver_change} |"
            )
        lines.extend(
            [
                "",
                "- `attention_reader.writers` contains joint ordered top-two role/value "
                "signatures, owner-conditional activation-rule F1, state-conditional "
                "routes, local/context scalar decomposition, and bounded causal deletion "
                "effects where available.",
                "- The swaps establish causal sensitivity inside the fitted Q-model; they "
                "do not establish that the source Transformer uses the same reader mechanism.",
            ]
        )
    else:
        lines.extend(["", "## Frozen attention-reader audit", "", reader["selection"]])
    writer_coverage = result["semantics"]["writer"]["coverage"]
    redundancy = result["semantics"]["writer"]["within_quantum_redundancy"]
    cluster_count = sum(
        len(bank["causal_signature_clusters"]) for bank in redundancy["banks"]
    )
    lines.extend(
        [
            "",
            "## Concept coverage",
            "",
            f"- Concepts audited: {len(result['concepts'])}",
            f"- Best single writer median held-out F1: "
            f"{writer_coverage['best_single_heldout_f1']['median']:.3f}",
            f"- Fraction of concepts with single-writer F1 >= 0.6: "
            f"{writer_coverage['fraction_single_f1_at_least_0p6']:.3f}",
            f"- Top-four union median held-out F1: "
            f"{writer_coverage['top4_union_heldout_f1']['median']:.3f}",
            f"- Within-quantum causal-logit signature clusters: {cluster_count}",
            "- `within_quantum_redundancy` in `audit.json` contains held-out support "
            "Jaccard/coactivation, decoder cosine, and bounded causal-logit signature "
            "cosine for every within-bank writer pair.",
            "",
            "See `writer_dashboards.json` for top and activation-quantile examples.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if int(args.min_concept_positives) <= 0:
        raise ValueError("min-concept-positives must be positive")
    if int(args.compound_top_concepts) <= 0:
        raise ValueError("compound-top-concepts must be positive")
    if float(args.compound_description_penalty) < 0.0:
        raise ValueError("compound-description-penalty must be non-negative")
    if int(args.redundancy_causal_examples) < 0:
        raise ValueError("redundancy-causal-examples must be non-negative")
    if int(args.attention_reader_examples) < 0:
        raise ValueError("attention-reader-examples must be non-negative")
    if int(args.attention_reader_min_events) <= 0:
        raise ValueError("attention-reader-min-events must be positive")
    if not -1.0 <= float(args.redundancy_cluster_cosine) <= 1.0:
        raise ValueError("redundancy-cluster-cosine must lie in [-1, 1]")
    if not 0.0 <= float(args.interpretability_f1) <= 1.0:
        raise ValueError("interpretability-f1 must lie in [0, 1]")
    attention_top_k_values = _parse_top_k_values(args.attention_top_k_values)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run_dir = args.run_dir.resolve()
    q_model_dir = args.q_model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config("quanta_discovery", args.config)
    print("loading task, source checkpoint, and Q-model", flush=True)
    (
        task,
        source,
        readout,
        modules,
        existences,
        edges,
        artifact_metadata,
    ) = _load_models(
        run_dir,
        q_model_dir,
        config,
        device=device,
        priority_dir=(
            None if args.priority_dir is None else args.priority_dir.resolve()
        ),
    )
    enforce_parent_gate_closure = not bool(
        artifact_metadata["global_edge_discovery"]
    )

    train_numbers = {int(example.number) for example in task.train}
    uncontaminated_eval = [
        example for example in task.eval if int(example.number) not in train_numbers
    ]
    calibration_examples, test_examples = _stratified_eval_split(
        uncontaminated_eval, seed=int(args.seed)
    )
    calibration_examples = _balanced_limit(
        calibration_examples,
        int(args.calibration_examples),
        seed=int(args.seed) + 1,
    )
    test_examples = _balanced_limit(
        test_examples, int(args.test_examples), seed=int(args.seed) + 2
    )
    print(
        f"collecting {len(calibration_examples)} calibration and "
        f"{len(test_examples)} held-out test examples",
        flush=True,
    )
    calibration = _collect_panel(
        source,
        readout,
        modules,
        task,
        calibration_examples,
        existences,
        edges,
        device=device,
        batch_size=int(args.batch_size),
        routing_mode=str(config.q_routing_mode),
        enforce_parent_gate_closure=enforce_parent_gate_closure,
    )
    test = _collect_panel(
        source,
        readout,
        modules,
        task,
        test_examples,
        existences,
        edges,
        device=device,
        batch_size=int(args.batch_size),
        routing_mode=str(config.q_routing_mode),
        enforce_parent_gate_closure=enforce_parent_gate_closure,
    )
    concept_names, calibration_concepts, test_concepts = _concept_matrices(
        calibration.events,
        test.events,
        minimum_positives=int(args.min_concept_positives),
    )
    continuous_names, calibration_continuous, test_continuous = _continuous_matrices(
        calibration.events, test.events
    )
    print(f"scoring {len(concept_names)} pre-specified concepts", flush=True)

    quantum_calibration = np.concatenate(calibration.quantum_scores, axis=1)
    quantum_test = np.concatenate(test.quantum_scores, axis=1)
    writer_calibration = np.concatenate(calibration.writer_scores, axis=1)
    writer_test = np.concatenate(test.writer_scores, axis=1)
    quantum_refs = _feature_refs(modules, kind="quantum")
    writer_refs = _feature_refs(modules, kind="writer")
    quantum_records = _fit_hypotheses(
        quantum_calibration,
        quantum_test,
        calibration_concepts,
        test_concepts,
        concept_names,
        quantum_refs,
    )
    writer_records = _fit_hypotheses(
        writer_calibration,
        writer_test,
        calibration_concepts,
        test_concepts,
        concept_names,
        writer_refs,
    )
    _add_conditional_writer_hypotheses(
        writer_records,
        writer_calibration,
        writer_test,
        calibration.quantum_scores,
        test.quantum_scores,
        calibration_concepts,
        test_concepts,
        concept_names,
    )
    quantum_compound = _fit_compound_hypotheses(
        quantum_calibration,
        quantum_test,
        calibration_concepts,
        test_concepts,
        concept_names,
        quantum_refs,
        top_concepts=int(args.compound_top_concepts),
        description_penalty=float(args.compound_description_penalty),
    )
    writer_compound = _fit_compound_hypotheses(
        writer_calibration,
        writer_test,
        calibration_concepts,
        test_concepts,
        concept_names,
        writer_refs,
        top_concepts=int(args.compound_top_concepts),
        description_penalty=float(args.compound_description_penalty),
    )
    for record, compound in zip(quantum_records, quantum_compound):
        record["compound_hypotheses"] = compound["hypotheses"]
    for record, compound in zip(writer_records, writer_compound):
        record["compound_hypotheses"] = compound["hypotheses"]
    _add_conditional_writer_compound_hypotheses(
        writer_records,
        writer_calibration,
        writer_test,
        calibration.quantum_scores,
        test.quantum_scores,
        calibration_concepts,
        test_concepts,
        concept_names,
        top_concepts=int(args.compound_top_concepts),
        description_penalty=float(args.compound_description_penalty),
    )
    quantum_amplitude = _fit_amplitude_hypotheses(
        quantum_calibration,
        quantum_test,
        calibration_continuous,
        test_continuous,
        continuous_names,
        quantum_refs,
    )
    writer_amplitude = _fit_amplitude_hypotheses(
        writer_calibration,
        writer_test,
        calibration_continuous,
        test_continuous,
        continuous_names,
        writer_refs,
    )
    for record, amplitude in zip(quantum_records, quantum_amplitude):
        record["amplitude_hypothesis"] = amplitude["hypothesis"]
    for record, amplitude in zip(writer_records, writer_amplitude):
        record["amplitude_hypothesis"] = amplitude["hypothesis"]
    _add_conditional_writer_amplitude_hypotheses(
        writer_records,
        writer_calibration,
        writer_test,
        calibration.quantum_scores,
        test.quantum_scores,
        calibration_continuous,
        test_continuous,
        continuous_names,
    )

    rng = np.random.default_rng(int(args.seed))
    shuffled_writer_records = _fit_hypotheses(
        writer_calibration[rng.permutation(len(writer_calibration))],
        writer_test[rng.permutation(len(writer_test))],
        calibration_concepts,
        test_concepts,
        concept_names,
        writer_refs,
    )
    shuffled_writer_compound = _fit_compound_hypotheses(
        writer_calibration[rng.permutation(len(writer_calibration))],
        writer_test[rng.permutation(len(writer_test))],
        calibration_concepts,
        test_concepts,
        concept_names,
        writer_refs,
        top_concepts=int(args.compound_top_concepts),
        description_penalty=float(args.compound_description_penalty),
    )
    for record, compound in zip(shuffled_writer_records, shuffled_writer_compound):
        record["compound_hypotheses"] = compound["hypotheses"]
    threshold = float(args.interpretability_f1)
    quantum_summary = {
        family: _feature_summary(
            quantum_records, family=family, threshold=threshold
        )
        for family in ("all", "algorithmic", "input", "output")
    }
    writer_summary = {
        family: _feature_summary(writer_records, family=family, threshold=threshold)
        for family in ("all", "algorithmic", "input", "output")
    }
    writer_conditional_summary = {
        family: _feature_summary(
            writer_records,
            family=family,
            threshold=threshold,
            hypothesis_key="conditional_on_owner",
        )
        for family in ("all", "algorithmic", "input", "output")
    }
    quantum_compound_summary = {
        family: _compound_feature_summary(
            quantum_records, family=family, threshold=threshold
        )
        for family in ("all", "algorithmic", "input", "output")
    }
    writer_compound_summary = {
        family: _compound_feature_summary(
            writer_records, family=family, threshold=threshold
        )
        for family in ("all", "algorithmic", "input", "output")
    }
    writer_conditional_compound_summary = {
        family: _compound_feature_summary(
            writer_records, family=family, threshold=threshold, conditional=True
        )
        for family in ("all", "algorithmic", "input", "output")
    }
    shuffled_summary = {
        family: _feature_summary(
            shuffled_writer_records, family=family, threshold=threshold
        )
        for family in ("all", "algorithmic", "input", "output")
    }
    quantum_coverage = _coverage(
        quantum_calibration,
        quantum_test,
        calibration_concepts,
        test_concepts,
        concept_names,
        quantum_refs,
    )
    writer_coverage = _coverage(
        writer_calibration,
        writer_test,
        calibration_concepts,
        test_concepts,
        concept_names,
        writer_refs,
    )

    counterfactual_examples = _balanced_limit(
        test_examples,
        int(args.counterfactual_examples),
        seed=int(args.seed) + 3,
    )
    print(
        f"running sparse/dense path and {int(args.causal_writers)} writer "
        f"counterfactuals on {len(counterfactual_examples)} examples",
        flush=True,
    )
    causal = _evaluate_counterfactuals(
        source,
        readout,
        modules,
        task,
        counterfactual_examples,
        existences,
        edges,
        writer_records,
        device=device,
        batch_size=int(args.batch_size),
        routing_mode=str(config.q_routing_mode),
        enforce_parent_gate_closure=enforce_parent_gate_closure,
        threshold=threshold,
        causal_writers=int(args.causal_writers),
    )
    print(
        f"running global-edge zero and message-shuffle controls on "
        f"{len(counterfactual_examples)} examples",
        flush=True,
    )
    edge_composition = _evaluate_edge_composition(
        source,
        readout,
        modules,
        task,
        counterfactual_examples,
        existences,
        edges,
        device=device,
        batch_size=int(args.batch_size),
        routing_mode=str(config.q_routing_mode),
        enforce_parent_gate_closure=enforce_parent_gate_closure,
        seed=int(args.seed) + 20,
    )
    redundancy_examples = (
        _balanced_limit(
            test_examples,
            int(args.redundancy_causal_examples),
            seed=int(args.seed) + 4,
        )
        if int(args.redundancy_causal_examples) > 0
        else []
    )
    if redundancy_examples:
        print(
            f"running per-writer causal logit signatures on {len(redundancy_examples)} "
            "held-out examples",
            flush=True,
        )
        causal_signatures = _writer_causal_logit_signatures(
            source,
            readout,
            modules,
            task,
            redundancy_examples,
            existences,
            edges,
            writer_records,
            device=device,
            batch_size=int(args.batch_size),
            routing_mode=str(config.q_routing_mode),
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
    else:
        causal_signatures = {}
    writer_redundancy = _within_quantum_redundancy(
        writer_calibration,
        writer_test,
        writer_refs,
        modules,
        causal_signatures,
        cluster_cosine=float(args.redundancy_cluster_cosine),
    )
    attention_reader: dict[str, Any]
    reader_calibration_examples: list[Any] = []
    reader_test_examples: list[Any] = []
    if int(args.attention_reader_examples) > 0:
        reader_calibration_examples = _balanced_limit(
            calibration_examples,
            int(args.attention_reader_examples),
            seed=int(args.seed) + 5,
        )
        reader_test_examples = _balanced_limit(
            test_examples,
            int(args.attention_reader_examples),
            seed=int(args.seed) + 6,
        )
        print(
            f"running frozen attention-reader audit on {len(reader_calibration_examples)} "
            f"calibration and {len(reader_test_examples)} test examples",
            flush=True,
        )
        reader_calibration = _collect_attention_reader_panel(
            source,
            readout,
            modules,
            task,
            reader_calibration_examples,
            existences,
            edges,
            device=device,
            batch_size=int(args.batch_size),
            routing_mode=str(config.q_routing_mode),
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        reader_test = _collect_attention_reader_panel(
            source,
            readout,
            modules,
            task,
            reader_test_examples,
            existences,
            edges,
            device=device,
            batch_size=int(args.batch_size),
            routing_mode=str(config.q_routing_mode),
            enforce_parent_gate_closure=enforce_parent_gate_closure,
        )
        attention_reader = _attention_reader_summary(
            reader_calibration,
            reader_test,
            writer_records,
            modules,
            threshold=threshold,
            minimum_events=int(args.attention_reader_min_events),
        )
        _attach_reader_causal_effects(attention_reader, causal)
        print("running frozen attention top-k and swap controls", flush=True)
        interventions = {
            "softmax": _attention_intervention_behavior(
                source,
                readout,
                modules,
                task,
                reader_test_examples,
                existences,
                edges,
                device=device,
                batch_size=int(args.batch_size),
                routing_mode=str(config.q_routing_mode),
                enforce_parent_gate_closure=enforce_parent_gate_closure,
                mode="softmax",
                seed=int(args.seed) + 7,
            ),
            "causal_prefix_position_shuffle_frozen_state": _attention_intervention_behavior(
                source,
                readout,
                modules,
                task,
                reader_test_examples,
                existences,
                edges,
                device=device,
                batch_size=int(args.batch_size),
                routing_mode=str(config.q_routing_mode),
                enforce_parent_gate_closure=enforce_parent_gate_closure,
                mode="position_shuffle",
                seed=int(args.seed) + 8,
            ),
            "matched_value_swap_frozen_state": _attention_intervention_behavior(
                source,
                readout,
                modules,
                task,
                reader_test_examples,
                existences,
                edges,
                device=device,
                batch_size=int(args.batch_size),
                routing_mode=str(config.q_routing_mode),
                enforce_parent_gate_closure=enforce_parent_gate_closure,
                mode="value_swap",
                seed=int(args.seed) + 9,
            ),
            "matched_local_score_swap_frozen_state": _attention_intervention_behavior(
                source,
                readout,
                modules,
                task,
                reader_test_examples,
                existences,
                edges,
                device=device,
                batch_size=int(args.batch_size),
                routing_mode=str(config.q_routing_mode),
                enforce_parent_gate_closure=enforce_parent_gate_closure,
                mode="local_swap",
                seed=int(args.seed) + 10,
            ),
        }
        for top_k in attention_top_k_values:
            interventions[f"top_{top_k}_renormalized"] = _attention_intervention_behavior(
                source,
                readout,
                modules,
                task,
                reader_test_examples,
                existences,
                edges,
                device=device,
                batch_size=int(args.batch_size),
                routing_mode=str(config.q_routing_mode),
                enforce_parent_gate_closure=enforce_parent_gate_closure,
                mode="top_k",
                seed=int(args.seed) + 11 + top_k,
                top_k=top_k,
            )
        attention_reader["interventions"] = interventions
        attention_reader["intervention_contract"] = (
            "Top-k uses the frozen checkpoint with masked-renormalized attention. "
            "Position shuffle permutes only causal source positions. Value and local "
            "swaps match examples by digit length and output-token count, freeze the "
            "receiver's quantum gates and writer masks, and report donor versus receiver "
            "target-logit movement."
        )
    else:
        attention_reader = {
            "enabled": False,
            "selection": "Pass --attention-reader-examples > 0 to run this diagnostic.",
        }

    result = {
        "method": (
            "heldout_exact_semantics_compound_amplitude_sparse_output_and_"
            "global_edge_counterfactuals_v3"
        ),
        "source_run": str(run_dir),
        "q_model_dir": str(q_model_dir),
        "config": str(args.config.resolve()),
        "split": {
            "source_eval_examples": len(task.eval),
            "train_overlaps_excluded": len(task.eval) - len(uncontaminated_eval),
            "uncontaminated_eval_examples": len(uncontaminated_eval),
            "calibration_examples": len(calibration_examples),
            "test_examples": len(test_examples),
            "calibration_events": len(calibration.events),
            "test_events": len(test.events),
            "disjoint_numbers": not bool(
                {int(example.number) for example in calibration_examples}
                & {int(example.number) for example in test_examples}
            ),
            "seed": int(args.seed),
            "minimum_concept_positives_per_panel": int(
                args.min_concept_positives
            ),
            "compound_top_concepts": int(args.compound_top_concepts),
            "compound_description_penalty": float(args.compound_description_penalty),
            "redundancy_causal_examples": len(redundancy_examples),
            "attention_reader_calibration_examples": len(reader_calibration_examples),
            "attention_reader_test_examples": len(reader_test_examples),
            "attention_top_k_values": list(attention_top_k_values),
            "attention_reader_min_events": int(args.attention_reader_min_events),
        },
        "concepts": concept_names,
        "artifact": {
            "quantum_counts_by_layer": artifact_metadata[
                "quantum_counts_by_layer"
            ],
            "writer_counts_by_layer": artifact_metadata["writer_counts_by_layer"],
            "routing_mode": str(config.q_routing_mode),
            "edge_discovery": str(config.q_edge_discovery),
            "enforce_parent_gate_closure": enforce_parent_gate_closure,
            "attention_direct_residual": bool(config.q_attention_direct_residual),
            "writer_sparsity": str(config.q_writer_sparsity),
            "writer_l0_target": float(config.q_writer_l0_target),
        },
        "panels": {
            "calibration": {
                "behavior": calibration.behavior,
                "path_energy_by_layer": calibration.path_energy,
            },
            "test": {
                "behavior": test.behavior,
                "path_energy_by_layer": test.path_energy,
            },
        },
        "semantics": {
            "quantum": {
                "summary": quantum_summary,
                "compound_summary": quantum_compound_summary,
                "amplitude_summary": _amplitude_summary(quantum_records),
                "features": quantum_records,
                "coverage": quantum_coverage,
            },
            "writer": {
                "summary": writer_summary,
                "conditional_summary": writer_conditional_summary,
                "compound_summary": writer_compound_summary,
                "conditional_compound_summary": writer_conditional_compound_summary,
                "amplitude_summary": _amplitude_summary(writer_records),
                "conditional_amplitude_summary": _amplitude_summary(
                    writer_records, conditional=True
                ),
                "features": writer_records,
                "coverage": writer_coverage,
                "row_permutation_control": {
                    "atomic": shuffled_summary,
                    "compound": {
                        family: _compound_feature_summary(
                            shuffled_writer_records, family=family, threshold=threshold
                        )
                        for family in ("all", "algorithmic", "input", "output")
                    },
                },
                "within_quantum_redundancy": writer_redundancy,
            },
        },
        "causal": causal,
        "edge_composition": edge_composition,
        "attention_reader": attention_reader,
        "claim_boundary": (
            "Causal interventions establish use inside the fitted Q-model, not "
            "identity with a unique or canonical decomposition of the source model."
        ),
    }
    dashboards = _writer_dashboards(
        writer_calibration,
        calibration.events,
        writer_records,
        top_examples=int(args.top_examples),
    )
    result = _json_ready(result)
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "writer_dashboards.json").write_text(
        json.dumps(_json_ready(dashboards), indent=2) + "\n"
    )
    (output_dir / "report.md").write_text(_markdown_report(result))
    print(output_dir / "report.md", flush=True)


if __name__ == "__main__":
    main()
