from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DependencyProbeExample:
    code_index: int
    code: int
    candidates: list[tuple[int, ...]]
    circuit_signals: torch.Tensor
    model_probability: torch.Tensor
    labels: torch.Tensor


def build_dependency_probe_example(
    *,
    code_index: int,
    code: int,
    parents: list[int],
    batch: dict[str, torch.Tensor],
    logits: torch.Tensor,
    output_slots: torch.Tensor,
) -> DependencyProbeExample:
    candidates = declared_parent_subsets(parents)
    model_probability = torch.sigmoid(
        _output_logit(logits, batch["slot_ids"], int(output_slots[code_index]))
    )
    parent_probabilities = {
        int(parent): torch.sigmoid(
            _output_logit(logits, batch["slot_ids"], int(output_slots[int(parent)]))
        )
        for parent in parents
    }
    local_nand = (
        ~batch["local_bits"][:, int(code_index)].bool().all(dim=1)
    ).to(dtype=torch.float32)
    signals = [
        _candidate_circuit_signal(candidate, parent_probabilities, local_nand)
        for candidate in candidates
    ]
    return DependencyProbeExample(
        code_index=int(code_index),
        code=int(code),
        candidates=candidates,
        circuit_signals=torch.stack(signals).detach().cpu(),
        model_probability=model_probability.detach().cpu(),
        labels=batch["true_values"][:, int(code_index)].to(dtype=torch.float32).detach().cpu(),
    )


def declared_parent_subsets(parents: list[int]) -> list[tuple[int, ...]]:
    ordered = sorted(set(int(parent) for parent in parents))
    return [
        tuple(combination)
        for size in range(len(ordered) + 1)
        for combination in itertools.combinations(ordered, size)
    ]


def evaluate_dependency_probes(
    examples: list[DependencyProbeExample],
    *,
    probe_steps: int = 100,
    probe_lr: float = 0.1,
    lambda_l2: float = 1e-4,
) -> tuple[dict[int, float], list[dict]]:
    if not examples:
        return {}, []
    n_examples = {int(example.circuit_signals.shape[1]) for example in examples}
    if len(n_examples) != 1:
        raise ValueError("Dependency probe examples must use the same sample count.")
    sample_count = n_examples.pop()
    if sample_count < 2:
        raise ValueError("Dependency probes require at least two examples.")

    candidate_signals = torch.cat(
        [example.circuit_signals for example in examples],
        dim=0,
    ).to(dtype=torch.float32)
    soft_targets = torch.cat(
        [
            example.model_probability.unsqueeze(0).expand(len(example.candidates), -1)
            for example in examples
        ],
        dim=0,
    ).to(dtype=torch.float32)
    split = max(1, sample_count // 2)
    selection_scores = _fit_scalar_probe_grid(
        candidate_signals[:, :split],
        soft_targets[:, :split],
        candidate_signals[:, split:],
        soft_targets[:, split:],
        probe_steps=probe_steps,
        probe_lr=probe_lr,
        lambda_l2=lambda_l2,
    )

    selected_signals = []
    selected_labels = []
    records = []
    offset = 0
    for example in examples:
        count = len(example.candidates)
        local_scores = selection_scores[offset : offset + count]
        best_local = min(
            range(count),
            key=lambda index: (
                float(local_scores[index]),
                len(example.candidates[index]),
                example.candidates[index],
            ),
        )
        selected_signals.append(example.circuit_signals[best_local])
        selected_labels.append(example.labels)
        records.append(
            {
                "code": int(example.code),
                "best_set": list(example.candidates[best_local]),
                "best_score": float(local_scores[best_local]),
                "candidate_scores": {
                    _candidate_label(candidate): float(local_scores[index])
                    for index, candidate in enumerate(example.candidates)
                },
            }
        )
        offset += count

    selected_signal_grid = torch.stack(selected_signals).to(dtype=torch.float32)
    label_grid = torch.stack(selected_labels).to(dtype=torch.float32)
    ground_truth_losses = _fit_scalar_probe_grid(
        selected_signal_grid[:, :split],
        label_grid[:, :split],
        selected_signal_grid[:, split:],
        label_grid[:, split:],
        probe_steps=probe_steps,
        probe_lr=probe_lr,
        lambda_l2=lambda_l2,
    )
    losses = {}
    for record, example, loss in zip(records, examples, ground_truth_losses):
        record["ground_truth_loss"] = float(loss)
        losses[int(example.code_index)] = float(loss)
    return losses, records


def _fit_scalar_probe_grid(
    train_signals: torch.Tensor,
    train_targets: torch.Tensor,
    eval_signals: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    probe_steps: int,
    probe_lr: float,
    lambda_l2: float,
) -> torch.Tensor:
    weights = torch.zeros(train_signals.shape[0], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(train_signals.shape[0], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=float(probe_lr))
    for _ in range(int(probe_steps)):
        optimizer.zero_grad()
        train_logits = train_signals * weights.unsqueeze(1) + bias.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(
            train_logits,
            train_targets,
            reduction="none",
        ).mean(dim=1)
        if lambda_l2:
            loss = loss + float(lambda_l2) * weights.square()
        loss.mean().backward()
        optimizer.step()
    with torch.no_grad():
        eval_logits = eval_signals * weights.unsqueeze(1) + bias.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(
            eval_logits,
            eval_targets,
            reduction="none",
        ).mean(dim=1)


def _output_logit(
    logits: torch.Tensor,
    slot_ids: torch.Tensor,
    output_slot: int,
) -> torch.Tensor:
    slot_mask = slot_ids == int(output_slot)
    if not bool(slot_mask.any(dim=1).all().item()):
        raise ValueError(f"Output slot {output_slot} is missing from a dependency probe batch.")
    logit_difference = logits[..., 1] - logits[..., 0]
    return (logit_difference * slot_mask.to(dtype=logit_difference.dtype)).sum(dim=1)


def _candidate_circuit_signal(
    candidate: tuple[int, ...],
    parent_probabilities: dict[int, torch.Tensor],
    local_nand: torch.Tensor,
) -> torch.Tensor:
    if not candidate:
        return local_nand
    parent_probability = torch.stack(
        [parent_probabilities[int(parent)] for parent in candidate],
        dim=1,
    )
    parent_nand = 1.0 - parent_probability.prod(dim=1)
    return 1.0 - parent_nand * local_nand


def _candidate_label(candidate: tuple[int, ...]) -> str:
    return "{}" if not candidate else "{" + ",".join(str(node) for node in candidate) + "}"
