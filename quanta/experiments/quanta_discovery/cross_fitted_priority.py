"""Cross-fitted finite-panel priority estimators and calibration diagnostics.

The canonical exact-GD discovery path remains unchanged. Sampled directions
reproduce estimator-calibration controls; complete grouped complements provide
the deterministic large-panel production estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class CrossFittedExampleSplit:
    """Disjoint source-example indices for two directions and a score panel."""

    direction_a: np.ndarray
    direction_b: np.ndarray
    score: np.ndarray


@dataclass(frozen=True)
class GroupedCrossFitFold:
    """One score fold and its complete canonical-number complement."""

    score: np.ndarray
    direction: np.ndarray


def digit_stratified_disjoint_split(
    numbers: Sequence[int],
    *,
    direction_examples: int,
    score_examples: int,
    seed: int,
) -> CrossFittedExampleSplit:
    """Draw disjoint, proportionally digit-stratified source-example subsets.

    Each stratum is shuffled independently, then its members are allocated to
    the three groups in proportion to their requested sizes.  This preserves
    the source digit mixture while ensuring that no scored event contributes to
    either direction estimate.
    """

    requested = (int(direction_examples), int(direction_examples), int(score_examples))
    if any(value <= 0 for value in requested):
        raise ValueError("direction_examples and score_examples must be positive")
    total = sum(requested)
    if total > len(numbers):
        raise ValueError(
            "cross-fitted direction and score examples must be disjoint and "
            "fit within the available training examples"
        )
    generator = np.random.default_rng(int(seed))
    strata: dict[int, list[int]] = {}
    for index, number in enumerate(numbers):
        strata.setdefault(len(str(abs(int(number)))), []).append(index)

    remaining_by_stratum: dict[int, list[int]] = {}
    for digit, members in strata.items():
        shuffled = np.asarray(members, dtype=np.int64)
        generator.shuffle(shuffled)
        remaining_by_stratum[digit] = shuffled.tolist()
    groups: list[list[int]] = []
    # Draw the three groups sequentially.  Each draw uses largest-remainder
    # proportional allocation over the currently unused examples, preserving
    # the source digit mixture while enforcing disjointness.
    for requested_count in requested:
        available = sum(len(values) for values in remaining_by_stratum.values())
        ideal = {
            digit: requested_count * len(values) / available
            for digit, values in remaining_by_stratum.items()
        }
        allocation = {digit: min(int(np.floor(value)), len(remaining_by_stratum[digit])) for digit, value in ideal.items()}
        while sum(allocation.values()) < requested_count:
            candidates = [digit for digit, values in remaining_by_stratum.items() if allocation[digit] < len(values)]
            digit = max(candidates, key=lambda value: (ideal[value] - allocation[value], -value))
            allocation[digit] += 1
        group: list[int] = []
        for digit in sorted(allocation):
            count = allocation[digit]
            group.extend(remaining_by_stratum[digit][:count])
            del remaining_by_stratum[digit][:count]
        groups.append(group)
    return CrossFittedExampleSplit(
        direction_a=np.sort(np.asarray(groups[0], dtype=np.int64)),
        direction_b=np.sort(np.asarray(groups[1], dtype=np.int64)),
        score=np.sort(np.asarray(groups[2], dtype=np.int64)),
    )


def digit_stratified_group_folds(
    numbers: Sequence[int], *, folds: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Partition occurrences while keeping equal-number duplicates together.

    ``digits_wise_uniform`` samples with replacement.  Grouping by number keeps
    an identical input out of the direction estimate used to score it, while a
    greedy per-digit assignment keeps occurrence counts approximately balanced.
    """

    if int(folds) < 2:
        raise ValueError("folds must be at least 2")
    strata: dict[int, dict[int, list[int]]] = {}
    for index, raw_number in enumerate(numbers):
        number = int(raw_number)
        digit = len(str(abs(number)))
        strata.setdefault(digit, {}).setdefault(number, []).append(index)
    generator = np.random.default_rng(int(seed))
    result: list[list[int]] = [[] for _ in range(int(folds))]
    for digit, grouped in sorted(strata.items()):
        groups = list(grouped.values())
        if len(groups) < int(folds):
            raise ValueError(
                f"digit stratum {digit} has fewer unique numbers than folds"
            )
        generator.shuffle(groups)
        groups.sort(key=len, reverse=True)
        loads = [0] * int(folds)
        for group in groups:
            target = min(range(int(folds)), key=lambda value: (loads[value], value))
            result[target].extend(group)
            loads[target] += len(group)
    return tuple(np.sort(np.asarray(value, dtype=np.int64)) for value in result)


def digit_stratified_group_crossfit(
    numbers: Sequence[int], *, folds: int, seed: int
) -> tuple[GroupedCrossFitFold, ...]:
    """Return grouped score folds and their complete out-of-fold directions."""

    score_folds = digit_stratified_group_folds(
        numbers, folds=int(folds), seed=int(seed)
    )
    all_indices = np.arange(len(numbers), dtype=np.int64)
    result: list[GroupedCrossFitFold] = []
    for score_indices in score_folds:
        score_numbers = {int(numbers[int(index)]) for index in score_indices}
        direction_indices = np.asarray(
            [
                index
                for index in all_indices
                if int(numbers[int(index)]) not in score_numbers
            ],
            dtype=np.int64,
        )
        if len(direction_indices) == 0:
            raise ValueError("every score fold must have a non-empty complement")
        result.append(
            GroupedCrossFitFold(
                score=np.asarray(score_indices, dtype=np.int64),
                direction=direction_indices,
            )
        )
    return tuple(result)


def _stratified_sample_indices(
    available: np.ndarray,
    numbers: Sequence[int],
    *,
    count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample occurrences without replacement, proportionally by digit length."""

    pool = np.asarray(available, dtype=np.int64)
    if not 0 < int(count) <= len(pool):
        raise ValueError("direction sample must fit in the available complement")
    strata: dict[int, list[int]] = {}
    for index in pool:
        strata.setdefault(len(str(abs(int(numbers[int(index)])))), []).append(int(index))
    ideal = {
        digit: int(count) * len(values) / len(pool)
        for digit, values in strata.items()
    }
    allocation = {
        digit: min(int(np.floor(value)), len(strata[digit]))
        for digit, value in ideal.items()
    }
    while sum(allocation.values()) < int(count):
        candidates = [
            digit
            for digit in strata
            if allocation[digit] < len(strata[digit])
        ]
        digit = max(
            candidates,
            key=lambda value: (ideal[value] - allocation[value], -value),
        )
        allocation[digit] += 1
    selected: list[int] = []
    for digit in sorted(strata):
        values = np.asarray(strata[digit], dtype=np.int64)
        generator.shuffle(values)
        selected.extend(values[: allocation[digit]].tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def analyze_cross_fitted_event_priority(
    model: Any,
    task: Any,
    train_panel: Any,
    eval_panel: Any,
    trajectory: Any,
    *,
    folds: int,
    direction_examples: int,
    direction_replicates: int,
    split_seed: int,
    full_complement: bool = False,
    max_components: int,
    alternating_steps: int,
    minimum_component_gain_fraction: float,
    save_dir: str,
    device: torch.device,
    on_checkpoint: Callable[[int, int, int, dict[str, float]], None] | None = None,
) -> Any:
    """Approximate population-GD credit with grouped cross-fitted directions.

    The source trajectory remains exact full-batch GD.  At each checkpoint,
    every training occurrence is scored against direction estimates drawn only
    from other canonical-number folds. Sampled mode averages disjoint replicate
    directions. Full-complement mode uses every out-of-fold occurrence once.
    Both remove event self-inclusion while retaining an artifact compatible
    with the canonical binary factorization and Q-model training path.
    """

    from .priority import (
        PriorityAnalysis,
        _binary_priority_factorization,
        _event_gradient_inner_products,
        _factorization_summary,
        _infer_binary_support,
        _population_block_gradients,
        write_priority_artifacts,
    )
    from .trajectory import build_prediction_event_panel, checkpoint_path

    checkpoint_steps = np.asarray(trajectory.checkpoint_steps, dtype=np.int64)
    learning_rates = np.asarray(trajectory.learning_rates, dtype=np.float64)
    interval_mass = np.asarray(
        [
            learning_rates[int(start) : int(end)].sum()
            for start, end in zip(checkpoint_steps[:-1], checkpoint_steps[1:])
        ],
        dtype=np.float64,
    )
    numbers = [int(example.number) for example in task.train]
    crossfit_folds = digit_stratified_group_crossfit(
        numbers, folds=int(folds), seed=int(split_seed)
    )
    generator = np.random.default_rng(int(split_seed) + 17_171)
    direction_sets: list[tuple[np.ndarray, ...]] = []
    direction_batches: list[tuple[Any, ...]] = []
    score_batches: list[Any] = []
    score_panels: list[Any] = []
    score_masks: list[np.ndarray] = []
    panel_example_indices = np.asarray(train_panel.example_indices, dtype=np.int64)
    for fold in crossfit_folds:
        score_indices = fold.score
        available = fold.direction
        if full_complement:
            replicas = [available]
        else:
            remaining = available.copy()
            replicas = []
            for _ in range(int(direction_replicates)):
                selected = _stratified_sample_indices(
                    remaining,
                    numbers,
                    count=int(direction_examples),
                    generator=generator,
                )
                replicas.append(selected)
                remaining = np.setdiff1d(remaining, selected, assume_unique=True)
        frozen_replicas = tuple(replicas)
        direction_sets.append(frozen_replicas)
        direction_batches.append(
            tuple(
                task.encode_examples(
                    [task.train[int(index)] for index in direction_indices],
                    device=device,
                )
                for direction_indices in frozen_replicas
            )
        )
        examples = [task.train[int(index)] for index in score_indices]
        score_batches.append(task.encode_examples(examples, device=device))
        score_panels.append(
            build_prediction_event_panel(task, split="train", examples=examples)
        )
        score_masks.append(np.isin(panel_example_indices, score_indices))
    if not np.all(np.sum(np.stack(score_masks), axis=0) == 1):
        raise RuntimeError("cross-fitted score folds do not partition training events")

    eval_batch = task.encode_examples(list(task.eval_examples), device=device)
    train_scores: list[np.ndarray] = []
    eval_scores: list[np.ndarray] = []
    disagreement: list[float] = []
    for checkpoint_index, optimizer_step in enumerate(checkpoint_steps[:-1]):
        state = torch.load(
            checkpoint_path(save_dir, trajectory, checkpoint_index),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        checkpoint_train = np.empty(
            (len(model.layers), len(train_panel)), dtype=np.float32
        )
        checkpoint_eval: list[np.ndarray] = []
        checkpoint_disagreement: list[float] = []
        for score_batch, score_panel, score_mask, replicas in zip(
            score_batches, score_panels, score_masks, direction_batches
        ):
            replica_fields: list[np.ndarray] = []
            for direction_batch in replicas:
                directions = _population_block_gradients(model, task, direction_batch)
                field = _event_gradient_inner_products(
                    model, score_batch, score_panel, directions
                ).cpu().numpy()
                replica_fields.append(field)
                checkpoint_eval.append(
                    _event_gradient_inner_products(
                        model, eval_batch, eval_panel, directions
                    ).cpu().numpy()
                )
            averaged = np.mean(replica_fields, axis=0)
            checkpoint_train[:, score_mask] = averaged
            if len(replica_fields) > 1:
                checkpoint_disagreement.append(
                    float(np.sqrt(np.mean(np.square(replica_fields[0] - replica_fields[1]))))
                )
        if full_complement:
            pairwise_disagreement = [
                float(
                    np.sqrt(
                        np.mean(
                            np.square(checkpoint_eval[left] - checkpoint_eval[right])
                        )
                    )
                )
                for left in range(len(checkpoint_eval))
                for right in range(left + 1, len(checkpoint_eval))
            ]
            checkpoint_disagreement.extend(pairwise_disagreement)
        train_scores.append(checkpoint_train)
        eval_scores.append(np.mean(checkpoint_eval, axis=0))
        disagreement.append(
            float(np.mean(checkpoint_disagreement, dtype=np.float64))
            if checkpoint_disagreement
            else 0.0
        )
        if on_checkpoint is not None:
            on_checkpoint(
                checkpoint_index + 1,
                len(checkpoint_steps) - 1,
                int(optimizer_step),
                {"priority_direction_disagreement_rmse": disagreement[-1]},
            )

    train_credit = np.transpose(np.stack(train_scores), (1, 2, 0))
    eval_credit = np.transpose(np.stack(eval_scores), (1, 2, 0))
    train_credit *= interval_mass[None, None]
    eval_credit *= interval_mass[None, None]
    train_weights = np.full(
        train_credit.shape[1], 1.0 / train_credit.shape[1], dtype=np.float64
    )
    eval_weights = np.full(
        eval_credit.shape[1], 1.0 / eval_credit.shape[1], dtype=np.float64
    )
    train_supports: list[np.ndarray] = []
    eval_supports: list[np.ndarray] = []
    temporal_priority: list[np.ndarray] = []
    train_residuals: list[np.ndarray] = []
    eval_residuals: list[np.ndarray] = []
    layer_summaries: list[dict[str, Any]] = []
    for layer in range(train_credit.shape[0]):
        fit = _binary_priority_factorization(
            train_credit[layer],
            train_weights,
            max_components=int(max_components),
            alternating_steps=int(alternating_steps),
            minimum_component_gain_fraction=float(
                minimum_component_gain_fraction
            ),
        )
        curves = fit["temporal_priority"]
        train_support = fit["support"]
        train_residual = fit["residual"]
        eval_support, eval_residual = _infer_binary_support(
            eval_credit[layer], curves
        )
        train_supports.append(train_support)
        eval_supports.append(eval_support)
        temporal_priority.append(curves)
        train_residuals.append(train_residual)
        eval_residuals.append(eval_residual)
        layer_summaries.append(
            {
                "host_layer": int(layer),
                "candidate_slot_ceiling": int(max_components),
                "nonzero_candidate_slots": int(curves.shape[0]),
                "count_status": "retained by source-relative marginal gain",
                "train": _factorization_summary(
                    train_credit[layer], train_residual, train_support, curves,
                    train_weights, checkpoint_steps[1:],
                ),
                "eval_diagnostic": _factorization_summary(
                    eval_credit[layer], eval_residual, eval_support, curves,
                    eval_weights, checkpoint_steps[1:],
                ),
                "candidate_improvements": fit["improvements"],
                "candidate_gain_fractions_of_original_sse": fit[
                    "gain_fractions_of_original_sse"
                ],
            }
        )
    summary = {
        "method": (
            "cross_fitted_full_complement_priority_v1"
            if full_complement
            else "cross_fitted_checkpoint_priority_v1"
        ),
        "scientific_status": (
            "Exact full-batch source trajectory with complete, grouped cross-fitted "
            "population-gradient directions. Every training event is scored using "
            "all occurrences outside its canonical-number fold."
            if full_complement
            else "Exact full-batch source trajectory with sampled, grouped cross-fitted "
            "population-gradient directions. Every training event is scored out "
            "of canonical-number fold; direction replicates are averaged."
        ),
        "priority_identity": (
            "cross-fitted full-complement approximation; exact global population-gradient "
            "closure is not asserted"
            if full_complement
            else "approximate; exact population-gradient closure is not asserted"
        ),
        "priority_crossfit_folds": int(folds),
        "priority_direction_mode": (
            "full_crossfit_complement" if full_complement else "sampled_replicates"
        ),
        "priority_direction_examples_by_fold": [
            [int(len(indices)) for indices in replicas]
            for replicas in direction_sets
        ],
        "priority_direction_examples": (
            None if full_complement else int(direction_examples)
        ),
        "priority_direction_replicates": (
            1 if full_complement else int(direction_replicates)
        ),
        "priority_split_seed": int(split_seed),
        "direction_disagreement_scope": (
            "pairwise full-complement directions evaluated on shared eval events"
            if full_complement
            else "within-fold sampled direction replicates evaluated on scored train events"
        ),
        "mean_direction_disagreement_rmse": float(np.mean(disagreement)),
        "minimum_component_gain_fraction_of_original_sse": float(
            minimum_component_gain_fraction
        ),
        "layers": layer_summaries,
    }
    result = PriorityAnalysis(
        checkpoint_steps=checkpoint_steps,
        interval_steps=checkpoint_steps[1:],
        interval_learning_mass=interval_mass,
        train_credit=train_credit.astype(np.float32),
        eval_credit=eval_credit.astype(np.float32),
        train_supports=tuple(train_supports),
        eval_supports=tuple(eval_supports),
        temporal_priority=tuple(temporal_priority),
        train_residuals=tuple(train_residuals),
        eval_residuals=tuple(eval_residuals),
        summary=summary,
    )
    write_priority_artifacts(save_dir, train_panel, eval_panel, result)
    return result


def field_pearson(exact: np.ndarray, estimate: np.ndarray) -> float:
    """Pearson correlation over all event-by-time cells of one host layer."""

    x = np.asarray(exact, dtype=np.float64).reshape(-1)
    y = np.asarray(estimate, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("exact and estimated fields must have matching shapes")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return 0.0 if denominator == 0.0 else float((x @ y) / denominator)


def best_support_matching(
    exact: np.ndarray, estimate: np.ndarray) -> list[tuple[int, int, float, float]]:
    """Greedily match binary factors by Jaccard, with deterministic ties.

    Factorization order is already greedy by source-relative marginal gain, so
    greedy matching is appropriate for the small calibration basis and avoids
    adding a SciPy dependency solely for reporting diagnostics.
    """

    reference = np.asarray(exact, dtype=bool)
    observed = np.asarray(estimate, dtype=bool)
    pairs: list[tuple[float, int, int, float]] = []
    for left in range(reference.shape[1]):
        for right in range(observed.shape[1]):
            intersection = int((reference[:, left] & observed[:, right]).sum())
            union = int((reference[:, left] | observed[:, right]).sum())
            jaccard = intersection / max(union, 1)
            f1 = 2.0 * intersection / max(
                int(reference[:, left].sum() + observed[:, right].sum()), 1
            )
            pairs.append((jaccard, left, right, f1))
    selected: list[tuple[int, int, float, float]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for jaccard, left, right, f1 in sorted(
        pairs, key=lambda value: (-value[0], value[1], value[2])
    ):
        if left not in used_left and right not in used_right:
            selected.append((left, right, jaccard, f1))
            used_left.add(left)
            used_right.add(right)
    return selected
