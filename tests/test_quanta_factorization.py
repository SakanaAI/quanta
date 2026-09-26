from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quanta.experiments.quanta_discovery.factorization import (
    _close_supports,
    _close_temporal_priority,
    _select_edges,
)
from quanta.experiments.quanta_discovery.cross_fitted_priority import (
    best_support_matching,
    digit_stratified_disjoint_split,
    digit_stratified_group_crossfit,
    digit_stratified_group_folds,
    field_pearson,
)


def test_cross_fitted_split_is_disjoint_deterministic_and_sized() -> None:
    split = digit_stratified_disjoint_split(
        [1, 2, 10, 11, 100, 101, 1000, 1001, 1002],
        direction_examples=3,
        score_examples=3,
        seed=7,
    )

    assert [len(value) for value in (split.direction_a, split.direction_b, split.score)] == [3, 3, 3]
    assert len(set(split.direction_a) | set(split.direction_b) | set(split.score)) == 9
    assert np.array_equal(split.score, digit_stratified_disjoint_split(
        [1, 2, 10, 11, 100, 101, 1000, 1001, 1002], direction_examples=3, score_examples=3, seed=7,
    ).score)


def test_cross_fitted_metrics_match_identical_factors() -> None:
    support = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=bool)
    assert field_pearson(support, support) == 1.0
    matches = best_support_matching(support, support[:, ::-1])
    assert {(left, right) for left, right, _, _ in matches} == {(0, 1), (1, 0)}
    assert all(jaccard == 1.0 and f1 == 1.0 for _, _, jaccard, f1 in matches)


def test_cross_fitted_group_folds_keep_duplicate_numbers_together() -> None:
    numbers = [1, 1, 2, 3, 10, 10, 11, 12]
    folds = digit_stratified_group_folds(numbers, folds=2, seed=3)

    assert sorted(np.concatenate(folds).tolist()) == list(range(len(numbers)))
    for number in set(numbers):
        containing = [
            fold_index
            for fold_index, fold in enumerate(folds)
            if any(numbers[int(index)] == number for index in fold)
        ]
        assert len(containing) == 1


def test_full_complement_crossfit_uses_every_out_of_fold_occurrence() -> None:
    numbers = [1, 1, 2, 3, 10, 10, 11, 12]
    folds = digit_stratified_group_crossfit(numbers, folds=2, seed=3)

    assert sorted(np.concatenate([fold.score for fold in folds]).tolist()) == list(
        range(len(numbers))
    )
    for fold in folds:
        score_numbers = {numbers[int(index)] for index in fold.score}
        direction_numbers = {numbers[int(index)] for index in fold.direction}
        assert score_numbers.isdisjoint(direction_numbers)
        assert sorted(np.concatenate([fold.score, fold.direction]).tolist()) == list(
            range(len(numbers))
        )


def test_simple_edge_rule_chooses_only_low_cost_best_parent() -> None:
    supports = (
        np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool),
        np.asarray([[1], [1], [0], [0]], dtype=bool),
    )

    edges, candidates = _select_edges(supports, maximum_closure_cost=0.1)

    assert edges == ((0, 0, 1, 0),)
    assert len(candidates) == 2
    assert candidates[0]["closure_cost"] == 0.0


def test_simple_closure_is_exact_and_preserves_curve_mass() -> None:
    supports = (
        np.asarray([[0], [1], [0]], dtype=bool),
        np.asarray([[1], [1], [0]], dtype=bool),
    )
    curves = (
        np.asarray([[0.0, 0.0, 2.0]]),
        np.asarray([[2.0, 0.0, 0.0]]),
    )
    edge = (0, 0, 1, 0)

    closed_supports = _close_supports(supports, (edge,))
    closed_curves = _close_temporal_priority(curves, (edge,))
    normalized_cumulative = tuple(
        np.cumsum(value, axis=1) / value.sum(axis=1, keepdims=True)
        for value in closed_curves
    )

    assert np.all(closed_supports[0][:, 0] | ~closed_supports[1][:, 0])
    assert np.min(
        normalized_cumulative[0][0] - normalized_cumulative[1][0]
    ) >= -1.0e-10
    np.testing.assert_allclose(closed_curves[0].sum(), curves[0].sum())


def test_build_simple_closure_graph_writes_a_self_contained_artifact(
    tmp_path: Path,
) -> None:
    from quanta.experiments.quanta_discovery.factorization import (
        build_simple_closure_graph,
    )

    supports = (
        np.asarray([[1], [1], [0]], dtype=bool),
        np.asarray([[1], [0], [0]], dtype=bool),
    )
    curves = (
        np.asarray([[2.0, 0.0]]),
        np.asarray([[0.0, 2.0]]),
    )
    np.savez(
        tmp_path / "priority_analysis.npz",
        train_supports=np.asarray(supports, dtype=object),
        temporal_priority=np.asarray(curves, dtype=object),
        train_credit=np.asarray(
            [
                supports[0].astype(float) @ curves[0],
                supports[1].astype(float) @ curves[1],
            ]
        ),
    )
    (tmp_path / "priority_summary.json").write_text(
        json.dumps({"minimum_component_gain_fraction_of_original_sse": 0.05})
    )

    summary_path = build_simple_closure_graph(
        tmp_path, maximum_closure_cost=0.5
    )

    summary = json.loads(summary_path.read_text())
    assert summary["edges"] == [[0, 0, 1, 0]]
    assert (tmp_path / "factorization" / "support_layer_0.npy").is_file()
    assert (tmp_path / "factorization" / "temporal_priority.npz").is_file()
