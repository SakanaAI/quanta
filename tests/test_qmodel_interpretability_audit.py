from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.audit_qmodel_interpretability import (
    AttentionReaderPanel,
    FeatureRef,
    _attention_reader_summary,
    _causal_prefix_permutation,
    _fit_amplitude_hypotheses,
    _fit_compound_hypotheses,
    _fit_hypotheses,
    _stratified_eval_split,
    _validated_writer_masks,
    _within_quantum_redundancy,
    average_precision,
    binary_metrics,
)


def test_binary_metrics_reports_precision_and_recall() -> None:
    record = binary_metrics(
        np.asarray([True, True, False, False]),
        np.asarray([True, False, True, False]),
    )

    assert record["precision"] == pytest.approx(0.5)
    assert record["recall"] == pytest.approx(0.5)
    assert record["f1"] == pytest.approx(0.5)
    assert record["balanced_accuracy"] == pytest.approx(0.5)


def test_average_precision_handles_tied_scores_as_one_threshold() -> None:
    score = average_precision(
        np.asarray([1.0, 0.5, 0.5, 0.0]),
        np.asarray([True, False, True, False]),
    )

    assert score == pytest.approx(5.0 / 6.0)


def test_causal_prefix_permutation_preserves_every_query_prefix() -> None:
    mask = torch.tensor([[1, 1, 1, 0]])
    permutation = _causal_prefix_permutation(
        mask, rng=np.random.default_rng(7)
    )

    for target in range(3):
        assert sorted(permutation[0, target, : target + 1].tolist()) == list(
            range(target + 1)
        )
    assert permutation[0, 3].tolist() == [0, 1, 2, 3]


def test_attention_reader_top_two_routes_are_joint_and_frozen() -> None:
    def panel(role_pairs: list[tuple[str, str]], active: list[bool]) -> AttentionReaderPanel:
        count = len(role_pairs)
        ones = np.ones((count, 1), dtype=np.float64)
        owner_active = np.ones((count, 1), dtype=np.float64)
        events = [
            {
                "functional_owner": "tens",
                "functional_subtype": "digit",
                "token_index": 1,
            }
            for _ in range(count)
        ]
        return AttentionReaderPanel(
            events=events,
            quantum_entropy=(ones,),
            quantum_effective_support=(ones,),
            quantum_top1_mass=(ones,),
            quantum_top2_mass=(ones,),
            quantum_top_role=([[pair[0]] for pair in role_pairs],),
            quantum_top_label=([["ignored"] for _ in range(count)],),
            quantum_top2_roles=([[pair] for pair in role_pairs],),
            quantum_top2_labels=(
                [[(f"{pair[0]}=3", f"{pair[1]}=four")] for pair in role_pairs],
            ),
            quantum_active=(ones,),
            writer_owner_active=(owner_active,),
            writer_scores=(np.asarray(active, dtype=np.float64).reshape(-1, 1),),
            writer_local_scores=(ones,),
            writer_context_scores=(ones,),
        )

    calibration = panel(
        [("input_digit[tens]", "output_prefix")] * 4
        + [("input_digit[units]", "output_prefix")] * 4,
        [True] * 4 + [False] * 4,
    )
    test = panel(
        [("input_digit[tens]", "output_prefix")] * 3
        + [("input_digit[units]", "output_prefix")] * 5,
        [True] * 3 + [False] * 5,
    )
    module = SimpleNamespace(
        quantum_count=1,
        writer_count=1,
        writer_owner=torch.tensor([0]),
    )
    writer_records = [
        {
            "name": "L0.Q0.W0",
            "layer": 0,
            "index": 0,
            "conditional_on_owner": {
                "compound_hypotheses": {"all": {"calibration": {"f1": 0.9}}}
            },
        }
    ]

    summary = _attention_reader_summary(
        calibration,
        test,
        writer_records,
        (module,),
        threshold=0.5,
        minimum_events=2,
    )

    row = summary["writers"][0]
    assert row["calibration_ordered_top2_roles"] == "input_digit[tens] -> output_prefix"
    assert row["calibration_ordered_top2_labels"] == (
        "input_digit[tens]=3 -> output_prefix=four"
    )
    assert row["test_ordered_top2_role_stability"] == pytest.approx(1.0)
    rule = row["ordered_top2_role_activation_rule"]
    assert rule is not None
    assert rule["signature"] == ["input_digit[tens]", "output_prefix"]
    assert rule["test"]["f1"] == pytest.approx(1.0)
    assert summary["writer_groups"]["rich_selected"]["writer_count"] == 1


def test_attention_reader_routes_condition_on_output_state() -> None:
    def panel(
        pairs: list[tuple[str, str]], subtypes: list[str]
    ) -> AttentionReaderPanel:
        count = len(pairs)
        ones = np.ones((count, 1), dtype=np.float64)
        return AttentionReaderPanel(
            events=[
                {
                    "functional_owner": "number",
                    "functional_subtype": subtype,
                    "token_index": 2,
                }
                for subtype in subtypes
            ],
            quantum_entropy=(ones,),
            quantum_effective_support=(ones,),
            quantum_top1_mass=(ones,),
            quantum_top2_mass=(ones,),
            quantum_top_role=([[pair[0]] for pair in pairs],),
            quantum_top_label=([[pair[0]] for pair in pairs],),
            quantum_top2_roles=([[pair] for pair in pairs],),
            quantum_top2_labels=([[pair] for pair in pairs],),
            quantum_active=(ones,),
            writer_owner_active=(ones,),
            writer_scores=(ones,),
            writer_local_scores=(ones,),
            writer_context_scores=(ones,),
        )

    pairs = [
        ("input_digit[tens]", "output_prefix"),
        ("input_digit[units]", "output_prefix"),
    ] * 3
    subtypes = ["tens_word", "units_word"] * 3
    module = SimpleNamespace(
        quantum_count=1, writer_count=1, writer_owner=torch.tensor([0])
    )
    record = {
        "name": "L0.Q0.W0",
        "layer": 0,
        "index": 0,
        "conditional_on_owner": {
            "compound_hypotheses": {"all": {"calibration": {"f1": 0.9}}}
        },
    }
    summary = _attention_reader_summary(
        panel(pairs, subtypes),
        panel(pairs, subtypes),
        [record],
        (module,),
        threshold=0.5,
        minimum_events=2,
    )

    routes = summary["writers"][0]["conditional_ordered_top2_role_routes"]
    assert len(routes) == 2
    assert {route["functional_subtype"] for route in routes} == {
        "tens_word",
        "units_word",
    }
    assert all(route["test_ordered_top2_stability"] == pytest.approx(1.0) for route in routes)


def test_hypothesis_is_selected_on_calibration_and_frozen_on_test() -> None:
    calibration_scores = np.asarray([[1.0], [1.0], [0.0], [0.0]])
    test_scores = np.asarray([[1.0], [0.0], [1.0], [0.0]])
    calibration_concepts = np.asarray(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
        ]
    )
    test_concepts = np.asarray(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
        ]
    )

    record = _fit_hypotheses(
        calibration_scores,
        test_scores,
        calibration_concepts,
        test_concepts,
        ["target=one", "target=two"],
        [FeatureRef("writer", 0, 0, 0)],
    )[0]

    assert record["hypotheses"]["all"]["concept"] == "target=one"
    assert record["hypotheses"]["all"]["calibration"]["f1"] == pytest.approx(1.0)
    assert record["hypotheses"]["all"]["test"]["f1"] == pytest.approx(0.5)


def test_compound_hypothesis_uses_calibration_selection_and_test_freeze() -> None:
    calibration_scores = np.asarray([[1.0], [1.0], [0.0], [0.0]])
    test_scores = np.asarray([[1.0], [0.0], [1.0], [0.0]])
    calibration_concepts = np.asarray(
        [
            [True, True],
            [True, True],
            [True, False],
            [False, True],
        ]
    )
    test_concepts = np.asarray(
        [
            [True, True],
            [True, True],
            [True, False],
            [False, True],
        ]
    )

    record = _fit_compound_hypotheses(
        calibration_scores,
        test_scores,
        calibration_concepts,
        test_concepts,
        ["target=one", "previous_target=START"],
        [FeatureRef("writer", 0, 0, 0)],
        top_concepts=2,
        description_penalty=0.02,
    )[0]

    hypothesis = record["hypotheses"]["all"]
    assert hypothesis["operator"] == "AND"
    assert hypothesis["terms"] == ["target=one", "previous_target=START"]
    assert hypothesis["calibration"]["f1"] == pytest.approx(1.0)
    assert hypothesis["test"]["f1"] == pytest.approx(0.5)


def test_amplitude_hypothesis_is_selected_on_calibration_and_frozen_on_test() -> None:
    records = _fit_amplitude_hypotheses(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([[0.0], [1.0], [0.0], [1.0]]),
        np.asarray([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]),
        np.asarray([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]),
        ["input_number", "remaining_output_tokens"],
        [FeatureRef("writer", 0, 0, 0)],
    )

    hypothesis = records[0]["hypothesis"]
    assert hypothesis["variable"] == "input_number"
    assert hypothesis["calibration"]["pearson_r"] == pytest.approx(1.0)
    assert hypothesis["test"]["pearson_r"] == pytest.approx(0.4472135955)


@dataclass(frozen=True)
class _Example:
    number: int


def test_stratified_split_is_disjoint_and_preserves_digit_buckets() -> None:
    examples = [_Example(number) for number in (1, 2, 10, 11, 100, 101)]

    calibration, test = _stratified_eval_split(examples, seed=7)

    assert {item.number for item in calibration}.isdisjoint(
        {item.number for item in test}
    )
    assert sorted(len(str(item.number)) for item in calibration) == [1, 2, 3]
    assert sorted(len(str(item.number)) for item in test) == [1, 2, 3]


def test_validated_masks_distinguish_global_and_owner_conditional_semantics() -> None:
    record = {
        "layer": 0,
        "index": 1,
        "hypotheses": {"all": {"calibration": {"f1": 0.4}}},
        "conditional_on_owner": {
            "hypotheses": {"all": {"calibration": {"f1": 0.8}}}
        },
    }
    modules = [SimpleNamespace(writer_count=3)]

    global_mask = _validated_writer_masks(
        modules, [record], threshold=0.6, conditional=False
    )
    conditional_mask = _validated_writer_masks(
        modules, [record], threshold=0.6, conditional=True
    )

    assert global_mask[0].tolist() == [False, False, False]
    assert conditional_mask[0].tolist() == [False, True, False]


def test_rich_validated_mask_uses_compound_calibration_not_test_score() -> None:
    record = {
        "layer": 0,
        "index": 1,
        "hypotheses": {"all": {"calibration": {"f1": 0.1}}},
        "compound_hypotheses": {"all": {"calibration": {"f1": 0.1}}},
        "conditional_on_owner": {
            "hypotheses": {"all": {"calibration": {"f1": 0.1}}},
            "compound_hypotheses": {
                "all": {"calibration": {"f1": 0.8}, "test": {"f1": 0.1}}
            },
        },
    }
    modules = [SimpleNamespace(writer_count=3)]

    rich_mask = _validated_writer_masks(
        modules, [record], threshold=0.6, conditional=True, rich=True
    )

    assert rich_mask[0].tolist() == [False, True, False]


def test_within_quantum_redundancy_clusters_identical_causal_signatures() -> None:
    references = [FeatureRef("writer", 0, 0, 0), FeatureRef("writer", 0, 1, 0)]
    module = SimpleNamespace(
        output_weight=torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    )

    redundancy = _within_quantum_redundancy(
        np.asarray([[1.0, 1.0], [1.0, 0.0]]),
        np.asarray([[1.0, 1.0], [0.0, 0.0]]),
        references,
        [module],
        {references[0].name: np.asarray([1.0, 2.0]), references[1].name: np.asarray([1.0, 2.0])},
        cluster_cosine=0.9,
    )

    bank = redundancy["banks"][0]
    assert bank["pairs"][0]["test_jaccard"] == pytest.approx(1.0)
    assert bank["pairs"][0]["decoder_cosine"] == pytest.approx(1.0)
    assert bank["causal_signature_clusters"] == [[references[0].name, references[1].name]]
