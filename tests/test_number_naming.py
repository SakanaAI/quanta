from collections import Counter
import hashlib

import pytest

from quanta.config import PosetsProbingConfig
from quanta.experiments.number_naming import NumberNamingTask
from quanta.experiments.number_naming.names import english_number_name


def test_english_number_names_are_space_tokenized() -> None:
    assert english_number_name(0) == "zero"
    assert english_number_name(123) == "one hundred twenty three"
    assert english_number_name(1_005) == "one thousand five"


def test_uniform_split_is_deterministic_and_contains_base_numbers() -> None:
    config = PosetsProbingConfig(
        training_size=25,
        eval_size=30,
        max_number=999,
        split_seed=3,
    )
    first = NumberNamingTask(config)
    second = NumberNamingTask(config)

    train_numbers = {example.number for example in first.train}
    eval_numbers = {example.number for example in first.eval}
    assert set(range(1, 100)).issubset(train_numbers)
    assert train_numbers & eval_numbers == set(range(1, 20))
    assert [item.number for item in first.train] == [item.number for item in second.train]
    assert [item.number for item in first.eval] == [item.number for item in second.eval]


def test_digit_uniform_split_balances_digit_lengths() -> None:
    config = PosetsProbingConfig(
        data_splits="digits_wise_uniform",
        training_size=30,
        eval_size=12,
        max_number=999,
        split_seed=7,
    )
    task = NumberNamingTask(config)

    assert Counter(len(str(item.number)) for item in task.train) == Counter(
        {1: 10, 2: 10, 3: 10}
    )
    assert Counter(len(str(item.number)) for item in task.eval) == Counter(
        {1: 4, 2: 4, 3: 4}
    )


def test_main_extreme_split_is_unique_and_covers_edge_cases() -> None:
    config = PosetsProbingConfig(
        data_splits="extreme_generalization_v2",
        training_size=300,
        eval_size=16_384,
        eval_strategy="balanced",
        max_number=999_999,
        split_seed=0,
    )
    task = NumberNamingTask(config)
    train_numbers = [item.number for item in task.train]
    eval_numbers = [item.number for item in task.eval]

    assert len(train_numbers) == len(set(train_numbers)) == 300
    assert Counter(len(str(number)) for number in train_numbers) == Counter(
        {1: 9, 2: 59, 3: 58, 4: 58, 5: 58, 6: 58}
    )
    assert {403_000, 500_000, 521_682, 834_275}.issubset(train_numbers)
    assert len(eval_numbers) == len(set(eval_numbers)) == 16_384
    assert set(range(1, 1_000)).issubset(eval_numbers)
    train_fingerprint = hashlib.sha256(
        ",".join(str(number) for number in train_numbers).encode()
    ).hexdigest()
    eval_fingerprint = hashlib.sha256(
        ",".join(str(number) for number in eval_numbers).encode()
    ).hexdigest()
    assert train_fingerprint == (
        "bc9ec752bca1a43af2241fcdce809678be05f5bd944b2ad08089af17371d4b3e"
    )
    assert eval_fingerprint == (
        "973d352eed74557310afba79fedd76356169f9750cb820742545c0ba8e628a97"
    )


def test_extreme_split_rejects_insufficient_edge_case_budget() -> None:
    config = PosetsProbingConfig(
        data_splits="extreme_generalization_v2",
        training_size=30,
        eval_size=1_005,
    )
    with pytest.raises(ValueError, match="training_size=30 is too small"):
        NumberNamingTask(config)
