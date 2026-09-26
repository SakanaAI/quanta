from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

from .names import number_name
from .probes import _virtual_token_roles

SPLIT_CACHE_VERSION = 13
_EXTREME_GENERALIZATION_LIMIT = 1_000_000
_EXTREME_TRAIN_PER_DIGIT_LENGTH = 50
_EXTREME_EVAL_DIGIT_LENGTHS = tuple(range(1, 7))
_EXTREME_BALANCED_EVAL_PREFIX_MAX = 999
_SOURCE_TRAIN_NUMBERS = frozenset(
    (
        *range(1, 103),
        *range(110, 120),
        *range(210, 220),
        1_000,
        1_001,
        1_002,
        1_021,
        1_321,
        2_000,
        10_000,
        11_000,
        12_000,
        20_000,
        21_000,
        100_000,
        200_000,
        1_000_000,
        1_000_001,
        1_000_002,
        2_000_000,
    )
)
_EXTREME_REQUIRED_TRAIN_NUMBERS = (
    700,
    403_000,
    500_000,
    521_682,
    834_275,
)
_EXTREME_V2_REQUIRED_TRAIN_NUMBERS = (
    *range(1, 10),
    *range(10, 22),
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    99,
    100,
    101,
    109,
    110,
    111,
    119,
    120,
    121,
    199,
    200,
    700,
    999,
    1_000,
    1_001,
    1_007,
    1_010,
    1_011,
    1_014,
    1_019,
    1_020,
    1_021,
    1_099,
    1_100,
    1_111,
    1_200,
    9_999,
    10_000,
    10_001,
    10_010,
    10_011,
    10_019,
    10_020,
    10_100,
    11_000,
    11_111,
    12_000,
    13_000,
    14_000,
    15_000,
    16_000,
    17_000,
    18_000,
    19_000,
    99_999,
    100_000,
    100_001,
    100_010,
    100_011,
    100_100,
    101_000,
    110_000,
    111_111,
    403_000,
    500_000,
    521_682,
    834_275,
    999_999,
)


@dataclass(frozen=True)
class NumberNamingExample:
    number: int
    text: str


@dataclass(frozen=True)
class NumberNamingSplits:
    train: list[NumberNamingExample]
    eval: list[NumberNamingExample]
    all_examples: list[NumberNamingExample]


def build_number_naming_splits(config) -> NumberNamingSplits:
    cache_path = _splits_cache_path(config)
    if cache_path is not None and cache_path.exists():
        logging.debug("number_naming loading cached splits from %s", cache_path)
        return _read_splits_cache(cache_path)

    if config.data_splits == "uniform":
        splits = _uniform(config)
    elif config.data_splits == "digits_wise_uniform":
        splits = _digits_wise_uniform(config)
    elif config.data_splits in {"extreme_generalization", "extreme_generalization_v2"}:
        splits = _extreme_generalization(config)
    else:
        raise ValueError(f"Unsupported data_splits: {config.data_splits!r}")

    splits = _apply_held_out_token_roles(config, splits)

    if cache_path is not None:
        _write_splits_cache(cache_path, splits)
        logging.debug("number_naming cached splits at %s", cache_path)
    return splits


def _uniform(config) -> NumberNamingSplits:
    max_number = int(config.max_number)
    examples = _generated_examples(range(1, max_number + 1), language=config.language)
    by_number = {
        example.number: example
        for example in examples
        if 0 < example.number <= max_number
    }
    base_numbers = set(range(1, min(99, max_number) + 1))
    missing_base = sorted(base_numbers - set(by_number))
    for number in missing_base:
        by_number[number] = NumberNamingExample(number, number_name(number, language=config.language))
    if missing_base:
        logging.debug(
            "number_naming filled missing base numbers from built-in generator: %s",
            missing_base,
        )
    missing_base = sorted(base_numbers - set(by_number))
    if missing_base:
        raise ValueError(f"NumberNaming dataset is missing required base numbers 1..99: {missing_base[:10]}")
    train_numbers = set(base_numbers)
    remaining = [number for number in by_number if number >= 100]
    rng = random.Random(int(config.split_seed))
    if int(config.training_size) > len(remaining):
        raise ValueError("training_size exceeds numbers available after 1..99 base set.")
    sampled_train = set(rng.sample(remaining, int(config.training_size)))
    train_numbers.update(sampled_train)
    eval_pool = [number for number in remaining if number not in sampled_train]
    eval_numbers = eval_pool if config.eval_size is None else rng.sample(eval_pool, min(int(config.eval_size), len(eval_pool)))
    lexical_overlap_numbers = set(range(1, min(19, max_number) + 1))
    eval_numbers = sorted(set(eval_numbers) | lexical_overlap_numbers)
    assert train_numbers.isdisjoint(set(eval_numbers) - lexical_overlap_numbers)
    train = [by_number[number] for number in sorted(train_numbers)]
    eval_examples = [by_number[number] for number in eval_numbers]
    logging.debug(
        "number_naming data=uniform source=%s all=%s train=%s eval=%s",
        "built_in_generator",
        len(examples),
        len(train),
        len(eval_examples),
    )
    all_examples = [by_number[number] for number in sorted(by_number)]
    return NumberNamingSplits(train=train, eval=eval_examples, all_examples=all_examples)


def _digits_wise_uniform(config) -> NumberNamingSplits:
    max_number = int(config.max_number)
    examples = _generated_examples(range(1, max_number + 1), language=config.language)
    by_number = {
        example.number: example
        for example in examples
        if 0 < example.number <= max_number
    }
    buckets: dict[int, list[NumberNamingExample]] = {}
    for example in by_number.values():
        buckets.setdefault(len(str(example.number)), []).append(example)
    if not buckets:
        raise ValueError("digits_wise_uniform requires at least one positive number.")
    for bucket in buckets.values():
        bucket.sort(key=lambda example: example.number)

    rng = random.Random(int(config.split_seed))
    train = _sample_digit_uniform_examples(
        buckets=buckets,
        total_count=int(config.training_size),
        rng=rng,
        split_name="train",
    )
    eval_total = len(by_number) if config.eval_size is None else int(config.eval_size)
    eval_examples = _sample_digit_uniform_examples(
        buckets=buckets,
        total_count=eval_total,
        rng=random.Random(int(config.split_seed) + 1_000_003),
        split_name="eval",
    )
    logging.debug(
        "number_naming data=digits_wise_uniform source=%s all=%s train=%s eval=%s digit_buckets=%s",
        "built_in_generator",
        len(examples),
        len(train),
        len(eval_examples),
        {digits: len(bucket) for digits, bucket in sorted(buckets.items())},
    )
    all_examples = [by_number[number] for number in sorted(by_number)]
    return NumberNamingSplits(train=train, eval=eval_examples, all_examples=all_examples)


def _sample_digit_uniform_examples(
    *,
    buckets: dict[int, list[NumberNamingExample]],
    total_count: int,
    rng: random.Random,
    split_name: str,
) -> list[NumberNamingExample]:
    if total_count < 0:
        raise ValueError(f"{split_name} size must be non-negative for digits_wise_uniform.")
    if total_count == 0:
        return []
    digit_lengths = sorted(buckets)
    if total_count < len(digit_lengths):
        raise ValueError(
            f"{split_name} size {total_count} is too small for digits_wise_uniform over "
            f"{len(digit_lengths)} digit lengths."
        )
    per_digit = total_count // len(digit_lengths)
    used_total = per_digit * len(digit_lengths)
    if used_total != total_count:
        logging.debug(
            "number_naming digits_wise_uniform %s size rounded down from %s to %s "
            "to keep equal counts per digit length",
            split_name,
            total_count,
            used_total,
        )
    sampled: list[NumberNamingExample] = []
    for digits in digit_lengths:
        bucket = buckets[digits]
        sampled.extend(rng.choice(bucket) for _ in range(per_digit))
    rng.shuffle(sampled)
    return sampled


def _extreme_generalization(config) -> NumberNamingSplits:
    source_by_number: dict[int, NumberNamingExample] = {}
    rng = random.Random(int(config.split_seed))
    if config.data_splits == "extreme_generalization_v2":
        train = _balanced_extreme_train_examples_v2(config, source_by_number, rng=rng)
    else:
        train = _balanced_extreme_train_examples(config, source_by_number, rng=rng)
    eval_rng = random.Random(int(config.split_seed) + 1_000_003)
    if str(getattr(config, "eval_strategy", "digits_wise")).lower() == "balanced":
        eval_numbers = _sample_generated_balanced_eval_numbers(config, rng=eval_rng)
    else:
        eval_numbers = _sample_generated_digit_uniform_eval_numbers(config, rng=eval_rng)
    eval_examples = _generated_examples(eval_numbers, language=config.language)
    all_examples_by_number = {
        example.number: example
        for example in train + eval_examples
    }
    logging.debug(
        "number_naming data=%s source=%s train_source=%s train=%s train_per_digit=%s test_source=%s test_filtered=%s eval_requested=%s eval=%s eval_per_digit=%s",
        config.data_splits,
        "built_in_generator",
        len(_SOURCE_TRAIN_NUMBERS),
        len(train),
        _digit_length_counts(train),
        _EXTREME_GENERALIZATION_LIMIT - 1 - len(
            [number for number in _SOURCE_TRAIN_NUMBERS if number < _EXTREME_GENERALIZATION_LIMIT]
        ),
        len(eval_numbers),
        config.eval_size,
        len(eval_examples),
        _digit_length_counts(eval_examples),
    )
    return NumberNamingSplits(
        train=train,
        eval=eval_examples,
        all_examples=[all_examples_by_number[number] for number in sorted(all_examples_by_number)],
    )


def _generated_examples(numbers, *, language: str) -> list[NumberNamingExample]:
    return [
        NumberNamingExample(int(number), number_name(int(number), language=language))
        for number in numbers
    ]


def _source_test_buckets(max_number: int) -> dict[int, list[int]]:
    buckets: dict[int, list[int]] = {}
    for digits in range(1, len(str(max_number)) + 1):
        start = 1 if digits == 1 else 10 ** (digits - 1)
        stop = min(max_number, 10**digits - 1)
        if start > stop:
            continue
        buckets[digits] = [
            number
            for number in range(start, stop + 1)
            if number not in _SOURCE_TRAIN_NUMBERS
        ]
    return buckets


def _sample_generated_digit_uniform_eval_numbers(config, *, rng: random.Random) -> list[int]:
    buckets = _source_test_buckets(_EXTREME_GENERALIZATION_LIMIT - 1)
    if config.eval_size is None:
        return [number for digits in sorted(buckets) for number in buckets[digits]]
    total_count = int(config.eval_size)
    if total_count < len(_EXTREME_EVAL_DIGIT_LENGTHS):
        raise ValueError(
            f"extreme_generalization eval_size={total_count} is too small for "
            f"digit-wise eval over {len(_EXTREME_EVAL_DIGIT_LENGTHS)} digit lengths."
        )
    per_digit, remainder = divmod(total_count, len(_EXTREME_EVAL_DIGIT_LENGTHS))
    selected: list[int] = []
    for index, digits in enumerate(_EXTREME_EVAL_DIGIT_LENGTHS):
        count = per_digit + (1 if index < remainder else 0)
        bucket = buckets[digits]
        if count <= len(bucket):
            selected.extend(rng.sample(bucket, count))
        else:
            selected.extend(rng.choice(bucket) for _ in range(count))
    return sorted(selected, key=lambda number: (len(str(number)), number))


def _sample_generated_balanced_eval_numbers(config, *, rng: random.Random) -> list[int]:
    buckets = _source_test_buckets(_EXTREME_GENERALIZATION_LIMIT - 1)
    if config.eval_size is None:
        prefix = list(range(1, _EXTREME_BALANCED_EVAL_PREFIX_MAX + 1))
        tail = [number for digits in range(4, 7) for number in buckets[digits]]
        return prefix + tail
    total_count = int(config.eval_size)
    prefix = list(range(1, _EXTREME_BALANCED_EVAL_PREFIX_MAX + 1))
    if total_count < len(prefix):
        raise ValueError(
            "extreme_generalization balanced eval_size must be at least "
            f"{len(prefix)} to include every 1-, 2-, and 3-digit number."
        )
    tail_count = total_count - len(prefix)
    if tail_count == 0:
        return prefix
    capacities = {digits: len(buckets[digits]) for digits in range(4, 7)}
    counts = _balanced_digit_counts(capacities, tail_count)
    selected = list(prefix)
    for digits in range(4, 7):
        selected.extend(rng.sample(buckets[digits], counts[digits]))
    return sorted(selected, key=lambda number: (len(str(number)), number))


def _sample_balanced_digit_examples(
    examples: list[NumberNamingExample],
    *,
    total_count: int,
    rng: random.Random,
    required_digit_lengths: tuple[int, ...] | None = None,
    split_name: str = "eval",
) -> list[NumberNamingExample]:
    if total_count < 0:
        raise ValueError(f"{split_name} size must be non-negative for balanced eval.")
    if total_count == 0:
        return []

    by_number: dict[int, NumberNamingExample] = {}
    for example in examples:
        by_number.setdefault(int(example.number), example)

    if required_digit_lengths is None:
        digit_lengths = tuple(sorted({len(str(number)) for number in by_number}))
    else:
        digit_lengths = tuple(sorted(required_digit_lengths))
    if not digit_lengths:
        raise ValueError(f"{split_name} requires at least one positive number.")

    buckets: dict[int, list[NumberNamingExample]] = {digits: [] for digits in digit_lengths}
    for number, example in by_number.items():
        digits = len(str(number))
        if digits in buckets:
            buckets[digits].append(example)
    missing = [digits for digits, bucket in buckets.items() if not bucket]
    if missing:
        raise ValueError(f"{split_name} is missing digit lengths: {missing}")
    for bucket in buckets.values():
        bucket.sort(key=lambda example: int(example.number))

    capacities = {digits: len(bucket) for digits, bucket in buckets.items()}
    total_available = sum(capacities.values())
    if total_count > total_available:
        raise ValueError(
            f"{split_name} requested {total_count} balanced examples without replacement, "
            f"but only {total_available} unique examples are available."
        )
    counts = _balanced_digit_counts(capacities, total_count)
    sampled: list[NumberNamingExample] = []
    for digits in digit_lengths:
        sampled.extend(rng.sample(buckets[digits], counts[digits]))
    sampled.sort(key=lambda example: (len(str(int(example.number))), int(example.number), example.text))
    return sampled


def _balanced_digit_counts(capacities: dict[int, int], total_count: int) -> dict[int, int]:
    counts = {digits: 0 for digits in capacities}
    active = sorted(digits for digits, capacity in capacities.items() if capacity > 0)
    remaining = int(total_count)
    while active and remaining > 0:
        base, remainder = divmod(remaining, len(active))
        saturated: list[int] = []
        for index, digits in enumerate(active):
            target = base + (1 if index < remainder else 0)
            if capacities[digits] <= target:
                counts[digits] = capacities[digits]
                remaining -= capacities[digits]
                saturated.append(digits)
        if not saturated:
            for index, digits in enumerate(active):
                counts[digits] = base + (1 if index < remainder else 0)
            remaining = 0
            break
        saturated_set = set(saturated)
        active = [digits for digits in active if digits not in saturated_set]
    return counts


def _balanced_extreme_train_examples(
    config,
    source_by_number: dict[int, NumberNamingExample],
    *,
    rng: random.Random,
    required_numbers: tuple[int, ...] = _EXTREME_REQUIRED_TRAIN_NUMBERS,
) -> list[NumberNamingExample]:
    examples: list[NumberNamingExample] = []
    required_by_digits: dict[int, set[int]] = {}
    for number in required_numbers:
        required_by_digits.setdefault(len(str(number)), set()).add(int(number))

    for digits in range(1, 7):
        start = 1 if digits == 1 else 10 ** (digits - 1)
        stop = min((10 ** digits) - 1, _EXTREME_GENERALIZATION_LIMIT - 1)
        required = sorted(number for number in required_by_digits.get(digits, set()) if start <= number <= stop)
        if len(required) > _EXTREME_TRAIN_PER_DIGIT_LENGTH:
            raise ValueError(
                f"extreme_generalization has {len(required)} required {digits}-digit train examples, "
                f"exceeding {_EXTREME_TRAIN_PER_DIGIT_LENGTH}."
            )
        if digits == 1:
            sampled = list(required)
            candidates = list(range(start, stop + 1))
            while len(sampled) < _EXTREME_TRAIN_PER_DIGIT_LENGTH:
                sampled.append(rng.choice(candidates))
            sampled.sort()
        else:
            required_set = set(required)
            candidates = [number for number in range(start, stop + 1) if number not in required_set]
            sampled = sorted(required + rng.sample(candidates, _EXTREME_TRAIN_PER_DIGIT_LENGTH - len(required)))
        examples.extend(_example_for_number(config, number, source_by_number) for number in sampled)
    return examples


def _balanced_extreme_train_examples_v2(
    config,
    source_by_number: dict[int, NumberNamingExample],
    *,
    rng: random.Random,
) -> list[NumberNamingExample]:
    target_count = int(config.training_size)
    if target_count < 0:
        raise ValueError("extreme_generalization_v2 training_size must be non-negative.")
    capacities = {}
    for digits in range(1, 7):
        start = 1 if digits == 1 else 10 ** (digits - 1)
        stop = min((10 ** digits) - 1, _EXTREME_GENERALIZATION_LIMIT - 1)
        capacities[digits] = stop - start + 1
    total_available = sum(capacities.values())
    if target_count > total_available:
        raise ValueError(
            "extreme_generalization_v2 requested more unique train examples than are available below one million."
        )

    counts = _balanced_digit_counts(capacities, target_count)
    required_by_digits: dict[int, set[int]] = {}
    for number in _EXTREME_V2_REQUIRED_TRAIN_NUMBERS:
        number = int(number)
        if 0 < number < _EXTREME_GENERALIZATION_LIMIT:
            required_by_digits.setdefault(len(str(number)), set()).add(number)

    examples: list[NumberNamingExample] = []
    for digits in range(1, 7):
        start = 1 if digits == 1 else 10 ** (digits - 1)
        stop = min((10 ** digits) - 1, _EXTREME_GENERALIZATION_LIMIT - 1)
        required = sorted(number for number in required_by_digits.get(digits, set()) if start <= number <= stop)
        count = counts[digits]
        if len(required) > count:
            raise ValueError(
                f"extreme_generalization_v2 training_size={target_count} is too small: "
                f"{digits}-digit edge cases require {len(required)} examples, but the balanced budget assigns {count}."
            )
        required_set = set(required)
        candidates = [number for number in range(start, stop + 1) if number not in required_set]
        sampled = sorted(required + rng.sample(candidates, count - len(required)))
        examples.extend(_example_for_number(config, number, source_by_number) for number in sampled)
    return examples


def _example_for_number(config, number: int, source_by_number: dict[int, NumberNamingExample]) -> NumberNamingExample:
    if int(number) in source_by_number:
        return source_by_number[int(number)]
    return NumberNamingExample(int(number), number_name(int(number), language=config.language))


def _digit_length_counts(examples: list[NumberNamingExample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for example in examples:
        counts[len(str(int(example.number)))] = counts.get(len(str(int(example.number))), 0) + 1
    return dict(sorted(counts.items()))


def _apply_held_out_token_roles(config, splits: NumberNamingSplits) -> NumberNamingSplits:
    held_out = getattr(config, "held_out_token_roles", None)
    if not held_out:
        return splits
    requested = {
        (str(item["token"]).lower(), str(item["role"]))
        for item in held_out
    }
    train = [
        example
        for example in splits.train
        if not _example_has_token_role(example, requested)
    ]
    held_out_eval = [
        example
        for example in splits.all_examples
        if _example_has_token_role(example, requested)
    ]
    if not held_out_eval:
        raise ValueError(f"held_out_token_roles did not match any examples: {sorted(requested)}")
    if len(train) == len(splits.train):
        raise ValueError(f"held_out_token_roles did not remove any training examples: {sorted(requested)}")
    eval_numbers = {example.number for example in splits.eval}
    eval_examples = list(splits.eval)
    for example in held_out_eval:
        if example.number not in eval_numbers:
            eval_examples.append(example)
            eval_numbers.add(example.number)
    logging.debug(
        "number_naming held_out_token_roles=%s train_before=%s train_after=%s held_out_eval=%s",
        sorted(requested),
        len(splits.train),
        len(train),
        len(held_out_eval),
    )
    return NumberNamingSplits(train=train, eval=eval_examples, all_examples=splits.all_examples)


def _example_has_token_role(example: NumberNamingExample, requested: set[tuple[str, str]]) -> bool:
    words = str(example.text).split()
    roles = _virtual_token_roles(int(example.number), words)
    return any((word.lower(), role) in requested for word, role in zip(words, roles))


def _splits_cache_path(config) -> Path | None:
    if not getattr(config, "save_dir", None):
        return None
    save_dir = Path(config.save_dir)
    cache_dir = save_dir.parent / "cache"
    key = {
        "version": SPLIT_CACHE_VERSION,
        "language": config.language,
        "data_splits": config.data_splits,
        "eval_strategy": getattr(config, "eval_strategy", None),
        "training_size": int(config.training_size),
        "eval_size": None if config.eval_size is None else int(config.eval_size),
        "split_seed": int(config.split_seed),
        "max_number": int(config.max_number),
        "held_out_token_roles": getattr(config, "held_out_token_roles", None),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    filename = f"{config.data_splits}-seed{int(config.split_seed)}-{digest}.json"
    return cache_dir / filename


def _write_splits_cache(path: Path, splits: NumberNamingSplits) -> None:
    os.makedirs(path.parent, exist_ok=True)
    payload = {
        "version": SPLIT_CACHE_VERSION,
        "train": [_example_to_dict(example) for example in splits.train],
        "eval": [_example_to_dict(example) for example in splits.eval],
        "all_examples": [_example_to_dict(example) for example in splits.all_examples],
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as handle:
        json.dump(payload, handle)
    os.replace(temp_path, path)


def _read_splits_cache(path: Path) -> NumberNamingSplits:
    with open(path, "r") as handle:
        payload = json.load(handle)
    if payload.get("version") != SPLIT_CACHE_VERSION:
        raise ValueError(f"Unsupported NumberNaming split cache version in {path}.")
    return NumberNamingSplits(
        train=[_example_from_dict(item) for item in payload["train"]],
        eval=[_example_from_dict(item) for item in payload["eval"]],
        all_examples=[_example_from_dict(item) for item in payload["all_examples"]],
    )


def _example_to_dict(example: NumberNamingExample) -> dict[str, object]:
    return {"number": int(example.number), "text": example.text}


def _example_from_dict(item: dict) -> NumberNamingExample:
    return NumberNamingExample(number=int(item["number"]), text=str(item["text"]))
