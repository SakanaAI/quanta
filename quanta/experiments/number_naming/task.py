from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import NumberNamingExample, _sample_balanced_digit_examples, build_number_naming_splits
from .names import english_number_name
from .probes import load_probe_suites, probe_words
from .tokenizer import DIGIT_TOKENS, SPECIAL_TOKENS, NumberNamingTokenizer

NATS_TO_BITS = 1.0 / math.log(2.0)


@dataclass
class NumberNamingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    numbers: list[int]
    texts: list[str]

    @property
    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


class NumberNamingTask:
    def __init__(self, config):
        self.config = config
        self.splits = build_number_naming_splits(config)
        self.train = self.splits.train
        self.eval = self.splits.eval
        self.eval_examples = _sample_eval_examples(config, self.eval)
        self.probe_suites = (
            load_probe_suites(config, self.eval_examples)
            if getattr(config, "probe_quanta_poset", None) is not None
            else []
        )
        self.probe_suite = self.probe_suites[0] if len(self.probe_suites) == 1 else None
        words = [
            word
            for example in self.splits.all_examples
            for word in example.text.split()
        ]
        for probe_suite in self.probe_suites:
            words.extend(probe_words(probe_suite))
        self.tokenizer = NumberNamingTokenizer(words)
        self.rng = random.Random(int(config.seed))
        self._validate_lengths()

    def make_batch(self, *, split: str, batch_size: int, device) -> NumberNamingBatch:
        source = self.train if split == "train" else self.eval_examples
        if not source:
            raise ValueError(f"{split} split is empty.")
        examples = [self.rng.choice(source) for _ in range(int(batch_size))] if split == "train" else source[: int(batch_size)]
        return self.encode_examples(examples, device=device)

    def encode_examples(self, examples: list[NumberNamingExample], *, device) -> NumberNamingBatch:
        encoded = [
            self.tokenizer.encode(
                example.number,
                example.text,
                max_seq_len=int(self.config.max_seq_len),
            )
            for example in examples
        ]
        max_len = max(len(item.input_ids) for item in encoded)
        input_ids = torch.full((len(encoded), max_len), self.tokenizer.pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
        labels = torch.full((len(encoded), max_len), -100, dtype=torch.long, device=device)
        for row, item in enumerate(encoded):
            length = len(item.input_ids)
            input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(item.labels, dtype=torch.long, device=device)
        return NumberNamingBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            numbers=[item.number for item in encoded],
            texts=[item.text for item in encoded],
        )

    def compute_loss(self, logits: torch.Tensor, batch: NumberNamingBatch) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch.labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def teacher_forced_metrics(self, logits: torch.Tensor, batch: NumberNamingBatch) -> dict[str, float]:
        shift_logits = logits[:, :-1, :]
        shift_labels = batch.labels[:, 1:]
        mask = shift_labels != -100
        if not bool(mask.any()):
            return {"token_accuracy": 0.0}
        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels) & mask
        return {"token_accuracy": float(correct.sum().item() / mask.sum().item())}

    def target_position_losses(self, logits: torch.Tensor, batch: NumberNamingBatch) -> dict[str, float]:
        shift_logits = logits[:, :-1, :]
        shift_labels = batch.labels[:, 1:]
        losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_labels)
        metrics = {}
        target_lengths = (shift_labels != -100).sum(dim=1)
        max_target_length = int(target_lengths.max().item()) if len(target_lengths) else 0
        for target_position in range(max_target_length):
            values = []
            for row in range(shift_labels.shape[0]):
                target_indices = torch.nonzero(shift_labels[row] != -100, as_tuple=False).flatten()
                if target_position < len(target_indices):
                    values.append(losses[row, target_indices[target_position]])
            if values:
                metrics[f"target_position_loss_{target_position}"] = _tensor_loss_bits(torch.stack(values).mean())
        return metrics

    def value_position_losses(self, logits: torch.Tensor, batch: NumberNamingBatch) -> dict[str, float]:
        shift_logits = logits[:, :-1, :]
        shift_labels = batch.labels[:, 1:]
        losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_labels)
        mask = shift_labels != -100
        metrics = {}
        roles_by_row = [
            _value_position_roles(number, text.split())
            for number, text in zip(batch.numbers, batch.texts)
        ]
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
            values = []
            for row, roles in enumerate(roles_by_row):
                role_indices = [index for index, value_role in enumerate(roles) if value_role == role]
                if not role_indices:
                    continue
                row_losses = losses[row][mask[row]]
                if not len(row_losses):
                    continue
                values.extend(row_losses[index] for index in role_indices if index < len(row_losses))
            if values:
                metrics[f"value_position_loss_{role}"] = _tensor_loss_bits(torch.stack(values).mean())
        return metrics

    def digit_length_losses(self, logits: torch.Tensor, batch: NumberNamingBatch) -> dict[str, float]:
        shift_logits = logits[:, :-1, :]
        shift_labels = batch.labels[:, 1:]
        losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_labels)
        mask = shift_labels != -100
        metrics = {}
        for digit_length in sorted({len(str(int(number))) for number in batch.numbers}):
            rows = [
                row
                for row, number in enumerate(batch.numbers)
                if len(str(int(number))) == digit_length and bool(mask[row].any())
            ]
            if not rows:
                continue
            values = [losses[row][mask[row]].mean() for row in rows]
            metrics[f"digit_length_loss_{digit_length}"] = _tensor_loss_bits(torch.stack(values).mean())
        return metrics

    def token_losses(self, logits: torch.Tensor, batch: NumberNamingBatch) -> dict[str, float]:
        shift_logits = logits[:, :-1, :]
        shift_labels = batch.labels[:, 1:]
        losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_labels)
        metrics = {}
        ignored_tokens = set(SPECIAL_TOKENS) | set(DIGIT_TOKENS)
        for token_id in sorted(set(int(value.item()) for value in shift_labels[shift_labels != -100])):
            token = self.tokenizer.id_to_token[token_id]
            if token in ignored_tokens:
                continue
            values = losses[shift_labels == int(token_id)]
            if len(values):
                metrics[f"token_loss_{token}"] = _tensor_loss_bits(values.mean())
        return metrics

    def greedy_decode(self, model, numbers: list[int], *, device) -> list[list[str]]:
        if not numbers:
            return []
        prefixes = [
            [self.tokenizer.bos_id]
            + [self.tokenizer.token_to_id[f"<D{digit}>"] for digit in self.input_digit_string(number)]
            + [self.tokenizer.sep_id]
            for number in numbers
        ]
        generated: list[list[int]] = [[] for _ in numbers]
        finished = [False for _ in numbers]
        max_seq_len = int(self.config.max_seq_len)
        model.eval()
        with torch.no_grad():
            for _ in range(max_seq_len - min(len(prefix) for prefix in prefixes)):
                active_rows = [
                    index
                    for index, prefix in enumerate(prefixes)
                    if not finished[index] and len(prefix) + len(generated[index]) < max_seq_len
                ]
                if not active_rows:
                    break
                sequences = [prefixes[index] + generated[index] for index in active_rows]
                current_len = max(len(sequence) for sequence in sequences)
                input_ids = torch.full(
                    (len(sequences), current_len),
                    self.tokenizer.pad_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row, sequence in enumerate(sequences):
                    input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
                    attention_mask[row, : len(sequence)] = 1
                last_indices = attention_mask.sum(dim=1).sub(1)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                row_indices = torch.arange(len(sequences), device=device)
                next_ids = logits[row_indices, last_indices].argmax(dim=-1).detach().cpu().tolist()
                for index, next_id in zip(active_rows, next_ids):
                    generated[index].append(int(next_id))
                    if int(next_id) == self.tokenizer.eos_id:
                        finished[index] = True
        return [self.tokenizer.decode_words(tokens) for tokens in generated]

    def decode_metrics(self, model, examples: list[NumberNamingExample], *, device) -> dict[str, float]:
        predictions = self.greedy_decode(model, [example.number for example in examples], device=device)
        exact = 0
        exact_wo_eos = 0
        exact_by_digit: dict[int, int] = {}
        exact_wo_eos_by_digit: dict[int, int] = {}
        count_by_digit: dict[int, int] = {}
        normalized = []
        for prediction, example in zip(predictions, examples):
            target = example.text.split()
            is_exact = int(prediction == target)
            is_exact_wo_eos = int(_exact_without_eos(prediction, target))
            digit_length = len(str(int(example.number)))
            exact += is_exact
            exact_wo_eos += is_exact_wo_eos
            exact_by_digit[digit_length] = exact_by_digit.get(digit_length, 0) + is_exact
            exact_wo_eos_by_digit[digit_length] = exact_wo_eos_by_digit.get(digit_length, 0) + is_exact_wo_eos
            count_by_digit[digit_length] = count_by_digit.get(digit_length, 0) + 1
            distance = _levenshtein(prediction, target)
            normalized.append(distance / max(len(prediction), len(target), 1))
        count = max(len(examples), 1)
        metrics = {
            "exact_accuracy": float(exact / count),
            "exact_accuracy_wo_eos": float(exact_wo_eos / count),
            "normalized_edit_distance": float(sum(normalized) / count),
        }
        for digit_length in sorted(count_by_digit):
            metrics[f"digit_length_exact_accuracy_{digit_length}"] = exact_by_digit[digit_length] / max(count_by_digit[digit_length], 1)
            metrics[f"digit_length_exact_accuracy_wo_eos_{digit_length}"] = exact_wo_eos_by_digit[digit_length] / max(count_by_digit[digit_length], 1)
        return metrics

    def _validate_lengths(self) -> None:
        for example in self.splits.all_examples:
            self.tokenizer.encode(
                example.number,
                example.text,
                max_seq_len=int(self.config.max_seq_len),
            )
        if self.probe_suite is not None:
            for probe in self.probe_suite.probes:
                for example in probe.examples:
                    length = 1 + len(self.input_digit_string(example.number)) + 1 + len(example.prefix)
                    if length > int(self.config.max_seq_len):
                        raise ValueError(
                            f"Probe example exceeds max_seq_len={self.config.max_seq_len}: "
                            f"{example.number} -> {' '.join(example.prefix)!r}"
                        )
                    for word in example.prefix:
                        if word not in self.tokenizer.token_to_id:
                            raise ValueError(f"Probe prefix token missing from vocabulary: {word!r}")
                    if example.target != "EOS" and example.target not in self.tokenizer.token_to_id:
                        raise ValueError(f"Probe target token missing from vocabulary: {example.target!r}")

    def input_digit_string(self, number: int) -> str:
        return str(int(number))


class NumberNaming(NumberNamingTask):
    pass


def _sample_eval_examples(config, examples: list[NumberNamingExample]) -> list[NumberNamingExample]:
    strategy = str(getattr(config, "eval_strategy", "digits_wise")).lower()
    if strategy == "uniform":
        return _sample_uniform_eval_examples(config, examples)
    if strategy == "digits_wise":
        return _sample_eval_examples_per_digit(config, examples)
    if strategy == "balanced":
        return _sample_balanced_eval_examples(config, examples)
    raise ValueError(f"Unsupported eval_strategy: {strategy!r}")


def _sample_uniform_eval_examples(config, examples: list[NumberNamingExample]) -> list[NumberNamingExample]:
    eval_samples = getattr(config, "eval_samples", None)
    if eval_samples is None:
        return list(examples)
    rng = random.Random(int(config.split_seed) + 17_029)
    selected = rng.sample(examples, min(int(eval_samples), len(examples)))
    selected.sort(key=lambda example: (int(example.number), example.text))
    return selected


def _sample_eval_examples_per_digit(config, examples: list[NumberNamingExample]) -> list[NumberNamingExample]:
    max_per_digit = getattr(config, "eval_samples_per_digit", None)
    if max_per_digit is None:
        return list(examples)
    by_digits: dict[int, list[NumberNamingExample]] = {}
    for example in examples:
        by_digits.setdefault(len(str(int(example.number))), []).append(example)
    rng = random.Random(int(config.split_seed) + 17_029)
    selected: list[NumberNamingExample] = []
    for digit_length in sorted(by_digits):
        bucket = by_digits[digit_length]
        count = min(int(max_per_digit), len(bucket))
        selected.extend(rng.sample(bucket, count))
    selected.sort(key=lambda example: (len(str(int(example.number))), int(example.number), example.text))
    return selected


def _sample_balanced_eval_examples(config, examples: list[NumberNamingExample]) -> list[NumberNamingExample]:
    eval_samples = getattr(config, "eval_samples", None)
    max_per_digit = getattr(config, "eval_samples_per_digit", None)
    if eval_samples is None and max_per_digit is None:
        return list(examples)
    by_digits = {len(str(int(example.number))) for example in examples}
    if not by_digits:
        return []
    total_count = int(eval_samples) if eval_samples is not None else int(max_per_digit) * len(by_digits)
    rng = random.Random(int(config.split_seed) + 17_029)
    return _sample_balanced_digit_examples(
        examples,
        total_count=total_count,
        rng=rng,
        split_name="balanced eval",
    )


def _exact_without_eos(prediction: list[str], target: list[str]) -> bool:
    return len(prediction) >= len(target) and prediction[: len(target)] == target


def _tensor_loss_bits(value: torch.Tensor) -> float:
    return float((value.detach() * NATS_TO_BITS).cpu().item())


def _levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _value_position_roles(number: int, words: list[str]) -> list[str]:
    if not words:
        return []
    if number < 0:
        raise ValueError("number must be non-negative.")
    if number < 20:
        return ["units"]
    if number < 100:
        tens, ones = divmod(number, 10)
        roles = ["tens"]
        if ones:
            roles.append("units")
        return roles
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        roles = ["hundreds", "hundreds"]
        if remainder:
            roles.extend(_value_position_roles(remainder, words[2:]))
        return roles
    for scale, role in [
        (1_000_000_000, "billions"),
        (1_000_000, "millions"),
        (1_000, "thousands"),
    ]:
        block, remainder = divmod(number, scale)
        if block:
            block_words = english_number_name(block).split()
            roles = [
                _scale_block_role(block_role, scale)
                for block_role in _value_position_roles(block, block_words)
            ] + [role]
            if remainder:
                roles.extend(_value_position_roles(remainder, english_number_name(remainder).split()))
            return roles
    raise ValueError(f"Cannot derive value positions for number: {number}")


def _scale_block_role(role: str, scale: int) -> str:
    scale_roles = {
        1_000: {
            "units": "thousands",
            "tens": "ten_thousands",
            "hundreds": "hundred_thousands",
        },
        1_000_000: {
            "units": "millions",
            "tens": "ten_millions",
            "hundreds": "hundred_millions",
        },
        1_000_000_000: {
            "units": "billions",
            "tens": "ten_billions",
            "hundreds": "hundred_billions",
        },
    }
    return scale_roles.get(int(scale), {}).get(role, role)
