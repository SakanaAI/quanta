from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from quanta.config import QuantaDiscoveryConfig
from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.number_naming.data import NumberNamingExample
from quanta.experiments.number_naming.task import NumberNamingTask


OBSERVATION_BATCH_SIZE = 256


@dataclass(frozen=True)
class PredictionEventPanel:
    """A fixed finite panel with one row per next-token prediction event."""

    event_ids: tuple[str, ...]
    splits: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prediction_positions: torch.Tensor
    target_ids: torch.Tensor
    example_indices: tuple[int, ...]
    target_positions: tuple[int, ...]
    numbers: tuple[int, ...]
    texts: tuple[str, ...]
    visible_token_ids: tuple[tuple[int, ...], ...]
    target_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.event_ids)
        if len(self.splits) != count:
            raise ValueError("prediction event split annotations have wrong length.")
        if any(split not in {"train", "eval"} for split in self.splits):
            raise ValueError("prediction event splits must be train or eval.")
        if len(set(self.event_ids)) != count:
            raise ValueError("prediction event identifiers must be unique.")
        if any(
            int(tensor.shape[0]) != count
            for tensor in (
                self.input_ids,
                self.attention_mask,
                self.prediction_positions,
                self.target_ids,
            )
        ):
            raise ValueError("prediction event tensors have inconsistent rows.")

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def batches(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for start in range(0, len(self), int(batch_size)):
            end = min(start + int(batch_size), len(self))
            selection = slice(start, end)
            yield (
                selection,
                self.input_ids[selection].to(device),
                self.attention_mask[selection].to(device),
                self.prediction_positions[selection].to(device),
                self.target_ids[selection].to(device),
            )

    def metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "event_id": self.event_ids[index],
                "split": self.splits[index],
                "example_index": int(self.example_indices[index]),
                "number": int(self.numbers[index]),
                "text": self.texts[index],
                "target_position": int(self.target_positions[index]),
                "prediction_position": int(self.prediction_positions[index]),
                "visible_token_ids": list(self.visible_token_ids[index]),
                "target_token_id": int(self.target_ids[index]),
                "target_token": self.target_tokens[index],
            }
            for index in range(len(self))
        ]


@dataclass(frozen=True)
class TrainingTrajectory:
    checkpoint_steps: tuple[int, ...]
    checkpoint_paths: tuple[str, ...]
    losses: np.ndarray
    optimizer_steps: tuple[int, ...] = ()
    learning_rates: np.ndarray | None = None
    layer_updates: np.ndarray | None = None
    mlp_layer_updates: np.ndarray | None = None
    training_losses: np.ndarray | None = None
    gd_identity_max_abs_error: float | None = None

    @property
    def num_checkpoints(self) -> int:
        return len(self.checkpoint_steps)


def build_model_and_task(
    config: QuantaDiscoveryConfig,
    *,
    device: torch.device,
) -> tuple[NumberNamingTask, DecoderTransformerLM]:
    task = NumberNamingTask(config)
    _resolve_training_steps(config, len(task.train))
    model = DecoderTransformerLM(
        vocab_size=task.tokenizer.vocab_size,
        max_seq_len=int(config.max_seq_len),
        d_model=int(config.d_model),
        n_layers=int(config.n_layers),
        n_heads=int(config.n_heads),
        dropout=float(config.dropout),
        pad_id=task.tokenizer.pad_id,
        mlp_ratio=float(config.mlp_ratio),
    ).to(device)
    return task, model


def build_prediction_event_panel(
    task: NumberNamingTask,
    *,
    split: str = "train",
    examples: Sequence[NumberNamingExample] | None = None,
) -> PredictionEventPanel:
    """Build train, eval, or combined event panels with stable split-local IDs."""

    normalized_split = str(split).lower()
    if examples is not None:
        if normalized_split not in {"train", "eval"}:
            raise ValueError(
                "an explicit example panel must be labelled train or eval"
            )
        sources = ((normalized_split, list(examples)),)
    elif normalized_split == "train":
        sources = (("train", list(task.train)),)
    elif normalized_split == "eval":
        sources = (("eval", list(task.eval_examples)),)
    elif normalized_split == "all":
        sources = (
            ("train", list(task.train)),
            ("eval", list(task.eval_examples)),
        )
    else:
        raise ValueError("prediction event split must be train, eval, or all.")
    if any(not examples for _, examples in sources):
        empty = [name for name, examples in sources if not examples]
        raise ValueError(
            "prediction event distributions are empty: "
            + ", ".join(empty)
        )
    examples = [
        example
        for _, source_examples in sources
        for example in source_examples
    ]
    example_splits = [
        split_name
        for split_name, source_examples in sources
        for _ in source_examples
    ]
    encoded = task.encode_examples(examples, device=torch.device("cpu"))
    event_ids: list[str] = []
    event_splits: list[str] = []
    input_ids: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    positions: list[int] = []
    target_ids: list[int] = []
    example_indices: list[int] = []
    target_positions: list[int] = []
    numbers: list[int] = []
    texts: list[str] = []
    visible_token_ids: list[tuple[int, ...]] = []
    target_tokens: list[str] = []
    split_event_counts = {name: 0 for name, _ in sources}
    for example_index in range(int(encoded.input_ids.shape[0])):
        split_name = example_splits[example_index]
        target_position = 0
        for prediction_position in range(int(encoded.input_ids.shape[1]) - 1):
            target_id = int(encoded.labels[example_index, prediction_position + 1])
            if target_id == -100:
                continue
            ids = encoded.input_ids[example_index].clone()
            mask = encoded.attention_mask[example_index].clone()
            ids[prediction_position + 1 :] = int(task.tokenizer.pad_id)
            mask[prediction_position + 1 :] = 0
            split_event_index = split_event_counts[split_name]
            event_id = f"{split_name}_z_{split_event_index:06d}"
            split_event_counts[split_name] += 1
            event_ids.append(event_id)
            event_splits.append(split_name)
            input_ids.append(ids)
            attention_masks.append(mask)
            positions.append(prediction_position)
            target_ids.append(target_id)
            example_indices.append(example_index)
            target_positions.append(target_position)
            numbers.append(int(encoded.numbers[example_index]))
            texts.append(str(encoded.texts[example_index]))
            visible_token_ids.append(
                tuple(int(value) for value in ids[: prediction_position + 1].tolist())
            )
            target_tokens.append(str(task.tokenizer.id_to_token[target_id]))
            target_position += 1
    if not event_ids:
        raise ValueError("the selected distributions have no prediction events.")
    return PredictionEventPanel(
        event_ids=tuple(event_ids),
        splits=tuple(event_splits),
        input_ids=torch.stack(input_ids),
        attention_mask=torch.stack(attention_masks),
        prediction_positions=torch.tensor(positions, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        example_indices=tuple(example_indices),
        target_positions=tuple(target_positions),
        numbers=tuple(numbers),
        texts=tuple(texts),
        visible_token_ids=tuple(visible_token_ids),
        target_tokens=tuple(target_tokens),
    )


@torch.no_grad()
def evaluate_prediction_events(
    model: DecoderTransformerLM,
    panel: PredictionEventPanel,
    *,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    loss_parts: list[torch.Tensor] = []
    for _, input_ids, attention_mask, positions, target_ids in panel.batches(
        OBSERVATION_BATCH_SIZE,
        device=device,
    ):
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        rows = torch.arange(len(input_ids), device=device)
        event_logits = logits[rows, positions]
        event_losses = F.cross_entropy(
            event_logits,
            target_ids,
            reduction="none",
        )
        loss_parts.append(event_losses.detach().cpu())
    model.train(was_training)
    return torch.cat(loss_parts).float().numpy()


def write_trajectory_artifacts(
    save_dir: str,
    panel: PredictionEventPanel,
    trajectory: TrainingTrajectory,
) -> None:
    np.save(os.path.join(save_dir, "losses.npy"), trajectory.losses)
    if trajectory.layer_updates is not None:
        np.save(
            os.path.join(save_dir, "layer_updates.npy"),
            trajectory.layer_updates,
        )
    if trajectory.mlp_layer_updates is not None:
        np.save(
            os.path.join(save_dir, "mlp_layer_updates.npy"),
            trajectory.mlp_layer_updates,
        )
    if trajectory.learning_rates is not None:
        np.save(
            os.path.join(save_dir, "learning_rates.npy"),
            trajectory.learning_rates,
        )
    if trajectory.training_losses is not None:
        np.save(
            os.path.join(save_dir, "training_losses.npy"),
            trajectory.training_losses,
        )
    _write_json(os.path.join(save_dir, "events.json"), panel.metadata())
    _write_json(
        os.path.join(save_dir, "checkpoint_metadata.json"),
        {
            "checkpoint_steps": list(trajectory.checkpoint_steps),
            "checkpoint_files": list(trajectory.checkpoint_paths),
            "num_prediction_events": len(panel),
            "num_train_prediction_events": panel.splits.count("train"),
            "num_eval_prediction_events": panel.splits.count("eval"),
            "losses_shape": list(trajectory.losses.shape),
            "optimizer_steps": list(trajectory.optimizer_steps),
            "layer_updates_shape": (
                None
                if trajectory.layer_updates is None
                else list(trajectory.layer_updates.shape)
            ),
            "mlp_layer_updates_shape": (
                None
                if trajectory.mlp_layer_updates is None
                else list(trajectory.mlp_layer_updates.shape)
            ),
            "training_losses_shape": (
                None
                if trajectory.training_losses is None
                else list(trajectory.training_losses.shape)
            ),
            "gd_identity_max_abs_error": trajectory.gd_identity_max_abs_error,
            "deterministic_evaluation": True,
        },
    )


@torch.no_grad()
def save_functional_residual_writes(
    save_dir: str,
    model: DecoderTransformerLM,
    panel: PredictionEventPanel,
    trajectory: TrainingTrajectory,
    *,
    device: torch.device,
) -> None:
    """Save source block writes at every checkpoint in event/write space."""
    writes = []
    original = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    for relative_path in trajectory.checkpoint_paths:
        state = torch.load(os.path.join(save_dir, relative_path), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        checkpoint = []
        for _, ids, mask, positions, _ in panel.batches(OBSERVATION_BATCH_SIZE, device=device):
            trace = model.residual_trace(ids, mask)
            rows = torch.arange(len(ids), device=device)
            checkpoint.append(torch.stack([
                trace.block_boundaries[layer + 1][rows, positions]
                - trace.block_boundaries[layer][rows, positions]
                for layer in range(len(model.layers))
            ], dim=1).cpu())
        writes.append(torch.cat(checkpoint).numpy())
    model.load_state_dict(original, strict=True)
    np.save(os.path.join(save_dir, "functional_residual_writes.npy"), np.stack(writes))


def checkpoint_path(save_dir: str, trajectory: TrainingTrajectory, index: int) -> str:
    return os.path.join(save_dir, trajectory.checkpoint_paths[int(index)])


def _resolve_training_steps(
    config: QuantaDiscoveryConfig,
    dataset_size: int,
) -> None:
    if config.steps is None:
        if config.epochs is None:
            raise ValueError("quanta_discovery requires steps or epochs.")
        config.steps = int(
            math.ceil(
                float(config.epochs)
                * int(dataset_size)
                / int(config.batch_size)
            )
        )
        config.epochs = None
    if int(config.steps) <= 0:
        raise ValueError("training steps must be positive.")


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
