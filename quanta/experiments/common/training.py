from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any

import torch

from quanta.utils import _jsonable


def build_adam_optimizer(model: torch.nn.Module, *, lr: float, weight_decay: float, eps: float = 1e-5):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        eps=float(eps),
    )


def set_optimizer_learning_rate(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def save_config(config: Any, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else dict(config)
    with open(os.path.join(save_dir, "config.json"), "w") as handle:
        json.dump(_jsonable(payload), handle, indent=4)


def save_checkpoint(model: torch.nn.Module, optimizer, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))


def save_results(results: dict[str, Any], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.pkl"), "wb") as handle:
        pickle.dump(results, handle)


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(_jsonable(record)) + "\n")
