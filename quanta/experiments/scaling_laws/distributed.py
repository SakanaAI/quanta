from __future__ import annotations

from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list


def broadcast_object(accelerator: Accelerator, value: Any) -> Any:
    if accelerator.num_processes == 1:
        return value
    values = [value]
    broadcast_object_list(values, from_process=0)
    return values[0]


def sum_across_processes(accelerator: Accelerator, value: torch.Tensor) -> torch.Tensor:
    if accelerator.num_processes == 1:
        return value
    gathered = accelerator.gather(value.reshape(1))
    return gathered.sum()


def zero_grad_loss(model: nn.Module) -> torch.Tensor:
    total = None
    for parameter in model.parameters():
        term = parameter.sum() * 0.0
        total = term if total is None else total + term
    if total is None:
        raise ValueError("model has no parameters.")
    return total
