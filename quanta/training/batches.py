from __future__ import annotations

import torch

from quanta.tasks import CXOR


def proportional_batch_sizes(batch_size: int, codes: list[int], frequencies: dict[int, float] | None):
    if not codes:
        raise ValueError("Weighted batch sizing needs at least one task code.")
    weights = [float(frequencies.get(int(code), 1.0)) if frequencies else 1.0 for code in codes]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Task frequencies must have positive total weight.")

    raw_sizes = [batch_size * weight / sum(weights) for weight in weights]
    sizes = [int(value) for value in raw_sizes]
    remainder = batch_size - sum(sizes)
    order = sorted(range(len(codes)), key=lambda index: raw_sizes[index] - sizes[index], reverse=True)
    for index in order[:remainder]:
        sizes[index] += 1
    return sizes


def local_batch_size_for_process(global_batch_size: int, num_processes: int, process_index: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive.")
    if num_processes <= 0:
        raise ValueError("num_processes must be positive.")
    if process_index < 0 or process_index >= num_processes:
        raise ValueError("process_index must be in [0, num_processes).")
    base = global_batch_size // num_processes
    remainder = global_batch_size % num_processes
    return base + (1 if process_index < remainder else 0)


def split_batch_sizes_for_process(global_sizes: list[int], num_processes: int, process_index: int) -> list[int]:
    if num_processes <= 0:
        raise ValueError("num_processes must be positive.")
    if process_index < 0 or process_index >= num_processes:
        raise ValueError("process_index must be in [0, num_processes).")

    local_sizes = []
    for size in global_sizes:
        if size < 0:
            raise ValueError("batch sizes must be non-negative.")
        base = size // num_processes
        remainder = size % num_processes
        local_sizes.append(base + (1 if process_index < remainder else 0))
    return local_sizes


def get_batch(
    task_type: str,
    n_tasks: int,
    n_bits: int,
    Ss: list,
    codes: list,
    sizes: list,
    device,
    dtype=torch.float32,
    graph_dependencies: dict | None = None,
):
    task_type_lower = task_type.lower()

    if "cxor" in task_type_lower:
        return CXOR.get_batch(
            n_tasks=n_tasks,
            n=n_bits,
            Ss=Ss,
            codes=codes,
            sizes=sizes,
            device=device,
            dtype=dtype,
            graph_dependencies=graph_dependencies,
        )

    raise ValueError(f"Unsupported task type for get_batch: {task_type}")
