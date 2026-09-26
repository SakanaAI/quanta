from __future__ import annotations

import torch

IGNORE_INDEX = -100
CNAND_TOKEN_OUT = 2
CNAND_TOKEN_PAD = 3
CNAND_TOKEN_SIGN_POS = 4
CNAND_TOKEN_SIGN_NEG = 5



def task_probability_tensor(codes: list[int], task_frequencies: dict[int, float] | None, device) -> torch.Tensor:
    weights = torch.tensor(
        [float((task_frequencies or {}).get(int(code), 1.0)) for code in codes],
        dtype=torch.float32,
        device=device,
    )
    if torch.any(weights < 0) or float(weights.sum().item()) <= 0:
        raise ValueError("Task frequencies must have positive total weight.")
    return weights / weights.sum()


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


def ancestral_closure(node: int, graph_dependencies: dict[int, list[int]] | None) -> set[int]:
    closure = {int(node)}
    for parent in (graph_dependencies or {}).get(int(node), []):
        closure.update(ancestral_closure(int(parent), graph_dependencies))
    return closure


def _node_depth(node: int, graph_dependencies: dict[int, list[int]] | None) -> int:
    parents = list((graph_dependencies or {}).get(int(node), []))
    if not parents:
        return 0
    return 1 + max(_node_depth(int(parent), graph_dependencies) for parent in parents)
