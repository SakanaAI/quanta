from __future__ import annotations

import torch
from accelerate import Accelerator

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.batch_common import local_batch_size_for_process
from quanta.experiments.scaling_laws.distributed import sum_across_processes, zero_grad_loss
from quanta.experiments.scaling_laws.batches import build_sampled_cnand_batch
from quanta.experiments.scaling_laws.training.trainer_batching import (
    _batch_size,
    _local_loss_sum,
    _permute_batch,
)


def process_local_batch_size(config: ScalingLawsConfig, accelerator: Accelerator) -> int:
    microbatch_size = int(config.batch_size) // int(config.gradient_accumulation_steps)
    return local_batch_size_for_process(
        microbatch_size,
        int(accelerator.num_processes),
        int(accelerator.process_index),
    )


def run_training_step(
    *,
    model,
    optimizer,
    loss_sum_fn,
    config: ScalingLawsConfig,
    task_spec,
    task_probabilities: torch.Tensor,
    batch_cache: dict[str, torch.Tensor],
    local_batch_size: int,
    device,
    accelerator: Accelerator,
    fixed_microbatches=None,
    return_task_counts: bool = False,
):
    optimizer.zero_grad()
    local_loss_sum_total = torch.zeros((), device=device)
    local_counts_total = torch.zeros(
        (len(task_spec.codes),), dtype=torch.long, device=device
    )
    ideal_index = None
    for microbatch_index in range(int(config.gradient_accumulation_steps)):
        if fixed_microbatches is None:
            x, y = build_sampled_cnand_batch(
                config=config, task_spec=task_spec, probabilities=task_probabilities,
                batch_cache=batch_cache, device=device, batch_size=local_batch_size,
                ideal_index=ideal_index,
            )
        else:
            x, y = fixed_microbatches[microbatch_index]
        if config.trace_sampling == "ideal_threshold" and local_batch_size:
            ideal_index = x["ideal_indices"][0].detach()
        permutation = torch.randperm(_batch_size(x), device=device)
        x = _permute_batch(x, permutation)
        y = y[permutation]
        if local_batch_size:
            with accelerator.autocast():
                local_loss_sum = _local_loss_sum(model, x, y, loss_sum_fn, config)
            loss = local_loss_sum * accelerator.num_processes / int(config.batch_size)
        else:
            local_loss_sum = torch.zeros((), device=device)
            loss = zero_grad_loss(model)
        accelerator.backward(loss)
        local_loss_sum_total += local_loss_sum.detach()
        if config.trace_sampling in {
            "ideal_threshold",
            "ideal_path",
        }:
            local_counts_total += x["active_node_mask"].to(dtype=torch.long).sum(dim=0)
        else:
            local_counts_total += torch.bincount(
                x["target_indices"].reshape(-1).to(dtype=torch.long),
                minlength=len(task_spec.codes),
            ).to(dtype=torch.long)
    optimizer.step()
    global_loss_sum = sum_across_processes(accelerator, local_loss_sum_total)
    loss_value = float((global_loss_sum / int(config.batch_size)).item())
    if not return_task_counts:
        return loss_value
    if accelerator.num_processes > 1:
        global_counts = accelerator.gather(local_counts_total).reshape(accelerator.num_processes, -1).sum(dim=0)
    else:
        global_counts = local_counts_total
    return loss_value, [int(value) for value in global_counts.detach().cpu().tolist()]
