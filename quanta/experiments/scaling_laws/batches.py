from __future__ import annotations

from quanta.experiments.scaling_laws.batch_common import (
    CNAND_TOKEN_OUT,
    CNAND_TOKEN_PAD,
    IGNORE_INDEX,
    ancestral_closure,
    local_batch_size_for_process,
    task_probability_tensor,
)
from quanta.experiments.scaling_laws.batch_evaluation import (
    cached_cnand_eval_batch,
    evaluate_weighted_task_loss,
)
from quanta.experiments.scaling_laws.batch_records import prediction_sample_records
from quanta.experiments.scaling_laws.cnand_batches import (
    build_cnand_batch_cache,
    build_cnand_batch_from_targets,
    build_cnand_batch_from_active_nodes,
    build_cnand_out_token_batch,
    build_sampled_cnand_batch,
    compact_cnand_active_tokens,
    evaluate_cnand_out_values,
    masked_token_accuracy,
    masked_token_cross_entropy,
    uses_cnand_tokens,
    uses_masked_token_supervision,
)
from quanta.experiments.scaling_laws.cnand_cache import build_seeded_cnand_batch_cache


__all__ = [
    "CNAND_TOKEN_OUT",
    "CNAND_TOKEN_PAD",
    "IGNORE_INDEX",
    "ancestral_closure",
    "build_cnand_batch_cache",
    "build_cnand_batch_from_targets",
    "build_cnand_batch_from_active_nodes",
    "build_cnand_out_token_batch",
    "build_seeded_cnand_batch_cache",
    "build_sampled_cnand_batch",
    "cached_cnand_eval_batch",
    "compact_cnand_active_tokens",
    "evaluate_cnand_out_values",
    "evaluate_weighted_task_loss",
    "local_batch_size_for_process",
    "masked_token_accuracy",
    "masked_token_cross_entropy",
    "prediction_sample_records",
    "build_sampled_cnand_batch",
    "task_probability_tensor",
    "uses_cnand_tokens",
    "uses_masked_token_supervision",
]
