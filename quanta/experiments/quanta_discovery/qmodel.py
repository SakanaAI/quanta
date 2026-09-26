from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn

@dataclass(frozen=True)
class AttentionQuantumCache:
    """Per-quantum scalar-value KV cache for one Q-layer."""

    keys: torch.Tensor
    values: torch.Tensor

class QuantumReadout(nn.Module):
    """Counted source-compatible readout over the reconstructed residual."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(int(d_model), int(vocab_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden))

    @torch.no_grad()
    def copy_source_readout(self, source: nn.Module) -> None:
        """Copy the current source normalization and token head exactly."""

        self.norm.load_state_dict(source.final_norm.state_dict(), strict=True)
        self.head.load_state_dict(source.head.state_dict(), strict=True)


class CheapCausalAttentionQuantumLayer(nn.Module):
    """Causal low-rank attention and a ragged writer bank per quantum.

    Attention supplies writer context and can also contribute a counted,
    per-quantum residual projection. Sparse writers remain independently gated
    so the two functional paths can still be isolated experimentally.
    """

    def __init__(
        self,
        d_model: int,
        quantum_count: int,
        *,
        attention_rank: int = 1,
        attention_value_dim: int = 1,
        attention_direct_residual: bool = False,
        share_attention_projections: bool = False,
        attention_query_offsets: bool = False,
        writer_counts: Sequence[int] | None = None,
        writer_average_k: float = 1.0,
        writer_activation: str = "jumprelu",
        writer_sparsity: str = "batch_topk",
        writer_jump_threshold: float = 0.1,
        writer_jump_bandwidth: float = 0.1,
    ) -> None:
        super().__init__()
        if int(d_model) <= 0 or int(quantum_count) <= 0:
            raise ValueError("d_model and quantum_count must be positive")
        if int(attention_rank) <= 0 or int(attention_value_dim) <= 0:
            raise ValueError("attention rank and value dimension must be positive")
        self.d_model = int(d_model)
        self.quantum_count = int(quantum_count)
        self.attention_rank = int(attention_rank)
        self.attention_value_dim = int(attention_value_dim)
        self.attention_direct_residual = bool(attention_direct_residual)
        self.share_attention_projections = bool(share_attention_projections)
        self.attention_query_offsets = bool(attention_query_offsets)
        if not math.isfinite(float(writer_average_k)) or float(writer_average_k) <= 0:
            raise ValueError("writer_average_k must be finite and positive")
        self.writer_average_k = float(writer_average_k)
        if writer_activation not in {"relu", "jumprelu"}:
            raise ValueError("writer_activation must be 'relu' or 'jumprelu'")
        if writer_sparsity not in {"batch_topk", "l0_target"}:
            raise ValueError("writer_sparsity must be 'batch_topk' or 'l0_target'")
        if writer_sparsity == "l0_target" and writer_activation != "jumprelu":
            raise ValueError("l0_target writer sparsity requires jumprelu activation")
        self.writer_activation = str(writer_activation)
        self.writer_sparsity = str(writer_sparsity)
        if (
            not math.isfinite(float(writer_jump_threshold))
            or float(writer_jump_threshold) < 0
        ):
            raise ValueError("writer_jump_threshold must be finite and non-negative")
        if (
            not math.isfinite(float(writer_jump_bandwidth))
            or float(writer_jump_bandwidth) <= 0
        ):
            raise ValueError("writer_jump_bandwidth must be finite and positive")
        self.writer_jump_bandwidth = float(writer_jump_bandwidth)
        counts = torch.as_tensor(
            tuple(writer_counts) if writer_counts is not None else (1,) * self.quantum_count,
            dtype=torch.long,
        )
        if counts.shape != (self.quantum_count,) or bool(torch.any(counts <= 0)):
            raise ValueError(
                "writer_counts must provide one positive count per quantum"
            )
        owners = torch.repeat_interleave(
            torch.arange(self.quantum_count, dtype=torch.long), counts
        )
        # Allocation is immutable architecture metadata and is recorded in the
        # run summary. Keeping these buffers non-persistent preserves exact
        # compatibility with historical one-writer state dicts.
        self.register_buffer("writer_counts", counts, persistent=False)
        self.register_buffer("writer_owner", owners, persistent=False)
        self.writer_count = int(owners.numel())
        self.norm = nn.LayerNorm(self.d_model, elementwise_affine=False)

        projection_shape = (
            () if self.share_attention_projections else (self.quantum_count,)
        )
        self.query_weight = nn.Parameter(
            torch.empty(*projection_shape, self.attention_rank, self.d_model)
        )
        self.key_weight = nn.Parameter(
            torch.empty(*projection_shape, self.attention_rank, self.d_model)
        )
        self.value_weight = nn.Parameter(
            torch.empty(
                *projection_shape, self.attention_value_dim, self.d_model
            )
        )
        if self.attention_query_offsets:
            # Preserve the shared-query model at initialization, then let each
            # quantum learn which causal keys it should prefer.
            self.query_offsets = nn.Parameter(
                torch.zeros(self.quantum_count, self.attention_rank)
            )
        else:
            self.register_parameter("query_offsets", None)
        self.local_weight = nn.Parameter(
            torch.empty(self.writer_count, self.d_model)
        )
        self.local_bias = nn.Parameter(torch.zeros(self.writer_count))
        self.context_scale = nn.Parameter(
            torch.ones(self.writer_count, self.attention_value_dim)
        )
        self.output_weight = nn.Parameter(
            torch.empty(self.writer_count, self.d_model)
        )
        if self.attention_direct_residual:
            self.attention_output_weight = nn.Parameter(
                torch.empty(
                    self.quantum_count, self.attention_value_dim, self.d_model
                )
            )
        else:
            self.register_parameter("attention_output_weight", None)
        self.register_buffer(
            "writer_jump_threshold",
            torch.full((self.writer_count,), float(writer_jump_threshold)),
        )
        # The cutoff is learned from training batches and frozen for all
        # evaluation/autoregressive executions.
        self.register_buffer(
            "writer_cutoffs", torch.full((self.quantum_count,), float("inf"))
        )

        self.gate_weight = nn.Parameter(
            torch.empty(self.quantum_count, self.d_model)
        )
        self.gate_context_weight = nn.Parameter(
            torch.zeros(self.quantum_count, self.attention_value_dim)
        )
        self.gate_bias = nn.Parameter(torch.zeros(self.quantum_count))

        projection_scale = 1.0 / math.sqrt(self.d_model)
        nn.init.normal_(self.query_weight, std=projection_scale)
        nn.init.normal_(self.key_weight, std=projection_scale)
        nn.init.normal_(self.value_weight, std=projection_scale)
        nn.init.normal_(self.local_weight, std=projection_scale)
        nn.init.normal_(self.gate_weight, std=projection_scale)
        nn.init.normal_(self.output_weight, std=0.02)
        if self.attention_output_weight is not None:
            nn.init.normal_(self.attention_output_weight, std=0.02)

    @property
    def parameters_per_writer(self) -> int:
        """Logical and physical parameters in one scalar transcoder writer."""

        return 2 * self.d_model + self.attention_value_dim + 1

    def gate_logits(
        self, normalized: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Compute one linear gate logit per quantum."""

        if normalized.ndim == 3:
            linear = torch.einsum("btd,qd->btq", normalized, self.gate_weight)
        elif normalized.ndim == 4:
            linear = torch.einsum("btqd,qd->btq", normalized, self.gate_weight)
        else:
            raise ValueError("gate normalized state must have rank 3 or 4")
        logits = (
            linear
            + (self.gate_context_weight[None, None] * context).sum(dim=-1)
            + self.gate_bias
        )
        return logits


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module
