from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerResidualTrace:
    boundaries: tuple[torch.Tensor, ...]
    logits: torch.Tensor

    @property
    def final_hidden(self) -> torch.Tensor:
        return self.boundaries[-1]

    @property
    def block_boundaries(self) -> tuple[torch.Tensor, ...]:
        """Residual stream before block 1 and after each complete attention+MLP block."""

        return self.boundaries[::2]


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = max(1, int(round(int(d_model) * float(mlp_ratio))))
        self.attention_norm = nn.LayerNorm(int(d_model))
        self.attention = nn.MultiheadAttention(
            int(d_model),
            int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(int(d_model))
        self.mlp_input = nn.Linear(int(d_model), hidden)
        self.mlp_output = nn.Linear(hidden, int(d_model))
        self.dropout = nn.Dropout(float(dropout))

    def attention_write(
        self,
        hidden: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return hidden + self.dropout(update)

    def mlp_write(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.mlp_norm(hidden)
        update = self.mlp_output(self.dropout(F.gelu(self.mlp_input(normalized))))
        return hidden + self.dropout(update)


class DecoderTransformerLM(nn.Module):
    """Causal transformer with explicit residual boundaries after every write."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        pad_id: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)
        self.token_embedding = nn.Embedding(int(vocab_size), self.d_model, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(self.d_model, int(n_heads), float(mlp_ratio), float(dropout))
                for _ in range(int(n_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, int(vocab_size))

    @property
    def residual_boundary_count(self) -> int:
        return 1 + 2 * len(self.layers)

    def residual_trace(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        injections: Mapping[int, torch.Tensor] | None = None,
    ) -> TransformerResidualTrace:
        batch, length = input_ids.shape
        if length > self.max_seq_len:
            raise ValueError(f"seq_len={length} exceeds max_seq_len={self.max_seq_len}.")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding_mask = attention_mask == 0 if attention_mask is not None else input_ids == self.pad_id
        injections = dict(injections or {})
        unknown = set(injections) - set(range(self.residual_boundary_count))
        if unknown:
            raise ValueError(f"injection boundaries out of range: {sorted(unknown)}")
        if 0 in injections:
            hidden = hidden + injections[0]
        boundaries = [hidden]
        boundary = 1
        for layer in self.layers:
            hidden = layer.attention_write(hidden, causal_mask=causal_mask, padding_mask=padding_mask)
            if boundary in injections:
                hidden = hidden + injections[boundary]
            boundaries.append(hidden)
            boundary += 1
            hidden = layer.mlp_write(hidden)
            if boundary in injections:
                hidden = hidden + injections[boundary]
            boundaries.append(hidden)
            boundary += 1
        final = self.final_norm(hidden)
        return TransformerResidualTrace(boundaries=tuple(boundaries), logits=self.head(final))

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.residual_trace(input_ids, attention_mask).final_hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.residual_trace(input_ids, attention_mask).logits
