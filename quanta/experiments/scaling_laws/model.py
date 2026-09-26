from __future__ import annotations

import torch
import torch.nn as nn

from quanta.config import ScalingLawsConfig
from quanta.models import MLP, SquareActivation


class MultitaskSparseParityMLP(nn.Module):
    def __init__(self, *, config: ScalingLawsConfig):
        super().__init__()
        activation = nn.ReLU() if config.activation == "relu" else SquareActivation()
        self.mlp = MLP(
            int(config.n_tasks) + int(config.n_bits),
            2,
            int(config.depth),
            int(config.width),
            activation=activation,
            layernorm=bool(config.layernorm),
        )

    @property
    def input_dim(self) -> int:
        return int(self.mlp.mlp[0].in_features)

    def forward(self, batch: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"] if isinstance(batch, dict) else batch
        return self.mlp(features)


class CNANDTransformerModel(nn.Module):
    def __init__(self, *, config: ScalingLawsConfig, n_slots: int):
        super().__init__()
        self.n_slots = int(n_slots)
        self.width = int(config.width)
        n_lut_functions = getattr(config, "n_lut_functions", None)
        if n_lut_functions is None:
            vocab_size = 4
        elif int(n_lut_functions) == -1:
            vocab_size = 4 + int(config.n_tasks)
        else:
            vocab_size = 4 + int(n_lut_functions)
        self.token_embedding = nn.Embedding(vocab_size, self.width)
        self.slot_embedding = nn.Embedding(self.n_slots, self.width)
        layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=int(config.n_heads),
            dim_feedforward=int(round(float(config.mlp_ratio) * self.width)),
            dropout=float(config.dropout),
            activation=str(config.transformer_activation),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config.depth),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(self.width, 2)

    @property
    def input_dim(self) -> int:
        return self.n_slots

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.residual_stream(batch)
        return self.head(encoded)

    def residual_stream(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.token_embedding(batch["input_ids"])
        hidden = hidden + self.slot_embedding(batch["slot_ids"])
        return self.encoder(hidden, src_key_padding_mask=~batch["active_mask"])

    def forward_with_residual_stream(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.residual_stream(batch)
        return self.head(encoded), encoded
