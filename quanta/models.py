import copy

import torch
import torch.nn as nn


class SquareActivation(nn.Module):
    def forward(self, x):
        return x.square()


def _activation_layer(activation):
    if activation is None:
        return SquareActivation()
    if isinstance(activation, type):
        return activation()
    return copy.deepcopy(activation)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, depth, width, activation=None, layernorm=False, dtype=torch.float32):
        super().__init__()
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(in_features, width))
                if layernorm:
                    layers.append(nn.LayerNorm(width))
                layers.append(_activation_layer(activation))
            elif i == depth - 1:
                layers.append(nn.Linear(width, out_features))
            else:
                layers.append(nn.Linear(width, width))
                if layernorm:
                    layers.append(nn.LayerNorm(width))
                layers.append(_activation_layer(activation))
        self.mlp = nn.Sequential(*layers).to(dtype)

    def forward(self, x):
        return self.mlp(x)
