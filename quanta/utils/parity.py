from __future__ import annotations

import torch


def bipolar_random_bits(shape, *, dtype=torch.float32, device="cpu") -> torch.Tensor:
    return torch.randint(0, 2, shape, dtype=dtype, device=device).mul(2).sub(1)


def bipolar_parity_labels(bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = torch.where(mask.bool(), bits, torch.ones((), dtype=bits.dtype, device=bits.device))
    return selected.prod(dim=1).lt(0).to(torch.int64)


def bipolar_parity_labels_for_indices(bits: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros((bits.shape[0],), dtype=torch.int64, device=bits.device)
    return bits[:, indices].prod(dim=1).lt(0).to(torch.int64)
