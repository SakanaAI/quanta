import random

import numpy as np
import torch
import torch.nn as nn

def set_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


get_device = lambda device=None: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def maybe_data_parallel(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


class ForkRNG:
    """
    Context manager to fork the random number generator states for Python random,
    numpy, and PyTorch (CPU, CUDA, and MPS). This guarantees that any evaluation
    actions performed inside the block do not affect the RNG state of training.
    """
    def __enter__(self):
        self.random_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()

        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.cuda_states = torch.cuda.get_rng_state_all()

        self.mps_available = hasattr(torch, "mps") and torch.backends.mps.is_available()
        if self.mps_available:
            self.mps_state = torch.mps.get_rng_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.random_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)

        if self.cuda_available:
            torch.cuda.set_rng_state_all(self.cuda_states)

        if self.mps_available:
            torch.mps.set_rng_state(self.mps_state)


import math
import numpy as np



def theoretical_alpha(rho: float, beta: float, delta: float = 0.0) -> float:
    return float((math.log(beta) / math.log(rho) - 1.0) / (1.0 + delta))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key) if isinstance(key, np.integer) else key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _slug_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def config_pair_slug(rho: float, beta: float, delta: float = 0.0) -> str:
    return f"rho{_slug_number(rho)}-beta{_slug_number(beta)}-delta{_slug_number(delta)}"


def input_slug(config) -> str | None:
    if config.task == "cxor":
        return "parents"
    if config.task == "cnand":
        return "out_tokens"
    return None
