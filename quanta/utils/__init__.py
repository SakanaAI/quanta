from quanta.figures import plot_discovery_trajectories
from .utils import get_device, maybe_data_parallel, set_seeds, unwrap_model, ForkRNG
from .parity import bipolar_random_bits, bipolar_parity_labels, bipolar_parity_labels_for_indices
from .utils import theoretical_alpha, _jsonable, _slug_number, config_pair_slug, input_slug

__all__ = [
    "set_seeds",
    "get_device",
    "maybe_data_parallel",
    "unwrap_model",
    "ForkRNG",
    "plot_discovery_trajectories",
    "bipolar_random_bits",
    "bipolar_parity_labels",
    "bipolar_parity_labels_for_indices"
]
