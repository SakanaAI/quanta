from .base import Experiment
from .quanta_discovery import QuantaDiscoveryExperiment
from .number_naming import PosetsProbingExperiment
from .quanta_net import QuantaNetExperiment, QuantaSteeringExperiment
from .scaling_laws import ScalingLawsExperiment


def create_experiment(name, config, plot_config):
    if name == "quanta_discovery":
        return QuantaDiscoveryExperiment(config, plot_config)
    if name == "quanta_net":
        return QuantaNetExperiment(config, plot_config)
    if name == "quanta_steering":
        return QuantaSteeringExperiment(config, plot_config)
    if name == "posets_probing":
        return PosetsProbingExperiment(config, plot_config)
    if name == "scaling_laws":
        return ScalingLawsExperiment(config, plot_config)
    raise ValueError(f"Unsupported experiment: {name!r}")


__all__ = [
    "Experiment",
    "QuantaDiscoveryExperiment",
    "PosetsProbingExperiment",
    "QuantaNetExperiment",
    "QuantaSteeringExperiment",
    "ScalingLawsExperiment",
    "create_experiment",
]
