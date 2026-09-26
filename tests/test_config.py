from pathlib import Path

import pytest

from quanta.config import (
    QuantaDiscoveryConfig,
    QuantaNetConfig,
    QuantaSteeringConfig,
    ScalingLawsConfig,
    load_experiment_config,
    load_plot_config,
)
from quanta.config.paths import (
    experiment_name_from_path,
    resolve_experiment_config,
    resolve_run_config,
    resolve_utility_config,
)


@pytest.mark.parametrize(
    ("experiment", "path", "expected_type"),
    (
        (
            "scaling_laws",
            "configs/scaling_laws/cnand/loss_decomposition.yaml",
            ScalingLawsConfig,
        ),
        (
            "quanta_discovery",
            "configs/quanta_discovery/number_naming/main.yaml",
            QuantaDiscoveryConfig,
        ),
        (
            "quanta_net",
            "configs/quanta_net/number_naming/main.yaml",
            QuantaNetConfig,
        ),
        (
            "quanta_steering",
            "configs/quanta_steering/number_naming/control.yaml",
            QuantaSteeringConfig,
        ),
        (
            "quanta_steering",
            "configs/quanta_steering/number_naming/aligned.yaml",
            QuantaSteeringConfig,
        ),
    ),
)
def test_release_configs_load(experiment: str, path: str, expected_type: type) -> None:
    assert isinstance(load_experiment_config(experiment, path), expected_type)


def test_run_config_resolves_release_shorthand() -> None:
    resolved = resolve_run_config("scaling_laws/cnand/loss_decomposition")
    assert resolved.experiment_name == "scaling_laws"
    assert resolved.experiment_path == Path(
        "configs/scaling_laws/cnand/loss_decomposition.yaml"
    )
    assert resolved.plot_path == Path("configs/plot/default.yaml")


def test_explicit_config_and_plot_paths_resolve() -> None:
    experiment = Path("configs/quanta_discovery/number_naming/main.yaml")
    plot = Path("configs/plot/default.yaml")
    assert resolve_experiment_config(experiment) == experiment
    assert resolve_utility_config("plot", plot) == plot
    assert experiment_name_from_path(experiment) == "quanta_discovery"
    assert load_plot_config(plot, experiment_name="quanta_discovery")


def test_unknown_config_reports_requested_path() -> None:
    with pytest.raises(FileNotFoundError, match="missing/config.yaml"):
        resolve_experiment_config("missing/config")


def test_non_plot_utility_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported utility config kind"):
        resolve_utility_config("queue", "default")
