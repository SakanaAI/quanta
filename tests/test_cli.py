import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from main import parse_args, run_cli


def test_cli_runs_resolved_experiment() -> None:
    experiment = mock.Mock()
    with (
        mock.patch.object(
            sys,
            "argv",
            ["main.py", "scaling_laws/cnand/loss_decomposition"],
        ),
        mock.patch("main.load_plot_config", return_value=SimpleNamespace()) as load_plot,
        mock.patch(
            "main.load_experiment_config",
            return_value=SimpleNamespace(),
        ) as load_experiment,
        mock.patch("main.create_experiment", return_value=experiment) as create,
    ):
        run_cli()

    load_plot.assert_called_once_with(
        Path("configs/plot/default.yaml"),
        experiment_name="scaling_laws",
    )
    load_experiment.assert_called_once_with(
        "scaling_laws",
        Path("configs/scaling_laws/cnand/loss_decomposition.yaml"),
    )
    create.assert_called_once_with(
        "scaling_laws",
        load_experiment.return_value,
        load_plot.return_value,
    )
    experiment.run.assert_called_once_with()


def test_cli_accepts_config_and_optional_plot() -> None:
    with mock.patch.object(
        sys,
        "argv",
        [
            "main.py",
            "quanta_discovery/number_naming/main",
            "--plot",
            "default",
        ],
    ):
        args = parse_args()

    assert args.config == "quanta_discovery/number_naming/main"
    assert args.plot == "default"
