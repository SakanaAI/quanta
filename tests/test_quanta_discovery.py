from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quanta.config import QuantaDiscoveryConfig
from quanta.config.loader import load_experiment_config
from quanta.experiments.quanta_discovery.experiment import (
    ALGORITHM_NAME,
    QuantaDiscoveryExperiment,
)
from quanta.experiments.quanta_discovery.priority import _binary_priority_factorization
from quanta.experiments.quanta_discovery.mlp_pilot import (
    MLPDiscoveryPilotConfig,
    _expected_primitive_supports,
    _mlp_config,
    factorize_priority_field,
    run as run_mlp_discovery_pilot,
)
from scripts.train_quanta_model import (
    _binary_gate_metrics,
    _load_priority,
    _reconstruction_coordinates,
    _uniform_replay_steps,
    _writer_l0_target_penalty,
)
from quanta.experiments.quanta_discovery.dynamics import (
    block_parameters_by_layer,
    flattened_block_parameters,
    flattened_mlp_parameters,
)
from quanta.experiments.number_naming.model import DecoderTransformerLM


def test_main_config_matches_release_operating_point() -> None:
    config = load_experiment_config(
        "quanta_discovery",
        Path("configs/quanta_discovery/number_naming/main.yaml"),
    )

    assert isinstance(config, QuantaDiscoveryConfig)
    assert config.batch_size == config.training_size == 300
    assert config.optimizer == "sgd"
    assert config.minimum_candidate_gain == pytest.approx(0.03)
    assert (config.q_attention_rank, config.q_attention_value_dim) == (8, 16)
    assert config.q_attention_direct_residual is False
    assert config.q_share_attention_projections is True
    assert config.q_training_mode == "iid_checkpoint_replay"
    assert config.q_iid_checkpoint_count == 200
    assert config.q_optimizer_steps == 5_000
    assert config.q_checkpoint_steps == (5_000,)
    assert config.q_routing_mode == "initial_residual_plus_parents"
    assert config.q_reconstruction_positions == "all_valid_positions"
    assert config.q_writer_minimum == 3
    assert config.q_writer_overcomplete_factor == pytest.approx(4.0)
    assert config.q_writer_maximum == 3
    assert config.q_writer_total_budget is None
    assert config.q_functional_rank_energy == pytest.approx(0.90)
    assert config.q_writer_activation == "jumprelu"
    assert config.q_writer_sparsity == "l0_target"
    assert config.q_writer_jump_threshold == pytest.approx(0.10)
    assert config.q_writer_jump_bandwidth == pytest.approx(0.10)
    assert config.q_writer_l0_scope == "global_event"
    assert config.q_writer_l0_target == pytest.approx(6.0)
    assert config.q_writer_l0_weight == pytest.approx(1.0)
    assert config.q_writer_l0_ramp_steps == 1_000
    assert config.q_output_kl_weight == pytest.approx(0.70)
    assert config.maximum_edge_closure_cost == pytest.approx(0.10)
    assert config.detach_gate_gradients is True


def test_architecture_agnostic_priority_factorization_recovers_singletons() -> None:
    values = np.asarray(
        [
            [4.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 6.0, 5.0],
        ]
    )
    fit = factorize_priority_field(
        values,
        np.ones(3),
        max_components=5,
        alternating_steps=20,
        minimum_component_gain_fraction=0.01,
        interval_steps=np.arange(10, 70, 10),
    )

    assert fit["support"].shape == (3, 3)
    assert sorted(map(tuple, fit["support"].T.astype(int))) == [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    ]
    assert fit["summary"]["field_explained_fraction"] == pytest.approx(1.0)


def test_priority_factorization_uses_event_initializations_after_broad_factor() -> None:
    pulse = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0])
    values = np.zeros((4, 35))
    for event in range(4):
        start = 10 * event
        values[event, start : start + len(pulse)] = pulse

    fit = factorize_priority_field(
        values,
        np.ones(4),
        max_components=6,
        alternating_steps=20,
        minimum_component_gain_fraction=0.05,
    )

    recovered = set(map(tuple, fit["support"].T.astype(int)))
    assert {
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
    } <= recovered


def test_signed_event_feature_factorization_recovers_singletons() -> None:
    values = np.asarray(
        [
            [2.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, 1.0],
        ]
    )
    fit = factorize_priority_field(
        values,
        np.ones(2),
        max_components=2,
        alternating_steps=20,
        minimum_component_gain_fraction=0.05,
        nonnegative_curves=False,
    )

    assert fit["support"].shape == (2, 2)
    assert sorted(map(tuple, fit["support"].T.astype(int))) == [(0, 1), (1, 0)]
    assert fit["summary"]["field_explained_fraction"] == pytest.approx(1.0)


def test_mlp_discovery_pilot_writes_complete_artifacts(tmp_path: Path) -> None:
    output = run_mlp_discovery_pilot(
        MLPDiscoveryPilotConfig(
            n_tasks=2,
            parity_degree=1,
            support_pool_bits=2,
            samples_per_task=2,
            width=4,
            hidden_layers=1,
            steps=4,
            checkpoint_every=2,
            max_components=3,
            alternating_steps=5,
            minimum_component_gain_fraction=0.0,
            output_dir=str(tmp_path / "mlp-discovery"),
        )
    )

    summary = json.loads((output / "summary.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete"
    assert summary["expected_functional_quanta"] == 2
    assert summary["expected_geometry"] == "antichain"
    assert summary["threshold_sensitivity"]
    assert (output / "priority_analysis.npz").is_file()
    assert (output / "discovery.png").is_file()


def test_mlp_compositional_pilot_declares_nested_primitive_supports() -> None:
    config = _mlp_config(
        MLPDiscoveryPilotConfig(
            n_tasks=8,
            function_group_size=4,
            subgroup_size=2,
            parity_degree=1,
            degree_b=1,
            degree_c=1,
            composition="nested_nand",
        )
    )
    records = _expected_primitive_supports(config)

    assert len(records) == 14
    assert [record["tasks"] for record in records if record["role"] == "A"] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert [record["tasks"] for record in records if record["role"] == "B"] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]
    assert sum(record["role"] == "C" for record in records) == 8


def test_iid_replay_grid_is_uniform_and_includes_source_final_state() -> None:
    steps = _uniform_replay_steps(5_000, 200)

    assert len(steps) == len(set(steps.tolist())) == 200
    assert int(steps[0]) == 1
    assert int(steps[-1]) == 5_000
    assert int(np.diff(steps).max() - np.diff(steps).min()) <= 1


def test_global_writer_l0_scope_counts_all_writers_per_event() -> None:
    quantum = SimpleNamespace(
        logits=torch.zeros(2, 1),
        writer_gates=(
            torch.tensor([[[1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
            torch.tensor([[[1.0, 1.0]], [[0.0, 1.0]]]),
        ),
        local_gates=(
            torch.ones(2, 1, 2),
            torch.ones(2, 1, 1),
        ),
    )
    modules = (
        SimpleNamespace(quantum_count=2, writer_owner=torch.tensor([0, 0, 1])),
        SimpleNamespace(quantum_count=1, writer_owner=torch.tensor([0, 0])),
    )

    penalty, observed = _writer_l0_target_penalty(
        quantum,
        modules,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        scope="global_event",
        target=3.0,
        microbatch_fraction=1.0,
    )

    assert observed.item() == pytest.approx(3.0)
    assert penalty.item() == pytest.approx(0.0)


def test_config_accepts_global_event_writer_l0_scope(tmp_path: Path) -> None:
    path = tmp_path / "global_writer_l0.yaml"
    path.write_text(
        "discovery:\n"
        "  q_writer_l0_scope: global_event\n"
        "  q_writer_l0_target: 6.0\n"
    )

    config = load_experiment_config("quanta_discovery", path)

    assert config.q_writer_l0_scope == "global_event"
    assert config.q_writer_l0_target == pytest.approx(6.0)


def test_global_edge_loader_accepts_explicit_raw_stage_b_directory(
    tmp_path: Path,
) -> None:
    priority_dir = tmp_path / "raw"
    priority_dir.mkdir()
    np.save(priority_dir / "support_layer_0.npy", np.asarray([[1], [0]]))
    np.savez(
        priority_dir / "temporal_priority.npz",
        layer_0=np.asarray([[0.0, 1.0]]),
    )
    (priority_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "threshold_refit",
                "candidate_counts": [1],
                "edges": [[0, 0, 1, 0]],
            }
        )
    )

    supports, temporal, edges, summary = _load_priority(
        tmp_path,
        priority_dir,
        raw_stage_b=True,
    )

    assert supports[0].shape == (2, 1)
    assert temporal[0].shape == (1, 2)
    assert edges == ()
    assert summary["method"] == "threshold_refit_raw_stage_b"


def test_binary_gate_metrics_distinguishes_local_and_effective_predictions() -> None:
    metrics = _binary_gate_metrics(
        np.asarray([[0.9, 0.8], [0.2, 0.1]]),
        np.asarray([[1, 0], [0, 1]], dtype=bool),
        effective_predictions=np.asarray([[1, 0], [0, 0]], dtype=bool),
    )

    assert metrics["local_gate"]["true_positive"] == 1
    assert metrics["local_gate"]["false_positive"] == 1
    assert metrics["local_gate"]["false_negative"] == 1
    assert metrics["local_gate"]["precision"] == pytest.approx(0.5)
    assert metrics["local_gate"]["recall"] == pytest.approx(0.5)
    assert metrics["effective_gate"]["precision"] == pytest.approx(1.0)
    assert metrics["effective_gate"]["recall"] == pytest.approx(0.5)


def test_q_checkpoint_steps_are_sorted_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "q_checkpoints.yaml"
    path.write_text(
        "training:\n  steps: 5000\n"
        "discovery:\n  q_training_mode: iid_checkpoint_replay\n"
        "  q_optimizer_steps: 15000\n"
        "  q_checkpoint_steps: [15000, 5000, 10000]\n"
    )

    config = load_experiment_config("quanta_discovery", path)

    assert config.q_checkpoint_steps == (5000, 10000, 15000)


def test_config_rejects_removed_global_writer_budget(tmp_path: Path) -> None:
    path = tmp_path / "removed_writer_budget.yaml"
    path.write_text("discovery:\n  q_parameter_budget: 0.9\n")
    with pytest.raises(ValueError, match="Unknown quanta_discovery config keys"):
        load_experiment_config("quanta_discovery", path)


def test_all_valid_reconstruction_coordinates_include_nonprediction_tokens() -> None:
    batch = SimpleNamespace(
        labels=torch.tensor([[-100, 7, -100, -100]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )

    event_rows, event_positions = _reconstruction_coordinates(
        batch, "prediction_positions"
    )
    all_rows, all_positions = _reconstruction_coordinates(
        batch, "all_valid_positions"
    )

    assert event_rows.tolist() == [0]
    assert event_positions.tolist() == [0]
    assert all_rows.tolist() == [0, 0, 0]
    assert all_positions.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "key",
    [
        "checkpoint_every",
        "max_candidates_per_layer",
        "candidate_fit_steps",
        "q_microbatch_size",
    ],
)
def test_config_rejects_nonpositive_integer_controls(
    tmp_path: Path, key: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"discovery:\n  {key}: 0\n")
    with pytest.raises(ValueError, match=key):
        load_experiment_config("quanta_discovery", path)


@pytest.mark.parametrize("value", [0.0, -0.01, 1.01, "bad"])
def test_config_rejects_invalid_candidate_gain(
    tmp_path: Path, value: float | str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"discovery:\n  minimum_candidate_gain: {value}\n")
    with pytest.raises(ValueError, match="minimum_candidate_gain"):
        load_experiment_config("quanta_discovery", path)


def test_removed_historical_control_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "removed.yaml"
    path.write_text("discovery:\n  priority_estimator: interval_parameter_update\n")
    with pytest.raises(ValueError, match="priority_estimator"):
        load_experiment_config("quanta_discovery", path)


def test_config_accepts_full_complement_cross_fitted_priority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "full_complement.yaml"
    path.write_text(
        "discovery:\n  priority_estimator: cross_fitted_full_complement\n"
    )

    config = load_experiment_config("quanta_discovery", path)

    assert config.priority_estimator == "cross_fitted_full_complement"


def test_exact_gd_rejects_a_subsampled_batch(tmp_path: Path) -> None:
    path = tmp_path / "sampled.yaml"
    path.write_text(
        "task:\n  training_size: 100\n"
        "training:\n  batch_size: 10\n  optimizer: sgd\n"
    )
    with pytest.raises(ValueError, match="batch_size equal to the realized training set size"):
        load_experiment_config("quanta_discovery", path)


def test_uniform_full_batch_counts_the_mandatory_base_set(tmp_path: Path) -> None:
    path = tmp_path / "uniform_full_batch.yaml"
    path.write_text(
        "task:\n  data_splits: uniform\n  training_size: 100\n  max_number: 999\n"
        "training:\n  batch_size: 199\n  optimizer: sgd\n"
    )

    config = load_experiment_config("quanta_discovery", path)

    assert config.batch_size == 199


def test_config_rejects_removed_realized_stream_control(tmp_path: Path) -> None:
    path = tmp_path / "realized.yaml"
    path.write_text("discovery:\n  dynamics: realized_adam\n")
    with pytest.raises(ValueError, match="Unknown quanta_discovery config keys"):
        load_experiment_config("quanta_discovery", path)


def test_default_path_uses_single_algorithm_name() -> None:
    experiment = QuantaDiscoveryExperiment(
        QuantaDiscoveryConfig(n_layers=1, d_model=8, n_heads=1, steps=1)
    )
    parts = Path(str(experiment.config.save_dir)).parts
    assert ALGORITHM_NAME in parts
    assert ALGORITHM_NAME == "quanta_discovery_v0"


def test_block_priority_groups_all_local_transformer_parameters() -> None:
    model = DecoderTransformerLM(
        vocab_size=17,
        max_seq_len=8,
        d_model=8,
        n_layers=2,
        n_heads=1,
        dropout=0.0,
        pad_id=0,
    )

    groups = block_parameters_by_layer(model)

    assert tuple(tuple(id(parameter) for parameter in group) for group in groups) == tuple(
        tuple(id(parameter) for parameter in layer.parameters())
        for layer in model.layers
    )
    assert all(
        all(parameter is not model.token_embedding.weight for parameter in group)
        for group in groups
    )
    assert all(
        all(parameter is not model.head.weight for parameter in group)
        for group in groups
    )
    assert flattened_block_parameters(model).shape == (
        len(model.layers),
        sum(parameter.numel() for parameter in model.layers[0].parameters()),
    )
    assert flattened_mlp_parameters(model).shape[1] < flattened_block_parameters(model).shape[1]


def test_binary_priority_factorization_has_no_event_amplitudes() -> None:
    first = np.asarray([1, 1, 0, 0, 1, 0], dtype=bool)
    second = ~first
    values = (
        first[:, None] * np.asarray([3.0, 2.0, 0.0, 0.0])[None]
        + second[:, None] * np.asarray([0.0, 0.0, 2.0, 4.0])[None]
    )

    fit = _binary_priority_factorization(
        values,
        np.full(len(values), 1.0 / len(values)),
        max_components=4,
        alternating_steps=20,
    )

    assert fit["support"].dtype == np.bool_
    assert fit["support"].shape == (6, 2)
    np.testing.assert_allclose(fit["residual"], 0.0, atol=1.0e-10)
