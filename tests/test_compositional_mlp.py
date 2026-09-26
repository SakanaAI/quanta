from __future__ import annotations

import math

import numpy as np
import torch

from quanta.experiments.scaling_laws.compositional_mlp import (
    CompositionalMLPConfig,
    EmbeddedSharedMLP,
    OneHotSharedMLP,
    conditional_bayes_losses,
    factor_specs,
    factor_support_pool_bits,
    labels_from_signs,
    resolved_sensor_bits,
    run_name,
    support_indices,
    target_mode_coefficients,
    validate_config,
)


def test_legacy_parallel_nand_supports_and_coefficients() -> None:
    config = validate_config(CompositionalMLPConfig())
    tasks = torch.tensor([0, 1, 2, 3, 4, 5])
    supports = support_indices(config, tasks)
    assert [spec.sharing for spec in factor_specs(config)] == [4, 2, 1]
    assert supports[0].tolist() == [[0, 1], [0, 1], [0, 1], [0, 1], [8, 9], [8, 9]]
    assert supports[1].tolist() == [
        [32, 33],
        [32, 33],
        [36, 37],
        [36, 37],
        [40, 41],
        [40, 41],
    ]
    assert supports[2].tolist() == [
        [64, 65],
        [66, 67],
        [68, 69],
        [70, 71],
        [72, 73],
        [74, 75],
    ]
    expected = np.asarray([0.75, 0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25])
    np.testing.assert_allclose(target_mode_coefficients(config), expected)
    bayes = conditional_bayes_losses(config)
    assert math.isclose(bayes[0], 0.5435644431995964)
    assert math.isclose(bayes[1], 0.4056390622295664)
    assert math.isclose(bayes[3], 0.25)
    assert bayes[7] == 0.0


def test_flat_is_balanced_private_parity() -> None:
    config = validate_config(
        CompositionalMLPConfig(
            group_size=1,
            subgroup_size=1,
            degree_a=2,
            degree_b=0,
            degree_c=0,
            support_layout="compact",
            task_conditioning="embedding",
            sensor_bits=0,
            distractor_bits=7,
            eval_samples_per_task=8,
        )
    )
    assert resolved_sensor_bits(config) == 2 * config.n_tasks + 7
    signs = torch.tensor([[-1.0], [1.0]])
    assert labels_from_signs(signs, config.composition).tolist() == [0, 1]
    np.testing.assert_allclose(target_mode_coefficients(config), [0.0, 1.0])
    assert conditional_bayes_losses(config) == {0: 1.0, 1: 0.0}


def test_compact_nested_supports_and_truth_table() -> None:
    config = validate_config(
        CompositionalMLPConfig(
            n_tasks=8,
            group_size=4,
            subgroup_size=2,
            degree_a=1,
            degree_b=2,
            degree_c=3,
            composition="nested_nand",
            event_scope="all_modes",
            support_layout="compact",
            task_conditioning="embedding",
            sensor_bits=0,
            distractor_bits=5,
            eval_samples_per_task=64,
        )
    )
    assert config.sensor_bits == 2 + 8 + 24 + 5
    tasks = torch.tensor([0, 1, 2, 3, 4])
    supports = support_indices(config, tasks)
    assert supports[0].tolist() == [[0], [0], [0], [0], [1]]
    assert supports[1].tolist() == [[2, 3], [2, 3], [4, 5], [4, 5], [6, 7]]
    assert supports[2][4].tolist() == [22, 23, 24]
    predicates = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
    )
    # NAND(NAND(A, B), C)
    assert labels_from_signs(predicates, "nested_nand").tolist() == [1, 1, 0]
    assert np.count_nonzero(np.abs(target_mode_coefficients(config)) > 1e-8) == 8


def test_shared_pool_matches_fixed_dimension_sparse_parity() -> None:
    config = validate_config(
        CompositionalMLPConfig(
            n_tasks=512,
            group_size=1,
            subgroup_size=1,
            degree_a=2,
            degree_b=0,
            degree_c=0,
            support_layout="shared_pool",
            support_pool_bits=100,
            support_seed=17,
            task_conditioning="embedding",
            sensor_bits=0,
            eval_samples_per_task=8,
        )
    )
    assert factor_support_pool_bits(config, factor_specs(config)[0]) == 100
    assert resolved_sensor_bits(config) == 100
    tasks = torch.arange(config.n_tasks)
    supports = support_indices(config, tasks)
    assert supports[0].shape == (512, 2)
    assert int(supports[0].min()) == 0
    assert int(supports[0].max()) < 100
    assert len({tuple(row) for row in supports[0].tolist()}) == config.n_tasks
    torch.testing.assert_close(supports[0], support_indices(config, tasks)[0])


def test_shared_pool_auto_sizes_each_nested_role() -> None:
    config = validate_config(
        CompositionalMLPConfig(
            n_tasks=16,
            group_size=4,
            subgroup_size=2,
            degree_a=1,
            degree_b=2,
            degree_c=3,
            composition="nested_nand",
            event_scope="all_modes",
            support_layout="shared_pool",
            task_conditioning="embedding",
            sensor_bits=0,
            eval_samples_per_task=64,
        )
    )
    assert config.sensor_bits == 4 + 5 + 6
    supports = support_indices(config, torch.arange(config.n_tasks))
    for index, spec in enumerate(factor_specs(config)):
        assert supports[index].shape == (config.n_tasks, spec.degree)
        role_offset = sum(
            factor_support_pool_bits(config, earlier)
            for earlier in factor_specs(config)[:index]
        )
        role_stop = role_offset + factor_support_pool_bits(config, spec)
        assert int(supports[index].min()) >= role_offset
        assert int(supports[index].max()) < role_stop


def test_embedding_first_layer_is_algebraically_equivalent_to_onehot() -> None:
    n_tasks, sensors, width = 5, 7, 11
    onehot = OneHotSharedMLP(n_tasks, sensors, width, hidden_layers=2)
    embedded = EmbeddedSharedMLP(n_tasks, sensors, width, hidden_layers=2)
    first = onehot.net[0]
    with torch.no_grad():
        embedded.task_embedding.weight.copy_(first.weight[:, :n_tasks].T)
        embedded.sensor_projection.weight.copy_(first.weight[:, n_tasks:])
        embedded.sensor_projection.bias.copy_(first.bias)
        embedded.hidden[0].weight.copy_(onehot.net[2].weight)
        embedded.hidden[0].bias.copy_(onehot.net[2].bias)
        embedded.readout.weight.copy_(onehot.net[4].weight)
        embedded.readout.bias.copy_(onehot.net[4].bias)
    tasks = torch.tensor([0, 3, 4, 1])
    values = torch.randn(len(tasks), sensors)
    torch.testing.assert_close(onehot(tasks, values), embedded(tasks, values))


def test_run_identity_includes_support_and_feature_controls() -> None:
    base = validate_config(
        CompositionalMLPConfig(
            group_size=1,
            subgroup_size=1,
            degree_a=2,
            degree_b=0,
            degree_c=0,
            support_layout="compact",
            sensor_bits=0,
            distractor_bits=7,
            eval_samples_per_task=8,
        )
    )
    sparse = validate_config(
        CompositionalMLPConfig(**{**base.__dict__, "feature_mode": "sparse"})
    )
    pooled = validate_config(
        CompositionalMLPConfig(
            **{
                **base.__dict__,
                "support_layout": "shared_pool",
                "support_pool_bits": 10,
                "support_seed": 3,
                "sensor_bits": 0,
            }
        )
    )
    assert run_name(base) != run_name(sparse)
    assert run_name(base) != run_name(pooled)


def test_optimizer_is_validated_and_part_of_run_identity() -> None:
    sgd = validate_config(CompositionalMLPConfig())
    adam = validate_config(CompositionalMLPConfig(optimizer="adam"))
    assert run_name(sgd) != run_name(adam)
    with np.testing.assert_raises_regex(ValueError, "optimizer"):
        validate_config(CompositionalMLPConfig(optimizer="rmsprop"))
