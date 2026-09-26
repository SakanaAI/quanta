from __future__ import annotations

import numpy as np

from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.quanta_discovery.capacity import (
    fixed_qmodel_parameter_count,
    allocate_parameter_budget,
    allocate_uniform_writer_ceiling,
    allocate_writer_counts,
    allocate_overcomplete_writer_counts,
    functional_writer_metrics,
    dynamics_writer_metrics,
    parameter_count,
)


def test_dynamics_packet_complexity_tracks_writer_group_participation() -> None:
    # d_model=2, hidden_size=2 gives 12 flattened source-MLP parameters.
    updates = np.zeros((2, 1, 12), dtype=np.float64)
    updates[0, 0, 0] = 1.0
    updates[1, 0, 0] = 1.0
    updates[1, 0, 2] = 1.0
    priorities = (np.asarray([[1.0, 0.0], [0.0, 1.0]]),)
    supports = (np.asarray([[1, 0], [1, 1]], dtype=bool),)

    metrics = dynamics_writer_metrics(
        updates,
        np.asarray([0, 1, 2]),
        priorities,
        supports,
        d_model=2,
        mlp_ratio=1.0,
    )[0]

    np.testing.assert_allclose(metrics.demand, [1.0, 0.5])
    np.testing.assert_allclose(metrics.tau_50_steps, [1.0, 2.0])
    np.testing.assert_allclose(metrics.packet_effective_groups, [1.0, 2.0])


def test_writer_allocation_reserves_one_then_uses_complexity() -> None:
    counts = allocate_writer_counts([1.0, 2.0, 4.0], 10)

    assert counts.tolist() == [2, 3, 5]


def test_functional_complexity_counts_residual_write_directions() -> None:
    writes = np.zeros((3, 2, 1, 2), dtype=np.float64)
    writes[1, 0, 0, 0] = 2.0
    writes[2, 1, 0, 1] = 2.0
    metrics = functional_writer_metrics(
        writes,
        (np.asarray([[1.0, 1.0]]),),
        (np.asarray([[1], [1]], dtype=bool),),
    )[0]
    assert metrics.complexity.tolist() == [2]
    assert not bool(metrics.zero_energy[0])
    assert allocate_overcomplete_writer_counts(metrics.complexity, minimum=4, factor=4, maximum=6).tolist() == [6]


def test_parameter_budget_never_exceeds_requested_fraction() -> None:
    allocation = allocate_parameter_budget(
        ([1.0, 2.0], [4.0]),
        source_parameter_count=1000,
        parameter_budget_fraction=0.2,
        fixed_parameter_count=100,
        parameters_per_writer=10,
    )

    assert [values.tolist() for values in allocation.writer_counts] == [[2, 3], [5]]
    assert allocation.total_writers == 10
    assert allocation.executable_parameter_count == 200
    assert allocation.unused_parameter_count == 0


def test_small_config_bandwidth_fits_one_writer_per_candidate() -> None:
    source = DecoderTransformerLM(
        vocab_size=44,
        max_seq_len=20,
        d_model=32,
        n_layers=3,
        n_heads=1,
        dropout=0.0,
        pad_id=0,
        mlp_ratio=2.0,
    )
    quantum_counts = (16, 16, 16)
    embedding_parameters = (
        source.token_embedding.weight.numel()
        + source.position_embedding.weight.numel()
    )
    fixed, per_writer = fixed_qmodel_parameter_count(
        quantum_counts,
        d_model=32,
        vocabulary_size=44,
        attention_rank=1,
        attention_value_dim=4,
        embedding_parameter_count=embedding_parameters,
    )

    allocation = allocate_parameter_budget(
        tuple(np.ones(count) for count in quantum_counts),
        source_parameter_count=parameter_count((source,)),
        parameter_budget_fraction=0.90,
        fixed_parameter_count=fixed,
        parameters_per_writer=per_writer,
    )

    assert allocation.total_writers >= sum(quantum_counts)
    assert allocation.executable_parameter_count <= allocation.target_parameter_count


def test_uniform_writer_ceiling_does_not_spend_the_budget_as_a_target() -> None:
    allocation = allocate_uniform_writer_ceiling(
        (2, 1),
        writers_per_quantum=4,
        source_parameter_count=1000,
        parameter_budget_fraction=0.9,
        fixed_parameter_count=100,
        parameters_per_writer=10,
    )

    assert [values.tolist() for values in allocation.writer_counts] == [[4, 4], [4]]
    assert allocation.executable_parameter_count == 220
    assert allocation.unused_parameter_count == 680
