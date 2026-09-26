from __future__ import annotations

import numpy as np
import torch

from quanta.experiments.quanta_discovery.mdl_graph import (
    MdlOwnershipGraphConfig,
    fit_mdl_ownership_graph,
    hard_ownership_provenance,
    ownership_codelength,
    temporal_edge_codelengths,
)


def test_inherited_provenance_shortens_predictable_parent_code() -> None:
    parent = torch.tensor([[1.0]] * 60 + [[0.0]] * 40)
    child = torch.tensor([[1.0]] * 50 + [[0.0]] * 50)
    no_cause = (torch.zeros_like(parent), torch.zeros_like(child))
    inherited = (child, torch.zeros_like(child))

    independent = ownership_codelength((parent, child), no_cause)
    compositional = ownership_codelength((parent, child), inherited)

    assert float(compositional) < float(independent)


def test_temporal_code_prefers_parent_acquired_before_child() -> None:
    early = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    late = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)

    compatible = temporal_edge_codelengths((early, late), 100)[(0, 1)][0, 0]
    incompatible = temporal_edge_codelengths((late, early), 100)[(0, 1)][0, 0]

    assert compatible < incompatible


def test_mdl_fit_selects_compression_edge_without_changing_dynamics() -> None:
    parent = np.asarray([1] * 60 + [0] * 40, dtype=bool)[:, None]
    child = np.asarray([1] * 50 + [0] * 50, dtype=bool)[:, None]
    parent_curve = np.asarray([[2.0, 1.0, 0.0]], dtype=np.float32)
    child_curve = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
    fields = (
        parent.astype(np.float32) @ parent_curve,
        child.astype(np.float32) @ child_curve,
    )
    config = MdlOwnershipGraphConfig(
        steps=400,
        learning_rate=5.0e-2,
        log_every=100,
    )

    fit = fit_mdl_ownership_graph(
        fields,
        (parent, child),
        (parent_curve, child_curve),
        config=config,
        device=torch.device("cpu"),
    )

    assert fit.edges == ((0, 0, 1, 0),)
    np.testing.assert_array_equal(fit.dynamics_supports[0], parent)
    np.testing.assert_array_equal(fit.dynamics_supports[1], child)
    assert fit.summary["fidelity"]["satisfied"]
    assert fit.summary["description_length"]["saved_bits"] > 0.0


def test_hard_ownership_adds_only_unmatched_child_demand() -> None:
    dynamics = (
        np.asarray([[1], [1], [0]], dtype=bool),
        np.asarray([[0], [1], [1]], dtype=bool),
    )

    local, inherited, effective = hard_ownership_provenance(
        dynamics, ((0, 0, 1, 0),)
    )

    np.testing.assert_array_equal(local[0][:, 0], [True, False, False])
    np.testing.assert_array_equal(inherited[0][:, 0], [False, True, True])
    np.testing.assert_array_equal(effective[0][:, 0], [True, True, True])
