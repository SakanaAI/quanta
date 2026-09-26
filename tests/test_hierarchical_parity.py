from __future__ import annotations

import numpy as np
import mup
import pytest
import torch

from quanta.experiments.scaling_laws.hierarchical_parity import (
    HierarchicalParityConfig,
    build_hierarchy,
    build_model,
    build_optimizer,
    build_supports,
    exact_depth_entropies,
    first_sustained_crossing,
    fit_depth_adjusted_clock,
    fit_log_clock,
    gf2_rank,
    nested_nand_sign,
    run,
    run_name,
    target_signs,
    validate_config,
)


def test_hierarchy_probabilities_recover_closure_demand() -> None:
    config = validate_config(
        HierarchicalParityConfig(
            base_tasks=4,
            branching_factor=2,
            max_depth=3,
            depth_mass_decay=1.0,
            root_frequency_exponent=2.0,
        )
    )
    hierarchy = build_hierarchy(config)
    depths = np.asarray(hierarchy["node_depths"])
    paths = np.asarray(hierarchy["paths"])
    terminal = np.asarray(hierarchy["terminal_probabilities"])
    demand = np.asarray(hierarchy["marginal_demand"])
    assert len(depths) == 4 + 8 + 16 + 32
    assert np.all(terminal > 0)
    np.testing.assert_allclose(terminal.sum(), 1.0)
    for node in range(len(depths)):
        descendants = np.any(paths == node, axis=1)
        np.testing.assert_allclose(terminal[descendants].sum(), demand[node])
    assert terminal[depths == 3].max() > terminal[depths == 0].min()


def test_rho_beta_demand_law_is_realized_exactly() -> None:
    rho = 2
    beta = 2.0 ** 1.2
    config = validate_config(
        HierarchicalParityConfig(
            base_tasks=4,
            branching_factor=rho,
            max_depth=4,
            demand_law="rho_beta",
            beta=beta,
            root_frequency_exponent=0.0,
            support_pool_bits=32,
            hidden_layers=4,
        )
    )
    hierarchy = build_hierarchy(config)
    depths = np.asarray(hierarchy["node_depths"])
    paths = np.asarray(hierarchy["paths"])
    terminal = np.asarray(hierarchy["terminal_probabilities"])
    marginal = np.asarray(hierarchy["marginal_demand"])
    depth_masses = np.asarray(hierarchy["depth_masses"])
    diagnostics = hierarchy["demand_law_diagnostics"]

    np.testing.assert_allclose(terminal.sum(), 1.0)
    assert np.all(terminal > 0)
    np.testing.assert_allclose(diagnostics["realized_rho_by_depth"], rho)
    np.testing.assert_allclose(diagnostics["realized_beta_by_depth"], beta)
    np.testing.assert_allclose(depth_masses[1:-1] / depth_masses[:-2], rho / beta)
    for depth in range(config.max_depth + 1):
        values = terminal[depths == depth]
        np.testing.assert_allclose(values, values[0])
        expected_marginal = 0.25 * beta**(-depth)
        np.testing.assert_allclose(marginal[depths == depth], expected_marginal)
    for node in range(len(depths)):
        descendants = np.any(paths == node, axis=1)
        np.testing.assert_allclose(terminal[descendants].sum(), marginal[node])
    assert diagnostics["max_rho_relative_error"] < 1e-12
    assert diagnostics["max_beta_relative_error"] < 1e-12
    assert diagnostics["implied_alpha"] == pytest.approx(0.2)


def test_rho_beta_demand_rejects_mixed_or_nondecaying_laws() -> None:
    with pytest.raises(ValueError, match="beta > branching_factor"):
        validate_config(
            HierarchicalParityConfig(
                branching_factor=2,
                demand_law="rho_beta",
                beta=2.0,
                root_frequency_exponent=0.0,
            )
        )
    with pytest.raises(ValueError, match="root_frequency_exponent=0"):
        validate_config(
            HierarchicalParityConfig(
                branching_factor=2,
                demand_law="rho_beta",
                beta=2.5,
                root_frequency_exponent=1.0,
            )
        )
    with pytest.raises(ValueError, match="only used"):
        validate_config(HierarchicalParityConfig(beta=2.5))


def test_supports_are_unique_and_independent_on_every_path() -> None:
    config = validate_config(
        HierarchicalParityConfig(
            base_tasks=3,
            branching_factor=2,
            max_depth=3,
            support_pool_bits=24,
            support_seed=7,
        )
    )
    hierarchy = build_hierarchy(config)
    supports = build_supports(config, hierarchy)
    paths = np.asarray(hierarchy["paths"])
    assert len({tuple(row) for row in supports.tolist()}) == len(supports)
    for path in paths:
        nodes = path[path >= 0]
        masks = [(1 << int(supports[node, 0])) | (1 << int(supports[node, 1])) for node in nodes]
        assert gf2_rank(masks) == len(nodes)


def test_nested_nand_sign_and_path_targets() -> None:
    parent = torch.tensor([1.0, 1.0, -1.0, -1.0])
    local = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert nested_nand_sign(parent, local).tolist() == [-1.0, 1.0, 1.0, 1.0]

    sensors = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
        ]
    )
    paths = torch.tensor([[0, -1], [0, 1]])
    supports = torch.tensor([[0, 1], [2, 3]])
    tasks = torch.tensor([0, 1])
    assert target_signs(tasks, sensors, paths, supports).tolist() == [1.0, 1.0]


def test_depth_entropy_and_sustained_crossing() -> None:
    np.testing.assert_allclose(
        exact_depth_entropies(2),
        [1.0, 0.8112781244591328, 0.954434002924965],
    )
    steps = np.arange(8) * 100
    curve = np.asarray([0.0, 0.9, 0.1, 0.81, 0.82, 0.83, 0.84, 0.85])
    assert first_sustained_crossing(steps, curve, 0.8, window=3) == 300.0
    assert first_sustained_crossing(steps, curve, 0.9, window=3) is None


def test_models_are_shared_binary_predictors_and_identity_is_complete() -> None:
    early_config = validate_config(
        HierarchicalParityConfig(
            base_tasks=2,
            branching_factor=2,
            max_depth=1,
            support_pool_bits=8,
            width=16,
            hidden_layers=2,
            task_conditioning="early",
        )
    )
    late_config = validate_config(
        HierarchicalParityConfig(
            **{**early_config.__dict__, "task_conditioning": "late"}
        )
    )
    n_tasks = 6
    tasks = torch.tensor([0, 1, 5])
    sensors = torch.randn(3, 8)
    assert build_model(early_config, n_tasks)(tasks, sensors).shape == (3, 2)
    assert build_model(late_config, n_tasks)(tasks, sensors).shape == (3, 2)
    assert run_name(early_config) != run_name(late_config)
    assert run_name(early_config) != run_name(
        HierarchicalParityConfig(**{**early_config.__dict__, "support_seed": 9})
    )


def test_mup_is_opt_in_and_builds_scaled_optimizer_groups() -> None:
    standard_config = validate_config(
        HierarchicalParityConfig(
            base_tasks=2,
            branching_factor=2,
            max_depth=1,
            support_pool_bits=8,
            width=32,
            hidden_layers=2,
            optimizer="sgd",
            learning_rate=0.1,
        )
    )
    mup_config = validate_config(
        HierarchicalParityConfig(
            **{
                **standard_config.__dict__,
                "parameterization": "mup",
                "mup_base_width": 8,
                "mup_delta_width": 16,
            }
        )
    )
    standard_model = build_model(standard_config, n_tasks=6)
    mup_model = build_model(mup_config, n_tasks=6)
    assert isinstance(standard_model.readout, torch.nn.Linear)
    assert not isinstance(standard_model.readout, mup.MuReadout)
    assert isinstance(mup_model.readout, mup.MuReadout)
    assert all(hasattr(parameter, "infshape") for parameter in mup_model.parameters())

    optimizer = build_optimizer(mup_model, mup_config)
    group_lrs = {float(group["lr"]) for group in optimizer.param_groups}
    assert len(group_lrs) > 1
    assert mup_config.learning_rate in group_lrs
    assert run_name(standard_config) != run_name(mup_config)


def test_checkpointing_is_opt_in_and_saves_resumable_state(tmp_path) -> None:
    base = HierarchicalParityConfig(
        base_tasks=2,
        branching_factor=1,
        max_depth=1,
        support_pool_bits=4,
        width=8,
        hidden_layers=2,
        batch_size=8,
        microbatch_size=4,
        steps=2,
        eval_every=1,
        checkpoint_every=1,
        eval_samples=32,
        eval_task_chunk=2,
        device="cpu",
        output_dir=str(tmp_path),
    )
    output = run(base)
    checkpoint_paths = sorted(output.glob("checkpoint_step_*.pt"))
    assert [path.name for path in checkpoint_paths] == [
        "checkpoint_step_0000001.pt",
        "checkpoint_step_0000002.pt",
    ]
    checkpoint = torch.load(checkpoint_paths[-1], weights_only=False)
    assert checkpoint["format_version"] == 1
    assert checkpoint["step"] == 2
    assert checkpoint["record_steps"] == [0, 1, 2]
    assert checkpoint["config"]["checkpoint_every"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "training_generator_state" in checkpoint
    assert run_name(base) != run_name(
        HierarchicalParityConfig(**{**base.__dict__, "checkpoint_every": 0})
    )


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
    common = dict(
        base_tasks=2,
        branching_factor=1,
        max_depth=1,
        support_pool_bits=4,
        width=8,
        hidden_layers=2,
        batch_size=8,
        microbatch_size=4,
        eval_every=1,
        checkpoint_every=1,
        eval_samples=32,
        eval_task_chunk=2,
        device="cpu",
    )
    source_config = HierarchicalParityConfig(
        **common, steps=2, output_dir=str(tmp_path / "source")
    )
    source = run(source_config)
    checkpoint = source / "checkpoint_step_0000002.pt"

    resumed_config = HierarchicalParityConfig(
        **common,
        steps=4,
        output_dir=str(tmp_path / "resumed"),
        resume_checkpoint=str(checkpoint),
    )
    resumed = run(resumed_config)
    uninterrupted = run(
        HierarchicalParityConfig(
            **common, steps=4, output_dir=str(tmp_path / "uninterrupted")
        )
    )

    resumed_trajectory = np.load(resumed / "trajectory.npz")
    uninterrupted_trajectory = np.load(uninterrupted / "trajectory.npz")
    np.testing.assert_array_equal(resumed_trajectory["steps"], [0, 1, 2, 3, 4])
    for key in (
        "task_losses_bits",
        "task_accuracy",
        "functional_coefficients",
        "weighted_losses_bits",
    ):
        np.testing.assert_array_equal(
            resumed_trajectory[key], uninterrupted_trajectory[key]
        )

    resumed_state = torch.load(
        resumed / "checkpoint_step_0000004.pt", weights_only=False
    )["model_state_dict"]
    uninterrupted_state = torch.load(
        uninterrupted / "checkpoint_step_0000004.pt", weights_only=False
    )["model_state_dict"]
    for name in resumed_state:
        assert torch.equal(resumed_state[name], uninterrupted_state[name]), name


def test_checkpoint_resume_rejects_changed_training_configuration(tmp_path) -> None:
    source_config = HierarchicalParityConfig(
        base_tasks=2,
        branching_factor=1,
        max_depth=1,
        support_pool_bits=4,
        width=8,
        hidden_layers=2,
        batch_size=8,
        microbatch_size=4,
        steps=1,
        eval_every=1,
        checkpoint_every=1,
        eval_samples=32,
        eval_task_chunk=2,
        device="cpu",
        output_dir=str(tmp_path / "source"),
    )
    source = run(source_config)
    checkpoint = source / "checkpoint_step_0000001.pt"
    changed = HierarchicalParityConfig(
        **{
            **source_config.__dict__,
            "steps": 2,
            "learning_rate": 0.2,
            "output_dir": str(tmp_path / "changed"),
            "resume_checkpoint": str(checkpoint),
        }
    )
    with pytest.raises(ValueError, match="learning_rate"):
        run(changed)


def test_checkpoint_resume_allows_changed_microbatch_size(tmp_path) -> None:
    source_config = HierarchicalParityConfig(
        base_tasks=2,
        branching_factor=1,
        max_depth=1,
        support_pool_bits=4,
        width=8,
        hidden_layers=2,
        batch_size=8,
        microbatch_size=4,
        steps=1,
        eval_every=1,
        checkpoint_every=1,
        eval_samples=32,
        eval_task_chunk=2,
        device="cpu",
        output_dir=str(tmp_path / "source"),
    )
    source = run(source_config)
    checkpoint = source / "checkpoint_step_0000001.pt"
    resumed = run(
        HierarchicalParityConfig(
            **{
                **source_config.__dict__,
                "microbatch_size": 8,
                "steps": 2,
                "output_dir": str(tmp_path / "resumed"),
                "resume_checkpoint": str(checkpoint),
            }
        )
    )
    assert (resumed / "checkpoint_step_0000002.pt").is_file()


def test_model_is_not_shallower_than_declared_composition() -> None:
    with pytest.raises(ValueError, match="at least max_depth"):
        validate_config(
            HierarchicalParityConfig(
                base_tasks=2,
                branching_factor=1,
                max_depth=3,
                hidden_layers=2,
            )
        )


def test_clock_fits_reject_unidentifiable_coordinates() -> None:
    times = np.asarray([100.0, 200.0, 300.0, 400.0])
    constant_probability = np.full(4, 0.25)
    assert fit_log_clock(constant_probability, times)["gamma"] is None
    depths = np.arange(4)
    collinear_probability = 0.1 * 2.0**depths
    fit = fit_depth_adjusted_clock(collinear_probability, depths, times)
    assert fit["gamma"] is None
    assert fit["depth_kappa"] is None
