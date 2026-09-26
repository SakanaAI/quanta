from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path

from quanta.config.types import (
    QuantaDiscoveryConfig,
    PosetsProbingConfig,
    QuantaNetConfig,
    QuantaSteeringConfig,
    ScalingLawsConfig,
    PlotConfig,
    TrainingConfig,
)
from quanta.config.utils import (
    _flatten_training_data,
    _load_yaml_mapping,
    _normalize_keys,
    _validate_discreteness_transition_error,
    _validate_task_name,
    normalize_float_list,
    normalize_int_list,
    normalize_str_list,
    resolve_steps_and_epochs,
)

_VALID_SCHEDULERS = frozenset(
    {
        "constant",
        "constant_with_warmup",
        "linear",
        "cosine",
        "linear_after_plateau",
    }
)
_SCHEDULER_ERROR = (
    "scheduler must be 'constant', 'constant_with_warmup', 'linear', "
    "'cosine', or 'linear_after_plateau'."
)

COMPOSITIONAL_SWEEP_FIELDS = (
    "rho",
    "beta",
    "delta",
    "width",
    "lr",
    "depth",
    "batch_size",
    "weight_decay",
    "scheduler",
    "warmup_phase",
    "plateau_phase",
    "activation",
    "layernorm",
    "architecture",
    "n_heads",
    "mlp_ratio",
    "dropout",
    "max_depth",
    "base_tasks",
    "base_freq",
)

COMPOSITIONAL_MLP_ONLY_SWEEP_FIELDS = {
    "activation",
    "layernorm",
}

COMPOSITIONAL_TRANSFORMER_ONLY_SWEEP_FIELDS = {
    "n_heads",
    "mlp_ratio",
    "dropout",
}


def _validate_scheduler(scheduler: str) -> None:
    if scheduler not in _VALID_SCHEDULERS:
        raise ValueError(_SCHEDULER_ERROR)


def _validate_phase(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")


def load_experiment_config(experiment_name: str, path: str | Path):
    if experiment_name == "quanta_discovery":
        return load_quanta_discovery_config(path)
    if experiment_name == "quanta_net":
        return load_quanta_net_config(path)
    if experiment_name == "quanta_steering":
        return load_quanta_steering_config(path)
    if experiment_name == "posets_probing":
        return load_posets_probing_config(path)
    if experiment_name == "scaling_laws":
        return load_scaling_laws_config(path)
    raise ValueError(f"Unsupported experiment: {experiment_name!r}")


def load_quanta_discovery_config(path: str | Path) -> QuantaDiscoveryConfig:
    raw = _load_yaml_mapping(path)
    data = _flatten_training_data(raw, extra_sections={"discovery"})
    if "save_dir" in data:
        raise ValueError("save_dir is generated automatically and cannot be set in experiment configs.")
    init_fields = {
        item.name for item in fields(QuantaDiscoveryConfig) if item.init
    }
    unknown = sorted(set(data) - init_fields)
    if unknown:
        raise ValueError(
            f"Unknown quanta_discovery config keys in {path}: "
            f"{', '.join(unknown)}"
        )
    if "epochs" in data and "steps" not in data:
        data["steps"] = None
    config = QuantaDiscoveryConfig(**data)
    config.task = config.task.lower()
    config.language = config.language.lower()
    config.data_splits = config.data_splits.lower()
    config.eval_strategy = config.eval_strategy.lower()
    if config.task != "number_naming":
        raise ValueError("quanta_discovery currently supports only number_naming.")
    if config.language != "english":
        raise ValueError("number_naming currently supports language: english.")
    if config.data_splits not in {
        "uniform",
        "digits_wise_uniform",
        "extreme_generalization",
        "extreme_generalization_v2",
    }:
        raise ValueError(
            "data_splits must be 'uniform', 'digits_wise_uniform', "
            "'extreme_generalization', or 'extreme_generalization_v2'."
        )
    if config.eval_strategy not in {"uniform", "digits_wise", "balanced"}:
        raise ValueError(
            "eval_strategy must be 'uniform', 'digits_wise', or 'balanced'."
        )
    if config.training_size <= 0:
        raise ValueError("training_size must be positive.")
    if config.eval_size is not None and config.eval_size <= 0:
        raise ValueError("eval_size must be positive or null.")
    if config.eval_samples is not None and config.eval_samples <= 0:
        raise ValueError("eval_samples must be positive or null.")
    if (
        config.eval_samples_per_digit is not None
        and config.eval_samples_per_digit <= 0
    ):
        raise ValueError("eval_samples_per_digit must be positive or null.")
    if config.max_number < 99:
        raise ValueError("max_number must be at least 99.")
    if config.architecture != "decoder_transformer":
        raise ValueError("architecture must be 'decoder_transformer'.")
    if config.n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if config.d_model <= 0:
        raise ValueError("d_model must be positive.")
    if config.n_heads <= 0:
        raise ValueError("n_heads must be positive.")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if config.mlp_ratio <= 0:
        raise ValueError("mlp_ratio must be positive.")
    if not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0).")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    if config.steps is not None and config.epochs is not None:
        raise ValueError("Set either steps or epochs, not both.")
    if config.steps is None and config.epochs is None:
        raise ValueError("quanta_discovery requires steps or epochs.")
    if config.steps is not None and config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.epochs is not None and config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    config.optimizer = config.optimizer.lower()
    if config.optimizer not in {"sgd", "adam"}:
        raise ValueError("quanta_discovery optimizer must be 'sgd' or 'adam'.")
    if config.optimizer != "sgd":
        raise ValueError(
            "exact priority discovery currently requires optimizer: sgd."
        )
    _validate_scheduler(config.scheduler)
    _validate_phase("warmup_phase", config.warmup_phase)
    _validate_phase("plateau_phase", config.plateau_phase)
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if config.weight_decay != 0:
        raise ValueError(
            "exact full-batch GD discovery currently requires weight_decay: 0."
        )
    for name in (
        "checkpoint_every",
        "max_candidates_per_layer",
        "candidate_fit_steps",
        "q_microbatch_size",
        "q_attention_rank",
        "q_attention_value_dim",
        "q_writer_minimum",
        "q_writer_maximum",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name in (
        "q_gate_entropy_start_step",
        "q_gate_entropy_ramp_steps",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if not isinstance(
        config.minimum_candidate_gain, (int, float)
    ) or isinstance(config.minimum_candidate_gain, bool):
        raise ValueError("minimum_candidate_gain must be numeric.")
    if not 0.0 < float(config.minimum_candidate_gain) <= 1.0:
        raise ValueError("minimum_candidate_gain must lie in (0, 1].")
    if not isinstance(config.detach_gate_gradients, bool):
        raise ValueError("detach_gate_gradients must be boolean.")
    for name in (
        "q_balance_gate_bce",
        "q_attention_direct_residual",
        "q_share_attention_projections",
        "q_attention_query_offsets",
    ):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be boolean.")
    if config.q_routing_mode not in {
        "layer_residual_roots",
        "initial_residual_plus_parents",
    }:
        raise ValueError(
            "q_routing_mode must be 'layer_residual_roots' or "
            "'initial_residual_plus_parents'."
        )
    if config.q_edge_discovery not in {"fixed_closure", "global_static"}:
        raise ValueError(
            "q_edge_discovery must be 'fixed_closure' or 'global_static'."
        )
    if (
        config.q_edge_discovery == "global_static"
        and config.q_routing_mode != "initial_residual_plus_parents"
    ):
        raise ValueError(
            "global_static edge discovery requires "
            "q_routing_mode: initial_residual_plus_parents."
        )
    if config.q_reconstruction_positions not in {
        "prediction_positions",
        "all_valid_positions",
    }:
        raise ValueError(
            "q_reconstruction_positions must be 'prediction_positions' or "
            "'all_valid_positions'."
        )
    if config.q_training_mode not in {
        "online_trajectory",
        "iid_checkpoint_replay",
    }:
        raise ValueError(
            "q_training_mode must be 'online_trajectory' or "
            "'iid_checkpoint_replay'."
        )
    if (
        not isinstance(config.q_iid_checkpoint_count, int)
        or isinstance(config.q_iid_checkpoint_count, bool)
        or not 1 <= config.q_iid_checkpoint_count <= int(config.steps or 0)
    ):
        raise ValueError(
            "q_iid_checkpoint_count must be an integer in [1, steps]."
        )
    if config.q_optimizer_steps is not None and (
        not isinstance(config.q_optimizer_steps, int)
        or isinstance(config.q_optimizer_steps, bool)
        or config.q_optimizer_steps <= 0
    ):
        raise ValueError("q_optimizer_steps must be a positive integer or null.")
    if config.q_writer_total_budget is not None and (
        not isinstance(config.q_writer_total_budget, int)
        or isinstance(config.q_writer_total_budget, bool)
        or config.q_writer_total_budget <= 0
    ):
        raise ValueError("q_writer_total_budget must be a positive integer or null.")
    if (
        config.q_training_mode == "online_trajectory"
        and config.q_optimizer_steps is not None
        and config.q_optimizer_steps != int(config.steps or 0)
    ):
        raise ValueError(
            "online_trajectory requires q_optimizer_steps to equal steps."
        )
    resolved_q_optimizer_steps = int(config.q_optimizer_steps or config.steps or 0)
    if not isinstance(config.q_checkpoint_steps, (list, tuple)):
        raise ValueError("q_checkpoint_steps must be a list of positive integers.")
    checkpoint_steps = tuple(config.q_checkpoint_steps)
    if any(
        not isinstance(step, int)
        or isinstance(step, bool)
        or step <= 0
        or step > resolved_q_optimizer_steps
        for step in checkpoint_steps
    ):
        raise ValueError(
            "q_checkpoint_steps must contain positive integers no larger than "
            "the resolved q_optimizer_steps."
        )
    if len(set(checkpoint_steps)) != len(checkpoint_steps):
        raise ValueError("q_checkpoint_steps must not contain duplicates.")
    config.q_checkpoint_steps = tuple(sorted(checkpoint_steps))
    for name in (
        "q_learning_rate",
        "q_writer_overcomplete_factor",
        "q_functional_rank_energy",
        "q_writer_average_k",
        "q_writer_jump_threshold",
        "q_writer_jump_bandwidth",
        "q_output_kl_weight",
        "q_gate_entropy_weight",
        "q_edge_l0_weight",
        "q_edge_entropy_weight",
        "maximum_edge_closure_cost",
    ):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be numeric.")
    if config.q_learning_rate <= 0.0:
        raise ValueError("q_learning_rate must be positive.")
    if config.q_writer_maximum < config.q_writer_minimum:
        raise ValueError("q_writer_maximum must be at least q_writer_minimum.")
    if config.q_writer_overcomplete_factor <= 0.0:
        raise ValueError("q_writer_overcomplete_factor must be positive.")
    if not 0.0 < config.q_functional_rank_energy <= 1.0:
        raise ValueError("q_functional_rank_energy must lie in (0, 1].")
    if config.q_writer_average_k <= 0.0:
        raise ValueError("q_writer_average_k must be positive.")
    if config.q_writer_activation not in {"relu", "jumprelu"}:
        raise ValueError("q_writer_activation must be 'relu' or 'jumprelu'.")
    if config.q_writer_sparsity not in {"batch_topk", "l0_target"}:
        raise ValueError(
            "q_writer_sparsity must be 'batch_topk' or 'l0_target'."
        )
    if (
        config.q_writer_sparsity == "l0_target"
        and config.q_writer_activation != "jumprelu"
    ):
        raise ValueError("l0_target writer sparsity requires jumprelu activation.")
    if config.q_writer_jump_threshold < 0.0:
        raise ValueError("q_writer_jump_threshold must be non-negative.")
    if config.q_writer_jump_bandwidth <= 0.0:
        raise ValueError("q_writer_jump_bandwidth must be positive.")
    if config.q_writer_l0_scope not in {"per_active_quantum", "global_event"}:
        raise ValueError(
            "q_writer_l0_scope must be 'per_active_quantum' or 'global_event'."
        )
    if config.q_writer_l0_target <= 0.0:
        raise ValueError("q_writer_l0_target must be positive.")
    if config.q_writer_l0_weight < 0.0:
        raise ValueError("q_writer_l0_weight must be non-negative.")
    if config.q_writer_l0_ramp_steps < 0:
        raise ValueError("q_writer_l0_ramp_steps must be non-negative.")
    if config.q_output_kl_weight < 0.0:
        raise ValueError("q_output_kl_weight must be non-negative.")
    if config.q_gate_entropy_weight < 0.0:
        raise ValueError("q_gate_entropy_weight must be non-negative.")
    if config.q_edge_l0_weight < 0.0:
        raise ValueError("q_edge_l0_weight must be non-negative.")
    if config.q_edge_entropy_weight < 0.0:
        raise ValueError("q_edge_entropy_weight must be non-negative.")
    for name in (
        "q_edge_l0_ramp_steps",
        "q_edge_entropy_start_step",
        "q_edge_entropy_ramp_steps",
        "q_edge_freeze_steps",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if config.q_edge_freeze_steps > resolved_q_optimizer_steps:
        raise ValueError("q_edge_freeze_steps cannot exceed q_optimizer_steps.")
    if not 0.0 <= config.maximum_edge_closure_cost <= 1.0:
        raise ValueError("maximum_edge_closure_cost must lie in [0, 1].")
    if config.priority_estimator not in {
        "exact_gd",
        "cross_fitted_checkpoint",
        "cross_fitted_full_complement",
    }:
        raise ValueError(
            "priority_estimator must be 'exact_gd', 'cross_fitted_checkpoint', "
            "or 'cross_fitted_full_complement'."
        )
    if config.priority_crossfit_folds < 2:
        raise ValueError("priority_crossfit_folds must be at least 2.")
    if config.priority_direction_examples <= 0:
        raise ValueError("priority_direction_examples must be positive.")
    if config.priority_direction_replicates <= 0:
        raise ValueError("priority_direction_replicates must be positive.")
    realized_training_size = int(config.training_size)
    if config.data_splits == "uniform":
        realized_training_size += min(99, int(config.max_number))
    if config.batch_size != realized_training_size:
        raise ValueError(
            "full-population source dynamics require batch_size equal to the "
            "realized training set size."
        )
    return config


def load_posets_probing_config(path: str | Path) -> PosetsProbingConfig:
    raw = _load_yaml_mapping(path)
    data = _flatten_training_data(raw, extra_sections=set())
    if "save_dir" in data:
        raise ValueError("save_dir is generated automatically and cannot be set in experiment configs.")
    init_fields = {item.name for item in fields(PosetsProbingConfig) if item.init}
    unknown = sorted(set(data) - init_fields)
    if unknown:
        raise ValueError(f"Unknown posets_probing config keys in {path}: {', '.join(unknown)}")
    if "epochs" in data and "steps" not in data:
        data["steps"] = None
    if "evals_per_epoch" in data and "eval_steps" not in data:
        data["eval_steps"] = None
    if data.get("width") is not None and "d_model" not in data:
        data["d_model"] = data["width"]
    config = PosetsProbingConfig(**data)
    config.task = config.task.lower()
    config.language = config.language.lower()
    config.data_splits = config.data_splits.lower()
    config.eval_strategy = config.eval_strategy.lower()
    if config.task != "number_naming":
        raise ValueError("posets_probing currently supports only number_naming.")
    if config.language != "english":
        raise ValueError("number_naming currently supports language: english.")
    valid_splits = {
        "uniform",
        "digits_wise_uniform",
        "extreme_generalization",
        "extreme_generalization_v2",
    }
    if config.data_splits not in valid_splits:
        raise ValueError(
            "data_splits must be 'uniform', 'digits_wise_uniform', "
            "'extreme_generalization', or 'extreme_generalization_v2'."
        )
    if config.eval_strategy not in {"uniform", "digits_wise", "balanced"}:
        raise ValueError("eval_strategy must be 'uniform', 'digits_wise', or 'balanced'.")
    if config.architecture != "decoder_transformer":
        raise ValueError("architecture must be 'decoder_transformer'.")
    if config.n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if config.d_model <= 0:
        raise ValueError("d_model must be positive.")
    if config.n_heads <= 0:
        raise ValueError("n_heads must be positive.")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if config.mlp_ratio <= 0:
        raise ValueError("mlp_ratio must be positive.")
    if not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0).")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    if config.steps is not None and config.epochs is not None:
        raise ValueError("Set either steps or epochs, not both.")
    if config.steps is None and config.epochs is None:
        config.steps = 10_000
    if config.steps is not None and config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.epochs is not None and config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    _validate_scheduler(config.scheduler)
    _validate_phase("warmup_phase", config.warmup_phase)
    _validate_phase("plateau_phase", config.plateau_phase)
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if config.eval_steps is not None and config.evals_per_epoch is not None:
        raise ValueError("Set either eval_steps or evals_per_epoch, not both.")
    if config.eval_steps is None and config.evals_per_epoch is None:
        config.eval_steps = 500
    if config.eval_steps is not None and config.eval_steps <= 0:
        raise ValueError("eval_steps must be positive.")
    if config.evals_per_epoch is not None and config.evals_per_epoch <= 0:
        raise ValueError("evals_per_epoch must be positive.")
    if config.training_size < 0:
        raise ValueError("training_size must be non-negative.")
    if config.eval_size is not None and config.eval_size <= 0:
        raise ValueError("eval_size must be positive or null.")
    if config.eval_samples is not None and config.eval_samples <= 0:
        raise ValueError("eval_samples must be positive or null.")
    if config.eval_samples_per_digit is not None and config.eval_samples_per_digit <= 0:
        raise ValueError("eval_samples_per_digit must be positive or null.")
    if config.max_number < 99:
        raise ValueError("max_number must be at least 99.")
    if config.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")
    valid_probe_posets = {
        "eng_virtual_tokens",
        "eng_tokens",
        "eng_contextual_tokens",
        "eng_factorized",
        "eng_functional",
    }
    if isinstance(config.probe_quanta_poset, str):
        raise ValueError("probe_quanta_poset must be null or a list of probe ids.")
    if config.probe_quanta_poset is not None:
        unknown_probe_posets = sorted(set(config.probe_quanta_poset) - valid_probe_posets)
        if unknown_probe_posets:
            raise ValueError(
                "probe_quanta_poset currently supports "
                "'eng_virtual_tokens', 'eng_tokens', 'eng_contextual_tokens', "
                "'eng_factorized', or 'eng_functional'."
            )
    if config.probe_max_size is not None and config.probe_max_size <= 0:
        raise ValueError("probe_max_size must be positive or null.")
    if config.factorized_learned_threshold <= 0:
        raise ValueError("factorized_learned_threshold must be positive.")
    if config.factorized_stability_window <= 0:
        raise ValueError("factorized_stability_window must be positive.")
    valid_audits = {"patching_compatibility", "held_out_transfer"}
    if config.audit_quanta_poset is not None:
        unknown_audits = sorted(set(config.audit_quanta_poset) - valid_audits)
        if unknown_audits:
            raise ValueError("audit_quanta_poset supports 'patching_compatibility' and 'held_out_transfer'.")
    if config.patching_compatibility_max_classes is not None and config.patching_compatibility_max_classes <= 0:
        raise ValueError("patching_compatibility_max_classes must be positive or null.")
    if config.patching_compatibility_examples_per_class <= 0:
        raise ValueError("patching_compatibility_examples_per_class must be positive.")
    if config.held_out_token_roles is not None:
        for item in config.held_out_token_roles:
            if not isinstance(item, dict) or set(item) != {"token", "role"}:
                raise ValueError("held_out_token_roles entries must contain exactly token and role.")
    return config


def load_quanta_net_config(path: str | Path) -> QuantaNetConfig:
    return _load_q_config(path, QuantaNetConfig, "quanta_net")


def load_quanta_steering_config(path: str | Path) -> QuantaSteeringConfig:
    config = _load_q_config(path, QuantaSteeringConfig, "quanta_steering")
    if config.q_checkpoint is None:
        raise ValueError("quanta_steering requires q_checkpoint.")
    if config.transformer_layers <= 0 or config.transformer_heads <= 0:
        raise ValueError("transformer_layers and transformer_heads must be positive.")
    if config.transformer_d_model % config.transformer_heads != 0:
        raise ValueError("transformer_d_model must be divisible by transformer_heads.")
    if config.transformer_mlp_ratio <= 0.0:
        raise ValueError("transformer_mlp_ratio must be positive.")
    if not 0.0 <= config.transformer_dropout < 1.0:
        raise ValueError("transformer_dropout must be in [0, 1).")
    if not math.isfinite(float(config.lambda_align)) or config.lambda_align < 0.0:
        raise ValueError("lambda_align must be finite and non-negative.")
    if config.alignment_scale_mode not in {"unit", "training_rms"}:
        raise ValueError("alignment_scale_mode must be 'unit' or 'training_rms'.")
    if not math.isfinite(float(config.alignment_scale_epsilon)) or config.alignment_scale_epsilon <= 0.0:
        raise ValueError("alignment_scale_epsilon must be finite and positive.")
    if config.alignment_scales is not None and (
        len(config.alignment_scales) == 0
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
            for value in config.alignment_scales
        )
    ):
        raise ValueError("alignment_scales must contain finite positive numbers.")
    for field_name in (
        "minimum_oracle_token_accuracy",
        "minimum_oracle_exact_accuracy",
    ):
        if not 0.0 <= getattr(config, field_name) <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1].")
    if config.alignment_target_control not in {"normal", "depth_shuffled", "example_shuffled"}:
        raise ValueError(
            "alignment_target_control must be 'normal', 'depth_shuffled', or 'example_shuffled'."
        )
    return config


def _load_q_config(path: str | Path, config_type, experiment_name: str):
    raw = _load_yaml_mapping(path)
    data = _flatten_training_data(raw, extra_sections=set())
    if "save_dir" in data:
        raise ValueError("save_dir is generated automatically and cannot be set in experiment configs.")
    init_fields = {item.name for item in fields(config_type) if item.init}
    unknown = sorted(set(data) - init_fields)
    if unknown:
        raise ValueError(f"Unknown {experiment_name} config keys in {path}: {', '.join(unknown)}")
    if "epochs" in data and "steps" not in data:
        data["steps"] = None
    if "evals_per_epoch" in data and "eval_steps" not in data:
        data["eval_steps"] = None
    config = config_type(**data)
    config.task = config.task.lower()
    config.language = config.language.lower()
    config.data_splits = config.data_splits.lower()
    config.eval_strategy = config.eval_strategy.lower()
    if config.task != "number_naming":
        raise ValueError(f"{experiment_name} currently supports only number_naming.")
    if config.language != "english":
        raise ValueError("number_naming currently supports language: english.")
    if config.data_splits not in {
        "uniform",
        "digits_wise_uniform",
        "extreme_generalization",
        "extreme_generalization_v2",
    }:
        raise ValueError(
            "data_splits must be 'uniform', 'digits_wise_uniform', "
            "'extreme_generalization', or 'extreme_generalization_v2'."
        )
    if config.eval_strategy not in {"uniform", "digits_wise", "balanced"}:
        raise ValueError("eval_strategy must be 'uniform', 'digits_wise', or 'balanced'.")
    if config.d_source <= 0 or config.d_quantum <= 0:
        raise ValueError("d_source and d_quantum must be positive.")
    if config.read_heads <= 0 or config.d_quantum % config.read_heads != 0:
        raise ValueError("read_heads must be positive and divide d_quantum.")
    if config.quantum_mlp_ratio <= 0:
        raise ValueError("quantum_mlp_ratio must be positive.")
    if config.activation not in {"gelu", "relu", "layernorm_gelu"}:
        raise ValueError("activation must be 'gelu', 'relu', or 'layernorm_gelu'.")
    if not isinstance(config.add_initial_state, bool):
        raise ValueError("add_initial_state must be true or false.")
    if not isinstance(config.all_quanta_output, bool):
        raise ValueError("all_quanta_output must be true or false.")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    if config.steps is not None and config.epochs is not None:
        raise ValueError("Set either steps or epochs, not both.")
    if config.steps is None and config.epochs is None:
        config.steps = 10_000
    if config.steps is not None and config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.epochs is not None and config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    _validate_scheduler(config.scheduler)
    _validate_phase("warmup_phase", config.warmup_phase)
    _validate_phase("plateau_phase", config.plateau_phase)
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if config.eval_steps is not None and config.evals_per_epoch is not None:
        raise ValueError("Set either eval_steps or evals_per_epoch, not both.")
    if config.eval_steps is None and config.evals_per_epoch is None:
        config.eval_steps = 500
    if config.eval_steps is not None and config.eval_steps <= 0:
        raise ValueError("eval_steps must be positive.")
    if config.evals_per_epoch is not None and config.evals_per_epoch <= 0:
        raise ValueError("evals_per_epoch must be positive.")
    if config.full_eval_steps is not None and config.full_eval_steps <= 0:
        raise ValueError("full_eval_steps must be positive or null.")
    if config.parent_audit_size < 0:
        raise ValueError("parent_audit_size must be non-negative.")
    if config.training_size < 0:
        raise ValueError("training_size must be non-negative.")
    if config.eval_size is not None and config.eval_size <= 0:
        raise ValueError("eval_size must be positive or null.")
    if config.eval_samples is not None and config.eval_samples <= 0:
        raise ValueError("eval_samples must be positive or null.")
    if config.eval_samples_per_digit is not None and config.eval_samples_per_digit <= 0:
        raise ValueError("eval_samples_per_digit must be positive or null.")
    if config.max_number < 99:
        raise ValueError("max_number must be at least 99.")
    if config.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")
    if config.activity_weight < 0.0:
        raise ValueError("activity_weight must be non-negative.")
    if (
        not isinstance(config.semantic_weight, (int, float))
        or isinstance(config.semantic_weight, bool)
        or config.semantic_weight < 0.0
    ):
        raise ValueError("semantic_weight must be a non-negative number.")
    if (
        not isinstance(config.max_supervised_classes, int)
        or isinstance(config.max_supervised_classes, bool)
        or config.max_supervised_classes < 0
    ):
        raise ValueError("max_supervised_classes must be a non-negative integer.")
    if experiment_name == "quanta_net" and config.compiled_program_path is None:
        raise ValueError(f"{experiment_name} requires compiled_program_path.")
    return config


def load_scaling_laws_config(path: str | Path) -> ScalingLawsConfig:
    raw = _load_yaml_mapping(path)
    required_section_keys = {
        "model": {"width"},
        "experiment": {"rho_beta_delta", "seed"},
        "training": {"lr"},
    }
    for section, required_keys in required_section_keys.items():
        section_data = raw.get(section)
        if not isinstance(section_data, dict):
            raise ValueError(
                f"scaling_laws config {path} must contain a {section} mapping."
            )
        missing = sorted(required_keys - set(section_data))
        if missing:
            raise ValueError(
                f"scaling_laws config {path} must set "
                f"{', '.join(f'{section}.{key}' for key in missing)}."
            )
    for section, required_keys in required_section_keys.items():
        for key in required_keys:
            misplaced = [
                candidate
                for candidate, section_data in raw.items()
                if candidate != section and isinstance(section_data, dict) and key in section_data
            ]
            if key in raw or misplaced:
                raise ValueError(
                    f"scaling_laws requires {section}.{key}; "
                    f"remove misplaced {key} from {path}."
                )

    data = _flatten_training_data(raw, extra_sections={"experiment"})
    if "epochs" in data and "steps" not in data:
        data["steps"] = None
    if "save_dir" in data:
        raise ValueError("save_dir is generated automatically and cannot be set in experiment configs.")
    legacy_keys = sorted(
        set(data)
        & {
            "rho",
            "beta",
            "delta",
            "graph_dependencies",
            "task_frequencies",
            "samples_per_task",
            "log_steps",
            "alternative_decomposition",
            "cxor_one_hot_task_encoding",
        }
    )
    if legacy_keys:
        raise ValueError(
            "scaling_laws uses only the canonical sweep interface; "
            f"remove legacy keys from {path}: {', '.join(legacy_keys)}"
        )

    init_fields = {item.name for item in fields(ScalingLawsConfig) if item.init}
    unknown = sorted(set(data) - init_fields)
    if unknown:
        raise ValueError(f"Unknown scaling laws config keys in {path}: {', '.join(unknown)}")

    for required_key in ("width", "rho_beta_delta", "seed", "lr"):
        if required_key not in data:
            raise ValueError(
                f"scaling_laws config {path} must set {required_key}."
            )
    data["width"] = normalize_int_list(data["width"], "width")
    data["seed"] = normalize_int_list(data["seed"], "seed")
    if not isinstance(data["lr"], list):
        raise ValueError("training.lr must be a list of learning rates.")
    data["lr"] = normalize_float_list(data["lr"], "lr")
    if not isinstance(data["rho_beta_delta"], list) or not data["rho_beta_delta"]:
        raise ValueError("rho_beta_delta must be a non-empty list of [rho, beta, delta] triplets.")
    triplets = []
    for triplet in data["rho_beta_delta"]:
        if not isinstance(triplet, list) or len(triplet) != 3:
            raise ValueError("Each rho_beta_delta entry must be [rho, beta, delta].")
        triplets.append([float(value) for value in triplet])
    data["rho_beta_delta"] = triplets
    if "task_bits" in data:
        data["task_bits"] = normalize_int_list(data["task_bits"], "task_bits")
    if data.get("attention_masking") is None:
        data["attention_masking"] = "none"

    config = ScalingLawsConfig(**data)
    config.rho = [row[0] for row in config.rho_beta_delta]
    config.beta = [row[1] for row in config.rho_beta_delta]
    config.delta = [row[2] for row in config.rho_beta_delta]
    _validate_compositional_sweep_lengths(config)
    _validate_task_name(config, path)
    _validate_discreteness_transition_error(config)
    _validate_demand_config(config)
    if config.task not in {"cnand", "multitask_sparse_parity"}:
        raise ValueError("scaling_laws supports only cNAND or multitask_sparse_parity.")
    if config.task_bits != [-1, 1]:
        raise ValueError("scaling_laws currently requires task_bits: [-1, 1].")
    if any(value not in {"relu", "square"} for value in _list_values(config.activation)):
        raise ValueError("activation must be 'relu' or 'square'.")
    if any(value not in {"mlp", "transformer"} for value in _list_values(config.architecture)):
        raise ValueError("architecture must be 'mlp' or 'transformer'.")
    for sweep_row in _compositional_sweep_rows(config):
        if sweep_row["architecture"] != "transformer":
            continue
        if sweep_row["width"] <= 0:
            raise ValueError("width must be positive.")
        if sweep_row["depth"] <= 0:
            raise ValueError("depth must be positive.")
        if sweep_row["n_heads"] <= 0:
            raise ValueError("n_heads must be positive.")
        if sweep_row["width"] % sweep_row["n_heads"] != 0:
            raise ValueError("width must be divisible by n_heads.")
        if sweep_row["mlp_ratio"] <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if sweep_row["dropout"] < 0.0 or sweep_row["dropout"] >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
    if config.transformer_activation not in {"relu", "gelu"}:
        raise ValueError("transformer_activation must be 'relu' or 'gelu'.")
    if config.task == "cnand":
        if any(row["architecture"] != "transformer" for row in _compositional_sweep_rows(config)):
            raise ValueError("cNAND requires architecture: transformer.")
        if config.n_noise_bits != 0:
            raise ValueError("cNAND does not support n_noise_bits; use n_local_bits for node-local inputs.")
    elif config.task == "multitask_sparse_parity":
        if any(row["architecture"] != "mlp" for row in _compositional_sweep_rows(config)):
            raise ValueError("multitask_sparse_parity requires architecture: mlp.")
        if config.parity_task_bits <= 0 or config.parity_subset_size <= 0:
            raise ValueError("multitask_sparse_parity requires positive parity_task_bits and parity_subset_size.")
        if config.parity_subset_size > config.parity_task_bits:
            raise ValueError("parity_subset_size cannot exceed parity_task_bits.")
    elif any(row["architecture"] == "transformer" for row in _compositional_sweep_rows(config)):
        raise ValueError("transformer currently supports only cNAND.")
    if any(
        row["scheduler"] not in _VALID_SCHEDULERS
        for row in _compositional_sweep_rows(config)
    ):
        raise ValueError(_SCHEDULER_ERROR)
    if config.mixed_precision is not None and config.mixed_precision not in {"no", "fp16", "bf16", "fp8"}:
        raise ValueError("mixed_precision must be null, 'no', 'fp16', 'bf16', or 'fp8'.")
    for name in ("warmup_phase", "plateau_phase"):
        if any(
            not 0.0 <= row[name] <= 1.0
            for row in _compositional_sweep_rows(config)
        ):
            raise ValueError(f"{name} must be between 0.0 and 1.0.")
    if any(value <= 0 or value == 1 for value in config.rho):
        raise ValueError("rho values must be positive and not equal to 1.")
    if any(value <= 0 for value in config.beta):
        raise ValueError("beta values must be positive.")
    if any(value <= -1 for value in config.delta):
        raise ValueError("delta values must be greater than -1.")
    if not config.width or any(w <= 0 for w in _list_values(config.width)):
        raise ValueError("width must contain positive integers.")
    if not config.lr or any(val <= 0 for val in config.lr):
        raise ValueError("lr must contain positive learning rates.")
    if not config.seed:
        raise ValueError("seed must contain at least one value.")
    if any(row["base_tasks"] <= 0 for row in _compositional_sweep_rows(config)):
        raise ValueError("base_tasks must be positive.")
    if any(row["base_freq"] <= 0 for row in _compositional_sweep_rows(config)):
        raise ValueError("base_freq must be positive.")
    if any(row["max_depth"] < 0 for row in _compositional_sweep_rows(config)):
        raise ValueError("max_depth must be non-negative.")
    if any(row["batch_size"] <= 0 for row in _compositional_sweep_rows(config)):
        raise ValueError("batch_size must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if config.batch_size % config.gradient_accumulation_steps != 0:
        raise ValueError(
            "batch_size must be divisible by gradient_accumulation_steps."
        )
    if config.m <= 0:
        raise ValueError("m must be positive.")
    if config.eval_steps <= 0:
        raise ValueError("eval_steps must be positive.")
    if isinstance(config.save_steps, list):
        raise ValueError("save_steps must be a scalar integer or null.")
    if config.save_steps is not None and config.save_steps <= 0:
        raise ValueError("save_steps must be positive or null.")
    if config.eval_samples_per_task <= 0:
        raise ValueError("eval_samples_per_task must be positive.")
    if (
        config.eval_seed is not None
        and (
            not isinstance(config.eval_seed, int)
            or isinstance(config.eval_seed, bool)
            or config.eval_seed < 0
        )
    ):
        raise ValueError("eval_seed must be a non-negative integer or null.")
    if config.eval_loss_formula not in {"quanta_weighted", "task_weighted"}:
        raise ValueError(
            "eval_loss_formula must be 'quanta_weighted' or 'task_weighted'."
        )
    if config.loss_supervision not in {"all", "target_only"}:
        raise ValueError(
            "loss_supervision must be 'all' or 'target_only'."
        )
    if config.trace_sampling not in {
        "principal",
        "ideal_threshold",
        "ideal_path",
    }:
        raise ValueError(
            "trace_sampling must be 'principal', 'ideal_threshold', "
            "or 'ideal_path'."
        )
    if config.trace_sampling == "ideal_threshold":
        if config.quanta_demand != "composition":
            raise ValueError("ideal_threshold trace sampling requires quanta_demand: composition.")
        if config.loss_supervision != "all":
            raise ValueError("ideal_threshold trace sampling requires loss_supervision: all.")
        if config.eval_loss_formula != "quanta_weighted":
            raise ValueError("ideal_threshold trace sampling requires eval_loss_formula: quanta_weighted.")
        if any(value < 1 for value in config.beta):
            raise ValueError("ideal_threshold trace sampling requires beta >= 1 for monotone depth demand.")
    if config.graph_family not in {
        "exponential_random",
        "exponential_paired",
        "exponential_depth_paired",
        "flat_roots",
    }:
        raise ValueError(
            "graph_family must be 'exponential_random', 'exponential_paired', "
            "or 'exponential_depth_paired', or 'flat_roots'."
        )
    if config.graph_family == "flat_roots":
        if config.max_depth != 0:
            raise ValueError("flat_roots requires task.max_depth: 0.")
        if config.trace_sampling != "principal":
            raise ValueError("flat_roots requires experiment.trace_sampling: principal.")
        if config.flat_frequency_exponent is None or config.flat_frequency_exponent <= 0:
            raise ValueError(
                "flat_roots requires experiment.flat_frequency_exponent > 0."
            )
    if config.target_alpha is not None and config.target_alpha <= 0:
        raise ValueError("target_alpha must be positive or null.")
    if config.graph_family in {
        "exponential_paired",
        "exponential_depth_paired",
    }:
        if config.trace_sampling != "ideal_path":
            raise ValueError(
                f"{config.graph_family} graphs require trace_sampling: ideal_path."
            )
        if config.graph_family == "exponential_paired" and config.target_alpha is None:
            raise ValueError("exponential_paired graphs require target_alpha.")
        if config.graph_family == "exponential_depth_paired" and config.target_alpha is not None:
            raise ValueError(
                "exponential_depth_paired derives alpha from rho and beta; "
                "target_alpha must be null."
            )
        if config.m != 2:
            raise ValueError(
                f"{config.graph_family} graphs currently require m: 2."
            )
        if config.base_tasks % config.m != 0:
            raise ValueError(
                f"{config.graph_family} graphs require base_tasks divisible by m."
            )
        if any(
            not math.isclose(float(row["rho"]), round(float(row["rho"])))
            or float(row["rho"]) < 1
            for row in _compositional_sweep_rows(config)
        ):
            raise ValueError(
                f"{config.graph_family} graphs require integer rho >= 1."
            )
        if any(
            not math.isclose(float(row["base_freq"]), 1.0)
            for row in _compositional_sweep_rows(config)
        ):
            raise ValueError(
                f"{config.graph_family} graphs currently require base_freq: 1.0."
            )
        if config.graph_family == "exponential_depth_paired" and any(
            float(row["beta"]) <= float(row["rho"])
            for row in _compositional_sweep_rows(config)
        ):
            raise ValueError(
                "exponential_depth_paired requires beta > rho so compact-path "
                "terminal probabilities are positive."
            )
    elif config.trace_sampling == "ideal_path":
        raise ValueError(
            "ideal_path trace sampling requires a paired exponential graph family."
        )
    if config.trace_sampling == "ideal_path":
        if config.quanta_demand != "composition":
            raise ValueError("ideal_path trace sampling requires quanta_demand: composition.")
        if config.loss_supervision != "all":
            raise ValueError("ideal_path trace sampling requires loss_supervision: all.")
        if config.eval_loss_formula != "quanta_weighted":
            raise ValueError("ideal_path trace sampling requires eval_loss_formula: quanta_weighted.")
        if config.attention_masking != "only_parent_outs":
            raise ValueError(
                "ideal_path requires attention_masking: only_parent_outs so each "
                "quantum is evaluated in the same causal context used in training."
            )
    if config.attention_masking not in {"none", "other_bits", "only_parent_outs"}:
        raise ValueError(
            "attention_masking must be 'none', 'other_bits', or 'only_parent_outs'."
        )
    if config.n_lut_functions is not None and int(config.n_lut_functions) != -1 and int(config.n_lut_functions) <= 0:
        raise ValueError("n_lut_functions must be null, -1, or a positive integer.")
    if config.lut_family not in {"random_balanced", "symmetry_orbit", "parity"}:
        raise ValueError(
            "lut_family must be 'random_balanced', 'symmetry_orbit', or 'parity'."
        )
    if config.lut_family != "random_balanced" and config.n_lut_functions is None:
        raise ValueError("non-default lut_family requires n_lut_functions.")
    if config.function_seed is not None and (
        not isinstance(config.function_seed, int)
        or isinstance(config.function_seed, bool)
        or config.function_seed < 0
    ):
        raise ValueError("function_seed must be a non-negative integer or null.")
    if config.exhaustive_local_eval:
        if (
            config.task != "cnand"
            or config.graph_family not in {"flat_roots", "exponential_random"}
            or int(config.max_depth) != 0
        ):
            raise ValueError(
                "exhaustive_local_eval currently requires depth-zero cNAND roots."
            )
        if int(config.n_local_bits) > 10:
            raise ValueError("exhaustive_local_eval supports at most 10 local bits.")
    dataset_size = (
        int(config.offline_dataset_size)
        if not config.dynamic and config.offline_dataset_size is not None
        else None
    )
    if not config.dynamic and (dataset_size is None or dataset_size <= 0):
        raise ValueError("offline cNAND requires training.offline_dataset_size > 0.")
    resolve_steps_and_epochs(config, dataset_size=dataset_size)
    return config


def _validate_demand_config(config: TrainingConfig) -> None:
    if config.quanta_demand not in {"shortcut", "composition", "uniform"}:
        raise ValueError(
            "quanta_demand must be 'shortcut', 'composition', or 'uniform'."
        )
    if config.quanta_demand_fit_tolerance < 0:
        raise ValueError("quanta_demand_fit_tolerance must be non-negative.")
    if config.quanta_demand == "composition" and not config.beta:
        raise ValueError("composition demand requires beta.")


def _validate_compositional_sweep_lengths(config: ScalingLawsConfig) -> None:
    relevant_fields = _compositional_relevant_sweep_fields(config)
    sweep_lengths = {
        field: len(values)
        for field in COMPOSITIONAL_SWEEP_FIELDS
        if field in relevant_fields
        if len(values := _compositional_sweep_values(field, getattr(config, field))) > 1
    }
    if len(set(sweep_lengths.values())) > 1:
        details = ", ".join(f"{field}={length}" for field, length in sorted(sweep_lengths.items()))
        raise ValueError(
            "Compositional scaling sweep parameters with multiple values must have the same length "
            f"({details})."
        )


def _compositional_sweep_rows(config: ScalingLawsConfig) -> list[dict[str, object]]:
    relevant_fields = _compositional_relevant_sweep_fields(config)
    values_by_field = {
        field: _compositional_sweep_values(field, getattr(config, field))
        if field in relevant_fields
        else _compositional_sweep_values(field, getattr(config, field))[:1]
        for field in COMPOSITIONAL_SWEEP_FIELDS
    }
    sweep_size = max((len(values) for values in values_by_field.values()), default=1)
    rows = []
    for index in range(sweep_size):
        rows.append(
            {
                field: values[index] if len(values) > 1 else values[0]
                for field, values in values_by_field.items()
            }
        )
    return rows


def _compositional_sweep_values(field: str, value: object) -> list[object]:
    if isinstance(value, list):
        if field == "delta" and not value:
            return [0.0]
        return value
    return [value]


def _list_values(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def _compositional_relevant_sweep_fields(config: ScalingLawsConfig) -> set[str]:
    architectures = {str(value) for value in _compositional_sweep_values("architecture", config.architecture)}
    return {
        field
        for architecture in architectures
        for field in COMPOSITIONAL_SWEEP_FIELDS
        if _compositional_field_relevant(field, architecture, config.task)
    }


def _compositional_field_relevant(field: str, architecture: str, task: str) -> bool:
    del task
    if architecture == "transformer":
        return field not in COMPOSITIONAL_MLP_ONLY_SWEEP_FIELDS
    if architecture == "mlp":
        return field not in COMPOSITIONAL_TRANSFORMER_ONLY_SWEEP_FIELDS
    return True


def load_plot_config(
    path: str | Path,
    experiment_name: str | None = None,
    subexperiment_name: str | None = None,
) -> PlotConfig:
    data = _plot_config_data(
        path,
        experiment_name=experiment_name,
        subexperiment_name=subexperiment_name,
    )
    unknown = sorted(set(data) - set(PlotConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown plotting config keys in {path}: {', '.join(unknown)}")
    config = PlotConfig(**data)
    if not 0.0 <= float(config.smoothing) <= 1.0:
        raise ValueError("smoothing must be between 0 and 1.")
    if config.x_scale not in {"log", "linear"}:
        raise ValueError("x_scale must be 'log' or 'linear'.")
    if config.x_axis not in {"steps", "samples", "effective_samples", "gradient_exposure"}:
        raise ValueError("x_axis must be 'steps', 'samples', 'effective_samples', or 'gradient_exposure'.")
    if float(config.x_start) < 0:
        raise ValueError("x_start must be non-negative.")
    if config.x_lim is not None and float(config.x_lim) <= 0:
        raise ValueError("x_lim must be positive or null.")
    if config.x_lim is not None and float(config.x_lim) <= float(config.x_start):
        raise ValueError("x_lim must be greater than x_start.")
    if config.ylim is not None and float(config.ylim) <= 0:
        raise ValueError("ylim must be positive or null.")
    if config.loss_decomposition not in {
        "tasks",
        "tasks_dependencies_ready",
        "quanta",
    }:
        raise ValueError(
            "loss_decomposition must be 'tasks', 'tasks_dependencies_ready', or 'quanta'."
        )
    if not isinstance(config.weighted_loss, bool):
        raise ValueError("weighted_loss must be true or false.")
    if not config.output_filename.strip():
        raise ValueError("output_filename must not be empty.")
    if not config.loss_decomposition_filename.strip():
        raise ValueError("loss_decomposition_filename must not be empty.")
    if not config.demand_diagnostics_filename.strip():
        raise ValueError("demand_diagnostics_filename must not be empty.")
    if not isinstance(config.animate_loss_decomposition, bool):
        raise ValueError("animate_loss_decomposition must be true or false.")
    return config


def _plot_config_data(
    path: str | Path,
    *,
    experiment_name: str | None,
    subexperiment_name: str | None,
) -> dict:
    raw = _normalize_keys(_load_yaml_mapping(path))
    fields = set(PlotConfig.__dataclass_fields__)
    if any(key in fields for key in raw):
        return raw
    known_sections = {
        "common",
        "general",
        "quanta_discovery",
        "scaling_laws",
        "posets_probing",
        "quanta_net",
        "quanta_steering",
    }
    if not any(key in known_sections for key in raw):
        return raw

    shared: dict = {}
    for key in ("common", "general"):
        section = raw.get(key)
        if section is not None:
            if not isinstance(section, dict):
                raise ValueError(f"Plot config section {key!r} in {path} must be a mapping.")
            shared.update(_normalize_keys(section))

    if experiment_name is None:
        section_names = set(raw) - {"common", "general"}
        if section_names:
            raise ValueError(
                f"Plot config {path} has experiment sections; pass experiment_name to select one."
            )
        return shared

    experiment_key = experiment_name.lower()
    selected = raw.get(experiment_key, {})
    if selected is None:
        selected = {}
    if not isinstance(selected, dict):
        raise ValueError(f"Plot config section {experiment_key!r} in {path} must be a mapping.")
    selected = _normalize_keys(selected)
    if subexperiment_name is not None:
        subexperiment_key = subexperiment_name.lower()
        subexperiment_section = selected.get(subexperiment_key, {})
        if subexperiment_section is None:
            subexperiment_section = {}
        if not isinstance(subexperiment_section, dict):
            raise ValueError(
                f"Plot config section {experiment_key}.{subexperiment_key!r} in {path} must be a mapping."
            )
        selected = {
            key: value
            for key, value in selected.items()
            if key in fields
        }
        selected.update(_normalize_keys(subexperiment_section))
    unknown_sections = sorted(set(raw) - known_sections)
    if unknown_sections:
        raise ValueError(f"Unknown plotting config sections in {path}: {', '.join(unknown_sections)}")
    return {**shared, **selected}


__all__ = [
    "QuantaDiscoveryConfig",
    "PosetsProbingConfig",
    "QuantaNetConfig",
    "QuantaSteeringConfig",
    "ScalingLawsConfig",
    "PlotConfig",
    "TrainingConfig",
    "load_experiment_config",
    "load_quanta_net_config",
    "load_quanta_steering_config",
    "load_quanta_discovery_config",
    "load_posets_probing_config",
    "load_scaling_laws_config",
    "load_plot_config",
    "resolve_steps_and_epochs",
]
