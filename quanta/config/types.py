from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    task: str = "cnand"
    width: int | list[int] = 128
    depth: int = 3
    layernorm: bool = False
    seed: int | list[int] = 2
    steps: int | None = 200000
    epochs: float | None = None
    samples_per_task: int | None = None
    eval_samples_per_task: int = 500
    n_tasks: int = field(default=0, init=False)
    n_bits: int = field(default=0, init=False)
    n_atomic_task_bits: int = 4
    n_local_bits: int = 2
    parity_task_bits: int = 100
    parity_subset_size: int = 3
    n_noise_bits: int = 0
    task_bits: list[int] = field(default_factory=lambda: [-1, 1])
    task_frequencies: dict[int, float] | None = None
    quanta_demand: str = "shortcut"
    quanta_demand_fit_tolerance: float = 0.02
    quanta_demand_diagnostics: dict[str, Any] | None = field(default=None, init=False)
    trace_sampling: str = "principal"
    lr: float = 1e-3
    scheduler: str = "constant"
    warmup_phase: float = 0.1
    plateau_phase: float = 0.0
    weight_decay: float = 1e-4
    eval_steps: int = 200
    save_steps: int | None = None
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    save_dir: str | None = field(default=None, init=False)
    verbose: bool = False
    dynamic: bool = True
    offline_dataset_size: int | None = None
    wandb_project: str | None = "cnand"
    wandb_mode: str | None = None
    graph_dependencies: dict[int, list[int]] | None = None
    discreteness_transition_error: bool = False
    rho: float | None = None
    beta: float | None = None
    delta: float | None = 0.0
    base_tasks: int = 4
    base_freq: float = 1.0
    max_depth: int | None = None
    m: int = 2
    graph_family: str = "exponential_random"
    flat_frequency_exponent: float | None = None
    target_alpha: float | None = None
    activation: str = "relu"
    architecture: str = "mlp"
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    transformer_activation: str = "gelu"
    alternative_decomposition: str = "dependencies_ready"
    attention_masking: str = "none"
    n_lut_functions: int | None = None
    lut_family: str = "random_balanced"
    function_seed: int | None = None
    exhaustive_local_eval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScalingLawsConfig(TrainingConfig):
    task: str = "cnand"
    wandb_project: str | None = None
    width: int | list[int] = 128
    rho: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    rho_beta_delta: list[list[float]] = field(default_factory=lambda: [[2.0, 4.0, 0.0]])
    seed: int | list[int] = 0
    eval_seed: int | None = None
    base_tasks: int = 4
    base_freq: float = 1.0
    max_depth: int = 2
    m: int = 2
    eval_loss_formula: str = "quanta_weighted"
    loss_supervision: str = "all"
    lr: list[float] = field(default_factory=lambda: [1e-3])
    mixed_precision: str | None = None
    run_tag: str | None = None


@dataclass
class PosetsProbingConfig:
    task: str = "number_naming"
    language: str = "english"
    data_splits: str = "uniform"
    training_size: int = 50_000
    eval_size: int | None = 10_000
    eval_strategy: str = "digits_wise"
    eval_samples: int | None = None
    eval_samples_per_digit: int | None = None
    split_seed: int = 0
    max_number: int = 999_999
    architecture: str = "decoder_transformer"
    n_layers: int = 2
    d_model: int = 128
    width: int | None = None
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_seq_len: int = 64
    steps: int | None = 10_000
    epochs: float | None = None
    batch_size: int = 128
    lr: float = 3e-4
    scheduler: str = "constant"
    warmup_phase: float = 0.1
    plateau_phase: float = 0.0
    weight_decay: float = 0.0
    eval_steps: int | None = 500
    evals_per_epoch: int | None = None
    seed: int = 0
    save_steps: int | None = None
    eval_batch_size: int = 512
    probe_quanta_poset: str | list[str] | None = None
    probe_max_size: int | None = None
    factorized_learned_threshold: float = 0.1
    factorized_stability_window: int = 2
    audit_quanta_poset: list[str] | None = None
    patching_compatibility_max_classes: int | None = 48
    patching_compatibility_examples_per_class: int = 8
    held_out_token_roles: list[dict[str, str]] | None = None
    device: str | None = None
    wandb_project: str | None = None
    wandb_mode: str | None = None
    save_dir: str | None = field(default=None, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantaNetConfig:
    task: str = "number_naming"
    language: str = "english"
    data_splits: str = "uniform"
    training_size: int = 50_000
    eval_size: int | None = 10_000
    eval_strategy: str = "digits_wise"
    eval_samples: int | None = None
    eval_samples_per_digit: int | None = None
    split_seed: int = 0
    max_number: int = 999_999
    compiled_program_path: str | None = None
    d_source: int = 32
    d_quantum: int = 32
    read_heads: int = 1
    quantum_mlp_ratio: float = 1.0
    activation: str = "gelu"
    add_initial_state: bool = True
    all_quanta_output: bool = True
    max_seq_len: int = 64
    steps: int | None = 10_000
    epochs: float | None = None
    batch_size: int = 128
    lr: float = 3e-4
    scheduler: str = "constant"
    warmup_phase: float = 0.1
    plateau_phase: float = 0.0
    weight_decay: float = 0.0
    eval_steps: int | None = 500
    evals_per_epoch: int | None = None
    full_eval_steps: int | None = None
    parent_audit_size: int = 128
    seed: int = 0
    save_steps: int | None = None
    eval_batch_size: int = 512
    activity_weight: float = 1.0
    semantic_weight: float = 1.0
    max_supervised_classes: int = 20
    device: str | None = None
    wandb_project: str | None = None
    wandb_mode: str | None = None
    save_dir: str | None = field(default=None, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantaSteeringConfig(QuantaNetConfig):
    q_checkpoint: str | None = None
    transformer_d_model: int = 32
    transformer_layers: int = 7
    transformer_heads: int = 1
    transformer_mlp_ratio: float = 2.0
    transformer_dropout: float = 0.0
    interface_seed: int = 0
    lambda_align: float = 1.0
    alignment_scale_mode: str = "unit"
    alignment_scales: list[float] | None = None
    alignment_scale_epsilon: float = 1.0e-8
    alignment_target_control: str = "normal"
    minimum_oracle_token_accuracy: float = 0.99
    minimum_oracle_exact_accuracy: float = 0.98
    control_seed: int = 0


@dataclass
class QuantaDiscoveryConfig:
    """Configuration for the single Quanta-discovery v0 pipeline."""

    task: str = "number_naming"
    language: str = "english"
    data_splits: str = "extreme_generalization_v2"
    training_size: int = 300
    eval_size: int | None = 16_384
    eval_strategy: str = "balanced"
    eval_samples: int | None = 300
    eval_samples_per_digit: int | None = None
    split_seed: int = 0
    max_number: int = 999_999
    architecture: str = "decoder_transformer"
    n_layers: int = 3
    d_model: int = 32
    n_heads: int = 1
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    max_seq_len: int = 20
    steps: int | None = 5_000
    epochs: float | None = None
    batch_size: int = 300
    lr: float = 0.03
    optimizer: str = "sgd"
    scheduler: str = "constant"
    warmup_phase: float = 0.1
    plateau_phase: float = 0.0
    weight_decay: float = 0.0
    seed: int = 0
    checkpoint_every: int = 125
    priority_estimator: str = "exact_gd"
    priority_crossfit_folds: int = 2
    priority_direction_examples: int = 100
    priority_direction_replicates: int = 2
    priority_split_seed: int = 0
    max_candidates_per_layer: int = 16
    candidate_fit_steps: int = 40
    minimum_candidate_gain: float = 0.05
    q_learning_rate: float = 3.0e-3
    q_microbatch_size: int = 300
    q_attention_rank: int = 1
    q_attention_value_dim: int = 4
    q_attention_direct_residual: bool = True
    q_share_attention_projections: bool = False
    q_attention_query_offsets: bool = False
    q_training_mode: str = "online_trajectory"
    q_iid_checkpoint_count: int = 200
    q_optimizer_steps: int | None = None
    q_checkpoint_steps: tuple[int, ...] = ()
    q_routing_mode: str = "layer_residual_roots"
    q_edge_discovery: str = "fixed_closure"
    q_edge_l0_weight: float = 0.0
    q_edge_l0_ramp_steps: int = 1_000
    q_edge_entropy_weight: float = 0.0
    q_edge_entropy_start_step: int = 2_500
    q_edge_entropy_ramp_steps: int = 1_000
    q_edge_freeze_steps: int = 0
    q_reconstruction_positions: str = "prediction_positions"
    q_writer_minimum: int = 4
    q_writer_overcomplete_factor: float = 4.0
    q_writer_maximum: int = 32
    q_writer_total_budget: int | None = None
    q_functional_rank_energy: float = 0.90
    q_writer_average_k: float = 1.0
    q_writer_activation: str = "jumprelu"
    q_writer_sparsity: str = "l0_target"
    q_writer_jump_threshold: float = 0.10
    q_writer_jump_bandwidth: float = 0.10
    q_writer_l0_scope: str = "per_active_quantum"
    q_writer_l0_target: float = 3.0
    q_writer_l0_weight: float = 1.0
    q_writer_l0_ramp_steps: int = 1_000
    q_output_kl_weight: float = 0.7
    q_gate_entropy_weight: float = 0.3
    q_gate_entropy_start_step: int = 550
    q_gate_entropy_ramp_steps: int = 275
    q_balance_gate_bce: bool = True
    maximum_edge_closure_cost: float = 0.10
    detach_gate_gradients: bool = True
    device: str | None = None
    save_dir: str | None = field(default=None, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlotConfig:
    window: int = 100
    smoothing: float = 0.0
    x_start: float = 10.0
    x_lim: float | None = None
    ylim: float | None = None
    x_scale: str = "linear"
    x_axis: str = "steps"
    loss_decomposition: str = "tasks"
    weighted_loss: bool = True
    record_train_loss: bool = False
    plot_total_loss: bool = False
    output_filename: str = "scaling_law.png"
    loss_decomposition_filename: str = "loss_decomposition.png"
    demand_diagnostics_filename: str = "rank_demand.png"
    animate_loss_decomposition: bool = True
    save_pdf: bool = False

    def trajectory_kwargs(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "smoothing": self.smoothing,
            "x_start": self.x_start,
            "x_lim": self.x_lim,
            "ylim": self.ylim,
            "x_scale": self.x_scale,
            "x_axis": self.x_axis,
            "loss_decomposition": self.loss_decomposition,
            "weighted_loss": self.weighted_loss,
            "record_train_loss": self.record_train_loss,
            "plot_total_loss": self.plot_total_loss,
            "save_pdf": self.save_pdf,
        }
