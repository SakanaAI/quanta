from quanta.experiments.scaling_laws.batches import (
    build_seeded_cnand_batch_cache as build_cnand_batch_cache,
    build_sampled_cnand_batch,
)
from quanta.experiments.scaling_laws.batch_common import task_probability_tensor, local_batch_size_for_process

import math
import os
import pickle
import json
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quanta.config import ScalingLawsConfig, PlotConfig
from quanta.experiments.scaling_laws.graph import generate_layered_poset
from quanta.experiments.scaling_laws import (
    ScalingLawsExperiment,
    aggregate_scaling_results,
    build_scaling_run_jobs,
    CNANDTransformerModel,
    scaling_run_save_dir,
    theoretical_alpha,
    _select_resume_state,
)
from quanta.experiments.scaling_laws.experiment import (
    _compact_summary,
)
from quanta.experiments.scaling_laws.aggregation import discover_all_runs, select_best_runs
from quanta.experiments.scaling_laws.metrics import (
    tail_median,
    trainable_parameter_counts,
)
from quanta.experiments.scaling_laws.run import _demand_record
from quanta.experiments.scaling_laws.run import masked_token_mean_loss
from quanta.tools.masked_cnand import MaskedCNANDTransformer, structured_attention_mask
from quanta.experiments.scaling_laws.resume import _resume_candidate_dirs
from quanta.experiments.scaling_laws.resume import _configs_match_for_resume
from quanta.experiments.scaling_laws.graph import generate_layered_poset
from quanta.experiments.scaling_laws.batches import (
    build_cnand_batch_from_targets,
    evaluate_weighted_task_loss,
    evaluate_cnand_out_values,
    masked_token_cross_entropy,
)
from quanta.experiments.scaling_laws.training.trainer import (
    _evaluation_rng,
    scheduled_learning_rate,
    train_until_budget_or_convergence,
)
from quanta.experiments.scaling_laws.training.trainer_state import (
    restore_process_rng_state,
)
from quanta.experiments.scaling_laws.training.trainer_curriculum import learned_task_percentages_by_depth
from quanta.experiments.scaling_laws.wandb_logging import log_scaling_wandb
from quanta.figures.scaling_laws import (
    _plot_title,
    _relative_error,
    _relative_error_label,
    _r_squared_label,
    _theoretical_alpha_prefix,
)
from quanta.experiments.task_specs import TaskSpecBuilder
from quanta.models import MLP, SquareActivation
from quanta.plot import plot_from_pkl
from quanta.utils import set_seeds


class ScalingLawsTests(unittest.TestCase):
    def test_tail_median_uses_at_most_the_final_six_evaluations(self):
        self.assertEqual(tail_median([100.0, 6.0, 1.0, 5.0, 2.0, 4.0, 3.0]), 3.5)
        self.assertEqual(tail_median([3.0, 1.0, 2.0]), 2.0)

    def test_trainable_parameter_counts_excludes_embedding_modules(self):
        model = nn.Sequential(nn.Embedding(5, 3), nn.Linear(3, 2))

        counts = trainable_parameter_counts(model)

        self.assertEqual(counts["n_parameters"], 23)
        self.assertEqual(counts["n_embedding_parameters"], 15)
        self.assertEqual(counts["n_non_embedding_parameters"], 8)

    def test_structured_attention_mask_reuses_shared_layout(self):
        slot_ids = torch.arange(6).unsqueeze(0).expand(4, -1)
        mask = structured_attention_mask(
            slot_ids,
            tokens_per_node=2,
            n_heads=2,
            mask_mode="only_parent_outs",
            parent_indices=torch.tensor([[0], [0], [1]], dtype=torch.long),
            parent_masks=torch.tensor([[False], [True], [True]]),
        )

        self.assertEqual(tuple(mask.shape), (6, 6))

    def test_masked_transformer_treats_null_mask_mode_as_unmasked(self):
        config = ScalingLawsConfig(
            width=8,
            depth=1,
            n_heads=1,
            mlp_ratio=1.0,
        )
        model = MaskedCNANDTransformer(
            config=config,
            n_slots=6,
            tokens_per_node=2,
            mask_mode=None,
            parent_indices=torch.zeros((3, 1), dtype=torch.long),
            parent_masks=torch.zeros((3, 1), dtype=torch.bool),
        )
        batch = {
            "input_ids": torch.zeros((1, 6), dtype=torch.long),
            "slot_ids": torch.arange(6).unsqueeze(0),
            "active_mask": torch.ones((1, 6), dtype=torch.bool),
        }

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (1, 6, 2))

    def test_resume_treats_null_and_none_attention_masking_as_equal(self):
        saved = {"attention_masking": None}
        current = {"attention_masking": "none"}

        self.assertTrue(_configs_match_for_resume(saved, current))

    def test_resume_distinguishes_loss_supervision(self):
        saved = {"loss_supervision": "all"}
        current = {"loss_supervision": "target_only"}

        self.assertFalse(_configs_match_for_resume(saved, current))

    def test_resume_distinguishes_trace_sampling(self):
        saved = {"trace_sampling": "principal"}
        current = {"trace_sampling": "ideal_threshold"}

        self.assertFalse(_configs_match_for_resume(saved, current))

    def test_ideal_threshold_batches_share_one_downward_closed_context(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            graph_dependencies={2: [0], 3: [0, 1], 4: [2, 3]},
            rho=[2.0],
            beta=[2.0],
            delta=[0.0],
            quanta_demand="composition",
            trace_sampling="ideal_threshold",
            width=8,
            depth=1,
            n_heads=1,
            mlp_ratio=1.0,
            n_local_bits=1,
            max_depth=2,
            seed=7,
            steps=1,
            batch_size=16,
            eval_samples_per_task=2,
            eval_steps=1,
            lr=1e-3,
        )
        task_spec = TaskSpecBuilder().build(config)
        batch_cache = build_cnand_batch_cache(config, task_spec, "cpu")
        probabilities = task_probability_tensor(
            task_spec.codes,
            config.task_frequencies,
            "cpu",
        )

        reconstructed = (
            batch_cache["ideal_probabilities"].to(dtype=torch.float64)
            @ batch_cache["ideal_masks"].to(dtype=torch.float64)
        )
        torch.testing.assert_close(
            reconstructed,
            batch_cache["theoretical_p_q"].to(dtype=torch.float64),
        )
        batch, _ = build_sampled_cnand_batch(
            config=config,
            task_spec=task_spec,
            probabilities=probabilities,
            batch_cache=batch_cache,
            device="cpu",
            batch_size=16,
        )

        self.assertTrue(
            bool((batch["active_node_mask"] == batch["active_node_mask"][0]).all())
        )
        active = set(batch["active_node_mask"][0].nonzero().flatten().tolist())
        for child, parents in config.graph_dependencies.items():
            if child in active:
                self.assertTrue(set(parents).issubset(active))
        expected_supervised = int(batch["active_node_mask"].sum().item())
        self.assertEqual(int(batch["loss_mask"].sum().item()), expected_supervised)

    def test_ideal_threshold_batch_can_reuse_an_ideal_across_microbatches(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            graph_dependencies={2: [0], 3: [0, 1]},
            rho=[2.0],
            beta=[2.0],
            delta=[0.0],
            quanta_demand="composition",
            trace_sampling="ideal_threshold",
            width=8,
            depth=1,
            n_heads=1,
            mlp_ratio=1.0,
            n_local_bits=1,
            max_depth=2,
            seed=7,
            steps=1,
            batch_size=8,
            eval_samples_per_task=2,
            eval_steps=1,
            lr=1e-3,
        )
        task_spec = TaskSpecBuilder().build(config)
        batch_cache = build_cnand_batch_cache(config, task_spec, "cpu")
        probabilities = task_probability_tensor(task_spec.codes, config.task_frequencies, "cpu")
        first, _ = build_sampled_cnand_batch(
            config=config,
            task_spec=task_spec,
            probabilities=probabilities,
            batch_cache=batch_cache,
            device="cpu",
            batch_size=4,
        )
        second, _ = build_sampled_cnand_batch(
            config=config,
            task_spec=task_spec,
            probabilities=probabilities,
            batch_cache=batch_cache,
            device="cpu",
            batch_size=4,
            ideal_index=first["ideal_indices"][0],
        )
        torch.testing.assert_close(
            first["active_node_mask"][0], second["active_node_mask"][0]
        )

    def test_ideal_threshold_training_runs_end_to_end(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            graph_dependencies={2: [0, 1]},
            rho=[2.0],
            beta=[2.53],
            delta=[0.0],
            quanta_demand="composition",
            trace_sampling="ideal_threshold",
            width=8,
            depth=1,
            n_heads=1,
            mlp_ratio=1.0,
            n_local_bits=1,
            max_depth=1,
            seed=7,
            steps=2,
            batch_size=4,
            gradient_accumulation_steps=2,
            eval_samples_per_task=2,
            eval_steps=1,
            lr=1e-3,
            wandb_project=None,
        )
        task_spec = TaskSpecBuilder().build(config)
        config.n_tasks = task_spec.n_tasks
        cache = build_cnand_batch_cache(config, task_spec, "cpu")
        model = CNANDTransformerModel(
            config=config,
            n_slots=int(cache["slot_ids"].shape[0]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            results = train_until_budget_or_convergence(
                model=model,
                loss_fn=masked_token_mean_loss,
                config=config,
                task_spec=task_spec,
                save_dir=temp_dir,
                node_depths={0: 0, 1: 0, 2: 1},
            )

        self.assertEqual(results["steps_run"], 2)
        self.assertEqual(results["trace_sampling"], "ideal_threshold")
        self.assertTrue(math.isfinite(results["final_eval_loss_bits"]))

    def test_cnand_loss_supervision_all_supervises_entire_active_closure(self):
        config, task_spec, _, _ = self._tiny_cnand_training_setup(
            steps=1,
            eval_steps=1,
        )
        config.loss_supervision = "all"
        batch_cache = build_cnand_batch_cache(config, task_spec, "cpu")

        batch = build_cnand_batch_from_targets(
            config=config,
            target_indices=torch.tensor([2]),
            batch_cache=batch_cache,
            device="cpu",
        )

        self.assertEqual(batch["active_mask"].sum().item(), 6)
        self.assertEqual(batch["loss_mask"].sum().item(), 3)
        self.assertEqual(
            (batch["loss_targets"] != -100).sum().item(),
            3,
        )

    def test_cnand_loss_supervision_target_only_keeps_closure_active(self):
        config, task_spec, _, _ = self._tiny_cnand_training_setup(
            steps=1,
            eval_steps=1,
        )
        config.loss_supervision = "target_only"
        batch_cache = build_cnand_batch_cache(config, task_spec, "cpu")

        batch = build_cnand_batch_from_targets(
            config=config,
            target_indices=torch.tensor([2]),
            batch_cache=batch_cache,
            device="cpu",
        )

        target_slot = int(batch["target_slots"].item())
        self.assertEqual(batch["active_mask"].sum().item(), 6)
        self.assertEqual(batch["loss_mask"].sum().item(), 1)
        self.assertTrue(bool(batch["loss_mask"][0, target_slot]))
        self.assertEqual(
            (batch["loss_targets"] != -100).sum().item(),
            1,
        )

    def _tiny_cnand_training_setup(self, *, steps: int, eval_steps: int):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            graph_dependencies={2: [0, 1]},
            task_frequencies={0: 1.0, 1: 1.0, 2: 1.0},
            width=8,
            depth=1,
            n_heads=1,
            mlp_ratio=1.0,
            n_local_bits=1,
            max_depth=1,
            seed=7,
            steps=steps,
            batch_size=4,
            eval_samples_per_task=2,
            eval_steps=eval_steps,
            lr=1e-3,
        )
        task_spec = TaskSpecBuilder().build(config)
        config.n_tasks = task_spec.n_tasks
        cache = build_cnand_batch_cache(config, task_spec, "cpu")
        set_seeds(config.seed)
        model = CNANDTransformerModel(
            config=config,
            n_slots=int(cache["slot_ids"].shape[0]),
        )
        node_depths = {0: 0, 1: 0, 2: 1}
        return config, task_spec, model, node_depths

    def test_cnand_evaluation_frequency_does_not_change_training_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = self._tiny_cnand_training_setup(steps=3, eval_steps=1)
            second = self._tiny_cnand_training_setup(steps=3, eval_steps=3)
            for index, (config, task_spec, model, node_depths) in enumerate((first, second)):
                train_until_budget_or_convergence(
                    model=model,
                    loss_fn=masked_token_mean_loss,
                    config=config,
                    task_spec=task_spec,
                    save_dir=os.path.join(temp_dir, str(index)),
                    node_depths=node_depths,
                )

            first_state = torch.load(os.path.join(temp_dir, "0", "model.pt"), map_location="cpu")
            second_state = torch.load(os.path.join(temp_dir, "1", "model.pt"), map_location="cpu")
            self.assertEqual(first_state.keys(), second_state.keys())
            for name in first_state:
                self.assertTrue(torch.equal(first_state[name], second_state[name]), name)

    def test_cnand_fixed_eval_seed_reuses_contexts_without_advancing_training_rng(self):
        config = ScalingLawsConfig(seed=7, eval_seed=123)
        set_seeds(99)
        expected_training_draw = torch.rand(4)

        set_seeds(99)
        with _evaluation_rng(config):
            first_eval_draw = torch.rand(4)
        actual_training_draw = torch.rand(4)

        torch.rand(11)
        with _evaluation_rng(config):
            second_eval_draw = torch.rand(4)

        self.assertTrue(torch.equal(first_eval_draw, second_eval_draw))
        self.assertTrue(torch.equal(expected_training_draw, actual_training_draw))

    def test_training_always_evaluates_before_the_first_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, task_spec, model, node_depths = self._tiny_cnand_training_setup(
                steps=2,
                eval_steps=5000,
            )

            results = train_until_budget_or_convergence(
                model=model,
                loss_fn=masked_token_mean_loss,
                config=config,
                task_spec=task_spec,
                save_dir=temp_dir,
                node_depths=node_depths,
            )

        self.assertEqual(results["eval_steps"], [0, 2])
        self.assertEqual(results["samples"], [0, 8])
        self.assertEqual(results["tail_median_eval_points"], 2)
        self.assertEqual(results["tail_median_start_step"], 0)
        self.assertEqual(results["tail_median_end_step"], 2)
        self.assertAlmostEqual(
            results["tail_median_eval_loss_bits"],
            float(np.median(results["eval_losses_bits"])),
        )
        self.assertEqual(
            results["n_parameters"],
            results["n_embedding_parameters"]
            + results["n_non_embedding_parameters"],
        )
        self.assertTrue(
            all(len(curve) == 2 for curve in results["subtask_losses"])
        )

    def test_cnand_resume_restores_rng_and_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            full_config, full_spec, full_model, depths = self._tiny_cnand_training_setup(
                steps=4,
                eval_steps=2,
            )
            full_dir = os.path.join(temp_dir, "full")
            train_until_budget_or_convergence(
                model=full_model,
                loss_fn=masked_token_mean_loss,
                config=full_config,
                task_spec=full_spec,
                save_dir=full_dir,
                node_depths=depths,
            )

            partial_config, partial_spec, partial_model, depths = self._tiny_cnand_training_setup(
                steps=2,
                eval_steps=2,
            )
            partial_dir = os.path.join(temp_dir, "partial")
            partial_results = train_until_budget_or_convergence(
                model=partial_model,
                loss_fn=masked_token_mean_loss,
                config=partial_config,
                task_spec=partial_spec,
                save_dir=partial_dir,
                node_depths=depths,
            )
            resume_config, resume_spec, resume_model, depths = self._tiny_cnand_training_setup(
                steps=4,
                eval_steps=2,
            )
            resumed_dir = os.path.join(temp_dir, "resumed")
            resumed_results = train_until_budget_or_convergence(
                model=resume_model,
                loss_fn=masked_token_mean_loss,
                config=resume_config,
                task_spec=resume_spec,
                save_dir=resumed_dir,
                node_depths=depths,
                resume_state={
                    "mode": "resume",
                    "run_dir": partial_dir,
                    "model_path": os.path.join(partial_dir, "model.pt"),
                    "optimizer_path": os.path.join(partial_dir, "optimizer.pt"),
                    "results": partial_results,
                    "steps_run": 2,
                    "saved_steps": 2,
                },
            )

            full_state = torch.load(os.path.join(full_dir, "model.pt"), map_location="cpu")
            resumed_state = torch.load(os.path.join(resumed_dir, "model.pt"), map_location="cpu")
            for name in full_state:
                self.assertTrue(torch.equal(full_state[name], resumed_state[name]), name)
            self.assertEqual(resumed_results["eval_steps"], [0, 2, 4])

    def test_restore_process_rng_state_keeps_cuda_rng_tensors_on_cpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": [torch.tensor([1, 2, 3], dtype=torch.uint8)],
            }
            torch.save(rng_state, os.path.join(temp_dir, "rng_state_rank3.pt"))

            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.set_rng_state_all") as set_cuda_rng,
            ):
                restored = restore_process_rng_state(
                    temp_dir,
                    process_index=3,
                    device="cuda:3",
                )

            self.assertTrue(restored)
            restored_cuda_states = set_cuda_rng.call_args.args[0]
            self.assertEqual(len(restored_cuda_states), 1)
            self.assertEqual(restored_cuda_states[0].device.type, "cpu")
            self.assertEqual(restored_cuda_states[0].dtype, torch.uint8)

    def test_default_save_dir_includes_batch_size(self):
        config = ScalingLawsConfig(steps=123, batch_size=456)

        experiment = ScalingLawsExperiment(config, plot_config=None)

        self.assertIn("depth3-steps123-batch456", experiment.config.save_dir)

    def test_default_save_dir_includes_base_tasks_and_delta(self):
        config = ScalingLawsConfig(
            steps=123,
            batch_size=456,
            rho=[2.0],
            beta=[2.53],
            delta=[0.0],
        )

        experiment = ScalingLawsExperiment(config, plot_config=None)

        self.assertIn("rho2-beta2p53-delta0-base4-maxdepth2-m2", experiment.config.save_dir)
        self.assertIn("depth3-steps123-batch456", experiment.config.save_dir)

    def test_task_weighted_eval_uses_distinct_default_save_dir(self):
        experiment = ScalingLawsExperiment(
            ScalingLawsConfig(
                steps=123,
                eval_loss_formula="task_weighted",
            ),
            plot_config=None,
        )

        self.assertIn("-evaltask_weighted", experiment.config.save_dir)

    def test_target_only_supervision_uses_distinct_default_save_dir(self):
        experiment = ScalingLawsExperiment(
            ScalingLawsConfig(
                steps=123,
                loss_supervision="target_only",
            ),
            plot_config=None,
        )

        self.assertIn("-suptarget_only", experiment.config.save_dir)

    def test_loss_decomposition_png_and_gif_are_always_generated_per_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "run")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                pickle.dump(
                    {
                        "codes": [0, 1],
                        "subtask_losses": [[0.8, 0.2], [0.9, 0.3]],
                        "eval_losses_bits": [0.75, 0.15],
                        "mean_task_losses_bits": [0.9, 0.4],
                        "quantum_subtask_losses": [[0.7, 0.1], [0.8, 0.2]],
                        "mean_quantum_subtask_losses": [[0.6, 0.2], [0.7, 0.3]],
                        "subtask_train_losses": [[0.85, 0.25], [0.95, 0.35]],
                        "eval_steps": [0, 10],
                        "samples": [100, 200],
                        "graph_dependencies": {1: [0]},
                        "task_frequencies": {0: 1.0, 1: 0.5},
                    },
                    handle,
                )
            experiment = ScalingLawsExperiment(
                ScalingLawsConfig(task="cnand"),
                PlotConfig(
                    window=7,
                    smoothing=0.25,
                    x_start=3.0,
                    x_lim=99.0,
                    ylim=0.75,
                    x_scale="log",
                    x_axis="steps",
                    loss_decomposition="quanta",
                    weighted_loss=False,
                    record_train_loss=True,
                    plot_total_loss=True,
                    loss_decomposition_filename="loss_decomposition.png",
                    save_pdf=True,
                ),
            )
            record = {
                "save_dir": run_dir,
                "architecture": "transformer",
                "depth": 1,
                "width": 32,
                "batch_size": 8,
            }

            with mock.patch(
                "quanta.experiments.scaling_laws.experiment.plot_discovery_trajectories"
            ) as plot_mock:
                experiment._plot_loss_decomposition_run(record)

            self.assertEqual(plot_mock.call_count, 2)
            static_call, animated_call = plot_mock.call_args_list
            self.assertEqual(
                static_call.kwargs["output_image_path"],
                os.path.join(run_dir, "loss_decomposition.png"),
            )
            self.assertEqual(
                animated_call.kwargs["output_image_path"],
                os.path.join(run_dir, "loss_decomposition.gif"),
            )
            self.assertEqual(static_call.kwargs["loss_decomposition"], "quanta")
            self.assertEqual(
                static_call.kwargs["quantum_subtask_losses"],
                [[0.7, 0.1], [0.8, 0.2]],
            )
            self.assertEqual(static_call.kwargs["overall_loss_bits"], [0.9, 0.4])
            self.assertEqual(
                static_call.kwargs["subtask_train_losses"],
                [[0.85, 0.25], [0.95, 0.35]],
            )
            self.assertEqual(static_call.kwargs["window"], 7)
            self.assertEqual(static_call.kwargs["smoothing"], 0.25)
            self.assertEqual(static_call.kwargs["x_start"], 3.0)
            self.assertEqual(static_call.kwargs["x_lim"], 99.0)
            self.assertEqual(static_call.kwargs["ylim"], 0.75)
            self.assertEqual(static_call.kwargs["x_scale"], "log")
            self.assertFalse(static_call.kwargs["weighted_loss"])
            self.assertTrue(static_call.kwargs["record_train_loss"])
            self.assertTrue(static_call.kwargs["plot_total_loss"])
            self.assertTrue(static_call.kwargs["save_pdf"])
            self.assertFalse(static_call.kwargs["legend"])
            self.assertFalse(animated_call.kwargs["legend"])
            self.assertTrue(animated_call.kwargs["animate"])

    def test_scheduled_learning_rate(self):
        self.assertEqual(scheduled_learning_rate(0.1, 1, 10, "constant"), 0.1)
        self.assertEqual(scheduled_learning_rate(0.1, 1, 10, "linear", warmup_phase=0.0), 0.1)
        self.assertAlmostEqual(scheduled_learning_rate(0.1, 10, 10, "linear", warmup_phase=0.0), 0.0)
        self.assertEqual(scheduled_learning_rate(0.1, 1, 10, "cosine", warmup_phase=0.0), 0.1)
        self.assertAlmostEqual(scheduled_learning_rate(0.1, 10, 10, "cosine", warmup_phase=0.0), 0.0)
        self.assertAlmostEqual(scheduled_learning_rate(0.1, 1, 10, "linear", warmup_phase=0.2), 0.05)
        self.assertEqual(scheduled_learning_rate(0.1, 2, 10, "linear", warmup_phase=0.2), 0.1)
        self.assertAlmostEqual(scheduled_learning_rate(0.1, 1, 10, "cosine", warmup_phase=0.2), 0.05)
        self.assertAlmostEqual(
            scheduled_learning_rate(
                0.1, 1, 10, "constant_with_warmup", warmup_phase=0.2
            ),
            0.05,
        )
        self.assertEqual(
            scheduled_learning_rate(
                0.1, 3, 10, "constant_with_warmup", warmup_phase=0.2
            ),
            0.1,
        )
        self.assertAlmostEqual(scheduled_learning_rate(0.1, 10, 10, "linear", warmup_phase=0.2), 0.0)
        self.assertEqual(
            scheduled_learning_rate(0.1, 1, 10, "linear_after_plateau", plateau_phase=0.3),
            0.1,
        )
        self.assertEqual(
            scheduled_learning_rate(0.1, 3, 10, "linear_after_plateau", plateau_phase=0.3),
            0.1,
        )
        self.assertAlmostEqual(
            scheduled_learning_rate(0.1, 4, 10, "linear_after_plateau", plateau_phase=0.3),
            0.1 * (1.0 - 1.0 / 7.0),
        )
        self.assertAlmostEqual(
            scheduled_learning_rate(0.1, 10, 10, "linear_after_plateau", plateau_phase=0.3),
            0.0,
        )

    def test_mlp_uses_square_activation_by_default(self):
        model = MLP(in_features=3, out_features=2, depth=3, width=4)

        activations = [module for module in model.mlp if isinstance(module, SquareActivation)]

        self.assertEqual(len(activations), 2)

    def test_learned_task_percentages_by_depth(self):
        diagnostics = {
            "task_losses": {
                0: {"depth": 0, "loss_bits": 0.01},
                1: {"depth": 0, "loss_bits": 0.10},
                2: {"depth": 1, "loss_bits": 0.04},
            }
        }

        percentages = learned_task_percentages_by_depth(diagnostics, threshold_bits=0.05)

        self.assertEqual(percentages, {0: 50.0, 1: 100.0})

    def test_select_resume_state_accepts_lower_step_matching_run(self):
        config = ScalingLawsConfig(steps=5, batch_size=8, width=[16], lr=[0.001], seed=[0])
        config.width = 16
        config.seed = 0
        config.lr = 0.001

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "run")
            os.makedirs(run_dir)
            saved_config = config.to_dict()
            saved_config.update({"steps": 3, "width": 16, "seed": 0, "lr": 0.001, "model": "MLP"})
            with open(os.path.join(run_dir, "config.json"), "w") as handle:
                json.dump(saved_config, handle)
            with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                pickle.dump({"steps_run": 3, "n_parameters": 10, "final_eval_loss_nats": 1.0, "final_eval_loss_bits": 1.0}, handle)
            torch.save({}, os.path.join(run_dir, "model.pt"))

            state = _select_resume_state([run_dir], config)

        self.assertIsNotNone(state)
        self.assertEqual(state["mode"], "resume")
        self.assertEqual(state["steps_run"], 3)

    def test_select_resume_state_rejects_scheduled_budget_extension(self):
        config = ScalingLawsConfig(
            steps=5,
            batch_size=8,
            lr=[0.001],
            scheduler="cosine",
        )
        config.width = 16
        config.seed = 0
        config.lr = 0.001

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "run")
            os.makedirs(run_dir)
            saved_config = config.to_dict()
            saved_config.update({"steps": 3, "model": "MLP"})
            with open(os.path.join(run_dir, "config.json"), "w") as handle:
                json.dump(saved_config, handle)
            with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                pickle.dump({"steps_run": 3}, handle)
            torch.save({}, os.path.join(run_dir, "model.pt"))

            self.assertIsNone(_select_resume_state([run_dir], config))

    def test_select_resume_state_rejects_different_cnand_lut_count(self):
        config = ScalingLawsConfig(
            steps=5,
            batch_size=8,
            width=16,
            lr=0.001,
            seed=0,
            n_lut_functions=4,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "run")
            os.makedirs(run_dir)
            saved_config = config.to_dict()
            saved_config.update({"steps": 3, "n_lut_functions": 2, "model": "MLP"})
            with open(os.path.join(run_dir, "config.json"), "w") as handle:
                json.dump(saved_config, handle)
            with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                pickle.dump({"steps_run": 3}, handle)
            torch.save({}, os.path.join(run_dir, "model.pt"))

            self.assertIsNone(_select_resume_state([run_dir], config))

    def test_select_resume_state_accepts_same_config_partial_checkpoint(self):
        config = ScalingLawsConfig(steps=5, batch_size=8, width=[16], lr=[0.001], seed=[0])
        config.width = 16
        config.seed = 0
        config.lr = 0.001

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "run")
            os.makedirs(run_dir)
            saved_config = config.to_dict()
            saved_config.update({"steps": 5, "width": 16, "seed": 0, "lr": 0.001, "model": "MLP"})
            with open(os.path.join(run_dir, "config.json"), "w") as handle:
                json.dump(saved_config, handle)
            with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                pickle.dump({"steps_run": 3, "n_parameters": 10, "final_eval_loss_nats": 1.0, "final_eval_loss_bits": 1.0}, handle)
            torch.save({}, os.path.join(run_dir, "model.pt"))

            state = _select_resume_state([run_dir], config)

        self.assertIsNotNone(state)
        self.assertEqual(state["mode"], "resume")
        self.assertEqual(state["steps_run"], 3)

    def test_select_resume_state_marks_same_config_complete_by_steps_run(self):
        config = ScalingLawsConfig(steps=5, batch_size=8, width=[16], lr=[0.001], seed=[0])
        config.width = 16
        config.seed = 0
        config.lr = 0.001

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "run")
            os.makedirs(run_dir)
            saved_config = config.to_dict()
            saved_config.update({"steps": 5, "width": 16, "seed": 0, "lr": 0.001, "model": "MLP"})
            with open(os.path.join(run_dir, "config.json"), "w") as handle:
                json.dump(saved_config, handle)
            with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                pickle.dump({"steps_run": 5, "n_parameters": 10, "final_eval_loss_nats": 1.0, "final_eval_loss_bits": 1.0}, handle)
            torch.save({}, os.path.join(run_dir, "model.pt"))

            state = _select_resume_state([run_dir], config)

        self.assertIsNotNone(state)
        self.assertEqual(state["mode"], "complete")

    def test_resume_candidates_include_sibling_step_budgets(self):
        config = ScalingLawsConfig(
            steps=200_000,
            batch_size=100_000,
            lr=[5e-5],
            max_depth=4,
            m=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            current_base = os.path.join(temp_dir, "depth2-steps200000-batch100000")
            prior_base = os.path.join(temp_dir, "depth2-steps100000-batch100000")
            different_batch = os.path.join(temp_dir, "depth2-steps100000-batch50000")
            os.makedirs(current_base)
            os.makedirs(prior_base)
            os.makedirs(different_batch)

            candidates = _resume_candidate_dirs(
                base_save_dir=current_base,
                pair_index=0,
                rho=2.0,
                beta=2.53,
                seed=0,
                width=2048,
                lr_val=5e-5,
                config=config,
            )

        self.assertEqual(
            candidates,
            [
                os.path.join(
                    current_base,
                    "runs",
                    "pair0",
                    "seed0-width2048-depth3-lr5em05",
                ),
                os.path.join(
                    prior_base,
                    "runs",
                    "pair0",
                    "seed0-width2048-depth3-lr5em05",
                ),
            ],
        )

    def test_resume_candidates_include_legacy_null_mask_directory(self):
        config = ScalingLawsConfig(
            steps=200_000,
            batch_size=4096,
            lr=[5e-5],
            attention_masking="none",
            eval_loss_formula="task_weighted",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            current_base = os.path.join(
                temp_dir,
                "depth1-steps200000-batch4096-evaltask_weighted",
            )
            legacy_base = os.path.join(
                temp_dir,
                "depth1-steps100000-batch4096-maskNone-evaltask_weighted",
            )
            os.makedirs(legacy_base)
            os.makedirs(
                os.path.join(
                    legacy_base,
                    "runs",
                    "pair0",
                    "seed0-width128-depth3",
                )
            )

            candidates = _resume_candidate_dirs(
                base_save_dir=current_base,
                pair_index=0,
                rho=2.0,
                beta=2.53,
                seed=0,
                width=128,
                lr_val=5e-5,
                config=config,
            )

        self.assertIn(
            os.path.join(
                legacy_base,
                "runs",
                "pair0",
                "seed0-width128-depth3",
            ),
            candidates,
        )

    def test_scaling_run_save_dir_uses_pair_index_only(self):
        config = ScalingLawsConfig(width=[2048], lr=[5e-5])

        save_dir = scaling_run_save_dir("/tmp/run", 0, 2.0, 2.53, 0, 2048, 5e-5, config)

        self.assertEqual(save_dir, "/tmp/run/runs/pair0/seed0-width2048-depth3-lr5em05")

    def test_scaling_run_save_dir_uses_transformer_model_scale(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            lr=[5e-5],
            width=64,
            depth=1,
            n_heads=2,
        )
        run_config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            width=256,
            depth=1,
            n_heads=2,
        )

        save_dir = scaling_run_save_dir("/tmp/run", 0, 2.0, 2.53, 0, 64, 5e-5, config, run_config)

        self.assertEqual(save_dir, "/tmp/run/runs/pair0/seed0-width64-depth1-heads2-lr5em05")

    def test_scaling_run_save_dir_separates_lr_grid_with_multiple_widths(self):
        config = ScalingLawsConfig(
            width=[8, 8, 32, 32],
            lr=[1e-4, 3e-4, 1e-4, 3e-4],
        )

        first = scaling_run_save_dir(
            "/tmp/run", 0, 2.0, 2.53, 0, 8, 1e-4, config
        )
        second = scaling_run_save_dir(
            "/tmp/run", 0, 2.0, 2.53, 0, 8, 3e-4, config
        )

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("-lr0p0001"))
        self.assertTrue(second.endswith("-lr0p0003"))

    def test_generate_layered_poset_uses_depth_counts_and_previous_depth_parents(self):
        graph = generate_layered_poset(
            rho=2.0,
            beta=4.0,
            base_tasks=3,
            base_freq=1.0,
            max_depth=2,
            m=2,
            seed=7,
        )

        self.assertEqual([len(graph["depth_nodes"][depth]) for depth in range(3)], [3, 6, 12])
        self.assertEqual(graph["task_frequencies"][0], 1.0)
        self.assertEqual(graph["task_frequencies"][3], 0.25)
        self.assertEqual(graph["task_frequencies"][9], 0.0625)
        self.assertEqual(graph["quanta_demand"]["mode"], "shortcut")

        for node, parents in graph["graph_dependencies"].items():
            depth = graph["node_depths"][node]
            self.assertGreater(depth, 0)
            self.assertEqual(len(parents), 2)
            for parent in parents:
                self.assertEqual(graph["node_depths"][parent], depth - 1)

    def test_flat_root_graph_has_matched_depth_and_rank_frequency(self):
        graph = generate_layered_poset(
            rho=1.0,
            beta=1.0,
            base_tasks=8,
            base_freq=1.0,
            max_depth=0,
            m=0,
            seed=7,
            graph_family="flat_roots",
            flat_frequency_exponent=1.0,
        )

        self.assertEqual(graph["graph_dependencies"], {})
        self.assertEqual(set(graph["node_depths"].values()), {0})
        self.assertEqual(graph["quanta_demand"]["mode"], "flat_frequency")
        self.assertEqual(
            sorted(graph["quanta_demand"]["frequency_ranks"].values()),
            list(range(1, 9)),
        )
        self.assertAlmostEqual(
            max(graph["task_frequencies"].values()),
            1.0,
        )

    def test_task_spec_keeps_an_explicit_empty_root_graph(self):
        config = ScalingLawsConfig(
            graph_dependencies={},
            task_frequencies={index: 1.0 for index in range(8)},
            n_local_bits=4,
        )

        task_spec = TaskSpecBuilder().build(config)

        self.assertEqual(task_spec.codes, list(range(8)))
        self.assertEqual(task_spec.graph_dependencies, {})

    def test_multitask_sparse_parity_uses_one_hot_selector_and_shared_bit_pool(self):
        config = ScalingLawsConfig(
            task="multitask_sparse_parity",
            base_tasks=4,
            graph_dependencies={},
            task_frequencies={index: 1.0 for index in range(4)},
            parity_task_bits=11,
            parity_subset_size=3,
            seed=0,
        )
        task_spec = TaskSpecBuilder().build(config)
        cache = build_cnand_batch_cache(config, task_spec, "cpu")
        probabilities = task_probability_tensor(task_spec.codes, config.task_frequencies, "cpu")
        batch, labels = build_sampled_cnand_batch(
            config=config,
            task_spec=task_spec,
            probabilities=probabilities,
            batch_cache=cache,
            device="cpu",
            batch_size=8,
        )

        self.assertEqual(task_spec.n_tasks, 4)
        self.assertEqual(config.n_bits, 11)
        self.assertTrue(torch.equal(cache["parity_masks"].sum(dim=1), torch.full((4,), 3)))
        self.assertEqual(tuple(batch["features"].shape), (8, 15))
        self.assertTrue(torch.equal(batch["features"][:, :4].sum(dim=1), torch.ones(8)))
        self.assertEqual(tuple(labels.shape), (8,))

    def test_exponential_paired_paths_are_feasible_and_match_finite_rank_law(self):
        graph = generate_layered_poset(
            rho=2.0,
            beta=2.53,
            base_tasks=8,
            base_freq=1.0,
            max_depth=6,
            m=2,
            seed=0,
            delta=0.0,
            quanta_demand="composition",
            trace_sampling="ideal_path",
            graph_family="exponential_paired",
            target_alpha=0.339,
        )

        self.assertEqual(
            [len(graph["depth_nodes"][depth]) for depth in range(7)],
            [8, 16, 32, 64, 128, 256, 512],
        )
        diagnostics = graph["quanta_demand"]
        sampler = diagnostics["ideal_sampler"]
        terminal_probabilities = np.concatenate(
            [np.asarray(values) for values in sampler["terminal_probabilities_by_depth"]]
        )
        self.assertGreaterEqual(float(terminal_probabilities.min()), 0.0)
        self.assertAlmostEqual(float(terminal_probabilities.sum()), 1.0)
        self.assertEqual(sampler["max_active_nodes"], 14)
        self.assertLess(
            abs(float(diagnostics["actual_rank_alpha"]) - 0.339) / 0.339,
            0.01,
        )

    def test_exponential_depth_paired_paths_exactly_realize_depth_law(self):
        graph = generate_layered_poset(
            rho=2.0,
            beta=2.53,
            base_tasks=8,
            base_freq=1.0,
            max_depth=8,
            m=2,
            seed=0,
            delta=0.0,
            quanta_demand="composition",
            trace_sampling="ideal_path",
            graph_family="exponential_depth_paired",
            target_alpha=None,
        )

        self.assertEqual(
            [len(graph["depth_nodes"][depth]) for depth in range(9)],
            [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        )
        diagnostics = graph["quanta_demand"]
        sampler = diagnostics["ideal_sampler"]
        terminal_probabilities = np.concatenate(
            [
                np.asarray(values)
                for values in sampler["terminal_probabilities_by_depth"]
            ]
        )
        self.assertGreater(float(terminal_probabilities.min()), 0.0)
        self.assertAlmostEqual(float(terminal_probabilities.sum()), 1.0)
        self.assertEqual(sampler["max_active_nodes"], 18)
        self.assertEqual(diagnostics["theory_alpha_source"], "depth_beta")
        self.assertEqual(diagnostics["demand_construction"], "depth_beta")
        self.assertAlmostEqual(diagnostics["actual_induced_beta"], 2.53)
        self.assertAlmostEqual(
            diagnostics["theoretical_alpha"],
            math.log(2.53, 2.0) - 1.0,
        )
        self.assertLess(diagnostics["depth_beta_relative_error"], 1e-12)

        depth_marginals = []
        for depth in range(9):
            values = np.asarray(
                [
                    graph["task_frequencies"][node]
                    for node in graph["depth_nodes"][depth]
                ]
            )
            self.assertAlmostEqual(float(values.min()), float(values.max()))
            depth_marginals.append(float(values.mean()))
        np.testing.assert_allclose(
            np.asarray(depth_marginals[:-1]) / np.asarray(depth_marginals[1:]),
            np.full(8, 2.53),
        )

    def test_ideal_path_batch_samples_independent_short_downward_closed_ideals(self):
        graph = generate_layered_poset(
            rho=2.0,
            beta=2.53,
            base_tasks=4,
            base_freq=1.0,
            max_depth=3,
            m=2,
            seed=0,
            delta=0.0,
            quanta_demand="composition",
            trace_sampling="ideal_path",
            graph_family="exponential_paired",
            target_alpha=0.339,
        )
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            graph_dependencies=graph["graph_dependencies"],
            task_frequencies=graph["task_frequencies"],
            quanta_demand="composition",
            trace_sampling="ideal_path",
            graph_family="exponential_paired",
            target_alpha=0.339,
            attention_masking="only_parent_outs",
            width=8,
            depth=4,
            n_heads=1,
            mlp_ratio=1.0,
            n_local_bits=1,
            base_tasks=4,
            max_depth=3,
            m=2,
            batch_size=256,
            eval_samples_per_task=1,
            n_lut_functions=-1,
            rho=[2.0],
            beta=[2.53],
            delta=[0.0],
        )
        config.quanta_demand_diagnostics = graph["quanta_demand"]
        task_spec = TaskSpecBuilder().build(config)
        config.n_tasks = task_spec.n_tasks
        cache = build_cnand_batch_cache(config, task_spec, "cpu")
        probabilities = task_probability_tensor(
            task_spec.codes, config.task_frequencies, "cpu"
        )
        set_seeds(11)
        batch, _ = build_sampled_cnand_batch(
            config=config,
            task_spec=task_spec,
            probabilities=probabilities,
            batch_cache=cache,
            device="cpu",
            batch_size=256,
        )

        self.assertGreater(int(torch.unique(batch["ideal_indices"]).numel()), 1)
        self.assertLessEqual(int(batch["active_node_mask"].sum(dim=1).max()), 8)
        self.assertLessEqual(int(batch["input_ids"].shape[1]), 8 * 3)
        for row in range(batch["active_node_mask"].shape[0]):
            active = set(
                batch["active_node_mask"][row]
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            for child, parents in graph["graph_dependencies"].items():
                if child in active:
                    self.assertTrue(set(parents).issubset(active))

    def test_wandb_logging_namespaces_sequential_lr_variants(self):
        config = ScalingLawsConfig(width=24, lr=[1e-4])
        config.lr = 1e-4
        diagnostics = {
            "eval_loss_bits": 0.25,
            "weighted_accuracy": 0.75,
            "mean_depth_loss": {0: {"loss_bits": 0.2}},
            "mean_depth_accuracy": {},
            "weighted_depth_loss": {},
            "weighted_depth_accuracy": {},
            "depth_learned_percentages": {0: 50.0},
        }
        with (
            mock.patch(
                "quanta.experiments.scaling_laws.wandb_logging.wandb.run",
                object(),
            ),
            mock.patch(
                "quanta.experiments.scaling_laws.wandb_logging.wandb.log"
            ) as wandb_log,
        ):
            log_scaling_wandb(
                config=config,
                step=100,
                samples_seen=51_200,
                diagnostics=diagnostics,
            )

        payload = wandb_log.call_args.args[0]
        self.assertIn("runs/width24_lr1em04/loss_bits", payload)
        self.assertEqual(payload["runs/width24_lr1em04/step"], 100)
        self.assertNotIn("step", wandb_log.call_args.kwargs)

    def test_run_jobs_build_composition_target_distribution(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            rho=[2.0],
            beta=[2.0],
            max_depth=2,
            base_tasks=2,
            quanta_demand="composition",
        )

        graph = build_scaling_run_jobs(config)[0]["graph"]

        self.assertEqual(graph["quanta_demand"]["mode"], "composition")
        self.assertAlmostEqual(sum(graph["task_frequencies"].values()), 1.0)
        self.assertIn("actual_induced_beta", graph["quanta_demand"])
        self.assertIn("actual_tail_alpha", graph["quanta_demand"])
        self.assertIn("theory_alpha_source", graph["quanta_demand"])

    def test_run_jobs_build_uniform_target_distribution(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            rho=[2.0],
            beta=[2.0],
            max_depth=2,
            base_tasks=2,
            base_freq=7.0,
            quanta_demand="uniform",
        )

        graph = build_scaling_run_jobs(config)[0]["graph"]

        self.assertEqual(graph["quanta_demand"]["mode"], "uniform")
        self.assertTrue(
            all(value == 7.0 for value in graph["task_frequencies"].values())
        )

    def test_demand_record_uses_actual_tail_when_fit_is_not_close(self):
        graph = {
            "quanta_demand": {
                "mode": "composition",
                "desired_beta": 2.0,
                "actual_induced_beta": 3.0,
                "actual_tail_alpha": 0.75,
                "comparison_beta": 3.0,
                "relative_rmse": 0.2,
                "close_fit": False,
            }
        }

        record = _demand_record(graph, rho=2.0, beta=2.0, delta=0.0)

        self.assertEqual(record["desired_theoretical_alpha"], 0.0)
        self.assertAlmostEqual(record["theoretical_alpha"], 0.75)
        self.assertEqual(record["actual_tail_alpha"], 0.75)
        self.assertEqual(record["theory_alpha_source"], "actual_induced_tail")
        self.assertEqual(record["comparison_beta"], 3.0)

    def test_demand_record_uses_depth_beta_for_nonzero_delta(self):
        graph = {
            "quanta_demand": {
                "mode": "composition",
                "actual_induced_beta": 4.0,
                "actual_tail_alpha": 0.75,
                "comparison_beta": 4.0,
                "relative_rmse": 0.2,
                "close_fit": False,
            }
        }

        record = _demand_record(graph, rho=2.0, beta=2.0, delta=1.0)

        self.assertAlmostEqual(record["theoretical_alpha"], 0.5)
        self.assertEqual(record["theory_alpha_source"], "depth_beta")

    def test_theoretical_alpha_matches_log_formula(self):
        self.assertAlmostEqual(theoretical_alpha(2.0, 4.0), 1.0)
        self.assertAlmostEqual(theoretical_alpha(3.0, 9.0), 1.0)
        self.assertAlmostEqual(theoretical_alpha(2.0, 4.0, delta=1.0), 0.5)

    def test_select_best_runs_averages_seeds_before_selecting_learning_rate(self):
        records = [
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 16,
                "n_parameters": 100,
                "seed": 0,
                "lr": 1e-3,
                "final_eval_loss_bits": 0.2,
            },
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 16,
                "n_parameters": 100,
                "seed": 1,
                "lr": 1e-3,
                "final_eval_loss_bits": 0.4,
            },
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 16,
                "n_parameters": 100,
                "seed": 0,
                "lr": 5e-4,
                "final_eval_loss_bits": 0.25,
            },
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 16,
                "n_parameters": 100,
                "seed": 1,
                "lr": 5e-4,
                "final_eval_loss_bits": 0.25,
            },
        ]

        selected = select_best_runs(records)
        summary = aggregate_scaling_results(selected)[0]

        self.assertEqual({record["lr"] for record in selected}, {5e-4})
        point = summary["width_summaries"][0]
        self.assertEqual(point["n_runs"], 2)
        self.assertAlmostEqual(point["final_eval_loss_bits_mean"], 0.25)
        self.assertEqual(point["final_eval_loss_bits_ci95"], 0.0)

    def test_aggregate_scaling_results_fits_positive_alpha_for_decreasing_loss(self):
        records = [
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 10,
                "n_parameters": 100,
                "final_eval_loss_bits": 1.0,
            },
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 20,
                "n_parameters": 200,
                "final_eval_loss_bits": 0.5,
            },
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 40,
                "n_parameters": 400,
                "final_eval_loss_bits": 0.25,
            },
        ]

        summary = aggregate_scaling_results(records)[0]

        self.assertTrue(math.isclose(summary["empirical_alpha"], 1.0, rel_tol=1e-6))
        self.assertEqual(summary["n_runs"], 3)

    def test_aggregate_scaling_results_builds_all_three_parameter_fits(self):
        records = [
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": width,
                "seed": 0,
                "lr": 1e-3,
                "n_parameters": parameters,
                "n_embedding_parameters": parameters // 5,
                "n_non_embedding_parameters": 4 * parameters // 5,
                "final_eval_loss_bits": loss,
                "tail_median_eval_loss_bits": 0.8 * loss,
            }
            for width, parameters, loss in [
                (8, 100, 1.0),
                (16, 200, 0.5),
                (32, 400, 0.25),
            ]
        ]

        summary = aggregate_scaling_results(records)[0]

        self.assertEqual(
            set(summary["fits"]),
            {
                "endpoint_all_parameters",
                "endpoint_non_embedding_parameters",
                "tail_median_all_parameters",
            },
        )
        for fit in summary["fits"].values():
            self.assertAlmostEqual(fit["empirical_alpha"], 1.0)
            self.assertAlmostEqual(fit["r_squared"], 1.0)
        width8 = next(
            point for point in summary["width_summaries"] if point["width"] == 8
        )
        self.assertEqual(width8["n_non_embedding_parameters_mean"], 80.0)

    def test_aggregate_scaling_results_can_use_mean_task_loss(self):
        records = [
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 10,
                "n_parameters": 100,
                "final_eval_loss_bits": 0.1,
                "final_mean_task_loss_bits": 1.0,
            },
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 20,
                "n_parameters": 200,
                "final_eval_loss_bits": 0.2,
                "final_mean_task_loss_bits": 0.5,
            },
        ]

        summary = aggregate_scaling_results(records, weighted_loss=False)[0]

        self.assertAlmostEqual(summary["empirical_alpha"], 1.0)
        self.assertEqual(
            [point["final_eval_loss_bits_mean"] for point in summary["width_summaries"]],
            [1.0, 0.5],
        )

    def test_compact_summary_keeps_fit_metrics_without_graph_payloads(self):
        compact = _compact_summary(
            {
                "experiment": "scaling_laws",
                "config": {"task": "cnand", "steps": 300_000, "graph_dependencies": {1: [0]}},
                "progress": {"stage_completed": 1, "stage_expected": 27},
                "runs": [
                    {
                        "seed": 0,
                        "width": 8,
                        "lr": 1e-3,
                        "n_parameters": 100,
                        "n_embedding_parameters": 20,
                        "n_non_embedding_parameters": 80,
                        "final_eval_loss_bits": 0.2,
                        "tail_median_eval_loss_bits": 0.19,
                        "graph": {"large": list(range(100))},
                    }
                ],
                "pair_summaries": [{"fits": {"endpoint_all_parameters": {"r_squared": 0.9}}}],
            }
        )

        self.assertEqual(compact["progress"]["completed_runs"], 1)
        self.assertNotIn("graph_dependencies", compact["config"])
        self.assertNotIn("graph", compact["runs"][0])
        self.assertEqual(
            compact["pair_summaries"][0]["fits"]["endpoint_all_parameters"]["r_squared"],
            0.9,
        )

    def test_aggregate_scaling_results_keeps_transformer_scales_with_same_width(self):
        records = [
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 64,
                "model_scale": "width64",
                "effective_width": 64,
                "n_parameters": 100,
                "final_eval_loss_bits": 1.0,
            },
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "width": 64,
                "model_scale": "width128",
                "effective_width": 128,
                "n_parameters": 200,
                "final_eval_loss_bits": 0.5,
            },
        ]

        summary = aggregate_scaling_results(records)[0]

        self.assertEqual(len(summary["width_summaries"]), 2)
        self.assertEqual(
            sorted(point["n_parameters_mean"] for point in summary["width_summaries"]),
            [100.0, 200.0],
        )

    def test_aggregate_scaling_results_uses_delta_for_theoretical_alpha(self):
        records = [
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 4.0,
                "delta": 1.0,
                "width": 10,
                "n_parameters": 100,
                "final_eval_loss_bits": 1.0,
            }
        ]

        summary = aggregate_scaling_results(records)[0]

        self.assertEqual(summary["delta"], 1.0)
        self.assertAlmostEqual(summary["theoretical_alpha"], 0.5)

    def test_aggregate_scaling_results_averages_seed_specific_tail_theory(self):
        records = [
            {
                "pair_index": 0,
                "rho": 2.0,
                "beta": 2.0,
                "quanta_demand": "composition",
                "actual_induced_beta": actual_beta,
                "actual_tail_alpha": actual_tail_alpha,
                "comparison_beta": actual_beta,
                "quanta_demand_relative_rmse": fit_error,
                "quanta_demand_close_fit": False,
                "theoretical_alpha": actual_tail_alpha,
                "theory_alpha_source": "actual_induced_tail",
                "width": 32,
                "n_parameters": 100,
                "final_eval_loss_bits": 1.0,
            }
            for actual_beta, actual_tail_alpha, fit_error in [
                (3.0, 0.25, 0.2),
                (5.0, 0.75, 0.4),
            ]
        ]

        summary = aggregate_scaling_results(records)[0]

        self.assertEqual(summary["actual_induced_beta"], 4.0)
        self.assertEqual(summary["actual_tail_alpha"], 0.5)
        self.assertEqual(summary["comparison_beta"], 4.0)
        self.assertAlmostEqual(summary["quanta_demand_relative_rmse"], 0.3)
        self.assertFalse(summary["quanta_demand_close_fit"])
        self.assertAlmostEqual(summary["theoretical_alpha"], 0.5)
        self.assertEqual(summary["theory_alpha_source"], "actual_induced_tail")

    def test_theoretical_alpha_prefix_reports_actual_tail_source(self):
        self.assertEqual(
            _theoretical_alpha_prefix("actual_induced_tail"),
            "Theoretical alpha (actual induced tail)",
        )
        self.assertEqual(_theoretical_alpha_prefix("depth_beta"), "Theoretical alpha")
        self.assertEqual(_r_squared_label(0.91234), r"$R^2=0.9123$")

    def test_discover_all_runs_uses_saved_pair_values_for_pair_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = os.path.join(temp_dir, "experiment")
            for width, loss in [(32, 0.25), (128, 0.125)]:
                run_dir = os.path.join(base_dir, "runs", "pair0", f"seed1-width{width}-depth1-heads2")
                os.makedirs(run_dir)
                config = {
                    "rho": [2.0],
                    "beta": [2.83],
                    "delta": [0.0],
                    "rho_beta_delta": [[2.0, 2.83, 0.0]],
                    "width": width,
                    "seed": 1,
                    "lr": 5e-5,
                    "depth": 1,
                    "architecture": "transformer",
                    "n_heads": 2,
                    "n_tasks": 10,
                    "n_bits": 4,
                    "steps": 100,
                }
                result = {
                    "steps_run": 100,
                    "n_parameters": width * 1000,
                    "final_eval_loss_nats": loss * math.log(2),
                    "final_eval_loss_bits": loss,
                }
                with open(os.path.join(run_dir, "config.json"), "w") as handle:
                    json.dump(config, handle)
                with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                    pickle.dump(result, handle)

            records = discover_all_runs(base_dir)
            summary = aggregate_scaling_results(records)[0]

        self.assertEqual(len(records), 2)
        self.assertEqual({record["beta"] for record in records}, {2.83})
        self.assertEqual(len(summary["width_summaries"]), 2)
        self.assertIsNotNone(summary["empirical_alpha"])

    def test_discover_all_runs_supports_missing_and_legacy_converged_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = os.path.join(temp_dir, "experiment")
            config = {
                "rho": [2.0],
                "beta": [3.0],
                "delta": [0.0],
                "rho_beta_delta": [[2.0, 3.0, 0.0]],
                "width": 32,
                "seed": 1,
                "lr": 1e-3,
                "depth": 1,
                "architecture": "transformer",
                "n_heads": 1,
                "n_tasks": 3,
                "n_bits": 4,
                "steps": 10,
            }
            results_by_run = {
                "modern": {
                    "steps_run": 10,
                    "n_parameters": 100,
                    "final_eval_loss_nats": 0.2,
                    "final_eval_loss_bits": 0.3,
                },
                "legacy": {
                    "steps_run": 5,
                    "n_parameters": 100,
                    "final_eval_loss_nats": 0.2,
                    "final_eval_loss_bits": 0.3,
                    "converged": True,
                },
                "incomplete": {
                    "steps_run": 5,
                    "n_parameters": 100,
                    "final_eval_loss_nats": 0.2,
                    "final_eval_loss_bits": 0.3,
                },
            }
            for name, results in results_by_run.items():
                run_dir = os.path.join(base_dir, "runs", "pair0", name)
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, "config.json"), "w") as handle:
                    json.dump(config, handle)
                with open(os.path.join(run_dir, "results.pkl"), "wb") as handle:
                    pickle.dump(results, handle)

            records = discover_all_runs(base_dir)

        self.assertEqual(len(records), 2)
        self.assertEqual({record["steps_run"] for record in records}, {5, 10})
        self.assertTrue(all("converged" not in record for record in records))

    def test_run_jobs_include_transformer_parameter_overrides(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            rho=[2.0],
            beta=[4.0],
            depth=[2, 3],
            n_heads=[2, 4],
        )

        jobs = build_scaling_run_jobs(config)

        self.assertEqual(
            [(job["overrides"]["depth"], job["overrides"]["n_heads"]) for job in jobs],
            [(2, 2), (3, 4)],
        )

    def test_run_jobs_are_seed_major_with_paired_width_learning_rates(self):
        config = ScalingLawsConfig(
            rho=[2.0],
            beta=[4.0],
            delta=[0.0],
            width=[8, 16],
            lr=[1e-3, 5e-4],
            seed=[0, 1, 2],
        )

        jobs = build_scaling_run_jobs(config)

        self.assertEqual(
            [(job["seed"], job["width"], job["lr"]) for job in jobs],
            [
                (0, 8, 1e-3),
                (0, 16, 5e-4),
                (1, 8, 1e-3),
                (1, 16, 5e-4),
                (2, 8, 1e-3),
                (2, 16, 5e-4),
            ],
        )

    def test_run_jobs_include_delta(self):
        config = ScalingLawsConfig(
            rho=[2.0],
            beta=[4.0],
            delta=[0.25],
        )

        jobs = build_scaling_run_jobs(config)

        self.assertEqual(jobs[0]["delta"], 0.25)

    def test_run_jobs_default_delta_does_not_truncate_pairs(self):
        config = ScalingLawsConfig(
            rho=[2.0, 3.0],
            beta=[4.0, 9.0],
        )

        jobs = build_scaling_run_jobs(config)

        self.assertEqual(
            [(job["rho"], job["beta"], job["delta"]) for job in jobs],
            [(2.0, 4.0, 0.0), (3.0, 9.0, 0.0)],
        )

    def test_run_scaling_job_uses_current_pair_delta_for_cnand_bits(self):
        from quanta.experiments.scaling_laws import run as run_module

        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            rho=[2.0, 3.0],
            beta=[4.0, 9.0],
            delta=[0.0, 1.0],
            max_depth=1,
            base_tasks=2,
            n_local_bits=2,
            depth=1,
            n_heads=4,
            lr=[1e-3, 1e-3],
            steps=1,
            batch_size=2,
        )
        job = build_scaling_run_jobs(config)[1]
        accelerator = mock.Mock()
        accelerator.is_main_process = True
        accelerator.num_processes = 1
        accelerator.device = "cpu"
        accelerator.wait_for_everyone.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            run_module,
            "train_until_budget_or_convergence",
            return_value={
                "n_parameters": 10,
                "final_eval_loss_nats": 0.1,
                "final_eval_loss_bits": 0.144,
                "steps_run": 1,
            },
        ):
            record = run_module.run_scaling_job(config, tmpdir, job, accelerator=accelerator)

        self.assertEqual(record["rho"], 3.0)
        self.assertEqual(record["delta"], 1.0)
        self.assertEqual(record["n_bits"], 40)
        self.assertNotIn("converged", record)

    def test_local_batch_size_partitions_global_batch(self):
        self.assertEqual(
            [local_batch_size_for_process(10, 4, index) for index in range(4)],
            [3, 3, 2, 2],
        )
        self.assertEqual(
            [local_batch_size_for_process(2, 4, index) for index in range(4)],
            [1, 1, 0, 0],
        )

    def test_cnand_delta_controls_private_bits_by_depth(self):
        config = ScalingLawsConfig(
            graph_dependencies={2: [0, 1]},
            n_atomic_task_bits=2,
            n_noise_bits=0,
            rho=[2.0],
            delta=[1.0],
        )

        task_spec = TaskSpecBuilder().build(config)

        self.assertEqual(task_spec.Ss_atomic, [[0, 1], [2, 3], [4, 5, 6, 7]])
        self.assertEqual(config.n_bits, 8)

    def test_cnand_delta_zero_matches_constant_private_bits(self):
        config = ScalingLawsConfig(
            graph_dependencies={2: [0, 1]},
            n_atomic_task_bits=2,
            n_noise_bits=0,
            rho=[2.0],
            delta=[0.0],
        )

        task_spec = TaskSpecBuilder().build(config)

        self.assertEqual(task_spec.Ss_atomic, [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(config.n_bits, 6)

    def test_cnand_uses_all_local_bits_as_signal(self):
        config = ScalingLawsConfig(
            task="cnand",
            architecture="transformer",
            graph_dependencies={0: []},
            task_frequencies={0: 1.0},
            batch_size=1,
            max_depth=0,
            n_local_bits=4,
        )
        task_spec = TaskSpecBuilder().build(config)
        config.n_tasks = task_spec.n_tasks
        batch_cache = build_cnand_batch_cache(config, task_spec, "cpu")
        local_bits = torch.tensor([[[1, 1, 0, 0]]], dtype=torch.long)
        changed_later_bits = torch.tensor([[[1, 1, 1, 1]]], dtype=torch.long)
        changed_first_bits = torch.tensor([[[1, 0, 0, 0]]], dtype=torch.long)

        base_values = evaluate_cnand_out_values(batch_cache=batch_cache, local_bits=local_bits)
        later_bit_values = evaluate_cnand_out_values(batch_cache=batch_cache, local_bits=changed_later_bits)
        first_bit_values = evaluate_cnand_out_values(batch_cache=batch_cache, local_bits=changed_first_bits)

        self.assertEqual(base_values.tolist(), [[1]])
        self.assertEqual(later_bit_values.tolist(), [[0]])
        self.assertEqual(first_bit_values.tolist(), [[1]])
        self.assertNotIn("local_signal_masks", batch_cache)

if __name__ == "__main__":
    unittest.main()
