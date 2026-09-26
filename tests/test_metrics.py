import os
import unittest

from quanta.metrics import (
    DEPENDENCY_CENSORED,
    DEPENDENCY_SATISFIED,
    DEPENDENCY_VIOLATED,
    compute_discreteness_transition_error,
    compute_poset_dynamics_metrics,
    compute_poset_dependencies_error,
    dependency_pair_violated,
    dependency_timing_status,
    learned_threshold_bits,
    random_prediction_loss_bits,
)


class ViolationMetricTests(unittest.TestCase):
    def test_env_learned_threshold_defaults_to_005_bits(self):
        previous = os.environ.pop("LEARNED_QUANTA_THRESHOLD", None)
        try:
            self.assertEqual(learned_threshold_bits(), 0.05)
        finally:
            if previous is not None:
                os.environ["LEARNED_QUANTA_THRESHOLD"] = previous

    def test_random_prediction_loss_matches_binary_uniform_cross_entropy(self):
        self.assertEqual(random_prediction_loss_bits(), 1.0)

    def test_dependency_must_cross_threshold_strictly_before_dependent_quantum(self):
        no_violation = compute_poset_dependencies_error(
            subtask_losses=[
                [1.0, 0.1, 0.1],
                [1.0, 0.1, 0.1],
                [1.0, 1.0, 0.1],
            ],
            codes=[0, 1, 4],
            task_type="cnand",
            graph_dependencies={4: [0, 1]},
            threshold=1.0,
            window=1,
        )
        tie_violation = compute_poset_dependencies_error(
            subtask_losses=[
                [1.0, 0.1],
                [0.1, 0.1],
                [1.0, 0.1],
            ],
            codes=[0, 1, 4],
            task_type="cnand",
            graph_dependencies={4: [0, 1]},
            threshold=1.0,
            window=1,
        )

        self.assertEqual(no_violation, (0.0, 0, 2))
        self.assertEqual(tie_violation, (0.5, 1, 2))

    def test_dependency_timing_status_requires_all_prerequisites_before_dependent(self):
        self.assertEqual(dependency_timing_status(60, [40, 50]), DEPENDENCY_SATISFIED)
        self.assertEqual(dependency_timing_status(60, [40, None]), DEPENDENCY_VIOLATED)
        self.assertEqual(dependency_timing_status(60, [40, 60]), DEPENDENCY_VIOLATED)
        self.assertEqual(dependency_timing_status(None, [40, None]), DEPENDENCY_CENSORED)
        self.assertTrue(dependency_pair_violated(60, None))

    def test_poset_dependencies_error_counts_only_learned_dependents(self):
        result = compute_poset_dependencies_error(
            subtask_losses=[
                [1.0, 0.1],
                [1.0, 0.1],
                [1.0, 0.1],
                [1.0, 1.0],
            ],
            codes=[0, 1, 2, 3],
            task_type="cnand",
            graph_dependencies={2: [0], 3: [1]},
            threshold=1.0,
            window=1,
        )

        self.assertEqual(result, (1.0, 1, 1))
        self.assertEqual(compute_poset_dependencies_error.last_candidate_dependencies, 1)
        self.assertEqual(compute_poset_dependencies_error.last_total_dependencies, 2)
        self.assertEqual(compute_poset_dependencies_error.last_candidate_dependency_fraction, 0.5)

    def test_poset_dependencies_error_counts_only_direct_parents(self):
        # CNAND task
        # 0 -> 1 -> 2
        # T: Node 0 (learned at 0), Node 1 (learned at 10), Node 2 (learned at 5)
        # PDE candidates: (2,1), (1,0) -> 2 deps. (2,1) is violated. PDE = 1/2
        subtask_losses = [
            [1.0, 0.1, 0.1],  # Node 0: learned at step 1 (window=1, threshold=1.0)
            [1.0, 1.0, 0.1],  # Node 1: learned at step 2
            [1.0, 0.1, 0.1],  # Node 2: learned at step 1
        ]
        pde_res = compute_poset_dependencies_error(
            subtask_losses=subtask_losses,
            codes=[0, 1, 2],
            task_type="cnand",
            graph_dependencies={1: [0], 2: [1]},
            threshold=1.0,
            window=1,
        )
        self.assertEqual(pde_res, (1/2, 1, 2))
        self.assertEqual(compute_poset_dependencies_error.last_candidate_dependencies, 2)
        self.assertEqual(compute_poset_dependencies_error.last_total_dependencies, 2)
        self.assertEqual(compute_poset_dependencies_error.last_candidate_dependency_fraction, 1.0)

    def test_poset_dependencies_error_non_cnand(self):
        # Non-CNAND task (Boolean lattice subset logic)
        # Codes: (1,), (1, 2), (1, 2, 3)
        # T: Node (1,) (learned at 1), Node (1,2) (learned at 2), Node (1,2,3) (learned at 1)
        # PDE candidates: ((1,2,3), (1,2)), ((1,2), (1,)) -> 2 deps. ((1,2,3), (1,2)) is violated. PDE = 1/2
        subtask_losses = [
            [1.0, 0.1, 0.1],
            [1.0, 1.0, 0.1],
            [1.0, 0.1, 0.1],
        ]
        pde_res = compute_poset_dependencies_error(
            subtask_losses=subtask_losses,
            codes=[(1,), (1, 2), (1, 2, 3)],
            task_type="other",
            threshold=1.0,
            window=1,
        )
        self.assertEqual(pde_res, (1/2, 1, 2))

    def test_compute_discreteness_transition_error_computes_both_metrics(self):
        import numpy as np

        subtask_losses = [
            [1.5 / np.log2(np.e), 1.5 / np.log2(np.e), 0.2 / np.log2(np.e)],
            [1.5 / np.log2(np.e), 0.8 / np.log2(np.e), 0.6 / np.log2(np.e), 0.6 / np.log2(np.e), 0.6 / np.log2(np.e), 0.4 / np.log2(np.e)],
        ]
        metrics = compute_discreteness_transition_error(
            subtask_losses=subtask_losses,
            codes=[0, 1],
            unlearned_quanta_threshold=1.0,
            learned_quanta_threshold=0.5,
        )

        self.assertEqual(metrics["total_quanta"], 2)
        self.assertEqual(metrics["candidate_quanta"], 2)
        self.assertEqual(metrics["total_candidates"], 2)
        self.assertEqual(metrics["candidate_quanta_fraction"], 1.0)
        self.assertAlmostEqual(metrics["dte_steps"], ((2 - 1) / 2 + (5 - 0) / 5) / 2)
        self.assertIn("dte_area", metrics)
        self.assertEqual(metrics["avg_transition_steps"], 3.0)
        self.assertEqual(metrics["min_transition_steps"], 1)
        self.assertEqual(metrics["max_transition_steps"], 5)

    def test_compute_discreteness_transition_error_missing_step_a(self):
        import numpy as np
        # Subtask 0: starts at 0.8 (< unlearned_threshold 1.0), then learned at 0.2. Step a does not exist.
        # Subtask 1: starts at 1.5 (> 1.0), then learned at 0.2. Step a exists.
        subtask_losses = [
            [0.8 / np.log2(np.e), 0.2 / np.log2(np.e)],
            [1.5 / np.log2(np.e), 0.2 / np.log2(np.e)],
        ]
        metrics = compute_discreteness_transition_error(
            subtask_losses=subtask_losses,
            codes=[0, 1],
            unlearned_quanta_threshold=1.0,
            learned_quanta_threshold=0.5,
        )
        # Subtask 0 is ignored because step a does not exist.
        self.assertEqual(metrics["total_quanta"], 1)
        self.assertEqual(metrics["dte_steps"], 1.0)
        self.assertEqual(metrics["avg_transition_steps"], 1.0)
        self.assertEqual(metrics["min_transition_steps"], 1)
        self.assertEqual(metrics["max_transition_steps"], 1)

    def test_compute_discreteness_distribution_weighted_error(self):
        import numpy as np
        subtask_losses = [
            [0.5 / np.log2(np.e), 1.5 / np.log2(np.e), 0.75 / np.log2(np.e), 0.25 / np.log2(np.e), 1.0 / np.log2(np.e)],
        ]
        metrics = compute_discreteness_transition_error(
            subtask_losses=subtask_losses,
            codes=[0],
            unlearned_quanta_threshold=1.0,
            learned_quanta_threshold=0.5,
        )

        self.assertAlmostEqual(metrics["dte_area"], 0.4375)
        self.assertAlmostEqual(metrics["dte_steps"], 2 / 3)
        self.assertEqual(metrics["total_quanta"], 1)
        self.assertEqual(metrics["avg_transition_steps"], 2.0)
        self.assertEqual(metrics["min_transition_steps"], 2)
        self.assertEqual(metrics["max_transition_steps"], 2)

    def test_compute_discreteness_distribution_weighted_error_is_bounded(self):
        import numpy as np
        subtask_losses = [
            [100.0 / np.log2(np.e), -100.0 / np.log2(np.e)],
        ]
        metrics = compute_discreteness_transition_error(
            subtask_losses=subtask_losses,
            codes=[0],
            unlearned_quanta_threshold=1.0,
            learned_quanta_threshold=0.5,
        )

        self.assertGreaterEqual(metrics["dte_area"], 0.0)
        self.assertLessEqual(metrics["dte_area"], 1.0)
        self.assertEqual(metrics["total_quanta"], 1)

    def test_poset_dynamics_metrics_reports_unit_and_exclusions(self):
        metrics = compute_poset_dynamics_metrics(
            loss_curves={
                "leaf": [2.0, 0.05, 0.01],
                "parent": [2.0, 2.0, 2.0],
            },
            steps=[0.0, 10.0, 20.0],
            graph={"parent": ["leaf"]},
            learned_threshold=0.1,
            unlearned_threshold=1.0,
            unit="effective samples",
        )

        self.assertEqual(metrics["unit"], "effective samples")
        self.assertEqual(metrics["dte_total_count"], 2)
        self.assertEqual(metrics["dte_excluded_count"], 1)
        self.assertEqual(metrics["dte_excluded_percent"], 50.0)
        self.assertEqual(metrics["pde"]["checked_pairs"], 0)
        self.assertEqual(metrics["pde"]["censored_pairs"], 1)
        self.assertEqual(metrics["pde_censored_percent"], 100.0)

    def test_fork_rng_no_side_effects(self):
        import torch
        import numpy as np
        import random
        from quanta.utils import ForkRNG, set_seeds

        set_seeds(42)

        # Generate a sequence of values outside the fork block
        val_py_1 = random.randint(0, 100)
        val_np_1 = np.random.randint(0, 100)
        val_torch_1 = torch.randint(0, 100, (1,)).item()

        # Enter ForkRNG block, make random calls inside it
        with ForkRNG():
            # These should not affect the sequence outside the block
            _ = random.randint(0, 100)
            _ = np.random.randint(0, 100)
            _ = torch.randint(0, 100, (1,))

        # The subsequent calls outside the block should continue the original sequence
        val_py_2 = random.randint(0, 100)
        val_np_2 = np.random.randint(0, 100)
        val_torch_2 = torch.randint(0, 100, (1,)).item()

        # Let's compare with generating without entering the fork block (fully deterministic sequence)
        set_seeds(42)
        val_py_ref_1 = random.randint(0, 100)
        val_np_ref_1 = np.random.randint(0, 100)
        val_torch_ref_1 = torch.randint(0, 100, (1,)).item()

        # Directly generate the next values in the sequence
        val_py_ref_2 = random.randint(0, 100)
        val_np_ref_2 = np.random.randint(0, 100)
        val_torch_ref_2 = torch.randint(0, 100, (1,)).item()

        # Verify that the sequence was preserved exactly
        self.assertEqual(val_py_1, val_py_ref_1)
        self.assertEqual(val_np_1, val_np_ref_1)
        self.assertEqual(val_torch_1, val_torch_ref_1)

        self.assertEqual(val_py_2, val_py_ref_2)
        self.assertEqual(val_np_2, val_np_ref_2)
        self.assertEqual(val_torch_2, val_torch_ref_2)


if __name__ == "__main__":
    unittest.main()
