import unittest

import numpy as np

from quanta.experiments.demand import (
    closure_matrix,
    induced_quanta_demand,
    resolve_task_demand,
    select_theoretical_alpha,
)
from quanta.experiments.demand_solver import compute_actual_tail_alpha
from quanta.experiments.ideal_sampling import threshold_ideal_mixture


class QuantaDemandTests(unittest.TestCase):
    def test_threshold_ideal_mixture_exactly_reconstructs_branching_marginals(self):
        graph = {2: [0], 3: [0, 1], 4: [2, 3]}
        marginals = {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5, 4: 0.25}

        mixture = threshold_ideal_mixture(
            marginals=marginals,
            graph_dependencies=graph,
        )

        masks = np.asarray(mixture["ideal_masks"], dtype=float)
        probabilities = np.asarray(mixture["probabilities"], dtype=float)
        reconstructed = probabilities @ masks
        np.testing.assert_allclose(
            reconstructed,
            [marginals[node] for node in mixture["nodes"]],
            atol=1e-10,
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        for mask in masks.astype(bool):
            active = {
                node for node, is_active in zip(mixture["nodes"], mask) if is_active
            }
            for child, parents in graph.items():
                if child in active:
                    self.assertTrue(set(parents).issubset(active))

    def test_threshold_ideal_mixture_rejects_non_monotone_marginals(self):
        with self.assertRaisesRegex(ValueError, "order-reversing"):
            threshold_ideal_mixture(
                marginals={0: 0.25, 1: 0.5},
                graph_dependencies={1: [0]},
            )

    def test_actual_tail_alpha_matches_direct_log_log_regression(self):
        demand = np.asarray([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
        weights = np.sort(demand / demand.sum())[::-1]
        capacities = np.arange(1, len(weights), dtype=float)
        tails = 1.0 - np.cumsum(weights)[:-1]
        expected = -np.polyfit(np.log(capacities), np.log(tails), 1)[0]

        self.assertAlmostEqual(compute_actual_tail_alpha(demand), expected)

    def test_actual_tail_alpha_rejects_invalid_demand(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compute_actual_tail_alpha(np.asarray([1.0, -0.1]))
        with self.assertRaisesRegex(ValueError, "positive total"):
            compute_actual_tail_alpha(np.zeros(3))

    def test_theory_selection_uses_tail_only_for_bad_delta_zero_fit(self):
        bad_fit = {
            "mode": "composition",
            "close_fit": False,
            "comparison_beta": 3.0,
            "actual_tail_alpha": 0.75,
        }
        close_fit = {**bad_fit, "close_fit": True}

        self.assertEqual(
            select_theoretical_alpha(
                demand=bad_fit,
                rho=2.0,
                beta=2.0,
                delta=0.0,
            ),
            (0.75, "actual_induced_tail"),
        )
        close_alpha, close_source = select_theoretical_alpha(
            demand=close_fit,
            rho=2.0,
            beta=2.0,
            delta=0.0,
        )
        self.assertAlmostEqual(close_alpha, np.log(3.0) / np.log(2.0) - 1.0)
        self.assertEqual(close_source, "depth_beta")

    def test_closure_matrix_marks_targets_and_all_ancestors(self):
        matrix = closure_matrix([0, 1, 2], {1: [0], 2: [1]})

        np.testing.assert_array_equal(
            matrix,
            np.asarray(
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_shortcut_mode_preserves_existing_depth_weights(self):
        result = resolve_task_demand(
            graph_dependencies={1: [0], 2: [1]},
            node_depths={0: 0, 1: 1, 2: 2},
            beta=4.0,
            base_freq=2.0,
            mode="shortcut",
        )

        self.assertEqual(
            result["target_frequencies"],
            {0: 2.0, 1: 0.5, 2: 0.125},
        )
        self.assertEqual(result["comparison_beta"], 4.0)

    def test_uniform_mode_assigns_identical_target_weights(self):
        result = resolve_task_demand(
            graph_dependencies={1: [0], 2: [1]},
            node_depths={0: 0, 1: 1, 2: 2},
            beta=4.0,
            base_freq=2.0,
            mode="uniform",
        )

        self.assertEqual(
            result["target_frequencies"],
            {0: 2.0, 1: 2.0, 2: 2.0},
        )
        self.assertEqual(result["mode"], "uniform")

    def test_composition_mode_exactly_inverts_feasible_chain(self):
        result = resolve_task_demand(
            graph_dependencies={1: [0], 2: [1]},
            node_depths={0: 0, 1: 1, 2: 2},
            beta=2.0,
            base_freq=1.0,
            mode="composition",
            fit_tolerance=1e-8,
        )

        self.assertTrue(result["close_fit"])
        self.assertLess(result["relative_rmse"], 1e-8)
        self.assertAlmostEqual(sum(result["target_frequencies"].values()), 1.0)
        self.assertTrue(all(value >= 0 for value in result["target_frequencies"].values()))
        self.assertIn("actual_tail_alpha", result)
        for node in result["desired_quanta_demand"]:
            self.assertAlmostEqual(
                result["desired_quanta_demand"][node],
                result["induced_quanta_demand"][node],
                places=8,
            )

    def test_ideal_threshold_mode_exactly_realizes_branching_depth_demand(self):
        result = resolve_task_demand(
            graph_dependencies={1: [0], 2: [0], 3: [0]},
            node_depths={0: 0, 1: 1, 2: 1, 3: 1},
            beta=2.0,
            base_freq=1.0,
            mode="composition",
            trace_sampling="ideal_threshold",
            fit_tolerance=1e-12,
        )

        self.assertTrue(result["close_fit"])
        self.assertEqual(result["relative_rmse"], 0.0)
        self.assertAlmostEqual(result["actual_induced_beta"], 2.0)
        self.assertEqual(
            result["induced_quanta_demand_raw"],
            {0: 1.0, 1: 0.5, 2: 0.5, 3: 0.5},
        )

    def test_infeasible_branching_reports_actual_distribution_and_beta(self):
        graph = {1: [0], 2: [0], 3: [0]}
        result = resolve_task_demand(
            graph_dependencies=graph,
            node_depths={0: 0, 1: 1, 2: 1, 3: 1},
            beta=2.0,
            base_freq=1.0,
            mode="composition",
            fit_tolerance=1e-6,
        )

        self.assertFalse(result["close_fit"])
        self.assertGreater(result["relative_rmse"], 1e-6)
        self.assertNotEqual(result["comparison_beta"], 2.0)
        self.assertGreaterEqual(result["actual_tail_alpha"], 0.0)
        self.assertAlmostEqual(
            sum(result["target_frequencies"].values()),
            1.0,
        )
        recomputed = induced_quanta_demand(result["target_frequencies"], graph)
        for node, value in recomputed.items():
            self.assertAlmostEqual(
                value,
                result["induced_quanta_demand_raw"][node],
                places=8,
            )

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "acyclic"):
            resolve_task_demand(
                graph_dependencies={0: [1], 1: [0]},
                node_depths={0: 0, 1: 1},
                beta=2.0,
                base_freq=1.0,
                mode="composition",
            )


if __name__ == "__main__":
    unittest.main()
