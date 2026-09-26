import os
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

from quanta.figures.discovery_trajectories import (
    _curves_y_limit,
    _ema_smooth,
    _dependencies_ready_loss_curves,
    _linear_time_frame_ends,
    plot_discovery_trajectories,
)
from quanta.figures.scaling_laws import (
    plot_scaling_laws,
)
from quanta.metrics.core import compute_learning_times
from quanta.effective_samples import effective_sample_data_from_results


class PlotCliTests(unittest.TestCase):
    def test_animation_frames_follow_training_time_not_display_axis(self):
        frame_ends = _linear_time_frame_ends(
            [0, 1, 2, 100],
            max_points=4,
            frame_count=5,
        )

        np.testing.assert_array_equal(frame_ends, [1, 3, 3, 3, 4])

    def test_all_nan_curve_does_not_emit_runtime_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = compute_learning_times(
                {0: np.array([np.nan, np.nan])},
                np.array([0.0, 1.0]),
                learned_threshold=0.1,
            )

        self.assertFalse(result[0]["learned"])
        self.assertTrue(np.isnan(result[0]["min_loss"]))

    def test_curve_y_limit_tracks_data_maximum_with_small_padding(self):
        lower, upper = _curves_y_limit(
            {0: np.array([0.2, 0.8]), 1: np.array([0.4, 1.0])}
        )

        self.assertGreaterEqual(lower, 0.0)
        self.assertGreater(upper, 1.0)
        self.assertLess(upper, 1.05)

    def test_quantum_decomposition_uses_saved_quantum_curves_in_one_panel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "quanta.png")
            figure = plot_discovery_trajectories(
                output_image_path=output,
                codes=[4, 5],
                subtask_losses=[[0.7, 0.4], [0.8, 0.5]],
                overall_loss_bits=[0.9, 0.1],
                quantum_subtask_losses=[[0.6, 0.2], [0.7, 0.3]],
                samples=[0, 100],
                graph_dependencies={4: [0, 1], 5: [1, 2]},
                x_start=0,
                x_lim=100,
                x_axis="steps",
                x_scale="linear",
                loss_decomposition="quanta",
                plot_total_loss=True,
            )

            self.assertTrue(os.path.exists(output))
            self.assertEqual(len(figure.axes), 1)

    def test_save_pdf_argument_writes_pdf_without_environment_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "loss.png")
            with mock.patch.dict(
                os.environ,
                {"SAVE_IMAGES_AS_PDF": "false"},
            ):
                plot_discovery_trajectories(
                    output_image_path=output,
                    codes=[0],
                    subtask_losses=[[0.7, 0.4]],
                    samples=[0, 100],
                    x_start=0,
                    x_lim=100,
                    x_scale="linear",
                    save_pdf=True,
                )

            self.assertTrue(os.path.exists(os.path.join(temp_dir, "loss.pdf")))

    def test_red_curve_is_saved_weighted_eval_loss_not_curve_mean(self):
        figure = plot_discovery_trajectories(
            output_image_path=None,
            codes=[0, 1],
            subtask_losses=[[1.0, 1.0], [3.0, 3.0]],
            quantum_subtask_losses=[[5.0, 5.0], [7.0, 7.0]],
            overall_loss_bits=[0.8, 0.2],
            samples=[0, 100],
            x_start=0,
            x_lim=100,
            x_axis="steps",
            x_scale="linear",
            loss_decomposition="quanta",
            plot_total_loss=True,
        )

        red_line = next(
            line
            for line in figure.axes[0].lines
            if line.get_label() == "Overall weighted loss"
        )
        np.testing.assert_allclose(red_line.get_ydata(), [0.8, 0.2])

    def test_unweighted_red_curve_uses_mean_task_loss(self):
        figure = plot_discovery_trajectories(
            output_image_path=None,
            codes=[0, 1],
            subtask_losses=[[1.0, 1.0], [3.0, 3.0]],
            mean_quantum_subtask_losses=[[4.0, 3.0], [6.0, 5.0]],
            overall_loss_bits=[2.0, 1.5],
            samples=[0, 100],
            x_start=0,
            x_lim=100,
            x_axis="steps",
            x_scale="linear",
            loss_decomposition="quanta",
            weighted_loss=False,
            plot_total_loss=True,
        )

        red_line = next(
            line
            for line in figure.axes[0].lines
            if line.get_label() == "Overall mean task loss"
        )
        np.testing.assert_allclose(red_line.get_ydata(), [2.0, 1.5])
        quantum_lines = [
            line for line in figure.axes[0].lines
            if line.get_label() != "Overall mean task loss"
        ]
        np.testing.assert_allclose(quantum_lines[0].get_ydata(), np.array([4.0, 3.0]) * np.log2(np.e))

    def test_discovery_trajectory_has_one_panel(self):
        figure = plot_discovery_trajectories(
            output_image_path=None,
            codes=[0, 1],
            subtask_losses=[[0.7, 0.4], [0.8, 0.5]],
            samples=[0, 100],
            x_start=0,
            x_lim=100,
            x_axis="steps",
            x_scale="linear",
            loss_decomposition="tasks_dependencies_ready",
        )

        self.assertEqual(len(figure.axes), 1)

    def test_samples_x_axis_uses_shared_training_sample_axis(self):
        figure = plot_discovery_trajectories(
            output_image_path=None,
            codes=[0, 1],
            subtask_losses=[[0.7, 0.4], [0.8, 0.5]],
            samples=[0, 100],
            x_start=0,
            x_lim=100,
            x_axis="samples",
            x_scale="linear",
            loss_decomposition="tasks_dependencies_ready",
        )

        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, 100])
        self.assertEqual(figure.axes[0].get_xlabel(), "Training samples")

    def test_effective_samples_use_a_distinct_axis_for_each_task(self):
        figure = plot_discovery_trajectories(
            output_image_path=None,
            codes=[0, 1],
            subtask_losses=[[0.7, 0.4], [0.8, 0.5]],
            samples=[0, 10],
            effective_samples={0: [0, 20], 1: [0, 80]},
            x_start=0,
            x_lim=80,
            x_axis="effective_samples",
            x_scale="linear",
        )

        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, 20])
        np.testing.assert_allclose(figure.axes[0].lines[1].get_xdata(), [0, 80])
        self.assertEqual(
            figure.axes[0].get_xlabel(),
            "Effective cumulative target samples per task",
        )

    def test_effective_sample_data_uses_the_decomposition_script_formula(self):
        data = effective_sample_data_from_results(
            {
                "codes": [0, 1],
                "eval_steps": [0, 10, 20],
                "task_probabilities": {0: 0.75, 1: 0.25},
                "node_depths": {0: 0, 1: 1},
                "training_samples": [8, 8],
            }
        )

        np.testing.assert_allclose(data.axes[0], [0, 60, 120])
        np.testing.assert_allclose(data.axes[1], [0, 20, 40])

    def test_effective_sample_data_prefers_exact_saved_counts(self):
        data = effective_sample_data_from_results(
            {
                "codes": [0, 1],
                "eval_steps": [0, 10, 20],
                "effective_samples": {0: [0, 17, 83], 1: [0, 63, 77]},
                "task_probabilities": {0: 0.75, 1: 0.25},
                "node_depths": {0: 0, 1: 1},
                "training_samples": [8, 8],
            }
        )

        np.testing.assert_allclose(data.axes[0], [0, 17, 83])
        np.testing.assert_allclose(data.axes[1], [0, 63, 77])

    def test_effective_samples_render_depth_and_combined_panels(self):
        data = effective_sample_data_from_results(
            {
                "codes": [0, 1],
                "eval_steps": [0, 10, 20],
                "task_probabilities": {0: 0.75, 1: 0.25},
                "node_depths": {0: 0, 1: 1},
                "training_samples": [8, 8],
            }
        )

        figure = plot_discovery_trajectories(
            output_image_path=None,
            codes=[0, 1],
            subtask_losses=[[0.8, 0.4, 0.02], [0.9, 0.6, 0.03]],
            samples=[0, 10, 20],
            effective_sample_data=data,
            x_start=0,
            x_lim=None,
            x_axis="effective_samples",
            x_scale="linear",
        )

        self.assertEqual(len(figure.axes), 3)
        self.assertIn("Depth 0", figure.axes[0].get_title())
        self.assertIn("Depth 1", figure.axes[1].get_title())
        self.assertEqual(figure.axes[2].get_title(), "All Depths: 2 tasks")
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, 60, 120])
        np.testing.assert_allclose(figure.axes[1].lines[0].get_xdata(), [0, 20, 40])

    def test_discovery_trajectory_ylim_sets_panel_upper_bounds(self):
        figure = plot_discovery_trajectories(
            output_image_path=None,
            codes=[0, 1],
            subtask_losses=[[0.7, 0.4], [0.8, 0.5]],
            samples=[0, 100],
            x_start=0,
            x_lim=100,
            x_axis="steps",
            x_scale="linear",
            ylim=0.6,
        )

        self.assertTrue(figure.axes)
        for axis in figure.axes:
            self.assertAlmostEqual(axis.get_ylim()[1], 0.6)

    def test_scaling_ylim_sets_log_axis_upper_bound(self):
        pair_summaries = [
            {
                "empirical_alpha": 0.5,
                "theoretical_alpha": 0.5,
                "fit_intercept": 1.0,
                "width_summaries": [
                    {
                        "width": 16,
                        "n_parameters_mean": 100.0,
                        "final_eval_loss_bits_mean": 0.8,
                        "final_eval_loss_bits_se": 0.0,
                        "n_runs": 1,
                    },
                    {
                        "width": 32,
                        "n_parameters_mean": 200.0,
                        "final_eval_loss_bits_mean": 0.5,
                        "final_eval_loss_bits_se": 0.0,
                        "n_runs": 1,
                    },
                ],
            }
        ]

        with mock.patch(
            "quanta.figures.scaling_laws.save_figure"
        ) as save_figure:
            plot_scaling_laws(
                "unused.png",
                pair_summaries,
                ylim=0.6,
            )

        figure = save_figure.call_args.args[0]
        self.assertEqual(len(figure.axes), 3)
        self.assertEqual(
            [axis.get_title() for axis in figure.axes],
            [
                "Endpoint / all parameters",
                "Endpoint / non-embedding parameters",
                "Tail median / all parameters",
            ],
        )
        for axis in figure.axes:
            self.assertAlmostEqual(axis.get_ylim()[1], 0.6)

    def test_ema_smooth_matches_wandb_style_weighted_history(self):
        np.testing.assert_allclose(_ema_smooth(np.array([1.0, 0.0, 0.0]), 0.5), [1.0, 0.5, 0.25])
        np.testing.assert_allclose(_ema_smooth(np.array([1.0, 0.0, 0.0]), 0.0), [1.0, 0.0, 0.0])

    def test_dependencies_ready_loss_waits_for_closure_dependencies(self):
        curves = {
            0: np.array([1.0, 0.2, 0.04, 0.03]),
            1: np.array([1.0, 0.9, 0.8, 0.04]),
            2: np.array([1.4, 1.0, 0.7, 0.3]),
        }

        proper = _dependencies_ready_loss_curves(
            raw_curves=curves,
            readiness_curves=curves,
            smoothing=0.0,
            codes=[0, 1, 2],
            graph_dependencies={1: [0], 2: [1]},
            threshold=0.05,
        )

        np.testing.assert_allclose(proper[0], [1.0, 0.2, 0.04, 0.03])
        np.testing.assert_allclose(proper[1], [1.0, 1.0, 0.8, 0.04])
        np.testing.assert_allclose(proper[2], [1.4, 1.4, 1.4, 0.3])

if __name__ == "__main__":
    unittest.main()
