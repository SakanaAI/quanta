import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.batch_evaluation import (
    evaluate_weighted_task_loss,
)
from quanta.experiments.scaling_laws.cnand_values import (
    masked_token_loss_statistics,
)


class BatchLogitModel(nn.Module):
    def forward(self, batch):
        return batch["logits"]


class ScalingLawsEvalLossTests(unittest.TestCase):
    def test_masked_token_loss_statistics_exposes_sum_count_and_mean(self):
        logits = torch.tensor([[[0.0, 0.0], [2.0, -1.0], [4.0, -2.0]]])
        batch = {
            "loss_targets": torch.tensor([[0, -100, 0]]),
        }

        statistics = masked_token_loss_statistics(logits, batch)

        expected_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2),
            batch["loss_targets"].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        self.assertEqual(statistics["token_count"], 2)
        self.assertTrue(torch.allclose(statistics["loss_sum_nats"], expected_sum))
        self.assertTrue(
            torch.allclose(statistics["mean_loss_nats"], expected_sum / 2)
        )

    def test_primary_eval_is_probability_weighted_global_token_mean(self):
        task_batches = [
            {
                "logits": torch.tensor([[[0.0, 0.0]]]),
                "loss_targets": torch.tensor([[0]]),
                "loss_mask": torch.tensor([[True]]),
            },
            {
                "logits": torch.tensor(
                    [[[4.0, -2.0], [4.0, -2.0], [4.0, -2.0]]]
                ),
                "loss_targets": torch.tensor([[0, 0, 0]]),
                "loss_mask": torch.tensor([[True, True, True]]),
            },
        ]
        config = ScalingLawsConfig(
            task="cnand",
            eval_samples_per_task=1,
        )
        task_spec = SimpleNamespace(codes=[0, 1])
        probabilities = torch.tensor([0.5, 0.5])

        with mock.patch(
            "quanta.experiments.scaling_laws.batch_evaluation.cached_cnand_eval_batch",
            side_effect=[(batch, batch["loss_targets"]) for batch in task_batches],
        ):
            diagnostics = evaluate_weighted_task_loss(
                model=BatchLogitModel(),
                loss_fn=nn.CrossEntropyLoss(),
                config=config,
                task_spec=task_spec,
                probabilities=probabilities,
                node_depths={0: 0, 1: 1},
                batch_cache={},
                device="cpu",
            )

        task_zero = diagnostics["task_losses"][0]
        task_one = diagnostics["task_losses"][1]
        expected_nats = (
            0.5 * task_zero["loss_sum_nats"] + 0.5 * task_one["loss_sum_nats"]
        ) / (
            0.5 * task_zero["token_count"] + 0.5 * task_one["token_count"]
        )
        old_weighted_task_mean = (
            0.5 * task_zero["mean_loss_nats"]
            + 0.5 * task_one["mean_loss_nats"]
        )

        self.assertAlmostEqual(diagnostics["weighted_token_loss_nats"], expected_nats)
        self.assertAlmostEqual(diagnostics["total_loss_nats"], expected_nats)
        self.assertAlmostEqual(
            diagnostics["weighted_token_loss_bits"],
            expected_nats / math.log(2),
        )
        self.assertAlmostEqual(
            diagnostics["total_loss_bits"],
            diagnostics["weighted_token_loss_bits"],
        )
        self.assertNotAlmostEqual(expected_nats, old_weighted_task_mean)
        self.assertEqual(task_zero["token_count"], 1)
        self.assertEqual(task_one["token_count"], 3)

    def test_task_weighted_eval_is_target_probability_weighted_task_mean(self):
        task_batches = [
            {
                "logits": torch.tensor([[[0.0, 0.0]]]),
                "loss_targets": torch.tensor([[0]]),
                "loss_mask": torch.tensor([[True]]),
            },
            {
                "logits": torch.tensor(
                    [[[4.0, -2.0], [4.0, -2.0], [4.0, -2.0]]]
                ),
                "loss_targets": torch.tensor([[0, 0, 0]]),
                "loss_mask": torch.tensor([[True, True, True]]),
            },
        ]
        config = ScalingLawsConfig(
            task="cnand",
            eval_samples_per_task=1,
            eval_loss_formula="task_weighted",
        )
        probabilities = torch.tensor([0.25, 0.75])

        with mock.patch(
            "quanta.experiments.scaling_laws.batch_evaluation.cached_cnand_eval_batch",
            side_effect=[(batch, batch["loss_targets"]) for batch in task_batches],
        ):
            diagnostics = evaluate_weighted_task_loss(
                model=BatchLogitModel(),
                loss_fn=nn.CrossEntropyLoss(),
                config=config,
                task_spec=SimpleNamespace(codes=[0, 1]),
                probabilities=probabilities,
                node_depths={0: 0, 1: 1},
                batch_cache={},
                device="cpu",
            )

        expected = sum(
            float(probabilities[index])
            * diagnostics["task_losses"][index]["mean_loss_nats"]
            for index in range(2)
        )
        self.assertEqual(diagnostics["eval_loss_formula"], "task_weighted")
        self.assertAlmostEqual(diagnostics["task_weighted_loss_nats"], expected)
        self.assertAlmostEqual(diagnostics["eval_loss_nats"], expected)
        self.assertAlmostEqual(diagnostics["total_loss_nats"], expected)
        self.assertNotAlmostEqual(
            diagnostics["weighted_token_loss_nats"],
            diagnostics["task_weighted_loss_nats"],
        )

    def test_quantum_loss_weights_its_out_token_across_active_target_contexts(self):
        task_batches = [
            {
                "logits": torch.tensor([[[2.0, 0.0]]]),
                "input_ids": torch.tensor([[2]]),
                "active_mask": torch.tensor([[True]]),
                "quantum_ids": torch.tensor([[0]]),
                "true_values": torch.tensor([[0, 0]]),
                "loss_targets": torch.tensor([[0]]),
                "loss_mask": torch.tensor([[True]]),
            },
            {
                "logits": torch.tensor([[[0.0, 1.0], [3.0, 0.0]]]),
                "input_ids": torch.tensor([[2, 2]]),
                "active_mask": torch.tensor([[True, True]]),
                "quantum_ids": torch.tensor([[0, 1]]),
                "true_values": torch.tensor([[1, 0]]),
                "loss_targets": torch.tensor([[-100, 0]]),
                "loss_mask": torch.tensor([[False, True]]),
            },
        ]
        probabilities = torch.tensor([0.25, 0.75])
        context_zero_loss = torch.nn.functional.cross_entropy(
            task_batches[0]["logits"][:, 0],
            torch.tensor([0]),
        ).item()
        context_one_q0_loss = torch.nn.functional.cross_entropy(
            task_batches[1]["logits"][:, 0],
            torch.tensor([1]),
        ).item()
        context_one_q1_loss = torch.nn.functional.cross_entropy(
            task_batches[1]["logits"][:, 1],
            torch.tensor([0]),
        ).item()

        with mock.patch(
            "quanta.experiments.scaling_laws.batch_evaluation.cached_cnand_eval_batch",
            side_effect=[(batch, batch["loss_targets"]) for batch in task_batches],
        ):
            diagnostics = evaluate_weighted_task_loss(
                model=BatchLogitModel(),
                loss_fn=nn.CrossEntropyLoss(),
                config=ScalingLawsConfig(
                    task="cnand",
                    eval_samples_per_task=1,
                    loss_supervision="target_only",
                ),
                task_spec=SimpleNamespace(codes=[0, 1]),
                probabilities=probabilities,
                node_depths={0: 0, 1: 1},
                batch_cache={},
                device="cpu",
            )

        expected_q0 = 0.25 * context_zero_loss + 0.75 * context_one_q0_loss
        self.assertAlmostEqual(
            diagnostics["quantum_losses"][0]["loss_nats"],
            expected_q0,
        )
        self.assertAlmostEqual(
            diagnostics["quantum_losses"][0]["context_probability"],
            1.0,
        )
        self.assertAlmostEqual(
            diagnostics["quantum_losses"][1]["loss_nats"],
            context_one_q1_loss,
        )
        self.assertAlmostEqual(
            diagnostics["quantum_losses"][1]["context_probability"],
            0.75,
        )
        self.assertAlmostEqual(
            diagnostics["quantum_mean_losses"][0]["loss_nats"],
            0.5 * (context_zero_loss + context_one_q0_loss),
        )
        self.assertEqual(
            diagnostics["quantum_mean_losses"][0]["context_count"],
            2,
        )

        with mock.patch(
            "quanta.experiments.scaling_laws.batch_evaluation.cached_cnand_eval_batch",
            side_effect=[(batch, batch["loss_targets"]) for batch in task_batches],
        ):
            zero_mass_diagnostics = evaluate_weighted_task_loss(
                model=BatchLogitModel(),
                loss_fn=nn.CrossEntropyLoss(),
                config=ScalingLawsConfig(
                    task="cnand",
                    eval_samples_per_task=1,
                    loss_supervision="target_only",
                ),
                task_spec=SimpleNamespace(codes=[0, 1]),
                probabilities=torch.tensor([1.0, 0.0]),
                node_depths={0: 0, 1: 1},
                batch_cache={},
                device="cpu",
            )

        self.assertNotIn(1, zero_mass_diagnostics["quantum_losses"])
        self.assertAlmostEqual(
            zero_mass_diagnostics["quantum_mean_losses"][1]["loss_nats"],
            context_one_q1_loss,
        )

    def test_distributed_eval_partitions_tasks_by_process(self):
        local_batch = {
            "logits": torch.tensor([[[0.0, 0.0]]]),
            "loss_targets": torch.tensor([[0]]),
            "loss_mask": torch.tensor([[True]]),
        }
        gathered_results = [
            {
                "index": 0,
                "code": 0,
                "loss_sum_nats": math.log(2),
                "token_count": 1,
                "mean_loss_nats": math.log(2),
                "accuracy": 1.0,
            },
            {
                "index": 1,
                "code": 1,
                "loss_sum_nats": math.log(2),
                "token_count": 1,
                "mean_loss_nats": math.log(2),
                "accuracy": 1.0,
            },
        ]
        accelerator = SimpleNamespace(
            process_index=1,
            num_processes=2,
            autocast=mock.MagicMock(return_value=torch.autocast("cpu", enabled=False)),
        )

        with (
            mock.patch(
                "quanta.experiments.scaling_laws.batch_evaluation.cached_cnand_eval_batch",
                return_value=(local_batch, local_batch["loss_targets"]),
            ) as build_batch,
            mock.patch(
                "quanta.experiments.scaling_laws.batch_evaluation.gather_object",
                return_value=gathered_results,
            ) as gather,
        ):
            diagnostics = evaluate_weighted_task_loss(
                model=BatchLogitModel(),
                loss_fn=nn.CrossEntropyLoss(),
                config=ScalingLawsConfig(
                    task="cnand",
                    eval_samples_per_task=1,
                ),
                task_spec=SimpleNamespace(codes=[0, 1]),
                probabilities=torch.tensor([0.5, 0.5]),
                node_depths={0: 0, 1: 1},
                batch_cache={},
                device="cpu",
                accelerator=accelerator,
            )

        build_batch.assert_called_once()
        self.assertEqual(build_batch.call_args.kwargs["task_index"], 1)
        gather.assert_called_once()
        self.assertAlmostEqual(diagnostics["weighted_token_loss_nats"], math.log(2))


if __name__ == "__main__":
    unittest.main()
