import unittest

import torch

from quanta.experiments.scaling_laws.decomposition import (
    DependencyProbeExample,
    declared_parent_subsets,
    evaluate_dependency_probes,
)


class ScalingLawsDecompositionTests(unittest.TestCase):
    def test_declared_parent_subsets_are_exhaustive_and_deterministic(self):
        self.assertEqual(
            declared_parent_subsets([4, 2]),
            [(), (2,), (4,), (2, 4)],
        )

    def test_probe_selects_best_declared_circuit_and_reports_label_loss(self):
        good_signal = torch.tensor([0.0, 0.0, 1.0, 1.0] * 32)
        bad_signal = torch.zeros_like(good_signal)
        labels = good_signal.clone()
        model_probability = 0.02 + 0.96 * good_signal
        example = DependencyProbeExample(
            code_index=2,
            code=2,
            candidates=[(), (0, 1)],
            circuit_signals=torch.stack([bad_signal, good_signal]),
            model_probability=model_probability,
            labels=labels,
        )

        losses, records = evaluate_dependency_probes(
            [example],
            probe_steps=200,
            probe_lr=0.1,
        )

        self.assertEqual(records[0]["best_set"], [0, 1])
        self.assertLess(losses[2], 0.1)
        self.assertLess(
            records[0]["candidate_scores"]["{0,1}"],
            records[0]["candidate_scores"]["{}"],
        )



if __name__ == "__main__":
    unittest.main()
