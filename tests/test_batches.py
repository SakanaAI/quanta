import unittest

from quanta.training.batches import proportional_batch_sizes


class BatchSupportTests(unittest.TestCase):
    def test_proportional_batch_sizes_preserve_total_and_frequency_order(self):
        sizes = proportional_batch_sizes(10, [0, 1, 2], {0: 1.0, 1: 2.0, 2: 1.0})

        self.assertEqual(sum(sizes), 10)
        self.assertGreater(sizes[1], sizes[0])
        self.assertGreater(sizes[1], sizes[2])

if __name__ == "__main__":
    unittest.main()
