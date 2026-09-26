import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quanta.config import ScalingLawsConfig
from quanta.experiments.scaling_laws.graph import generate_layered_poset
from quanta.experiments.scaling_laws.jobs import build_scaling_run_jobs


def _cache_test_config(*, seeds):
    return ScalingLawsConfig(
        rho=[2.0],
        beta=[3.0],
        delta=[0.0],
        rho_beta_delta=[[2.0, 3.0, 0.0]],
        seed=seeds,
        width=[16],
        lr=[1e-3],
        base_tasks=2,
        max_depth=2,
        m=2,
        quanta_demand="shortcut",
    )


class ScalingLawsGraphCacheTests(unittest.TestCase):
    def test_repeated_build_loads_graph_from_disk_cache(self):
        config = _cache_test_config(seeds=[7])

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "quanta.experiments.scaling_laws.jobs.generate_layered_poset",
            wraps=generate_layered_poset,
        ) as generate:
            first = build_scaling_run_jobs(config, graph_cache_dir=temp_dir)
            second = build_scaling_run_jobs(config, graph_cache_dir=temp_dir)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(first[0]["graph"], second[0]["graph"])
            self.assertEqual(len(list(Path(temp_dir).glob("*.pkl"))), 1)

    def test_cache_key_separates_seeds(self):
        config = _cache_test_config(seeds=[3, 4])

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "quanta.experiments.scaling_laws.jobs.generate_layered_poset",
            wraps=generate_layered_poset,
        ) as generate:
            build_scaling_run_jobs(config, graph_cache_dir=temp_dir)

            self.assertEqual(generate.call_count, 2)
            self.assertEqual(
                sorted(path.name.split("-", 1)[0] for path in Path(temp_dir).glob("*.pkl")),
                ["seed3", "seed4"],
            )

    def test_invalid_cache_is_regenerated(self):
        config = _cache_test_config(seeds=[5])

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "quanta.experiments.scaling_laws.jobs.generate_layered_poset",
            wraps=generate_layered_poset,
        ) as generate:
            build_scaling_run_jobs(config, graph_cache_dir=temp_dir)
            cache_path = next(Path(temp_dir).glob("*.pkl"))
            cache_path.write_bytes(b"invalid")

            build_scaling_run_jobs(config, graph_cache_dir=temp_dir)

            self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
