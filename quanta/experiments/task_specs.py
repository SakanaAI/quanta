from __future__ import annotations

from dataclasses import dataclass

from quanta.config import TrainingConfig


@dataclass
class TaskSpec:
    Ss_atomic: list
    resolved_Ss: list
    codes: list
    graph_dependencies: dict[int, list[int]] | None
    n_tasks: int
    quanta_demand: dict | None = None


class TaskSpecBuilder:
    def build(self, config: TrainingConfig) -> TaskSpec:
        if config.task in {"cxor", "cnand"}:
            return self._cxor(config)
        if config.task == "multitask_sparse_parity":
            return self._multitask_sparse_parity(config)
        raise ValueError(f"Unsupported task: {config.task}")

    def _multitask_sparse_parity(self, config: TrainingConfig) -> TaskSpec:
        frequencies = config.task_frequencies or {
            index: 1.0 for index in range(int(config.base_tasks))
        }
        codes = sorted(int(code) for code in frequencies)
        if codes != list(range(len(codes))):
            raise ValueError("multitask_sparse_parity task codes must be contiguous from zero.")
        if int(config.parity_subset_size) <= 0:
            raise ValueError("parity_subset_size must be positive.")
        if int(config.parity_subset_size) > int(config.parity_task_bits):
            raise ValueError("parity_subset_size cannot exceed parity_task_bits.")
        config.n_tasks = len(codes)
        config.n_bits = int(config.parity_task_bits)
        return TaskSpec(
            Ss_atomic=[list(range(int(config.parity_task_bits))) for _ in codes],
            resolved_Ss=[list(range(int(config.parity_task_bits))) for _ in codes],
            codes=codes,
            graph_dependencies={},
            n_tasks=len(codes),
            quanta_demand=config.quanta_demand_diagnostics,
        )

    def _cxor(self, config: TrainingConfig) -> TaskSpec:
        if config.graph_dependencies is None and config.rho is not None and config.beta is not None and config.max_depth is not None:
            from quanta.experiments.scaling_laws import generate_layered_poset
            graph = generate_layered_poset(
                rho=self._first_float(config.rho, default=1.0),
                beta=self._first_float(config.beta, default=1.0),
                delta=self._first_float(config.delta, default=0.0),
                base_tasks=config.base_tasks,
                base_freq=config.base_freq,
                max_depth=config.max_depth,
                m=config.m,
                seed=self._first_int(config.seed, default=0),
                quanta_demand=config.quanta_demand,
                quanta_demand_fit_tolerance=config.quanta_demand_fit_tolerance,
                trace_sampling=config.trace_sampling,
                graph_family=config.graph_family,
                flat_frequency_exponent=config.flat_frequency_exponent,
                target_alpha=config.target_alpha,
            )
            config.graph_dependencies = graph["graph_dependencies"]
            config.task_frequencies = graph["task_frequencies"]
            config.quanta_demand_diagnostics = graph["quanta_demand"]

        graph_dependencies = (
            config.graph_dependencies
            if config.graph_dependencies is not None
            else {4: [0, 1, 2, 3]}
        )
        all_nodes = set(graph_dependencies.keys())
        for parents in graph_dependencies.values():
            all_nodes.update(parents)
        if not all_nodes and config.task_frequencies is not None:
            all_nodes.update(int(node) for node in config.task_frequencies)

        max_graph_node = max(all_nodes) if all_nodes else 0
        n_tasks = max_graph_node + 1

        node_depths = {
            task_index: self._cxor_node_depth(task_index, graph_dependencies)
            for task_index in range(n_tasks)
        }
        rho = self._first_float(getattr(config, "rho", None), default=1.0)
        beta = self._first_float(config.beta, default=1.0)
        delta = self._first_float(getattr(config, "delta", None), default=0.0)
        if config.quanta_demand in {"composition", "uniform"}:
            from quanta.experiments.demand import (
                resolve_task_demand,
                select_theoretical_alpha,
            )

            if config.trace_sampling == "ideal_path":
                demand = config.quanta_demand_diagnostics
                if not demand or not demand.get("ideal_sampler"):
                    raise ValueError(
                        f"{config.trace_sampling} requires graph-provided sampler diagnostics."
                    )
            else:
                demand = resolve_task_demand(
                    graph_dependencies=graph_dependencies,
                    node_depths=node_depths,
                    beta=beta,
                    base_freq=config.base_freq,
                    mode=config.quanta_demand,
                    fit_tolerance=config.quanta_demand_fit_tolerance,
                    trace_sampling=config.trace_sampling,
                )
            selected_alpha, theory_alpha_source = select_theoretical_alpha(
                demand=demand,
                rho=rho,
                beta=beta,
                delta=delta,
            )
            demand.update(
                {
                    "theoretical_alpha": float(selected_alpha),
                    "theory_alpha_source": theory_alpha_source,
                }
            )
            config.task_frequencies = demand["target_frequencies"]
            config.quanta_demand_diagnostics = demand

        Ss_atomic = [[] for _ in range(n_tasks)]
        bit_index = 0
        bits_per_node = config.n_local_bits if config.task == "cnand" else config.n_atomic_task_bits
        for task_index in range(n_tasks):
            depth = node_depths[task_index]
            bit_count = max(1, int(round(bits_per_node * (rho ** (delta * depth)))))
            Ss_atomic[task_index] = list(range(bit_index, bit_index + bit_count))
            bit_index += bit_count
        config.n_bits = bit_index + config.n_noise_bits

        return TaskSpec(
            Ss_atomic=Ss_atomic,
            resolved_Ss=Ss_atomic,
            codes=list(range(n_tasks)),
            graph_dependencies=graph_dependencies,
            n_tasks=n_tasks,
            quanta_demand=config.quanta_demand_diagnostics,
        )

    @staticmethod
    def _first_float(value, *, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, list):
            if not value:
                return default
            return float(value[0])
        return float(value)

    @staticmethod
    def _first_int(value, *, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, list):
            if not value:
                return default
            return int(value[0])
        return int(value)


    @classmethod
    def _cxor_node_depth(cls, node: int, graph_dependencies: dict[int, list[int]]) -> int:
        parents = graph_dependencies.get(int(node), [])
        if not parents:
            return 0
        return 1 + max(cls._cxor_node_depth(int(parent), graph_dependencies) for parent in parents)
