from __future__ import annotations

import itertools

import torch

from quanta.config import ScalingLawsConfig
from quanta.utils import ForkRNG, set_seeds
from quanta.experiments.scaling_laws.batch_common import (
    _node_depth,
    ancestral_closure,
    task_probability_tensor,
    CNAND_TOKEN_SIGN_POS,
    CNAND_TOKEN_SIGN_NEG,
)

CNAND_CACHE_SEED_OFFSET = 104_729


def build_cnand_batch_cache(config: ScalingLawsConfig, task_spec, device) -> dict[str, torch.Tensor]:
    node_depths = torch.tensor(
        [_node_depth(int(code), task_spec.graph_dependencies) for code in task_spec.codes],
        dtype=torch.long,
        device=device,
    )
    parent_lists = [list((task_spec.graph_dependencies or {}).get(int(code), [])) for code in task_spec.codes]
    max_parents = max((len(parents) for parents in parent_lists), default=0)
    parent_indices = torch.full((len(task_spec.codes), max_parents), -1, dtype=torch.long, device=device)
    parent_masks = torch.zeros((len(task_spec.codes), max_parents), dtype=torch.bool, device=device)
    closure_masks = torch.zeros((len(task_spec.codes), len(task_spec.codes)), dtype=torch.bool, device=device)
    max_local_bits = max((len(task_spec.Ss_atomic[int(code)]) for code in task_spec.codes), default=int(config.n_local_bits))
    local_bit_indices = torch.zeros((len(task_spec.codes), max_local_bits), dtype=torch.long, device=device)
    local_bit_masks = torch.zeros((len(task_spec.codes), max_local_bits), dtype=torch.bool, device=device)
    for index, code in enumerate(task_spec.codes):
        parents = parent_lists[index]
        if parents:
            parent_indices[index, : len(parents)] = torch.tensor(parents, dtype=torch.long, device=device)
            parent_masks[index, : len(parents)] = True
        closure_masks[index, sorted(ancestral_closure(int(code), task_spec.graph_dependencies))] = True
        bits = list(task_spec.Ss_atomic[int(code)])
        local_bit_indices[index, : len(bits)] = torch.tensor(bits, dtype=torch.long, device=device)
        local_bit_masks[index, : len(bits)] = True

    probabilities = task_probability_tensor(task_spec.codes, config.task_frequencies, device)
    theoretical_p_q = probabilities
    ideal_masks = torch.empty((0, len(task_spec.codes)), dtype=torch.bool, device=device)
    ideal_probabilities = torch.empty((0,), dtype=torch.float32, device=device)
    if config.trace_sampling == "ideal_threshold":
        diagnostics = config.quanta_demand_diagnostics or {}
        mixture = diagnostics.get("ideal_mixture")
        if not mixture:
            raise ValueError("ideal_threshold trace sampling requires ideal-mixture diagnostics.")
        mixture_nodes = [int(node) for node in mixture["nodes"]]
        if mixture_nodes != [int(code) for code in task_spec.codes]:
            raise ValueError("ideal-mixture nodes must match the cNAND task codes.")
        ideal_masks = torch.tensor(mixture["ideal_masks"], dtype=torch.bool, device=device)
        ideal_probabilities = torch.tensor(
            mixture["probabilities"], dtype=torch.float32, device=device
        )
        theoretical_p_q = torch.tensor(
            [float(mixture["marginals"][int(code)]) for code in task_spec.codes],
            dtype=torch.float32,
            device=device,
        )
    path_ideal_masks = torch.empty(
        (0, len(task_spec.codes)), dtype=torch.bool, device=device
    )
    path_ideal_probabilities = torch.empty((0,), dtype=torch.float32, device=device)
    path_terminal_quantum_ids = torch.empty((0,), dtype=torch.long, device=device)
    if config.trace_sampling == "ideal_path":
        diagnostics = config.quanta_demand_diagnostics or {}
        sampler = diagnostics.get("ideal_sampler")
        if not sampler or sampler.get("type") != "paired_module_path":
            raise ValueError(
                "ideal_path trace sampling requires paired-module path diagnostics."
            )
        modules_by_depth = sampler["modules_by_depth"]
        parent_modules_by_depth = sampler["parent_modules_by_depth"]
        terminal_probabilities_by_depth = sampler[
            "terminal_probabilities_by_depth"
        ]
        depth_offsets: list[int] = []
        module_offset = 0
        for modules in modules_by_depth:
            depth_offsets.append(module_offset)
            module_offset += len(modules)

        flat_modules: list[list[int]] = []
        flat_parents: list[int] = []
        flat_probabilities: list[float] = []
        for depth, modules in enumerate(modules_by_depth):
            for local_index, module in enumerate(modules):
                flat_modules.append([int(node) for node in module])
                parent_local_index = int(parent_modules_by_depth[depth][local_index])
                flat_parents.append(
                    -1
                    if depth == 0
                    else depth_offsets[depth - 1] + parent_local_index
                )
                flat_probabilities.append(
                    float(terminal_probabilities_by_depth[depth][local_index])
                )

        path_ideal_masks = torch.zeros(
            (len(flat_modules), len(task_spec.codes)),
            dtype=torch.bool,
            device=device,
        )
        for module_index, (module, parent_index) in enumerate(
            zip(flat_modules, flat_parents)
        ):
            if parent_index >= 0:
                path_ideal_masks[module_index] = path_ideal_masks[parent_index]
            path_ideal_masks[module_index, module] = True
        path_ideal_probabilities = torch.tensor(
            flat_probabilities, dtype=torch.float32, device=device
        )
        path_ideal_probabilities /= path_ideal_probabilities.sum()
        path_terminal_quantum_ids = torch.tensor(
            [module[0] for module in flat_modules],
            dtype=torch.long,
            device=device,
        )
        theoretical_p_q = torch.tensor(
            [float(sampler["marginals"][int(code)]) for code in task_spec.codes],
            dtype=torch.float32,
            device=device,
        )
        reconstructed_marginals = (
            path_ideal_probabilities.unsqueeze(1)
            * path_ideal_masks.to(dtype=torch.float32)
        ).sum(dim=0)
        if not torch.allclose(
            reconstructed_marginals,
            theoretical_p_q,
            atol=2e-6,
            rtol=2e-5,
        ):
            raise RuntimeError(
                "Compact ideal-path mixture does not reconstruct its target marginals."
            )
    n_lut_functions = getattr(config, "n_lut_functions", None)
    uses_lut_functions = n_lut_functions is not None
    tokens_per_node = max_local_bits + 2 if uses_lut_functions else max_local_bits + 1
    output_token_slots = torch.arange(tokens_per_node - 1, len(task_spec.codes) * tokens_per_node, tokens_per_node, dtype=torch.long, device=device)
    token_node_ids = torch.arange(len(task_spec.codes), dtype=torch.long, device=device).repeat_interleave(tokens_per_node)

    node_functions = torch.empty((0, 0), dtype=torch.long, device=device)
    node_signs = torch.zeros((len(task_spec.codes),), dtype=torch.long, device=device)
    if uses_lut_functions:
        node_functions = torch.zeros((len(task_spec.codes), 2**max_local_bits), dtype=torch.long, device=device)
        n_nodes = len(task_spec.codes)
        num_unique = n_nodes if int(n_lut_functions) == -1 else int(n_lut_functions)

        n_inputs = 2 ** max_local_bits
        base = torch.cat([
            torch.zeros(n_inputs // 2, dtype=torch.long, device=device),
            torch.ones(n_inputs // 2, dtype=torch.long, device=device)
        ])

        unique_funcs = _build_lut_functions(
            family=str(getattr(config, "lut_family", "random_balanced")),
            num_unique=num_unique,
            n_inputs=n_inputs,
            local_bit_width=max_local_bits,
            base=base,
            device=device,
        )

        if int(n_lut_functions) == -1:
            node_fn_indices = torch.arange(n_nodes, device=device)
        else:
            node_fn_indices = torch.randint(0, num_unique, (n_nodes,), device=device)

        node_signs = 4 + node_fn_indices

        for index, code in enumerate(task_spec.codes):
            n_active = len(task_spec.Ss_atomic[int(code)])
            n_inputs_node = 2 ** n_active
            func_idx = node_fn_indices[index].item()
            node_functions[index, :n_inputs_node] = unique_funcs[func_idx, :n_inputs_node]

    return {
        "input_format": "node_bit_out_tokens",
        "prediction_mode": "out_tokens",
        "node_depths": node_depths,
        "parent_indices": parent_indices,
        "parent_masks": parent_masks,
        "closure_masks": closure_masks,
        "local_bit_indices": local_bit_indices,
        "local_bit_masks": local_bit_masks,
        "tokens_per_node": torch.tensor(tokens_per_node, dtype=torch.long, device=device),
        "output_token_slots": output_token_slots,
        "token_node_ids": token_node_ids,
        "slot_ids": torch.arange(len(task_spec.codes) * tokens_per_node, dtype=torch.long, device=device),
        "quantum_ids": torch.tensor([int(code) for code in task_spec.codes], dtype=torch.long, device=device),
        "kappa_values": torch.zeros((len(task_spec.codes),), dtype=torch.long, device=device),
        "theoretical_p_q": theoretical_p_q,
        "ideal_masks": ideal_masks,
        "ideal_probabilities": ideal_probabilities,
        "path_ideal_masks": path_ideal_masks,
        "path_ideal_probabilities": path_ideal_probabilities,
        "path_terminal_quantum_ids": path_terminal_quantum_ids,
        "node_functions": node_functions,
        "node_signs": node_signs,
    }


def _build_lut_functions(
    *,
    family: str,
    num_unique: int,
    n_inputs: int,
    local_bit_width: int,
    base: torch.Tensor,
    device,
) -> torch.Tensor:
    """Generate local rules while keeping the seeded cache deterministic."""
    if family == "parity":
        assignments = torch.arange(n_inputs, device=device)
        bits = (assignments.unsqueeze(1) >> torch.arange(local_bit_width, device=device)) & 1
        parity = bits.sum(dim=1).remainder(2).to(dtype=torch.long)
        # Input flips only complement parity.  Retaining this orbit gives each
        # root a distinct rule token without changing its spectrum or balance.
        phases = torch.randint(0, 2, (num_unique, 1), device=device)
        return parity.unsqueeze(0).expand(num_unique, -1).bitwise_xor(phases)

    if family == "symmetry_orbit":
        # Draw one balanced base rule, then use only its symmetries.  This
        # keeps every root's Boolean spectrum and truth-table difficulty fixed.
        base_rule = base[torch.randperm(n_inputs, device=device)]
        orbit = _balanced_lut_symmetry_orbit(base_rule, local_bit_width)
        if len(orbit) < num_unique:
            raise ValueError(
                "symmetry-orbit LUT family has fewer distinct functions than requested "
                f"({len(orbit)} < {num_unique})."
            )
        chosen = torch.randperm(len(orbit), device=device)[:num_unique]
        return torch.tensor(
            [orbit[int(index)] for index in chosen.detach().cpu().tolist()],
            dtype=torch.long,
            device=device,
        )

    if family != "random_balanced":
        raise ValueError(f"Unknown LUT family: {family!r}.")
    unique_funcs = torch.zeros((num_unique, n_inputs), dtype=torch.long, device=device)
    seen_functions = set()
    for i in range(num_unique):
        for _ in range(10000):
            perm = torch.randperm(n_inputs, device=device)
            func_tuple = tuple(base[perm].tolist())
            if func_tuple not in seen_functions:
                seen_functions.add(func_tuple)
                unique_funcs[i] = base[perm]
                break
        else:
            unique_funcs[i] = base[perm]
    return unique_funcs


def _balanced_lut_symmetry_orbit(base: torch.Tensor, local_bit_width: int) -> list[tuple[int, ...]]:
    """Return the distinct input-permutation/bit-flip/output-flip transforms."""
    values = base.detach().cpu().tolist()
    assignments = list(range(1 << local_bit_width))
    orbit: set[tuple[int, ...]] = set()
    for permutation in itertools.permutations(range(local_bit_width)):
        for input_flip_mask in range(1 << local_bit_width):
            transformed = []
            for assignment in assignments:
                source = 0
                for output_bit, input_bit in enumerate(permutation):
                    value = ((assignment >> output_bit) & 1) ^ ((input_flip_mask >> output_bit) & 1)
                    source |= value << input_bit
                transformed.append(values[source])
            transformed_tuple = tuple(transformed)
            orbit.add(transformed_tuple)
            orbit.add(tuple(1 - value for value in transformed_tuple))
    return sorted(orbit)


def build_seeded_cnand_batch_cache(
    config: ScalingLawsConfig,
    task_spec,
    device,
) -> dict[str, torch.Tensor]:
    function_seed = (
        int(config.function_seed)
        if getattr(config, "function_seed", None) is not None
        else int(config.seed)
    )
    if config.task == "multitask_sparse_parity":
        with ForkRNG():
            set_seeds(function_seed + CNAND_CACHE_SEED_OFFSET)
            masks = torch.zeros(
                (len(task_spec.codes), int(config.parity_task_bits)),
                dtype=torch.bool,
            )
            for index in range(len(task_spec.codes)):
                selected = torch.randperm(int(config.parity_task_bits))[
                    : int(config.parity_subset_size)
                ]
                masks[index, selected] = True
        return {"parity_masks": masks.to(device)}
    with ForkRNG():
        set_seeds(function_seed + CNAND_CACHE_SEED_OFFSET)
        cache = build_cnand_batch_cache(config, task_spec, "cpu")
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in cache.items()
    }
