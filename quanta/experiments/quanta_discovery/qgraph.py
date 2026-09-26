from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from quanta.experiments.number_naming.model import DecoderTransformerLM

from .qmodel import (
    AttentionQuantumCache,
    CheapCausalAttentionQuantumLayer,
    QuantumReadout,
)


QuantumGraphEdge = tuple[int, int, int, int]

ROUTING_MODES = frozenset(
    {"layer_residual_roots", "initial_residual_plus_parents"}
)


@dataclass(frozen=True)
class QuantaTrace:
    """Autonomous execution with graph-routed inputs and structural gates."""

    boundaries: tuple[torch.Tensor, ...]
    logits: torch.Tensor
    quantum_state: torch.Tensor
    updates: tuple[torch.Tensor, ...]
    contributions: tuple[torch.Tensor, ...]
    gate_logits: tuple[torch.Tensor, ...]
    local_gates: tuple[torch.Tensor, ...]
    effective_gates: tuple[torch.Tensor, ...]
    writer_local_scores: tuple[torch.Tensor, ...]
    writer_context_scores: tuple[torch.Tensor, ...]
    writer_preactivations: tuple[torch.Tensor, ...]
    writer_gate_logits: tuple[torch.Tensor, ...]
    writer_gates: tuple[torch.Tensor, ...]
    attention_contexts: tuple[torch.Tensor, ...]
    routed_inputs: tuple[torch.Tensor, ...]
    target_updates: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class QuantaStep:
    """One cached autoregressive execution step of the closed Q-program."""

    logits: torch.Tensor
    quantum_state: torch.Tensor
    updates: tuple[torch.Tensor, ...]
    contributions: tuple[torch.Tensor, ...]
    gate_logits: tuple[torch.Tensor, ...]
    local_gates: tuple[torch.Tensor, ...]
    effective_gates: tuple[torch.Tensor, ...]
    writer_gate_logits: tuple[torch.Tensor, ...]
    writer_gates: tuple[torch.Tensor, ...]
    attention_contexts: tuple[torch.Tensor, ...]
    routed_inputs: tuple[torch.Tensor, ...]
    caches: tuple[AttentionQuantumCache, ...]


def validate_quantum_graph(
    quantum_counts: Sequence[int], edges: Sequence[QuantumGraphEdge]
) -> tuple[QuantumGraphEdge, ...]:
    """Validate a layer-ordered graph and return canonical integer edges."""

    counts = tuple(int(value) for value in quantum_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("quantum_counts must be nonempty and positive")
    canonical: list[QuantumGraphEdge] = []
    seen: set[QuantumGraphEdge] = set()
    for raw_edge in edges:
        if len(raw_edge) != 4:
            raise ValueError("each edge must be (parent_layer, parent, child_layer, child)")
        parent_layer, parent, child_layer, child = (
            int(value) for value in raw_edge
        )
        if not 0 <= parent_layer < child_layer < len(counts):
            raise ValueError("edges must point strictly to a later host layer")
        if not 0 <= parent < counts[parent_layer]:
            raise ValueError("parent candidate is out of range")
        if not 0 <= child < counts[child_layer]:
            raise ValueError("child candidate is out of range")
        edge = (parent_layer, parent, child_layer, child)
        if edge in seen:
            raise ValueError("duplicate graph edge")
        seen.add(edge)
        canonical.append(edge)
    return tuple(sorted(canonical))


def graph_parents_by_child(
    quantum_counts: Sequence[int], edges: Sequence[QuantumGraphEdge]
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """Return the earlier-layer parents of every candidate."""

    canonical = validate_quantum_graph(quantum_counts, edges)
    parents: list[list[list[tuple[int, int]]]] = [
        [[] for _ in range(int(count))] for count in quantum_counts
    ]
    for parent_layer, parent, child_layer, child in canonical:
        parents[child_layer][child].append((parent_layer, parent))
    return tuple(
        tuple(tuple(sorted(candidate)) for candidate in layer)
        for layer in parents
    )


def _per_quantum_attention_components(
    module: CheapCausalAttentionQuantumLayer,
    normalized: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    query_offsets: torch.Tensor | None = None,
    attention_top_k: int | None = None,
    attention_position_permutation: torch.Tensor | None = None,
    attention_value_permutation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return causal attention context, weights, and values.

    ``query_offsets`` changes only which keys each quantum attends to.  In
    particular, K/V can remain shared, so this is a low-parameter way to give
    otherwise identical quantum inputs distinct causal summaries.

    The optional interventions are evaluation diagnostics. ``attention_top_k``
    masks and renormalizes each causal read. ``attention_position_permutation``
    has shape ``[batch, target_position, source_position]`` and must permute
    only valid causal source positions in each row. ``attention_value_permutation``
    has shape ``[batch]`` and swaps value streams across examples while leaving
    the receiver's queries and keys fixed.
    """
    if query_offsets is None:
        query_offsets = module.query_offsets
    if query_offsets is not None:
        if query_offsets.shape != (module.quantum_count, module.attention_rank):
            raise ValueError(
                "query_offsets must have shape [quantum_count, attention_rank]"
            )
        query_offsets = query_offsets.to(
            dtype=normalized.dtype, device=normalized.device
        )
    if module.share_attention_projections:
        queries = torch.einsum("btqd,rd->btqr", normalized, module.query_weight)
        keys = torch.einsum("btqd,rd->btqr", normalized, module.key_weight)
        values = torch.einsum("btqd,vd->btqv", normalized, module.value_weight)
    else:
        queries = torch.einsum(
            "btqd,qrd->btqr", normalized, module.query_weight
        )
        keys = torch.einsum("btqd,qrd->btqr", normalized, module.key_weight)
        values = torch.einsum(
            "btqd,qvd->btqv", normalized, module.value_weight
        )
    if query_offsets is not None:
        queries = queries + query_offsets[None, None]
    scores = torch.einsum("btqr,bsqr->bqts", queries, keys)
    scores = scores / math.sqrt(module.attention_rank)
    length = normalized.shape[1]
    causal = torch.tril(
        torch.ones((length, length), dtype=torch.bool, device=normalized.device)
    )
    key_valid = attention_mask.to(torch.bool)[:, None, None, :]
    allowed = causal[None, None] & key_valid
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if attention_top_k is not None:
        if int(attention_top_k) <= 0:
            raise ValueError("attention_top_k must be positive when specified")
        top_k = min(int(attention_top_k), int(weights.shape[-1]))
        _, indices = weights.topk(top_k, dim=-1)
        keep = torch.zeros_like(weights, dtype=torch.bool)
        keep.scatter_(-1, indices, True)
        weights = weights * keep
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-24)
    if attention_position_permutation is not None:
        expected = (
            normalized.shape[0],
            normalized.shape[1],
            normalized.shape[1],
        )
        if attention_position_permutation.shape != expected:
            raise ValueError(
                "attention_position_permutation must have shape [batch, length, length]"
            )
        permutation = attention_position_permutation.to(
            dtype=torch.long, device=weights.device
        )
        weights = weights.gather(
            -1,
            permutation[:, None].expand(-1, module.quantum_count, -1, -1),
        )
    if attention_value_permutation is not None:
        if attention_value_permutation.shape != (normalized.shape[0],):
            raise ValueError("attention_value_permutation must have shape [batch]")
        permutation = attention_value_permutation.to(
            dtype=torch.long, device=values.device
        )
        if bool((permutation < 0).any()) or bool((permutation >= len(values)).any()):
            raise ValueError("attention_value_permutation indices are out of range")
        values = values.index_select(0, permutation)
    context = torch.einsum("bqts,bsqv->btqv", weights, values)
    return context, weights, values


def _per_quantum_attention_context(
    module: CheapCausalAttentionQuantumLayer,
    normalized: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    query_offsets: torch.Tensor | None = None,
    attention_top_k: int | None = None,
    attention_position_permutation: torch.Tensor | None = None,
    attention_value_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return causal attention context under an optional frozen intervention."""

    context, _, _ = _per_quantum_attention_components(
        module,
        normalized,
        attention_mask,
        query_offsets=query_offsets,
        attention_top_k=attention_top_k,
        attention_position_permutation=attention_position_permutation,
        attention_value_permutation=attention_value_permutation,
    )
    return context


def _per_quantum_writes_and_local_gates(
    module: CheapCausalAttentionQuantumLayer,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    attention_scale: float,
    hard_gates: bool,
    straight_through_gates: bool,
    local_gate_override: torch.Tensor | None,
    writer_gate_override: torch.Tensor | None,
    writer_local_score_override: torch.Tensor | None,
    attention_top_k: int | None,
    attention_position_permutation: torch.Tensor | None,
    attention_value_permutation: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if inputs.ndim != 4 or inputs.shape[2:] != (
        module.quantum_count,
        module.d_model,
    ):
        raise ValueError("routed inputs must have shape [batch, length, quantum, d]")
    normalized = module.norm(inputs)
    context = _per_quantum_attention_context(
        module,
        normalized,
        attention_mask,
        attention_top_k=attention_top_k,
        attention_position_permutation=attention_position_permutation,
        attention_value_permutation=attention_value_permutation,
    )
    context = float(attention_scale) * context

    logits = module.gate_logits(normalized, context)
    probabilities = torch.sigmoid(logits)
    if local_gate_override is not None:
        if local_gate_override.shape != logits.shape:
            raise ValueError("local gate override must match [batch, length, quantum]")
        local_gates = local_gate_override.to(dtype=inputs.dtype, device=inputs.device)
    elif straight_through_gates:
        hard = (probabilities >= 0.5).to(inputs.dtype)
        local_gates = hard + probabilities - probabilities.detach()
    elif hard_gates:
        local_gates = (probabilities >= 0.5).to(inputs.dtype)
    else:
        local_gates = probabilities
    writer_inputs = normalized.index_select(2, module.writer_owner)
    local = (
        torch.einsum("btwd,wd->btw", writer_inputs, module.local_weight)
        + module.local_bias
    )
    if writer_local_score_override is not None:
        if writer_local_score_override.shape != local.shape:
            raise ValueError(
                "writer_local_score_override must match [batch, length, writer]"
            )
        local = writer_local_score_override.to(dtype=inputs.dtype, device=inputs.device)
    writer_context = context.index_select(2, module.writer_owner)
    contextual_scalar = (
        module.context_scale[None, None] * writer_context
    ).sum(dim=-1)
    preactivations = local + contextual_scalar
    features = _writer_features(module, preactivations, training=module.training)
    if writer_gate_override is None:
        writer_gates = _writer_mask(
            module,
            preactivations,
            features,
            local_gates,
            attention_mask,
            training=module.training,
        )
    else:
        if writer_gate_override.shape != preactivations.shape:
            raise ValueError(
                "writer_gate_override must match [batch, length, writer]"
            )
        writer_gates = writer_gate_override.to(dtype=inputs.dtype, device=inputs.device)
    writer_writes = (
        writer_gates[..., None]
        * features[..., None]
        * torch.nn.functional.normalize(module.output_weight, dim=-1)[None, None]
    )
    writes = writer_writes.new_zeros(
        *writer_writes.shape[:2], module.quantum_count, module.d_model
    ).index_add(2, module.writer_owner, writer_writes)
    if module.attention_output_weight is not None:
        writes = writes + torch.einsum(
            "btqv,qvd->btqd", context, module.attention_output_weight
        )
    return (
        writes,
        logits,
        local_gates,
        local,
        contextual_scalar,
        features,
        writer_gates,
        context,
    )


def _per_quantum_step(
    module: CheapCausalAttentionQuantumLayer,
    inputs: torch.Tensor,
    cache: AttentionQuantumCache | None,
    *,
    attention_scale: float,
    hard_gates: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AttentionQuantumCache,
]:
    if inputs.ndim != 3 or inputs.shape[1:] != (
        module.quantum_count,
        module.d_model,
    ):
        raise ValueError("step routed inputs must have shape [batch, quantum, d]")
    normalized = module.norm(inputs)
    if module.share_attention_projections:
        queries = torch.einsum("bqd,rd->bqr", normalized, module.query_weight)
        keys = torch.einsum("bqd,rd->bqr", normalized, module.key_weight)[:, None]
        values = torch.einsum("bqd,vd->bqv", normalized, module.value_weight)[:, None]
    else:
        queries = torch.einsum("bqd,qrd->bqr", normalized, module.query_weight)
        keys = torch.einsum("bqd,qrd->bqr", normalized, module.key_weight)[:, None]
        values = torch.einsum("bqd,qvd->bqv", normalized, module.value_weight)[:, None]
    if module.query_offsets is not None:
        queries = queries + module.query_offsets[None]
    if cache is None:
        all_keys = keys
        all_values = values
    else:
        if cache.keys.shape[0] != len(inputs):
            raise ValueError("cache and step inputs must have the same batch size")
        all_keys = torch.cat((cache.keys, keys), dim=1)
        all_values = torch.cat((cache.values, values), dim=1)
    scores = torch.einsum("bqr,bsqr->bqs", queries, all_keys)
    scores = scores / math.sqrt(module.attention_rank)
    weights = torch.softmax(scores, dim=-1)
    context = torch.einsum("bqs,bsqv->bqv", weights, all_values)
    context = float(attention_scale) * context

    logits = module.gate_logits(normalized[:, None], context[:, None])[:, 0]
    probabilities = torch.sigmoid(logits)
    local_gates = (
        (probabilities >= 0.5).to(inputs.dtype) if hard_gates else probabilities
    )
    writer_inputs = normalized.index_select(1, module.writer_owner)
    local = (
        torch.einsum("bwd,wd->bw", writer_inputs, module.local_weight)
        + module.local_bias
    )
    writer_context = context.index_select(1, module.writer_owner)
    contextual_scalar = (module.context_scale[None] * writer_context).sum(dim=-1)
    preactivations = local + contextual_scalar
    features = _writer_features(module, preactivations, training=module.training)
    writer_gates = _writer_mask(
        module,
        preactivations,
        features,
        local_gates,
        torch.ones(
            features.shape[:2], dtype=torch.bool, device=features.device
        ),
        training=False,
    )
    writer_writes = (
        writer_gates[..., None]
        * features[..., None]
        * torch.nn.functional.normalize(module.output_weight, dim=-1)[None]
    )
    writes = writer_writes.new_zeros(
        len(inputs), module.quantum_count, module.d_model
    ).index_add(1, module.writer_owner, writer_writes)
    if module.attention_output_weight is not None:
        writes = writes + torch.einsum(
            "bqv,qvd->bqd", context, module.attention_output_weight
        )
    return (
        writes,
        logits,
        local_gates,
        features,
        writer_gates,
        context,
        AttentionQuantumCache(keys=all_keys, values=all_values),
    )


def _frozen_writer_mask(
    module: CheapCausalAttentionQuantumLayer, scores: torch.Tensor, local_gates: torch.Tensor
) -> torch.Tensor:
    """Use stored per-quantum cutoffs for evaluation and autoregressive steps."""
    cutoffs = module.writer_cutoffs.index_select(0, module.writer_owner)
    owners = local_gates.index_select(-1, module.writer_owner)
    return ((scores >= cutoffs) & (scores > 0)).to(scores.dtype) * owners


def _jump_relu(
    preactivations: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    bandwidth: float,
    training: bool,
) -> torch.Tensor:
    """Hard JumpReLU with a sigmoid straight-through threshold gradient."""

    threshold = thresholds.abs().to(
        dtype=preactivations.dtype, device=preactivations.device
    )
    view_shape = (1,) * (preactivations.ndim - 1) + (len(threshold),)
    threshold = threshold.view(view_shape)
    hard = (preactivations > threshold).to(preactivations.dtype)
    if training:
        soft = torch.sigmoid((preactivations - threshold) / float(bandwidth))
        mask = hard + soft - soft.detach()
    else:
        mask = hard
    return preactivations * mask


def _writer_features(
    module: CheapCausalAttentionQuantumLayer,
    preactivations: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    """Apply the configured writer activation without selecting writers."""

    if module.writer_sparsity == "l0_target":
        # The independent support mask below supplies the JumpReLU. Keeping the
        # magnitude path linear avoids applying the same STE twice.
        return preactivations
    if module.writer_activation == "relu":
        return torch.relu(preactivations)
    return _jump_relu(
        preactivations,
        module.writer_jump_threshold,
        bandwidth=module.writer_jump_bandwidth,
        training=training,
    )


def _independent_jumprelu_writer_mask(
    module: CheapCausalAttentionQuantumLayer,
    preactivations: torch.Tensor,
    local_gates: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    """Independently threshold writers, retaining an STE for the L0 objective."""

    threshold = module.writer_jump_threshold.abs().to(
        dtype=preactivations.dtype, device=preactivations.device
    )
    view_shape = (1,) * (preactivations.ndim - 1) + (len(threshold),)
    threshold = threshold.view(view_shape)
    hard = (preactivations > threshold).to(preactivations.dtype)
    if training:
        soft = torch.sigmoid(
            (preactivations - threshold) / float(module.writer_jump_bandwidth)
        )
        support = hard + soft - soft.detach()
    else:
        support = hard
    owners = (
        local_gates.index_select(-1, module.writer_owner).detach() > 0.5
    ).to(preactivations.dtype)
    return support * owners


def _writer_mask(
    module: CheapCausalAttentionQuantumLayer,
    preactivations: torch.Tensor,
    features: torch.Tensor,
    local_gates: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    if module.writer_sparsity == "l0_target":
        return _independent_jumprelu_writer_mask(
            module, preactivations, local_gates, training=training
        )
    return _batch_topk_writer_mask(
        module, features, local_gates, attention_mask, training=training
    )


def _batch_topk_writer_mask(
    module: CheapCausalAttentionQuantumLayer,
    scores: torch.Tensor,
    local_gates: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    """Select a mean k writers per active quantum-position, independently by Q."""
    if not training:
        return _frozen_writer_mask(module, scores, local_gates)
    result = torch.zeros_like(scores)
    valid = attention_mask.to(torch.bool)
    for quantum in range(module.quantum_count):
        indices = torch.nonzero(module.writer_owner == quantum, as_tuple=False).flatten()
        active = valid & (local_gates[..., quantum].detach() > 0.5)
        active_count = int(active.sum())
        if active_count == 0:
            continue
        flat = scores[..., indices][active].reshape(-1)
        selected = min(flat.numel(), max(1, int(round(module.writer_average_k * active_count))))
        values, _ = torch.topk(flat, selected)
        cutoff = values[-1].detach()
        module.writer_cutoffs[quantum].mul_(0.95).add_(cutoff * 0.05) if torch.isfinite(module.writer_cutoffs[quantum]) else module.writer_cutoffs.__setitem__(quantum, cutoff)
        selected_mask = (scores[..., indices] >= cutoff) & active[..., None] & (scores[..., indices] > 0)
        result[..., indices] = selected_mask.to(scores.dtype)
    return result


def _routed_layer_inputs(
    *,
    layer: int,
    quantum_count: int,
    shared_hidden: torch.Tensor,
    initial_hidden: torch.Tensor,
    parents: tuple[tuple[tuple[int, int], ...], ...],
    contributions: Sequence[torch.Tensor],
    edge_scales: Mapping[QuantumGraphEdge, float | torch.Tensor] | None,
    edge_replacements: Mapping[QuantumGraphEdge, torch.Tensor] | None,
    routing_mode: str,
) -> torch.Tensor:
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"unsupported Q routing mode: {routing_mode!r}")
    candidate_inputs = []
    for child, candidate_parents in enumerate(parents[layer]):
        if routing_mode == "layer_residual_roots":
            routed = (
                torch.zeros_like(shared_hidden)
                if candidate_parents
                else shared_hidden
            )
        else:
            routed = initial_hidden
        for parent_layer, parent in candidate_parents:
            edge = (parent_layer, parent, layer, child)
            factual = contributions[parent_layer][:, :, parent]
            replacement = factual
            if edge_replacements is not None and edge in edge_replacements:
                candidate_replacement = edge_replacements[edge]
                if candidate_replacement.shape != factual.shape:
                    raise ValueError("edge replacement must match its parent message")
                replacement = candidate_replacement.to(
                    dtype=factual.dtype, device=factual.device
                )
            scale = 1.0 if edge_scales is None else edge_scales.get(edge, 1.0)
            routed = routed + scale * replacement
        candidate_inputs.append(routed)
    return torch.stack(candidate_inputs, dim=2)


def forward_quanta(
    source: DecoderTransformerLM,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    *,
    collect_teacher_targets: bool = False,
    attention_scale: float = 1.0,
    hard_gates: bool = False,
    straight_through_gates: bool = False,
    detach_gates_from_writes: bool = False,
    scales: Sequence[torch.Tensor | None] | None = None,
    local_gate_overrides: Sequence[torch.Tensor | None] | None = None,
    writer_gate_overrides: Sequence[torch.Tensor | None] | None = None,
    writer_local_score_overrides: Sequence[torch.Tensor | None] | None = None,
    edge_scales: Mapping[QuantumGraphEdge, float | torch.Tensor] | None = None,
    edge_replacements: Mapping[QuantumGraphEdge, torch.Tensor] | None = None,
    attention_top_k: int | None = None,
    attention_position_permutation: torch.Tensor | None = None,
    attention_value_permutation: torch.Tensor | None = None,
    routing_mode: str = "layer_residual_roots",
    enforce_parent_gate_closure: bool = True,
) -> QuantaTrace:
    """Execute the Q-program under one explicit input-routing mode.

    ``layer_residual_roots`` preserves the original carrier: roots read the
    ordinary incoming residual and graph children read only named messages.
    ``initial_residual_plus_parents`` gives every quantum the initial embedding
    residual and gives graph children their named messages as additional input.
    In both modes, edge interventions leave the parent's residual contribution
    intact while changing only the child's routed input.  By default, a graph
    edge additionally imposes parent-gate availability on the child.  The
    global-message graph uses ``enforce_parent_gate_closure=False`` so that
    topology constrains information access but not local activation.
    """

    if len(modules) != len(source.layers) or len(existences) != len(modules):
        raise ValueError("source, modules, and existences must have the same layers")
    quantum_counts = tuple(module.quantum_count for module in modules)
    canonical_edges = validate_quantum_graph(quantum_counts, edges)
    parents = graph_parents_by_child(quantum_counts, canonical_edges)
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"unsupported Q routing mode: {routing_mode!r}")
    if scales is None:
        scales = (None,) * len(modules)
    if local_gate_overrides is None:
        local_gate_overrides = (None,) * len(modules)
    if writer_gate_overrides is None:
        writer_gate_overrides = (None,) * len(modules)
    if writer_local_score_overrides is None:
        writer_local_score_overrides = (None,) * len(modules)
    if (
        len(scales) != len(modules)
        or len(local_gate_overrides) != len(modules)
        or len(writer_gate_overrides) != len(modules)
        or len(writer_local_score_overrides) != len(modules)
    ):
        raise ValueError("scales and overrides must match the module layers")

    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)[None].expand(batch, -1)
    embedding = source.token_embedding(input_ids) + source.position_embedding(positions)
    hidden = embedding
    quantum_state = torch.zeros_like(hidden)
    causal_mask = torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=input_ids.device),
        diagonal=1,
    )
    padding_mask = attention_mask == 0
    boundaries = [hidden]
    updates: list[torch.Tensor] = []
    contributions: list[torch.Tensor] = []
    logits_by_layer: list[torch.Tensor] = []
    local_by_layer: list[torch.Tensor] = []
    effective_by_layer: list[torch.Tensor] = []
    writer_local_scores_by_layer: list[torch.Tensor] = []
    writer_context_scores_by_layer: list[torch.Tensor] = []
    writer_preactivations_by_layer: list[torch.Tensor] = []
    writer_logits_by_layer: list[torch.Tensor] = []
    writer_gates_by_layer: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    routed_inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for layer, (
        source_layer,
        module,
        existence,
        scale,
        gate_override,
        writer_gate_override,
        writer_local_override,
    ) in enumerate(
        zip(
            source.layers,
            modules,
            existences,
            scales,
            local_gate_overrides,
            writer_gate_overrides,
            writer_local_score_overrides,
        )
    ):
        if collect_teacher_targets:
            with torch.no_grad():
                teacher_input = hidden.detach()
                teacher_after_attention = source_layer.attention_write(
                    teacher_input,
                    causal_mask=causal_mask,
                    padding_mask=padding_mask,
                )
                teacher_after = source_layer.mlp_write(teacher_after_attention)
                targets.append((teacher_after - teacher_input).detach())
        inputs = _routed_layer_inputs(
            layer=layer,
            quantum_count=module.quantum_count,
            shared_hidden=hidden,
            initial_hidden=embedding,
            parents=parents,
            contributions=contributions,
            edge_scales=edge_scales,
            edge_replacements=edge_replacements,
            routing_mode=routing_mode,
        )
        (
            writes,
            logits,
            local_gates,
            writer_local_scores,
            writer_context_scores,
            writer_logits,
            writer_gates,
            context,
        ) = _per_quantum_writes_and_local_gates(
            module,
            inputs,
            attention_mask,
            attention_scale=attention_scale,
            hard_gates=hard_gates,
            straight_through_gates=straight_through_gates,
            local_gate_override=gate_override,
            writer_gate_override=writer_gate_override,
            writer_local_score_override=writer_local_override,
            attention_top_k=attention_top_k,
            attention_position_permutation=attention_position_permutation,
            attention_value_permutation=attention_value_permutation,
        )
        if enforce_parent_gate_closure:
            effective_candidates = []
            for candidate, candidate_parents in enumerate(parents[layer]):
                availability = torch.ones_like(local_gates[..., candidate])
                for parent_layer, parent in candidate_parents:
                    availability = availability * effective_by_layer[parent_layer][
                        ..., parent
                    ]
                effective_candidates.append(
                    local_gates[..., candidate] * availability
                )
            effective_gates = torch.stack(effective_candidates, dim=-1)
        else:
            effective_gates = local_gates
        if existence.shape != (batch, module.quantum_count):
            raise ValueError("existence must have shape [batch, quantum]")
        execution_gates = (
            effective_gates.detach()
            if detach_gates_from_writes
            else effective_gates
        )
        layer_contributions = (
            existence[:, None, :, None]
            * execution_gates[..., None]
            * writes
        )
        if scale is not None:
            if scale.shape != (module.quantum_count,):
                raise ValueError("quantum scale must have shape [quantum]")
            layer_contributions = (
                layer_contributions * scale[None, None, :, None]
            )
        update = layer_contributions.sum(dim=2)
        hidden = hidden + update
        quantum_state = quantum_state + update
        boundaries.append(hidden)
        updates.append(update)
        contributions.append(layer_contributions)
        logits_by_layer.append(logits)
        local_by_layer.append(local_gates)
        effective_by_layer.append(effective_gates)
        writer_local_scores_by_layer.append(writer_local_scores)
        writer_context_scores_by_layer.append(writer_context_scores)
        writer_preactivations_by_layer.append(
            writer_local_scores + writer_context_scores
        )
        writer_logits_by_layer.append(writer_logits)
        writer_gates_by_layer.append(writer_gates)
        contexts.append(context)
        routed_inputs.append(inputs)

    return QuantaTrace(
        boundaries=tuple(boundaries),
        logits=readout(hidden),
        quantum_state=quantum_state,
        updates=tuple(updates),
        contributions=tuple(contributions),
        gate_logits=tuple(logits_by_layer),
        local_gates=tuple(local_by_layer),
        effective_gates=tuple(effective_by_layer),
        writer_local_scores=tuple(writer_local_scores_by_layer),
        writer_context_scores=tuple(writer_context_scores_by_layer),
        writer_preactivations=tuple(writer_preactivations_by_layer),
        writer_gate_logits=tuple(writer_logits_by_layer),
        writer_gates=tuple(writer_gates_by_layer),
        attention_contexts=tuple(contexts),
        routed_inputs=tuple(routed_inputs),
        target_updates=tuple(targets),
    )


def forward_quanta_step(
    source: DecoderTransformerLM,
    readout: QuantumReadout,
    modules: Sequence[CheapCausalAttentionQuantumLayer],
    input_ids: torch.Tensor,
    position: int,
    existences: Sequence[torch.Tensor],
    edges: Sequence[QuantumGraphEdge],
    caches: Sequence[AttentionQuantumCache | None] | None = None,
    *,
    attention_scale: float = 1.0,
    hard_gates: bool = True,
    routing_mode: str = "layer_residual_roots",
    enforce_parent_gate_closure: bool = True,
) -> QuantaStep:
    """Append one token to every graph-routed quantum attention cache."""

    if input_ids.ndim != 1:
        raise ValueError("step input_ids must have shape [batch]")
    if not 0 <= int(position) < source.position_embedding.num_embeddings:
        raise ValueError("position is outside the source embedding table")
    if len(modules) != len(source.layers) or len(existences) != len(modules):
        raise ValueError("source, modules, and existences must have the same layers")
    if caches is None:
        caches = (None,) * len(modules)
    if len(caches) != len(modules):
        raise ValueError("one optional cache is required per module")
    quantum_counts = tuple(module.quantum_count for module in modules)
    canonical_edges = validate_quantum_graph(quantum_counts, edges)
    parents = graph_parents_by_child(quantum_counts, canonical_edges)
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"unsupported Q routing mode: {routing_mode!r}")
    positions = torch.full_like(input_ids, int(position))
    embedding = source.token_embedding(input_ids) + source.position_embedding(positions)
    hidden = embedding
    quantum_state = torch.zeros_like(hidden)
    updates: list[torch.Tensor] = []
    contributions: list[torch.Tensor] = []
    logits_by_layer: list[torch.Tensor] = []
    local_by_layer: list[torch.Tensor] = []
    effective_by_layer: list[torch.Tensor] = []
    writer_logits_by_layer: list[torch.Tensor] = []
    writer_gates_by_layer: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    routed_inputs: list[torch.Tensor] = []
    next_caches: list[AttentionQuantumCache] = []
    for layer, (module, existence, cache) in enumerate(
        zip(modules, existences, caches)
    ):
        candidate_inputs = []
        for candidate, candidate_parents in enumerate(parents[layer]):
            if routing_mode == "layer_residual_roots":
                routed = torch.zeros_like(hidden) if candidate_parents else hidden
            else:
                routed = embedding
            for parent_layer, parent in candidate_parents:
                routed = routed + contributions[parent_layer][:, parent]
            candidate_inputs.append(routed)
        inputs = torch.stack(candidate_inputs, dim=1)
        (
            writes,
            logits,
            local_gates,
            writer_logits,
            writer_gates,
            context,
            next_cache,
        ) = _per_quantum_step(
            module,
            inputs,
            cache,
            attention_scale=attention_scale,
            hard_gates=hard_gates,
        )
        if enforce_parent_gate_closure:
            effective_candidates = []
            for candidate, candidate_parents in enumerate(parents[layer]):
                availability = torch.ones_like(local_gates[:, candidate])
                for parent_layer, parent in candidate_parents:
                    availability = availability * effective_by_layer[parent_layer][
                        :, parent
                    ]
                effective_candidates.append(local_gates[:, candidate] * availability)
            effective_gates = torch.stack(effective_candidates, dim=-1)
        else:
            effective_gates = local_gates
        if existence.shape != (len(input_ids), module.quantum_count):
            raise ValueError("step existence must have shape [batch, quantum]")
        layer_contributions = (
            existence[:, :, None] * effective_gates[:, :, None] * writes
        )
        update = layer_contributions.sum(dim=1)
        hidden = hidden + update
        quantum_state = quantum_state + update
        updates.append(update)
        contributions.append(layer_contributions)
        logits_by_layer.append(logits)
        local_by_layer.append(local_gates)
        effective_by_layer.append(effective_gates)
        writer_logits_by_layer.append(writer_logits)
        writer_gates_by_layer.append(writer_gates)
        contexts.append(context)
        routed_inputs.append(inputs)
        next_caches.append(next_cache)
    return QuantaStep(
        logits=readout(hidden),
        quantum_state=quantum_state,
        updates=tuple(updates),
        contributions=tuple(contributions),
        gate_logits=tuple(logits_by_layer),
        local_gates=tuple(local_by_layer),
        effective_gates=tuple(effective_by_layer),
        writer_gate_logits=tuple(writer_logits_by_layer),
        writer_gates=tuple(writer_gates_by_layer),
        attention_contexts=tuple(contexts),
        routed_inputs=tuple(routed_inputs),
        caches=tuple(next_caches),
    )
