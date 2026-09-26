from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.quanta_discovery.qgraph import (
    _per_quantum_attention_components,
    _per_quantum_attention_context,
    _independent_jumprelu_writer_mask,
    _jump_relu,
    _writer_features,
    forward_quanta,
    forward_quanta_step,
    validate_quantum_graph,
)
from quanta.experiments.quanta_discovery.qmodel import (
    CheapCausalAttentionQuantumLayer,
    QuantumReadout,
)


def _parts() -> tuple[
    DecoderTransformerLM,
    tuple[CheapCausalAttentionQuantumLayer, ...],
    QuantumReadout,
]:
    source = DecoderTransformerLM(
        vocab_size=13,
        max_seq_len=6,
        d_model=8,
        n_layers=2,
        n_heads=1,
        dropout=0.0,
        pad_id=0,
        mlp_ratio=2.0,
    )
    modules = tuple(
        CheapCausalAttentionQuantumLayer(
            8,
            2,
            attention_rank=1,
            attention_value_dim=2,
            writer_counts=(1, 2),
        )
        for _ in source.layers
    )
    return source, modules, QuantumReadout(8, 13)


def test_effective_gate_is_downward_closed() -> None:
    torch.manual_seed(0)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)
    parent_gates = torch.ones(1, 3, 2)
    parent_gates[..., 0] = 0.0

    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        ((0, 0, 1, 1),),
        local_gate_overrides=(parent_gates, torch.ones(1, 3, 2)),
    )

    assert bool(torch.all(trace.local_gates[1][..., 1] == 1.0))
    assert bool(torch.all(trace.effective_gates[1][..., 1] == 0.0))


def test_global_message_edge_does_not_close_child_activation() -> None:
    torch.manual_seed(0)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)
    parent_gates = torch.ones(1, 3, 2)
    parent_gates[..., 0] = 0.0

    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        ((0, 0, 1, 1),),
        local_gate_overrides=(parent_gates, torch.ones(1, 3, 2)),
        routing_mode="initial_residual_plus_parents",
        enforce_parent_gate_closure=False,
    )

    assert bool(torch.all(trace.local_gates[1][..., 1] == 1.0))
    assert bool(torch.all(trace.effective_gates[1][..., 1] == 1.0))
    torch.testing.assert_close(
        trace.routed_inputs[1][:, :, 1], trace.boundaries[0]
    )


def test_detached_execution_gates_only_block_reconstruction_gradients() -> None:
    torch.manual_seed(1)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    detached = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (),
        straight_through_gates=True,
        detach_gates_from_writes=True,
    )
    sum(value.square().sum() for value in detached.updates).backward()

    assert detached.effective_gates[0].requires_grad
    assert modules[0].gate_weight.grad is None
    assert modules[0].gate_bias.grad is None
    assert modules[0].output_weight.grad is not None


def test_query_offsets_specialize_shared_attention_queries() -> None:
    torch.manual_seed(11)
    module = CheapCausalAttentionQuantumLayer(
        8,
        2,
        attention_rank=2,
        attention_value_dim=2,
        share_attention_projections=True,
        attention_query_offsets=True,
    )
    normalized = torch.randn(1, 3, 2, 8)
    mask = torch.ones(1, 3, dtype=torch.long)

    baseline = _per_quantum_attention_context(module, normalized, mask)
    zero_offset = _per_quantum_attention_context(
        module,
        normalized,
        mask,
        query_offsets=torch.zeros(2, 2),
    )
    with torch.no_grad():
        module.query_offsets.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    offset = _per_quantum_attention_context(module, normalized, mask)

    torch.testing.assert_close(zero_offset, baseline)
    assert not torch.allclose(offset[:, :, 0], offset[:, :, 1])


def test_edge_child_reads_only_parent_message() -> None:
    torch.manual_seed(2)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)
    overrides = tuple(torch.ones(1, 3, 2) for _ in modules)
    edge = (0, 0, 1, 1)

    factual = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (edge,),
        local_gate_overrides=overrides,
    )
    deleted = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (edge,),
        local_gate_overrides=overrides,
        edge_scales={edge: 0.0},
    )

    torch.testing.assert_close(deleted.contributions[0], factual.contributions[0])
    torch.testing.assert_close(
        deleted.routed_inputs[1][:, :, 1],
        torch.zeros_like(deleted.routed_inputs[1][:, :, 1]),
    )
    torch.testing.assert_close(
        factual.routed_inputs[1][:, :, 1],
        factual.contributions[0][:, :, 0],
    )
    assert not torch.allclose(
        factual.routed_inputs[1][:, :, 1], factual.boundaries[1]
    )


def test_multiple_parent_messages_are_summed_without_a_residual_bypass() -> None:
    torch.manual_seed(3)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)
    overrides = tuple(torch.ones(1, 3, 2) for _ in modules)
    edges = ((0, 0, 1, 1), (0, 1, 1, 1))

    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        edges,
        local_gate_overrides=overrides,
    )

    torch.testing.assert_close(
        trace.routed_inputs[1][:, :, 1],
        trace.contributions[0][:, :, 0] + trace.contributions[0][:, :, 1],
    )


def test_initial_residual_plus_parents_routes_only_declared_messages() -> None:
    torch.manual_seed(12)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)
    edge = (0, 0, 1, 1)

    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (edge,),
        local_gate_overrides=tuple(torch.ones(1, 3, 2) for _ in modules),
        routing_mode="initial_residual_plus_parents",
    )

    initial = trace.boundaries[0]
    torch.testing.assert_close(trace.routed_inputs[1][:, :, 0], initial)
    torch.testing.assert_close(
        trace.routed_inputs[1][:, :, 1], initial + trace.contributions[0][:, :, 0]
    )


def test_writers_have_independent_binary_execution_gates() -> None:
    torch.manual_seed(4)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    trace = forward_quanta(
        source, readout, modules, inputs, mask, existences, (), hard_gates=True
    )

    for layer, module in enumerate(modules):
        assert trace.writer_gates[layer].shape == (1, 3, module.writer_count)
        assert set(trace.writer_gates[layer].unique().tolist()) <= {0.0, 1.0}


def test_attention_top_k_and_trace_scalar_decomposition() -> None:
    torch.manual_seed(13)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    normalized = modules[0].norm(source.token_embedding(inputs))[:, :, None, :]
    normalized = normalized.expand(-1, -1, modules[0].quantum_count, -1)
    _, weights, _ = _per_quantum_attention_components(
        modules[0], normalized, mask, attention_top_k=1
    )
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    assert torch.all((weights > 0).sum(dim=-1) == 1)

    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (),
        hard_gates=True,
        attention_top_k=1,
    )
    for layer in range(len(modules)):
        torch.testing.assert_close(
            trace.writer_local_scores[layer] + trace.writer_context_scores[layer],
            trace.writer_preactivations[layer],
        )


def test_jump_relu_is_hard_in_forward_with_surrogate_input_gradient() -> None:
    values = torch.tensor([[-0.2, 0.2, 0.8]], requires_grad=True)
    thresholds = torch.tensor([0.1, 0.3, 0.5])

    output = _jump_relu(values, thresholds, bandwidth=0.1, training=True)

    torch.testing.assert_close(output.detach(), torch.tensor([[-0.0, 0.0, 0.8]]))
    output.sum().backward()
    assert values.grad is not None
    assert bool(torch.isfinite(values.grad).all())
    assert float(values.grad[0, 1]) != 0.0


def test_relu_batchtopk_arm_removes_the_jump_threshold() -> None:
    module = CheapCausalAttentionQuantumLayer(
        8,
        1,
        writer_counts=(2,),
        writer_activation="relu",
        writer_sparsity="batch_topk",
        writer_jump_threshold=0.1,
    )
    values = torch.tensor([[0.05, -0.05]])

    features = _writer_features(module, values, training=True)

    torch.testing.assert_close(features, torch.tensor([[0.05, 0.0]]))


def test_jumprelu_l0_arm_has_independent_hard_support_and_surrogate_gradient() -> None:
    module = CheapCausalAttentionQuantumLayer(
        8,
        1,
        writer_counts=(2,),
        writer_activation="jumprelu",
        writer_sparsity="l0_target",
        writer_jump_threshold=0.1,
        writer_jump_bandwidth=0.1,
    )
    values = torch.tensor([[0.05, 0.2]], requires_grad=True)
    local_gates = torch.ones(1, 1)

    mask = _independent_jumprelu_writer_mask(
        module, values, local_gates, training=True
    )

    torch.testing.assert_close(mask.detach(), torch.tensor([[0.0, 1.0]]))
    mask.sum().backward()
    assert values.grad is not None
    assert float(values.grad[0, 0]) != 0.0


def test_attention_context_cannot_write_without_an_active_writer() -> None:
    torch.manual_seed(5)
    source, modules, readout = _parts()
    for module in modules:
        module.writer_jump_threshold.data.fill_(1.0e6)
        module.writer_cutoffs.fill_(0.0)
        module.eval()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    trace = forward_quanta(
        source, readout, modules, inputs, mask, existences, (), hard_gates=True
    )

    assert all(bool(torch.all(write == 0)) for write in trace.contributions)


def test_diagnostic_attention_residual_can_write_without_an_active_writer() -> None:
    torch.manual_seed(5)
    source, _, readout = _parts()
    modules = tuple(
        CheapCausalAttentionQuantumLayer(
            8,
            2,
            attention_rank=1,
            attention_value_dim=2,
            attention_direct_residual=True,
            writer_counts=(1, 2),
        )
        for _ in source.layers
    )
    for module in modules:
        module.writer_jump_threshold.data.fill_(1.0e6)
        module.writer_cutoffs.fill_(0.0)
        module.eval()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    trace = forward_quanta(
        source, readout, modules, inputs, mask, existences, (), hard_gates=True
    )

    assert any(bool(torch.any(write != 0)) for write in trace.contributions)


def test_execution_is_autonomous_without_teacher_targets() -> None:
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    with patch.object(
        source.layers[0], "attention_write", side_effect=AssertionError
    ), patch.object(source.layers[0], "mlp_write", side_effect=AssertionError):
        trace = forward_quanta(
            source, readout, modules, inputs, mask, existences, ()
        )

    assert trace.target_updates == ()


def test_cached_execution_matches_full_causal_execution() -> None:
    torch.manual_seed(3)
    source, modules, readout = _parts()
    inputs = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.rand(2, 2) for _ in modules)
    edge = (0, 0, 1, 1)
    full = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (edge,),
        hard_gates=True,
        routing_mode="initial_residual_plus_parents",
    )

    caches = None
    steps = []
    for position in range(inputs.shape[1]):
        step = forward_quanta_step(
            source,
            readout,
            modules,
            inputs[:, position],
            position,
            existences,
            (edge,),
            caches,
            hard_gates=True,
            routing_mode="initial_residual_plus_parents",
        )
        caches = step.caches
        steps.append(step)

    torch.testing.assert_close(full.logits, torch.stack([step.logits for step in steps], dim=1))
    for layer in range(len(modules)):
        torch.testing.assert_close(
            full.contributions[layer],
            torch.stack([step.contributions[layer] for step in steps], dim=1),
        )


def test_shared_qkv_cached_execution_matches_full_causal_execution() -> None:
    torch.manual_seed(11)
    source, _, readout = _parts()
    modules = tuple(
        CheapCausalAttentionQuantumLayer(
            8,
            2,
            attention_rank=2,
            attention_value_dim=3,
            attention_direct_residual=True,
            share_attention_projections=True,
            writer_counts=(1, 2),
        )
        for _ in source.layers
    )
    inputs = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.rand(2, 2) for _ in modules)
    edges = ((0, 0, 1, 1),)
    full = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        edges,
        hard_gates=True,
        routing_mode="initial_residual_plus_parents",
    )

    caches = None
    steps = []
    for position in range(inputs.shape[1]):
        step = forward_quanta_step(
            source,
            readout,
            modules,
            inputs[:, position],
            position,
            existences,
            edges,
            caches,
            hard_gates=True,
            routing_mode="initial_residual_plus_parents",
        )
        caches = step.caches
        steps.append(step)

    torch.testing.assert_close(
        full.logits, torch.stack([step.logits for step in steps], dim=1)
    )


def test_graph_validation_rejects_backward_and_duplicate_edges() -> None:
    assert validate_quantum_graph((2, 3), ((0, 1, 1, 2),)) == ((0, 1, 1, 2),)
    with pytest.raises(ValueError, match="strictly"):
        validate_quantum_graph((2, 3), ((1, 1, 0, 1),))
    with pytest.raises(ValueError, match="duplicate"):
        validate_quantum_graph((2, 3), ((0, 1, 1, 2), (0, 1, 1, 2)))
