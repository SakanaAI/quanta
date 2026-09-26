from __future__ import annotations

from unittest.mock import patch

import torch

from quanta.experiments.number_naming.model import DecoderTransformerLM
from quanta.experiments.quanta_discovery.qmodel import (
    CheapCausalAttentionQuantumLayer,
    QuantumReadout,
)
from quanta.experiments.quanta_discovery.qgraph import (
    forward_quanta,
)
from quanta.experiments.quanta_discovery.capacity import (
    fixed_qmodel_parameter_count,
)


def _model() -> DecoderTransformerLM:
    return DecoderTransformerLM(
        vocab_size=13,
        max_seq_len=6,
        d_model=8,
        n_layers=2,
        n_heads=1,
        dropout=0.0,
        pad_id=0,
        mlp_ratio=2.0,
    )


def test_parameter_accounting_matches_ragged_module_storage() -> None:
    layer = CheapCausalAttentionQuantumLayer(
        8,
        3,
        attention_rank=2,
        attention_value_dim=3,
        writer_counts=[1, 3, 2],
    )
    readout = QuantumReadout(8, 13)
    fixed, per_writer = fixed_qmodel_parameter_count(
        [3],
        d_model=8,
        vocabulary_size=13,
        attention_rank=2,
        attention_value_dim=3,
        embedding_parameter_count=17,
    )

    measured = 17 + sum(parameter.numel() for parameter in layer.parameters())
    measured += sum(parameter.numel() for parameter in readout.parameters())

    assert per_writer == layer.parameters_per_writer
    assert measured == fixed + 6 * per_writer


def test_writer_parameter_count_excludes_a_separate_writer_gate() -> None:
    layer = CheapCausalAttentionQuantumLayer(
        8,
        2,
        attention_rank=2,
        attention_value_dim=3,
        writer_counts=[1, 2],
    )
    fixed, per_writer = fixed_qmodel_parameter_count(
        [2],
        d_model=8,
        vocabulary_size=13,
        attention_rank=2,
        attention_value_dim=3,
        embedding_parameter_count=17,
    )
    readout = QuantumReadout(8, 13)

    measured = 17 + sum(parameter.numel() for parameter in layer.parameters())
    measured += sum(parameter.numel() for parameter in readout.parameters())

    assert per_writer == layer.parameters_per_writer
    assert measured == fixed + 3 * per_writer


def test_parameter_accounting_includes_optional_attention_residual() -> None:
    layer = CheapCausalAttentionQuantumLayer(
        8,
        2,
        attention_rank=2,
        attention_value_dim=3,
        attention_direct_residual=True,
        writer_counts=[1, 2],
    )
    fixed, per_writer = fixed_qmodel_parameter_count(
        [2],
        d_model=8,
        vocabulary_size=13,
        attention_rank=2,
        attention_value_dim=3,
        embedding_parameter_count=17,
        attention_direct_residual=True,
    )
    readout = QuantumReadout(8, 13)
    measured = 17 + sum(parameter.numel() for parameter in layer.parameters())
    measured += sum(parameter.numel() for parameter in readout.parameters())

    assert measured == fixed + 3 * per_writer


def test_parameter_accounting_supports_shared_qkv_per_layer() -> None:
    layer = CheapCausalAttentionQuantumLayer(
        8,
        3,
        attention_rank=2,
        attention_value_dim=3,
        attention_direct_residual=True,
        share_attention_projections=True,
        writer_counts=[1, 3, 2],
    )
    fixed, per_writer = fixed_qmodel_parameter_count(
        [3],
        d_model=8,
        vocabulary_size=13,
        attention_rank=2,
        attention_value_dim=3,
        embedding_parameter_count=17,
        attention_direct_residual=True,
        share_attention_projections=True,
    )
    readout = QuantumReadout(8, 13)
    measured = 17 + sum(parameter.numel() for parameter in layer.parameters())
    measured += sum(parameter.numel() for parameter in readout.parameters())

    assert layer.query_weight.shape == (2, 8)
    assert layer.key_weight.shape == (2, 8)
    assert layer.value_weight.shape == (3, 8)
    assert measured == fixed + 6 * per_writer


def test_parameter_accounting_includes_quantum_query_offsets() -> None:
    layer = CheapCausalAttentionQuantumLayer(
        8,
        3,
        attention_rank=2,
        attention_value_dim=3,
        share_attention_projections=True,
        attention_query_offsets=True,
        writer_counts=[1, 3, 2],
    )
    fixed, per_writer = fixed_qmodel_parameter_count(
        [3],
        d_model=8,
        vocabulary_size=13,
        attention_rank=2,
        attention_value_dim=3,
        embedding_parameter_count=17,
        share_attention_projections=True,
        attention_query_offsets=True,
    )
    readout = QuantumReadout(8, 13)
    measured = 17 + sum(parameter.numel() for parameter in layer.parameters())
    measured += sum(parameter.numel() for parameter in readout.parameters())

    assert layer.query_offsets is not None
    assert layer.query_offsets.shape == (3, 2)
    assert measured == fixed + 6 * per_writer


def test_autonomous_execution_does_not_call_source_blocks() -> None:
    torch.manual_seed(2)
    source = _model()
    modules = tuple(
        CheapCausalAttentionQuantumLayer(8, 2) for _ in source.layers
    )
    readout = QuantumReadout(8, 13)
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    with patch.object(
        source.layers[0], "attention_write", side_effect=AssertionError
    ), patch.object(source.layers[0], "mlp_write", side_effect=AssertionError):
        trace = forward_quanta(
            source, readout, modules, inputs, mask, existences, ()
        )

    assert trace.logits.shape == (1, 3, 13)
    assert trace.target_updates == ()


def test_readout_uses_reconstructed_residual_with_source_compatible_head() -> None:
    torch.manual_seed(3)
    source = _model()
    modules = tuple(
        CheapCausalAttentionQuantumLayer(8, 2) for _ in source.layers
    )
    readout = QuantumReadout(8, 13)
    readout.copy_source_readout(source)
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(2, 2) for _ in modules)
    scales = tuple(torch.zeros(2) for _ in modules)

    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (),
        scales=scales,
        hard_gates=True,
    )

    torch.testing.assert_close(trace.quantum_state, torch.zeros_like(trace.quantum_state))
    embedded = (
        source.token_embedding(inputs)
        + source.position_embedding(torch.arange(inputs.shape[1]))[None]
    )
    torch.testing.assert_close(trace.logits, source.head(source.final_norm(embedded)))
    assert not torch.allclose(trace.logits[0, 0], trace.logits[1, 0])


def test_teacher_target_is_complete_source_block_write() -> None:
    torch.manual_seed(4)
    source = _model()
    modules = tuple(
        CheapCausalAttentionQuantumLayer(8, 2) for _ in source.layers
    )
    readout = QuantumReadout(8, 13)
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    existences = tuple(torch.ones(1, 2) for _ in modules)

    trace = forward_quanta(
        source,
        readout,
        modules,
        inputs,
        mask,
        existences,
        (),
        collect_teacher_targets=True,
    )
    hidden = trace.boundaries[0]
    causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    after_attention = source.layers[0].attention_write(
        hidden, causal_mask=causal_mask, padding_mask=torch.zeros_like(mask).bool()
    )
    expected = source.layers[0].mlp_write(after_attention) - hidden

    torch.testing.assert_close(trace.target_updates[0], expected)
