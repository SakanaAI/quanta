import torch

from quanta.experiments.number_naming.model import DecoderTransformerLM
from scripts.analyze_quanta_steering import AlignmentAccumulator, BlockAblatedTransformer


def test_block_ablation_wrapper_without_skips_preserves_logits() -> None:
    model = DecoderTransformerLM(
        vocab_size=8,
        max_seq_len=5,
        d_model=4,
        n_layers=2,
        n_heads=1,
        dropout=0.0,
        pad_id=0,
        mlp_ratio=1.0,
    ).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    expected = model(input_ids, attention_mask)
    observed = BlockAblatedTransformer(model, set())(input_ids, attention_mask)

    assert torch.allclose(observed, expected)


def test_alignment_accumulator_recovers_exact_depth_correspondence() -> None:
    accumulator = AlignmentAccumulator(depth=2)
    targets = torch.tensor(
        [[[[1.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 3.0]]]]
    )
    accumulator.add(targets, targets, torch.tensor([[True, True]]))

    summary = accumulator.summary()

    assert summary["diagonal_mean_mse"] == 0.0
    assert summary["correct_depth_is_best_fraction"] == 1.0
    assert [row["best_matching_q_depth"] for row in summary["per_depth"]] == [1, 2]
