from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pytest
from pathlib import Path

from scripts import plot_qmodel_recovery_figure
from scripts import visualize_quanta
from quanta.experiments.quanta_discovery.visualization import (
    normalized_cumulative_priority,
    load_dynamics_diagnostics,
    writer_box_layout,
    write_dynamics_figures,
)


def test_canonical_tokenizer_matches_saved_number_naming_vocabulary() -> None:
    tokenizer = visualize_quanta._canonical_tokenizer()

    assert tokenizer.vocab_size == 44
    assert tokenizer.encode(132, "one hundred thirty two", max_seq_len=20).target_ids
    assert visualize_quanta._prompt_ids(tokenizer, 132) == [
        tokenizer.bos_id,
        tokenizer.token_to_id["<D1>"],
        tokenizer.token_to_id["<D3>"],
        tokenizer.token_to_id["<D2>"],
        tokenizer.sep_id,
    ]


def test_layout_has_one_node_per_quantum() -> None:
    positions = visualize_quanta._positions([9, 10, 7])

    assert len(positions) == 26
    assert positions[(0, 0)][0] == 0.36
    assert positions[(2, 6)][0] == pytest.approx(2.68)


def test_blend_respects_endpoints() -> None:
    assert visualize_quanta._blend("#ffffff", "#e5821f", 0.0) == "#ffffff"
    assert visualize_quanta._blend("#ffffff", "#e5821f", 1.0) == "#e5821f"
    assert visualize_quanta._blend("#ffffff", "#e5821f", np.nan) == "#ffffff"


def test_paper_figure_humanizes_composite_semantic_labels() -> None:
    concept = (
        "(previous_target=hundred) OR "
        "(active_factorized_family=CHUNK_HAS_TAIL)"
    )

    assert (
        plot_qmodel_recovery_figure.humanize_concept(concept)
        == "after hundred or chunk has tail"
    )


def test_paper_figure_maps_layer_writer_offsets_to_owner_quanta() -> None:
    rows = plot_qmodel_recovery_figure._quantum_rows([[2, 1], [1]])
    frame = {"writer_activities": [[0.0, 2.0, 3.0], [4.0]]}

    assert rows == [(0, 0, 0, 2), (0, 1, 2, 1), (1, 0, 0, 1)]
    assert plot_qmodel_recovery_figure._active_writers(frame, rows) == [
        ("L0.Q0.W1", 2.0),
        ("L0.Q1.W2", 3.0),
        ("L1.Q0.W0", 4.0),
    ]


def test_global_static_edges_do_not_enforce_parent_gate_closure() -> None:
    assert not visualize_quanta._enforce_parent_gate_closure(
        {"edge_discovery": {"mode": "global_static"}}
    )
    assert visualize_quanta._enforce_parent_gate_closure({})


def test_elbow_arrow_is_always_solid() -> None:
    figure, axis = plt.subplots()
    visualize_quanta._draw_elbow_arrow(
        axis,
        start=(0.0, 0.0),
        end=(1.0, 1.0),
        width=1.0,
        color="#e5821f",
        alpha=1.0,
    )

    assert len(axis.patches) == 1
    assert axis.patches[0].get_linestyle() == "solid"
    plt.close(figure)


def test_cumulative_priority_is_normalized_per_quantum() -> None:
    curves = np.asarray([[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]])

    cumulative = normalized_cumulative_priority(curves)

    np.testing.assert_allclose(cumulative[0], [0.0, 0.25, 0.5, 1.0])
    np.testing.assert_allclose(cumulative[1], 0.0)


def test_writer_boxes_stay_inside_fixed_footprint() -> None:
    boxes = writer_box_layout(351, center=(1.0, 0.5))

    assert len(boxes) == 351
    assert max(x + size for x, _y, size in boxes) - min(x for x, _y, _size in boxes) <= 0.141
    assert max(y + size for _x, y, size in boxes) - min(y for _x, y, _size in boxes) <= 0.046


def test_dynamics_figures_use_held_out_checkpoint_improvement(tmp_path: Path) -> None:
    factorization = tmp_path / "factorization"
    factorization.mkdir()
    np.savez(
        factorization / "temporal_priority.npz",
        layer_0=np.asarray([[1.0, 2.0], [2.0, 0.0]]),
    )
    (tmp_path / "checkpoint_metadata.json").write_text(
        '{"checkpoint_steps":[0,5,10],"num_train_prediction_events":2}\n'
    )
    np.save(
        tmp_path / "losses.npy",
        np.asarray(
            [
                [3.0, 2.0, 1.0],
                [2.0, 1.5, 1.0],
                [4.0, 3.0, 2.0],
                [2.0, 1.0, 0.5],
            ]
        ),
    )

    diagnostics = load_dynamics_diagnostics(tmp_path)
    manifest = write_dynamics_figures(tmp_path, tmp_path / "visualizations")

    np.testing.assert_allclose(diagnostics.model_improvement, [0.0, 1.0, 1.75])
    assert diagnostics.improvement_label.startswith("held-out")
    assert Path(manifest["cumulative_priority"]).exists()
    assert Path(manifest["priority_model_correlation"]).exists()
