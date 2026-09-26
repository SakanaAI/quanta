from __future__ import annotations

import numpy as np

from quanta.experiments.quanta_discovery.dynamics import (
    batch_prediction_event_ids,
    close_existence,
    cumulative_existence_at_step,
    prediction_event_offsets,
)


def test_batch_event_ids_follow_example_and_target_order() -> None:
    offsets = prediction_event_offsets(("one", "twenty one", "three"))

    event_ids = batch_prediction_event_ids(np.asarray([1, 0, 1]), offsets)

    np.testing.assert_array_equal(offsets, [0, 2, 5, 7])
    np.testing.assert_array_equal(event_ids, [2, 3, 4, 0, 1, 2, 3, 4])


def test_exact_step_existence_is_interpolated_then_downward_closed() -> None:
    parent_curve = np.asarray([[1.0, 1.0]], dtype=np.float64)
    child_curve = np.asarray([[0.0, 2.0]], dtype=np.float64)
    parent = cumulative_existence_at_step(parent_curve, step=25, total_steps=100)
    child = cumulative_existence_at_step(child_curve, step=75, total_steps=100)

    closed = close_existence((parent, child), ((0, 0, 1, 0),))

    np.testing.assert_allclose(parent, [0.25])
    np.testing.assert_allclose(child, [0.5])
    np.testing.assert_allclose(closed[0], [0.5])
    np.testing.assert_allclose(closed[1], [0.5])
