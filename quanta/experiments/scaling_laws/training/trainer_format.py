from __future__ import annotations

import json
from typing import Any


def _format_depth_loss_bits(mean_depth_loss: dict[int, dict[str, float]]) -> str:
    return " ".join(
        f"depth_{depth}={values['loss_bits']:.6g}"
        for depth, values in sorted(mean_depth_loss.items())
    )


def _format_prediction_sample(record: dict[str, Any] | None) -> str:
    if record is None:
        return "none"
    fields = [
        f"task={record['task']}",
        f"depth={record['depth']}",
        f"true={record['true_label']}",
        f"pred={record['predicted_label']}",
        f"correct={str(record['correct']).lower()}",
        f"loss_bits={record['loss_bits']}",
    ]
    if "true_label_probability" in record:
        fields.append(f"p_true={record['true_label_probability']}")
    if "masked_slots" in record:
        fields.append(f"masked_slots={len(record['masked_slots'])}")
    elif "true_state_bits" in record:
        fields.append(f"true_state_bits={json.dumps(record['true_state_bits'])}")
        fields.append(f"pred_state_bits={json.dumps(record['predicted_state_bits'])}")
    return " ".join(fields)
