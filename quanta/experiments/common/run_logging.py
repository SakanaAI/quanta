from __future__ import annotations

import json
import logging
import math
from typing import Any


def log_event(experiment: str, event: str, /, **fields: Any) -> None:
    """Emit one stable, compact experiment log record."""
    parts = [f"experiment={experiment}", f"event={event}"]
    parts.extend(f"{key}={_format_value(value)}" for key, value in fields.items())
    logging.info(" ".join(parts))


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    if value is None:
        return "null"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=isinstance(value, dict))
    text = str(value)
    return json.dumps(text) if any(character.isspace() for character in text) else text
