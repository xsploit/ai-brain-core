from __future__ import annotations

from typing import Any


QUALITATIVE_SCORES = {
    "none": 0.0,
    "low": 0.3,
    "medium": 0.5,
    "normal": 0.5,
    "high": 0.8,
    "critical": 1.0,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in QUALITATIVE_SCORES:
            return QUALITATIVE_SCORES[normalized]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
