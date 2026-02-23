"""Time parsing helpers."""

from __future__ import annotations

from datetime import datetime


def parse_timestamp(value: str) -> datetime:
    """Parse ISO8601 timestamp from weather payload."""
    return datetime.fromisoformat(value)

