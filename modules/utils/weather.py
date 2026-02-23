"""Weather data loading helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_weather_data(path: str | Path) -> list[dict]:
    weather_path = Path(path)
    with weather_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Weather file must contain a JSON list of timestep records.")
    return data

