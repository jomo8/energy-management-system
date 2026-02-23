"""Load orchestration for essential and non-essential demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import numpy as np

from .load import Load


@dataclass
class LoadManager:
    """Manages active loads and shedding/restoration decisions."""

    loads: list[Load] = field(default_factory=list)
    random_seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    @classmethod
    def from_config(cls, load_items: Iterable[dict], random_seed: int = 42) -> "LoadManager":
        loads = []
        for item in load_items:
            loads.append(
                Load(
                    name=item["name"],
                    rated_power_W=float(item["rated_power_W"]),
                    is_essential=bool(item["is_essential"]),
                    priority=int(item["priority"]),
                    is_connected=bool(item.get("is_connected", True)),
                    schedule={int(k): float(v) for k, v in item.get("schedule", {}).items()},
                    noise_factor=float(item.get("noise_factor", 0.0)),
                )
            )
        return cls(loads=loads, random_seed=random_seed)

    def _demand(self, hour: int, essential_only: bool | None = None) -> float:
        total = 0.0
        for load in self.loads:
            if essential_only is None or load.is_essential == essential_only:
                total += load.get_demand(hour=hour, rng=self.rng)
        return total

    def get_total_demand(self, timestamp: datetime) -> float:
        return self._demand(timestamp.hour, essential_only=None)

    def get_essential_demand(self, timestamp: datetime) -> float:
        return self._demand(timestamp.hour, essential_only=True)

    def get_non_essential_demand(self, timestamp: datetime) -> float:
        return self._demand(timestamp.hour, essential_only=False)

    def shed_loads(self, target_reduction_W: float, timestamp: datetime) -> list[str]:
        if target_reduction_W <= 0:
            return []
        shed = []
        reduced = 0.0
        candidates = sorted(
            [l for l in self.loads if not l.is_essential and l.is_connected],
            key=lambda l: l.priority,
            reverse=True,
        )
        for load in candidates:
            load_demand = load.get_demand(timestamp.hour, rng=self.rng)
            if load.is_connected:
                load.is_connected = False
                load.last_disconnected_at = timestamp
            reduced += load_demand
            shed.append(load.name)
            if reduced >= target_reduction_W:
                break
        return shed

    def restore_loads(self) -> list[str]:
        restored = []
        for load in sorted(self.loads, key=lambda l: l.priority):
            if not load.is_connected and not load.is_essential:
                load.is_connected = True
                load.last_disconnected_at = None
                restored.append(load.name)
        return restored

    def shed_non_essential_loads(self, timestamp: datetime) -> list[str]:
        """Disconnect all non-essential loads immediately."""
        shed = []
        for load in self.loads:
            if not load.is_essential and load.is_connected:
                load.is_connected = False
                load.last_disconnected_at = timestamp
                shed.append(load.name)
        return shed

    def shed_all_loads(self, timestamp: datetime) -> list[str]:
        """Disconnect all loads immediately."""
        shed = []
        for load in self.loads:
            if load.is_connected:
                load.is_connected = False
                load.last_disconnected_at = timestamp
                shed.append(load.name)
        return shed

    def restore_essential_loads(self, timestamp: datetime, buffer_minutes: int = 30) -> list[str]:
        """Restore essential loads only if they have respected the off-time buffer."""
        restored = []
        min_off_seconds = max(0, buffer_minutes) * 60
        for load in sorted(self.loads, key=lambda l: l.priority):
            if not load.is_essential or load.is_connected:
                continue
            off_time_ok = (
                load.last_disconnected_at is None
                or (timestamp - load.last_disconnected_at).total_seconds() >= min_off_seconds
            )
            if off_time_ok:
                load.is_connected = True
                load.last_disconnected_at = None
                restored.append(load.name)
        return restored

    def restore_non_essential_loads(self, timestamp: datetime, buffer_minutes: int = 60) -> list[str]:
        """Restore non-essential loads only if they have respected the off-time buffer."""
        restored = []
        min_off_seconds = max(0, buffer_minutes) * 60
        for load in sorted(self.loads, key=lambda l: l.priority):
            if load.is_essential or load.is_connected:
                continue
            off_time_ok = (
                load.last_disconnected_at is None
                or (timestamp - load.last_disconnected_at).total_seconds() >= min_off_seconds
            )
            if off_time_ok:
                load.is_connected = True
                load.last_disconnected_at = None
                restored.append(load.name)
        return restored

    def get_connected_loads(self) -> list[str]:
        return [load.name for load in self.loads if load.is_connected]

    def get_load_breakdown(self, timestamp: datetime) -> dict[str, float]:
        essential = self.get_essential_demand(timestamp)
        non_essential = self.get_non_essential_demand(timestamp)
        return {
            "essential_W": essential,
            "non_essential_W": non_essential,
            "total_W": essential + non_essential,
        }

