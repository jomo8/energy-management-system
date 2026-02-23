"""Individual load model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import numpy as np


@dataclass
class Load:
    """Represents a single controllable load."""

    name: str
    rated_power_W: float
    is_essential: bool
    priority: int
    is_connected: bool = True
    schedule: Dict[int, float] = field(default_factory=dict)
    noise_factor: float = 0.0
    last_disconnected_at: datetime | None = None

    def _hour_factor(self, hour: int) -> float:
        if not self.schedule:
            return 1.0
        return float(np.clip(self.schedule.get(hour, 1.0), 0.0, 2.0))

    def get_demand(self, hour: int, rng: np.random.Generator | None = None) -> float:
        """Return instantaneous load demand in watts."""
        if not self.is_connected:
            return 0.0
        base = self.rated_power_W * self._hour_factor(hour)
        if self.noise_factor <= 0.0:
            return float(base)
        generator = rng if rng is not None else np.random.default_rng()
        noise = generator.uniform(-self.noise_factor, self.noise_factor)
        return float(max(0.0, base * (1.0 + noise)))

