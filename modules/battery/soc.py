"""State of Charge coordinator that couples battery backend and SoH."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BatteryModel
from .soh import StateOfHealth


@dataclass
class StateOfCharge:
    """SoC service that applies SoH scaling before stepping the backend."""

    backend: BatteryModel
    soh: StateOfHealth

    def update(self, power_W: float, dt_s: float, ambient_temp_C: float) -> dict[str, float]:
        voltage = max(self.backend.get_terminal_voltage(), 1e-6)
        current_A = power_W / voltage

        self.backend.set_capacity_scale(self.soh.get_soh())
        self.backend.set_resistance_scale(self.soh.get_resistance_factor())

        result = self.backend.step(current_A=current_A, dt_s=dt_s, ambient_temp_C=ambient_temp_C)
        self.soh.update(
            current_A=result["current_A"],
            dt_s=dt_s,
            temperature_C=result["cell_temperature"],
            soc=result["soc"],
        )
        result["soh"] = self.soh.get_soh()
        result["soh_percent"] = self.soh.get_soh_percent()
        result["fec"] = self.soh.get_full_equivalent_cycles()
        return result

    def get_soc(self) -> float:
        return self.backend.get_soc()

