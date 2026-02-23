"""State of Health module with semi-empirical degradation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ChemistryParams, get_chemistry_params

R_GAS = 8.314462618
KELVIN_OFFSET = 273.15


@dataclass
class StateOfHealth:
    """Tracks capacity fade and resistance growth over lifetime."""

    chemistry: str
    initial_soh_percent: float = 100.0
    initial_cycle_count: float = 0.0

    def __post_init__(self) -> None:
        self.params: ChemistryParams = get_chemistry_params(self.chemistry)
        self.soh = float(np.clip(self.initial_soh_percent / 100.0, 0.5, 1.0))
        self.cycle_count = float(max(0.0, self.initial_cycle_count))
        self.ah_throughput = 0.0
        self.elapsed_days = 0.0
        self.q_loss_cyc = 1.0 - self.soh
        self.q_loss_cal = 0.0
        self.resistance_growth_factor = 1.0
        self._equivalent_full_cycles = self.cycle_count

    def update(self, current_A: float, dt_s: float, temperature_C: float, soc: float) -> None:
        """Update SoH terms using throughput + calendar aging laws."""
        dt_hours = dt_s / 3600.0
        dt_days = dt_s / 86400.0
        temp_K = max(temperature_C + KELVIN_OFFSET, 250.0)
        soc_stress = float(np.exp(self.params.soh_soc_stress_k * (float(np.clip(soc, 0.0, 1.0)) - 0.5)))

        self.elapsed_days += dt_days
        ah_increment = abs(current_A) * dt_hours
        self.ah_throughput += ah_increment

        cyc_term = self.params.soh_k_cyc * np.exp(-self.params.soh_ea_cyc_j_per_mol / (R_GAS * temp_K))
        cal_term = self.params.soh_k_cal * np.exp(-self.params.soh_ea_cal_j_per_mol / (R_GAS * temp_K))

        self.q_loss_cyc = float(cyc_term * np.sqrt(max(self.ah_throughput, 0.0)) * soc_stress)
        self.q_loss_cal = float(cal_term * np.sqrt(max(self.elapsed_days, 0.0)) * soc_stress)

        total_loss = float(np.clip(self.q_loss_cyc + self.q_loss_cal, 0.0, 0.5))
        self.soh = 1.0 - total_loss

        self._equivalent_full_cycles = self.ah_throughput / (2.0 * self.params.cell_capacity_ah)
        self.cycle_count = self.initial_cycle_count + self._equivalent_full_cycles

        self.resistance_growth_factor = 1.0 + self.params.soh_resistance_growth_alpha * total_loss

    def get_soh(self) -> float:
        return self.soh

    def get_soh_percent(self) -> float:
        return self.soh * 100.0

    def get_effective_capacity(self, nominal_pack_capacity_ah: float) -> float:
        return nominal_pack_capacity_ah * self.soh

    def get_resistance_factor(self) -> float:
        return self.resistance_growth_factor

    def get_full_equivalent_cycles(self) -> float:
        return self._equivalent_full_cycles

