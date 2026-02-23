"""Fast Thevenin ECM battery backend for interactive simulations."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from .base import BatteryModel
from .params import ChemistryParams, get_chemistry_params, interpolate_ocv

R_GAS = 8.314462618
KELVIN_OFFSET = 273.15


@dataclass
class TheveninECM(BatteryModel):
    """Second-order Thevenin battery model scaled to pack level."""

    chemistry: str
    num_cells_series: int
    num_strings_parallel: int
    initial_soc: float = 0.5
    cell_capacity_ah: float | None = None

    def __post_init__(self) -> None:
        self.params: ChemistryParams = get_chemistry_params(self.chemistry)
        self.soc = float(np.clip(self.initial_soc, 0.0, 1.0))
        self.cell_capacity_ah = self.cell_capacity_ah or self.params.cell_capacity_ah
        self.nominal_pack_capacity_ah = self.cell_capacity_ah * self.num_strings_parallel
        self.capacity_scale = 1.0
        self.resistance_scale = 1.0

        self.v_rc1 = 0.0
        self.v_rc2 = 0.0
        self.cell_temp_C = 25.0
        self.pack_voltage = self.get_ocv()

    def _arrhenius_scale(self, temp_C: float) -> float:
        temp_K = max(temp_C + KELVIN_OFFSET, 250.0)
        ref_K = 25.0 + KELVIN_OFFSET
        return exp(self.params.arrhenius_ea_j_per_mol / R_GAS * (1.0 / temp_K - 1.0 / ref_K))

    def _effective_params(self, temp_C: float) -> tuple[float, float, float, float, float]:
        temp_scale = self._arrhenius_scale(temp_C)
        r0 = self.params.r0_ohm * temp_scale * self.resistance_scale
        r1 = self.params.r1_ohm * temp_scale * self.resistance_scale
        r2 = self.params.r2_ohm * temp_scale * self.resistance_scale
        c1 = self.params.c1_f / max(temp_scale, 1e-6)
        c2 = self.params.c2_f / max(temp_scale, 1e-6)
        return r0, r1, c1, r2, c2

    def _update_thermal(self, current_cell_A: float, r0: float, dt_s: float, ambient_temp_C: float) -> None:
        heat_w = (current_cell_A**2) * r0
        thermal_tau = self.params.thermal_mass_j_per_k * self.params.thermal_resistance_k_per_w
        cooling = (self.cell_temp_C - ambient_temp_C) / max(thermal_tau, 1e-9)
        dtemp = dt_s * ((heat_w / self.params.thermal_mass_j_per_k) - cooling)
        self.cell_temp_C += dtemp

    def _update_soc(self, current_pack_A: float, dt_s: float) -> None:
        effective_capacity = max(self.nominal_pack_capacity_ah * self.capacity_scale, 1e-6)
        eff = self.params.coulombic_eff_discharge if current_pack_A >= 0 else self.params.coulombic_eff_charge
        delta_soc = (current_pack_A * dt_s) / (effective_capacity * 3600.0 * eff)

        self.soc = float(np.clip(self.soc - delta_soc, 0.0, 1.0))

        if abs(current_pack_A) < 1e-6:
            leakage = self.params.self_discharge_per_day * (dt_s / 86400.0)
            self.soc = float(np.clip(self.soc - leakage, 0.0, 1.0))

    def step(self, current_A: float, dt_s: float, ambient_temp_C: float) -> dict[str, float]:
        """
        Advance one timestep.

        Args:
            current_A: Pack current in Amps. Positive means discharging.
            dt_s: Timestep in seconds.
            ambient_temp_C: Ambient temperature in Celsius.
        """
        current_cell_A = current_A / max(self.num_strings_parallel, 1)
        r0, r1, c1, r2, c2 = self._effective_params(self.cell_temp_C)

        alpha1 = exp(-dt_s / max(r1 * c1, 1e-9))
        alpha2 = exp(-dt_s / max(r2 * c2, 1e-9))
        self.v_rc1 = self.v_rc1 * alpha1 + current_cell_A * r1 * (1.0 - alpha1)
        self.v_rc2 = self.v_rc2 * alpha2 + current_cell_A * r2 * (1.0 - alpha2)

        self._update_soc(current_A, dt_s)
        self._update_thermal(current_cell_A, r0, dt_s, ambient_temp_C)

        ocv_cell = interpolate_ocv(self.soc, self.params)
        terminal_cell = ocv_cell - current_cell_A * r0 - self.v_rc1 - self.v_rc2
        terminal_cell = float(
            np.clip(terminal_cell, self.params.min_cell_voltage_v, self.params.max_cell_voltage_v)
        )

        self.pack_voltage = terminal_cell * self.num_cells_series
        power_W = self.pack_voltage * current_A

        return {
            "soc": self.soc,
            "terminal_voltage": self.pack_voltage,
            "cell_temperature": self.cell_temp_C,
            "current_A": current_A,
            "power_W": power_W,
            "ocv": self.get_ocv(),
        }

    def get_soc(self) -> float:
        return self.soc

    def get_terminal_voltage(self) -> float:
        return self.pack_voltage

    def get_ocv(self) -> float:
        return interpolate_ocv(self.soc, self.params) * self.num_cells_series

    def set_capacity_scale(self, scale: float) -> None:
        self.capacity_scale = float(np.clip(scale, 0.4, 1.2))

    def set_resistance_scale(self, scale: float) -> None:
        self.resistance_scale = float(np.clip(scale, 0.8, 3.0))

    def reset(self, initial_soc: float) -> None:
        self.soc = float(np.clip(initial_soc, 0.0, 1.0))
        self.v_rc1 = 0.0
        self.v_rc2 = 0.0
        self.cell_temp_C = 25.0
        self.pack_voltage = self.get_ocv()

