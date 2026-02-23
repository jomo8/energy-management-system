"""PyBaMM SPMe backend adapter."""

from __future__ import annotations

import numpy as np
import pybamm

from .base import BatteryModel


class PyBaMMBackend(BatteryModel):
    """High-fidelity physics backend using PyBaMM SPMe."""

    def __init__(
        self,
        num_cells_series: int,
        num_strings_parallel: int,
        initial_soc: float = 0.5,
        nominal_cell_capacity_ah: float = 50.0,
    ) -> None:
        self.num_cells_series = num_cells_series
        self.num_strings_parallel = num_strings_parallel
        self.initial_soc = float(np.clip(initial_soc, 0.0, 1.0))
        self.capacity_scale = 1.0
        self.resistance_scale = 1.0

        self.model = pybamm.lithium_ion.SPMe()
        self.param_values = pybamm.ParameterValues("Chen2020")
        self.param_values["Current function [A]"] = "[input]"
        self.param_values["Nominal cell capacity [A.h]"] = nominal_cell_capacity_ah
        self.param_values["Number of electrodes connected in parallel to make a cell"] = (
            1 * self.num_strings_parallel
        )
        self.sim = pybamm.Simulation(self.model, parameter_values=self.param_values)
        self.sim.solve(t_eval=[0, 1], initial_soc=self.initial_soc, inputs={"Current function [A]": 0.0})
        self._last_result = self._extract_result(current_A=0.0)

    def _extract_result(self, current_A: float) -> dict[str, float]:
        cell_voltage = float(self.sim.solution["Terminal voltage [V]"].data[-1])
        soc = float(self.sim.solution["X-averaged negative particle stoichiometry"].entries[-1][-1])
        pack_voltage = cell_voltage * self.num_cells_series
        return {
            "soc": float(np.clip(soc, 0.0, 1.0)),
            "terminal_voltage": pack_voltage,
            "cell_temperature": 25.0,
            "current_A": current_A,
            "power_W": pack_voltage * current_A,
            "ocv": pack_voltage,
        }

    def step(self, current_A: float, dt_s: float, ambient_temp_C: float) -> dict[str, float]:
        _ = ambient_temp_C  # Thermal coupling can be added later.
        # Approximate SoH capacity fade by increasing equivalent current demand.
        scaled_current_A = current_A / max(self.capacity_scale, 1e-6)
        cell_current_A = scaled_current_A / max(self.num_strings_parallel, 1)
        self.sim.step(dt=dt_s, inputs={"Current function [A]": cell_current_A})
        self._last_result = self._extract_result(current_A=current_A)
        # Approximate resistance growth as additional voltage sag.
        self._last_result["terminal_voltage"] /= max(self.resistance_scale, 1e-6)
        self._last_result["power_W"] = self._last_result["terminal_voltage"] * current_A
        return self._last_result

    def get_soc(self) -> float:
        return self._last_result["soc"]

    def get_terminal_voltage(self) -> float:
        return self._last_result["terminal_voltage"]

    def get_ocv(self) -> float:
        return self._last_result["ocv"]

    def set_capacity_scale(self, scale: float) -> None:
        self.capacity_scale = float(np.clip(scale, 0.4, 1.2))

    def set_resistance_scale(self, scale: float) -> None:
        self.resistance_scale = float(np.clip(scale, 0.8, 3.0))

    def reset(self, initial_soc: float) -> None:
        self.initial_soc = float(np.clip(initial_soc, 0.0, 1.0))
        self.sim = pybamm.Simulation(self.model, parameter_values=self.param_values)
        self.sim.solve(t_eval=[0, 1], initial_soc=self.initial_soc, inputs={"Current function [A]": 0.0})
        self._last_result = self._extract_result(current_A=0.0)

