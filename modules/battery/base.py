"""Abstract battery model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class BatteryModel(ABC):
    """Shared interface for all battery simulation backends."""

    @abstractmethod
    def step(self, current_A: float, dt_s: float, ambient_temp_C: float) -> Dict[str, float]:
        """
        Advance model state by one time step.

        Args:
            current_A: Pack current in Amps. Positive means discharge.
            dt_s: Timestep in seconds.
            ambient_temp_C: Ambient temperature in Celsius.
        """

    @abstractmethod
    def get_soc(self) -> float:
        """Return current state of charge [0.0, 1.0]."""

    @abstractmethod
    def get_terminal_voltage(self) -> float:
        """Return pack terminal voltage in volts."""

    @abstractmethod
    def get_ocv(self) -> float:
        """Return pack open-circuit voltage in volts."""

    @abstractmethod
    def set_capacity_scale(self, scale: float) -> None:
        """Scale available capacity to account for SoH fade."""

    @abstractmethod
    def set_resistance_scale(self, scale: float) -> None:
        """Scale internal resistance to account for aging."""

    @abstractmethod
    def reset(self, initial_soc: float) -> None:
        """Reset model state to the requested SoC."""

