"""Energy management coordinator scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from modules.battery.soc import StateOfCharge
from modules.loads.load_manager import LoadManager
from modules.solar.solar_array import SolarArray


@dataclass
class EMSDecision:
    """Output action of one EMS control interval."""

    charge_discharge_power_W: float
    loads_to_shed: list[str]
    loads_to_restore: list[str]
    system_state: str
    reason: str


class EnergyManagementSystem:
    """
    EMS dispatcher with intentionally simple placeholder control logic.

    Detailed policy can replace this class later without changing module interfaces.
    """

    def __init__(
        self,
        battery: StateOfCharge,
        solar: SolarArray,
        load_mgr: LoadManager,
        config: dict,
    ) -> None:
        self.battery = battery
        self.solar = solar
        self.load_mgr = load_mgr
        self.config = config
        self.last_status: dict = {}
        self.system_locked = False

    def step(self, timestamp: datetime, weather_data: dict, dt_s: float) -> EMSDecision:
        solar_out = self.solar.calculate_power(weather_data=weather_data, timestamp=timestamp)
        load_breakdown = self.load_mgr.get_load_breakdown(timestamp)
        total_demand = load_breakdown["total_W"]
        essential_demand = load_breakdown["essential_W"]

        soc = self.battery.get_soc()
        solar_ac = solar_out["ac_power_W"]
        net_generation = solar_ac - total_demand

        loads_to_shed: list[str] = []
        loads_to_restore: list[str] = []
        state = "normal"
        reason = "Balanced operation."

        if soc < 0.05:
            self.system_locked = True
            state = "system_lock"
            reason = "SoC below 5%; all loads shed and system lock is active."
            loads_to_shed = self.load_mgr.shed_all_loads(timestamp)
        elif soc < 0.30:
            state = "critical"
            reason = "SoC below 30%; shedding all loads."
            loads_to_shed = self.load_mgr.shed_all_loads(timestamp)
        elif soc < 0.40:
            state = "emergency"
            reason = "SoC below 40%; shedding all non-essential loads."
            loads_to_shed = self.load_mgr.shed_non_essential_loads(timestamp)
        else:
            sufficient_solar_for_essential = solar_ac >= essential_demand
            if (soc > 0.50 or sufficient_solar_for_essential) and not self.system_locked:
                loads_to_restore.extend(
                    self.load_mgr.restore_essential_loads(timestamp=timestamp, buffer_minutes=30)
                )

            if soc > 0.70 and not self.system_locked:
                loads_to_restore.extend(
                    self.load_mgr.restore_non_essential_loads(timestamp=timestamp, buffer_minutes=60)
                )

        # Recompute demand after any shed/restore operation.
        load_breakdown = self.load_mgr.get_load_breakdown(timestamp)
        total_demand = load_breakdown["total_W"]
        essential_demand = load_breakdown["essential_W"]
        net_generation = solar_ac - total_demand

        battery_power_command_W = -net_generation
        result = self.battery.update(
            power_W=battery_power_command_W,
            dt_s=dt_s,
            ambient_temp_C=weather_data["air_temp"],
        )

        self.last_status = {
            "timestamp": timestamp.isoformat(),
            "state": state,
            "reason": reason,
            "system_locked": self.system_locked,
            "soc": result["soc"],
            "soh": result["soh"],
            "soh_percent": result.get("soh_percent", result["soh"] * 100.0),
            "cycle_count": self.battery.soh.cycle_count,
            "full_equivalent_cycles": result.get("fec", 0.0),
            "terminal_voltage": result["terminal_voltage"],
            "battery_power_W": result["power_W"],
            "solar_ac_W": solar_ac,
            "load_total_W": total_demand,
            "load_essential_W": essential_demand,
            "load_non_essential_W": max(0.0, total_demand - essential_demand),
            "connected_essential_count": sum(
                1 for load in self.load_mgr.loads if load.is_essential and load.is_connected
            ),
            "connected_non_essential_count": sum(
                1 for load in self.load_mgr.loads if (not load.is_essential) and load.is_connected
            ),
            "shed_count": len(loads_to_shed),
        }

        return EMSDecision(
            charge_discharge_power_W=battery_power_command_W,
            loads_to_shed=loads_to_shed,
            loads_to_restore=loads_to_restore,
            system_state=state,
            reason=reason,
        )

    def get_system_status(self) -> dict:
        return self.last_status

