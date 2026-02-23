"""Main simulation runner for modular EMS system."""

from __future__ import annotations

import json
from pathlib import Path

from modules.battery.pybamm_backend import PyBaMMBackend
from modules.battery.soc import StateOfCharge
from modules.battery.soh import StateOfHealth
from modules.battery.thevenin_ecm import TheveninECM
from modules.ems.ems import EnergyManagementSystem
from modules.loads.load_manager import LoadManager
from modules.solar.solar_array import SolarArray
from modules.utils.time_utils import parse_timestamp
from modules.utils.weather import load_weather_data


def load_configuration(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    return config


def _build_battery(config: dict) -> StateOfCharge:
    battery_cfg = config["battery"]
    chemistry = battery_cfg["chemistry"]
    initial_soc = battery_cfg["initial_soc_percent"] / 100.0
    num_cells_series = int(battery_cfg["num_cells_series"])
    num_strings_parallel = int(battery_cfg["num_strings_parallel"])
    cell_capacity_ah = float(battery_cfg["cell_capacity_ah"])

    soh = StateOfHealth(
        chemistry=chemistry,
        initial_soh_percent=float(battery_cfg.get("initial_soh_percent", 100.0)),
        initial_cycle_count=float(battery_cfg.get("initial_cycle_count", 0.0)),
    )

    backend_name = battery_cfg.get("backend", "ecm").lower()
    if backend_name == "pybamm":
        backend = PyBaMMBackend(
            num_cells_series=num_cells_series,
            num_strings_parallel=num_strings_parallel,
            initial_soc=initial_soc,
            nominal_cell_capacity_ah=cell_capacity_ah,
        )
    else:
        backend = TheveninECM(
            chemistry=chemistry,
            num_cells_series=num_cells_series,
            num_strings_parallel=num_strings_parallel,
            initial_soc=initial_soc,
            cell_capacity_ah=cell_capacity_ah,
        )
    return StateOfCharge(backend=backend, soh=soh)


def run_simulation(config_path: str | Path | None = None) -> list[dict]:
    root = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path is not None else root / "config" / "load_config.json"
    config = load_configuration(cfg_path)

    battery = _build_battery(config)
    location_cfg = config["location"]
    solar_cfg = config["solar"]
    load_cfg = config["loads"]
    sim_cfg = config["simulation"]

    solar = SolarArray(
        latitude=float(location_cfg["latitude"]),
        longitude=float(location_cfg["longitude"]),
        num_panels=int(solar_cfg["number_of_panels"]),
        panel_watt_peak=float(solar_cfg["panel_capacity_watts"]),
        tilt=float(solar_cfg["tilt"]),
        azimuth=float(solar_cfg["azimuth"]),
        inverter_efficiency=float(solar_cfg.get("inverter_efficiency", 0.96)),
        years_in_service=float(solar_cfg.get("years_in_service", 0.0)),
    )
    loads = LoadManager.from_config(load_cfg)
    ems = EnergyManagementSystem(battery=battery, solar=solar, load_mgr=loads, config=config)

    weather_path = root / config["simulation"]["weather_file"]
    weather = load_weather_data(weather_path)
    start_index = int(sim_cfg.get("start_index", 0))
    num_steps = int(sim_cfg.get("num_steps", 96))
    dt_s = float(sim_cfg.get("dt_seconds", 900))

    end_index = min(len(weather), start_index + num_steps)
    results: list[dict] = []
    for i in range(start_index, end_index):
        weather_step = weather[i]
        timestamp = parse_timestamp(weather_step["period_end"])
        _ = ems.step(timestamp=timestamp, weather_data=weather_step, dt_s=dt_s)
        results.append(ems.get_system_status())
    return results


if __name__ == "__main__":
    output = run_simulation()
    print(f"Simulation complete. Generated {len(output)} records.")

