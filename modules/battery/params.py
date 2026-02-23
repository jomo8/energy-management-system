"""Battery chemistry parameters for ECM and degradation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ChemistryParams:
    """Chemistry-specific electrical and aging parameters."""

    name: str
    cell_capacity_ah: float
    nominal_cell_voltage_v: float
    min_cell_voltage_v: float
    max_cell_voltage_v: float
    r0_ohm: float
    r1_ohm: float
    c1_f: float
    r2_ohm: float
    c2_f: float
    coulombic_eff_charge: float
    coulombic_eff_discharge: float
    self_discharge_per_day: float
    thermal_mass_j_per_k: float
    thermal_resistance_k_per_w: float
    arrhenius_ea_j_per_mol: float
    soh_k_cyc: float
    soh_ea_cyc_j_per_mol: float
    soh_k_cal: float
    soh_ea_cal_j_per_mol: float
    soh_soc_stress_k: float
    soh_resistance_growth_alpha: float
    ocv_soc_points: np.ndarray
    ocv_voltage_points: np.ndarray


def _build_nmc_params() -> ChemistryParams:
    soc = np.array(
        [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        dtype=float,
    )
    ocv = np.array(
        [3.00, 3.30, 3.45, 3.60, 3.70, 3.76, 3.82, 3.89, 3.97, 4.05, 4.14, 4.20],
        dtype=float,
    )
    return ChemistryParams(
        name="NMC",
        cell_capacity_ah=50.0,
        nominal_cell_voltage_v=3.7,
        min_cell_voltage_v=3.0,
        max_cell_voltage_v=4.2,
        r0_ohm=0.0018,
        r1_ohm=0.0010,
        c1_f=2400.0,
        r2_ohm=0.0014,
        c2_f=12000.0,
        coulombic_eff_charge=0.98,
        coulombic_eff_discharge=0.995,
        self_discharge_per_day=0.001,
        thermal_mass_j_per_k=150.0,
        thermal_resistance_k_per_w=1.8,
        arrhenius_ea_j_per_mol=24000.0,
        soh_k_cyc=2.6e-5,
        soh_ea_cyc_j_per_mol=31500.0,
        soh_k_cal=9.0e-5,
        soh_ea_cal_j_per_mol=22000.0,
        soh_soc_stress_k=2.2,
        soh_resistance_growth_alpha=2.5,
        ocv_soc_points=soc,
        ocv_voltage_points=ocv,
    )


def _build_lfp_params() -> ChemistryParams:
    soc = np.array(
        [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        dtype=float,
    )
    ocv = np.array(
        [2.50, 3.10, 3.20, 3.27, 3.30, 3.32, 3.34, 3.35, 3.37, 3.40, 3.48, 3.65],
        dtype=float,
    )
    return ChemistryParams(
        name="LFP",
        cell_capacity_ah=100.0,
        nominal_cell_voltage_v=3.2,
        min_cell_voltage_v=2.5,
        max_cell_voltage_v=3.65,
        r0_ohm=0.0012,
        r1_ohm=0.0008,
        c1_f=3000.0,
        r2_ohm=0.0010,
        c2_f=15000.0,
        coulombic_eff_charge=0.99,
        coulombic_eff_discharge=0.997,
        self_discharge_per_day=0.0008,
        thermal_mass_j_per_k=170.0,
        thermal_resistance_k_per_w=2.0,
        arrhenius_ea_j_per_mol=22000.0,
        soh_k_cyc=1.8e-5,
        soh_ea_cyc_j_per_mol=28000.0,
        soh_k_cal=6.0e-5,
        soh_ea_cal_j_per_mol=20000.0,
        soh_soc_stress_k=1.5,
        soh_resistance_growth_alpha=1.8,
        ocv_soc_points=soc,
        ocv_voltage_points=ocv,
    )


CHEMISTRY_DATABASE: Dict[str, ChemistryParams] = {
    "NMC": _build_nmc_params(),
    "LFP": _build_lfp_params(),
}


def get_chemistry_params(name: str) -> ChemistryParams:
    """Return chemistry parameters by normalized chemistry name."""
    normalized = name.strip().upper()
    if normalized not in CHEMISTRY_DATABASE:
        raise ValueError(f"Unsupported chemistry '{name}'. Valid options: {list(CHEMISTRY_DATABASE)}")
    return CHEMISTRY_DATABASE[normalized]


def interpolate_ocv(soc: float, params: ChemistryParams) -> float:
    """Interpolate open-circuit voltage from SoC lookup table."""
    clipped_soc = float(np.clip(soc, 0.0, 1.0))
    return float(np.interp(clipped_soc, params.ocv_soc_points, params.ocv_voltage_points))

