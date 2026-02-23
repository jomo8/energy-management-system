"""Solar array model based on pvlib and single-diode physics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pvlib


@dataclass
class SolarArray:
    """Encapsulates PV power calculations for the full array."""

    latitude: float
    longitude: float
    num_panels: int
    panel_watt_peak: float
    tilt: float
    azimuth: float
    inverter_efficiency: float = 0.96
    years_in_service: float = 0.0

    def __post_init__(self) -> None:
        self.site = pvlib.location.Location(self.latitude, self.longitude)
        self.module_params = self._fit_module_params()

    def _fit_module_params(self) -> dict[str, float]:
        # CEC-equivalent fit based on representative 325W poly-Si module.
        i_sc_ref = 9.17
        v_oc_ref = 46.38
        i_mp_ref = 8.69
        v_mp_ref = 37.39
        gamma_pmp_ref = -0.4
        cells_in_series = 72
        t_ref = 25.0

        alpha_sc = (0.058 / 100.0) * i_sc_ref
        beta_oc = (-0.330 / 100.0) * v_oc_ref

        try:
            il_ref, i0_ref, rs_ref, rsh_ref, a_ref, adjust = pvlib.ivtools.sdm.fit_cec_sam(
                v_oc=v_oc_ref,
                i_sc=i_sc_ref,
                v_mp=v_mp_ref,
                i_mp=i_mp_ref,
                alpha_sc=alpha_sc,
                beta_voc=beta_oc,
                cells_in_series=cells_in_series,
                temp_ref=t_ref,
                celltype="polySi",
                gamma_pmp=gamma_pmp_ref,
            )
        except Exception:
            # Safe fallback when CEC fitting backends are unavailable at runtime.
            il_ref, i0_ref, rs_ref, rsh_ref, a_ref, adjust = 9.17, 1.5e-10, 0.35, 450.0, 1.8, 0.0
        return {
            "I_L_ref": il_ref,
            "I_o_ref": i0_ref,
            "R_s": rs_ref,
            "R_sh_ref": rsh_ref,
            "a_ref": a_ref,
            "Adjust": adjust,
            "alpha_sc": alpha_sc,
            "cells_in_series": cells_in_series,
        }

    def _inverter_efficiency_curve(self, dc_power_w: float) -> float:
        rated = self.num_panels * self.panel_watt_peak
        loading = float(np.clip(dc_power_w / max(rated, 1e-6), 0.0, 1.2))
        # Mildly lower efficiency at low loading.
        return float(np.clip(0.93 + 0.04 * loading, 0.90, self.inverter_efficiency))

    def calculate_power(
        self,
        weather_data: dict,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Compute DC/AC array power output at one timestep."""
        ts = pd.DatetimeIndex([timestamp])
        solar_position = self.site.get_solarposition(times=ts)

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt,
            surface_azimuth=self.azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=float(weather_data.get("dni", 0.0)),
            ghi=float(weather_data.get("ghi", 0.0)),
            dhi=float(weather_data.get("dhi", 0.0)),
            albedo=float(weather_data.get("albedo", 0.2)),
        )

        temp_cell = pvlib.temperature.pvsyst_cell(
            poa_global=poa["poa_global"],
            temp_air=float(weather_data.get("air_temp", 25.0)),
            wind_speed=float(weather_data.get("wind_speed_10m", 1.0)),
        )

        iam = pvlib.iam.ashrae(solar_position["zenith"]).fillna(0.0)
        effective_irradiance = float(poa["poa_global"].iloc[0] * iam.iloc[0])

        il, i0, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance=effective_irradiance,
            temp_cell=float(temp_cell.iloc[0]),
            alpha_sc=self.module_params["alpha_sc"],
            a_ref=self.module_params["a_ref"],
            I_L_ref=self.module_params["I_L_ref"],
            I_o_ref=self.module_params["I_o_ref"],
            R_sh_ref=self.module_params["R_sh_ref"],
            R_s=self.module_params["R_s"],
        )
        diode = pvlib.pvsystem.singlediode(il, i0, rs, rsh, nnsvth)

        panel_dc_power_w = float(diode["p_mp"])
        degradation_factor = float((1.0 - 0.005) ** max(self.years_in_service, 0.0))
        array_dc_power_w = panel_dc_power_w * self.num_panels * degradation_factor
        inv_eff = self._inverter_efficiency_curve(array_dc_power_w)
        array_ac_power_w = array_dc_power_w * inv_eff

        return {
            "dc_power_W": array_dc_power_w,
            "ac_power_W": array_ac_power_w,
            "cell_temp_C": float(temp_cell.iloc[0]),
            "poa_irradiance": float(poa["poa_global"].iloc[0]),
            "inverter_efficiency": inv_eff,
        }

