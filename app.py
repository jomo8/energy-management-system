"""Streamlit dashboard for the Energy Management System simulation."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from modules.battery.pybamm_backend import PyBaMMBackend
from modules.battery.soc import StateOfCharge
from modules.battery.soh import StateOfHealth
from modules.battery.thevenin_ecm import TheveninECM
from modules.ems.ems import EnergyManagementSystem
from modules.loads.load_manager import LoadManager
from modules.solar.solar_array import SolarArray
from modules.utils.time_utils import parse_timestamp
from modules.utils.weather import load_weather_data
from simulation import load_configuration

alt.data_transformers.disable_max_rows()


def _build_soc(config: dict) -> StateOfCharge:
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


@st.cache_data(show_spinner=False)
def run_dashboard_simulation(config_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parent
    config = load_configuration(config_path)
    weather = load_weather_data(root / config["simulation"]["weather_file"])

    battery = _build_soc(config)
    solar_cfg = config["solar"]
    location_cfg = config["location"]
    load_mgr = LoadManager.from_config(config["loads"])
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
    ems = EnergyManagementSystem(battery=battery, solar=solar, load_mgr=load_mgr, config=config)

    sim_cfg = config["simulation"]
    configured_start_index = int(sim_cfg.get("start_index", 0))
    dt_s = float(sim_cfg.get("dt_seconds", 900))
    end_index = len(weather)
    one_week_steps = int((7 * 24 * 3600) / max(dt_s, 1.0))
    start_index = max(configured_start_index, end_index - one_week_steps)

    rows: list[dict] = []
    weather_rows: list[dict] = []
    for idx in range(start_index, end_index):
        w = weather[idx]
        timestamp = parse_timestamp(w["period_end"])
        _ = ems.step(timestamp=timestamp, weather_data=w, dt_s=dt_s)
        status = ems.get_system_status()
        status["timestamp"] = pd.to_datetime(status["timestamp"], utc=True)
        rows.append(status)
        weather_rows.append(
            {
                "timestamp": pd.to_datetime(w["period_end"], utc=True),
                "air_temp": float(w.get("air_temp", 0.0)),
                "wind_speed_10m": float(w.get("wind_speed_10m", 0.0)),
                "wind_direction_10m": float(w.get("wind_direction_10m", 0.0)),
            }
        )

    status_df = pd.DataFrame(rows).sort_values("timestamp")
    weather_df = pd.DataFrame(weather_rows).sort_values("timestamp")
    return status_df, weather_df


def _notification_messages(current: pd.Series) -> list[str]:
    notes: list[str] = []
    if current.get("connected_non_essential_count", 0) > 0:
        notes.append("All non-essential loads are active.")
    else:
        notes.append("All non-essential loads are currently shed.")

    if current.get("connected_essential_count", 0) > 0:
        notes.append("All essential loads are active.")
    else:
        notes.append("Essential loads are currently shed.")

    if current.get("solar_ac_W", 0.0) > 0:
        notes.append("Solar generation is active.")
    else:
        notes.append("Solar generation is currently inactive.")

    soc_percent = float(current.get("soc", 0.0)) * 100.0
    if soc_percent >= 60.0:
        notes.append("Normalizing: SoC is 60% or higher. Loads are being restored incrementally.")

    reason = current.get("reason", "")
    if reason:
        notes.append(f"EMS: {reason}")

    return notes


def _build_shed_intervals(chart_df: pd.DataFrame, latest_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Build contiguous time spans for load-shed highlights.

    Yellow: non-essential loads shed (essential still active).
    Red: all loads shed.
    """
    if chart_df.empty:
        return pd.DataFrame(columns=["start", "end", "event"])

    df = chart_df.sort_values("timestamp").copy()

    def classify(row: pd.Series) -> str:
        essential_connected = int(row.get("connected_essential_count", 0))
        non_essential_connected = int(row.get("connected_non_essential_count", 0))
        if essential_connected == 0 and non_essential_connected == 0:
            return "all_shed"
        if essential_connected > 0 and non_essential_connected == 0:
            return "non_essential_shed"
        return "normal"

    df["shed_state"] = df.apply(classify, axis=1)
    df["next_timestamp"] = df["timestamp"].shift(-1).fillna(latest_ts)

    spans: list[dict] = []
    current_state = None
    current_start = None
    current_end = None
    for row in df.itertuples(index=False):
        state = row.shed_state
        start = row.timestamp
        end = row.next_timestamp

        if state == "normal":
            if current_state in {"all_shed", "non_essential_shed"}:
                spans.append({"start": current_start, "end": current_end, "event": current_state})
                current_state = None
                current_start = None
                current_end = None
            continue

        if current_state is None:
            current_state = state
            current_start = start
            current_end = end
        elif state == current_state:
            current_end = end
        else:
            spans.append({"start": current_start, "end": current_end, "event": current_state})
            current_state = state
            current_start = start
            current_end = end

    if current_state in {"all_shed", "non_essential_shed"}:
        spans.append({"start": current_start, "end": current_end, "event": current_state})

    return pd.DataFrame(spans)


def main() -> None:
    st.set_page_config(page_title="Energy Management System Dashboard", layout="wide")
    st.title("Energy Management System Dashboard")

    root = Path(__file__).resolve().parent
    config_path = str(root / "load_config.json")
    status_df, weather_df = run_dashboard_simulation(config_path)

    if status_df.empty:
        st.error("No simulation data available.")
        return

    current = status_df.iloc[-1]
    current_weather = weather_df.iloc[-1]

    st.subheader("Location and Weather")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Location", "Dodoma")
    c2.metric("Temperature", f"{current_weather['air_temp']:.1f} °C")
    c3.metric("Wind Speed", f"{current_weather['wind_speed_10m']:.1f} m/s")
    c4.metric("Wind Direction", f"{current_weather['wind_direction_10m']:.0f}°")

    st.subheader("Battery and Power Status")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Battery SoC", f"{float(current['soc']) * 100.0:.1f}%")
    m2.metric("Battery Health", f"{float(current['soh_percent']):.1f}%")
    m3.metric("Battery Cycle Count", f"{float(current['cycle_count']):.1f}")
    m4.metric("Current Power Generation", f"{float(current['solar_ac_W']) / 1000.0:.2f} kW")
    m5.metric("Current Power Consumption", f"{float(current['load_total_W']) / 1000.0:.2f} kW")

    st.subheader("Real-Time Notifications")
    for message in _notification_messages(current):
        st.info(message)

    st.subheader("Graph Time Window")
    window_options = {
        "1 Day": 1,
        "3 Days": 3,
        "7 Days": 7,
    }
    selected_window = st.selectbox(
        "Select graph window",
        options=list(window_options.keys()),
        index=0,
    )

    window_days = window_options[selected_window]
    latest_ts = status_df["timestamp"].max()
    window_start = latest_ts - pd.Timedelta(days=window_days)
    chart_window_df = status_df[status_df["timestamp"] >= window_start].copy()
    if chart_window_df.empty:
        chart_window_df = status_df.copy()

    st.subheader(f"Load Profile ({selected_window})")
    load_profile = chart_window_df[["timestamp", "load_total_W"]].rename(columns={"load_total_W": "Load (W)"})
    shed_spans = _build_shed_intervals(chart_window_df, latest_ts=latest_ts)

    load_line = (
        alt.Chart(load_profile)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("Load (W):Q", title="Load (W)"),
            tooltip=["timestamp:T", alt.Tooltip("Load (W):Q", format=".2f")],
        )
        .properties(height=320)
    )
    if shed_spans.empty:
        load_chart = load_line
    else:
        span_rects = (
            alt.Chart(shed_spans)
            .mark_rect(opacity=0.18)
            .encode(
                x=alt.X("start:T"),
                x2=alt.X2("end:T"),
                y=alt.value(0),
                y2=alt.value(320),
                color=alt.Color(
                    "event:N",
                    scale=alt.Scale(
                        domain=["non_essential_shed", "all_shed"],
                        range=["#ffd54f", "#ef5350"],
                    ),
                    legend=alt.Legend(title="Shed State"),
                ),
            )
        )
        load_chart = alt.layer(span_rects, load_line)
    st.altair_chart(load_chart, use_container_width=True)
    st.caption("🔵 Load profile | 🟡 Non-essential loads shed | 🔴 All loads shed")

    st.subheader(f"SoC vs Solar Generation ({selected_window})")
    chart_df = chart_window_df[["timestamp", "soc", "solar_ac_W"]].copy()
    chart_df["soc_percent"] = chart_df["soc"] * 100.0

    soc_chart = (
        alt.Chart(chart_df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("soc_percent:Q", title="SoC (%)"),
            tooltip=["timestamp:T", alt.Tooltip("soc_percent:Q", format=".2f")],
        )
        .properties(height=320)
    )
    solar_chart = (
        alt.Chart(chart_df)
        .mark_line(color="#ff7f0e")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("solar_ac_W:Q", title="Solar Generation (W)"),
            tooltip=["timestamp:T", alt.Tooltip("solar_ac_W:Q", format=".2f")],
        )
    )
    if shed_spans.empty:
        layered = alt.layer(soc_chart, solar_chart).resolve_scale(y="independent")
    else:
        span_rects_soc = (
            alt.Chart(shed_spans)
            .mark_rect(opacity=0.18)
            .encode(
                x=alt.X("start:T"),
                x2=alt.X2("end:T"),
                y=alt.value(0),
                y2=alt.value(320),
                color=alt.Color(
                    "event:N",
                    scale=alt.Scale(
                        domain=["non_essential_shed", "all_shed"],
                        range=["#ffd54f", "#ef5350"],
                    ),
                    legend=alt.Legend(title="Shed State"),
                ),
            )
        )
        layered = alt.layer(span_rects_soc, soc_chart, solar_chart).resolve_scale(y="independent")
    st.altair_chart(layered, use_container_width=True)
    st.caption("🔵 SoC (%) | 🟠 Solar generation (W) | 🟡 Non-essential loads shed | 🔴 All loads shed")


if __name__ == "__main__":
    main()

