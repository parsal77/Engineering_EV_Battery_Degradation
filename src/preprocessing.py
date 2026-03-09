"""Preprocessing pipeline for NASA EV battery degradation data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    DEFAULT_BATTERY_IDS,
    ParsedCycle,
    download_battery_mat_files,
    parse_battery_cycles,
    project_root,
)

NOMINAL_CAPACITY_AH = 2.0
EOL_CAPACITY_AH = 1.4
RANDOM_SEED = 42


def calculate_capacity_ah(current: np.ndarray, time_seconds: np.ndarray) -> float:
    """Compute discharge capacity in ampere-hours using trapezoidal integration.

    Parameters
    ----------
    current : numpy.ndarray
        Current series in amps.
    time_seconds : numpy.ndarray
        Time series in seconds.

    Returns
    -------
    float
        Capacity in Ah.
    """

    current_arr = np.asarray(current, dtype=float).reshape(-1)
    time_arr = np.asarray(time_seconds, dtype=float).reshape(-1)
    if current_arr.size < 2 or time_arr.size < 2:
        return float("nan")
    # Current is often negative during discharge; absolute value gives delivered charge.
    capacity_ah = np.trapezoid(np.abs(current_arr), time_arr) / 3600.0
    return float(max(capacity_ah, 0.0))


def calculate_energy_wh(
    voltage: np.ndarray, current: np.ndarray, time_seconds: np.ndarray
) -> float:
    """Compute cycle energy in watt-hours using trapezoidal integration.

    Parameters
    ----------
    voltage : numpy.ndarray
        Voltage series in volts.
    current : numpy.ndarray
        Current series in amps.
    time_seconds : numpy.ndarray
        Time series in seconds.

    Returns
    -------
    float
        Energy in Wh.
    """

    voltage_arr = np.asarray(voltage, dtype=float).reshape(-1)
    current_arr = np.asarray(current, dtype=float).reshape(-1)
    time_arr = np.asarray(time_seconds, dtype=float).reshape(-1)
    if min(voltage_arr.size, current_arr.size, time_arr.size) < 2:
        return float("nan")
    power_w = voltage_arr * np.abs(current_arr)
    energy_wh = np.trapezoid(power_w, time_arr) / 3600.0
    return float(max(energy_wh, 0.0))


def compute_soh(
    capacity_ah: float, nominal_capacity_ah: float = NOMINAL_CAPACITY_AH
) -> float:
    """Compute state of health (SOH) percentage.

    Parameters
    ----------
    capacity_ah : float
        Capacity for a cycle in Ah.
    nominal_capacity_ah : float, optional
        Nominal capacity in Ah.

    Returns
    -------
    float
        SOH in percent.
    """

    return float((capacity_ah / nominal_capacity_ah) * 100.0)


def find_eol_cycle(
    capacity_series: pd.Series, eol_capacity_ah: float = EOL_CAPACITY_AH
) -> int:
    """Find the first cycle index where capacity falls below EOL threshold.

    Parameters
    ----------
    capacity_series : pandas.Series
        Capacity values indexed by discharge cycle order.
    eol_capacity_ah : float, optional
        End-of-life capacity threshold.

    Returns
    -------
    int
        EOL cycle number (1-based discharge cycle count). If no EOL is reached,
        returns the last available cycle number.
    """

    below_threshold = capacity_series <= eol_capacity_ah
    if below_threshold.any():
        return int(capacity_series.index[below_threshold][0])
    return int(capacity_series.index.max())


def _cycle_duration_seconds(time_series: np.ndarray) -> float:
    """Compute cycle duration in seconds from a time series."""

    arr = np.asarray(time_series, dtype=float).reshape(-1)
    if arr.size < 2:
        return float("nan")
    return float(arr[-1] - arr[0])


def _safe_array(data: dict[str, object], key: str) -> np.ndarray:
    """Extract a numeric array from a parsed cycle data dictionary."""

    value = data.get(key)
    if value is None:
        return np.array([], dtype=float)
    if np.isscalar(value):
        return np.array([float(value)], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def extract_discharge_records(
    cycles: list[ParsedCycle],
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
    eol_capacity_ah: float = EOL_CAPACITY_AH,
) -> pd.DataFrame:
    """Extract one row per discharge cycle with engineered summary statistics.

    Parameters
    ----------
    cycles : list[ParsedCycle]
        Parsed cycle records for one battery.
    nominal_capacity_ah : float, optional
        Nominal battery capacity.
    eol_capacity_ah : float, optional
        End-of-life capacity threshold.

    Returns
    -------
    pandas.DataFrame
        Discharge-cycle summary and profile columns.
    """

    records: list[dict[str, object]] = []
    discharge_counter = 0
    last_charge_duration = float("nan")
    last_charge_energy = float("nan")

    for cycle in cycles:
        time_series = _safe_array(cycle.data, "Time")
        voltage_series = _safe_array(cycle.data, "Voltage_measured")
        current_series = _safe_array(cycle.data, "Current_measured")
        temp_series = _safe_array(cycle.data, "Temperature_measured")

        if cycle.cycle_type == "charge":
            last_charge_duration = _cycle_duration_seconds(time_series)
            last_charge_energy = calculate_energy_wh(
                voltage_series, current_series, time_series
            )
            continue

        if cycle.cycle_type != "discharge":
            continue

        discharge_counter += 1
        capacity_ah = calculate_capacity_ah(
            current=current_series, time_seconds=time_series
        )

        records.append(
            {
                "battery_id": cycle.battery_id,
                "cycle_number": discharge_counter,
                "capacity_Ah": capacity_ah,
                "avg_voltage": (
                    float(np.nanmean(voltage_series))
                    if voltage_series.size
                    else float("nan")
                ),
                "avg_current": (
                    float(np.nanmean(np.abs(current_series)))
                    if current_series.size
                    else float("nan")
                ),
                "avg_temperature": (
                    float(np.nanmean(temp_series)) if temp_series.size else float("nan")
                ),
                "charge_time": last_charge_duration,
                "discharge_time": _cycle_duration_seconds(time_series),
                "energy_charged_Wh": last_charge_energy,
                "discharge_voltage_profile": voltage_series,
                "discharge_current_profile": current_series,
                "discharge_temperature_profile": temp_series,
                "discharge_time_profile": time_series,
            }
        )

    discharge_df = pd.DataFrame.from_records(records)
    discharge_df = discharge_df.sort_values("cycle_number").reset_index(drop=True)

    discharge_df["SOH"] = discharge_df["capacity_Ah"].map(
        lambda cap: compute_soh(
            capacity_ah=float(cap), nominal_capacity_ah=nominal_capacity_ah
        )
    )
    indexed_caps = discharge_df.set_index("cycle_number")["capacity_Ah"]
    eol_cycle = find_eol_cycle(indexed_caps, eol_capacity_ah=eol_capacity_ah)
    discharge_df["RUL"] = (
        (eol_cycle - discharge_df["cycle_number"]).clip(lower=0).astype(int)
    )

    return discharge_df


def preprocess_battery_file(
    mat_path: Path,
    processed_dir: Path,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
    eol_capacity_ah: float = EOL_CAPACITY_AH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process one battery MAT file and export its clean CSV.

    Parameters
    ----------
    mat_path : pathlib.Path
        Input MAT file path.
    processed_dir : pathlib.Path
        Output directory for processed CSV files.
    nominal_capacity_ah : float, optional
        Nominal capacity in Ah.
    eol_capacity_ah : float, optional
        End-of-life threshold in Ah.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(summary_df, detailed_df)`` where summary has only final export columns,
        and detailed retains profile columns for feature engineering.
    """

    processed_dir.mkdir(parents=True, exist_ok=True)
    battery_id = mat_path.stem
    cycles = parse_battery_cycles(mat_path=mat_path, battery_id=battery_id)
    detailed_df = extract_discharge_records(
        cycles=cycles,
        nominal_capacity_ah=nominal_capacity_ah,
        eol_capacity_ah=eol_capacity_ah,
    )

    summary_columns = [
        "battery_id",
        "cycle_number",
        "capacity_Ah",
        "SOH",
        "RUL",
        "avg_voltage",
        "avg_current",
        "avg_temperature",
        "charge_time",
        "discharge_time",
    ]
    summary_df = detailed_df.loc[:, summary_columns].copy()
    summary_df.to_csv(processed_dir / f"{battery_id}_processed.csv", index=False)

    return summary_df, detailed_df


def run_preprocessing_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    battery_ids: tuple[str, ...] = DEFAULT_BATTERY_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run preprocessing for all batteries and export merged CSV.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Directory containing MAT files.
    processed_dir : pathlib.Path
        Directory to save processed CSV outputs.
    battery_ids : tuple[str, ...], optional
        Battery IDs to include.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(all_summary_df, all_detailed_df)`` across all batteries.
    """

    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    detailed_frames: list[pd.DataFrame] = []

    for battery_id in battery_ids:
        mat_path = raw_dir / f"{battery_id}.mat"
        summary_df, detailed_df = preprocess_battery_file(
            mat_path=mat_path,
            processed_dir=processed_dir,
            nominal_capacity_ah=NOMINAL_CAPACITY_AH,
            eol_capacity_ah=EOL_CAPACITY_AH,
        )
        summary_frames.append(summary_df)
        detailed_frames.append(detailed_df)

    all_summary_df = pd.concat(summary_frames, ignore_index=True)
    all_detailed_df = pd.concat(detailed_frames, ignore_index=True)

    all_summary_df.to_csv(processed_dir / "all_batteries.csv", index=False)
    return all_summary_df, all_detailed_df


def main() -> None:
    """CLI entry point for preprocessing and base CSV generation."""

    root = project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    download_battery_mat_files(output_dir=raw_dir, battery_ids=DEFAULT_BATTERY_IDS)
    run_preprocessing_pipeline(
        raw_dir=raw_dir, processed_dir=processed_dir, battery_ids=DEFAULT_BATTERY_IDS
    )


if __name__ == "__main__":
    main()
