"""Feature engineering utilities for battery degradation modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_BATTERY_IDS
from src.preprocessing import calculate_energy_wh, run_preprocessing_pipeline


def compute_internal_resistance_proxy(
    voltage: np.ndarray, current: np.ndarray
) -> float:
    """Estimate an internal resistance proxy from early discharge dynamics.

    Parameters
    ----------
    voltage : numpy.ndarray
        Voltage profile for one discharge cycle.
    current : numpy.ndarray
        Current profile for one discharge cycle.

    Returns
    -------
    float
        Proxy resistance estimate in ohms. Returns NaN when undefined.
    """

    voltage_arr = np.asarray(voltage, dtype=float).reshape(-1)
    current_arr = np.asarray(current, dtype=float).reshape(-1)
    if min(voltage_arr.size, current_arr.size) < 2:
        return float("nan")

    delta_v = voltage_arr[1] - voltage_arr[0]
    delta_i = current_arr[1] - current_arr[0]

    if np.isclose(delta_i, 0.0):
        diffs = np.diff(current_arr)
        valid = np.where(np.abs(diffs) > 1e-6)[0]
        if valid.size == 0:
            return float("nan")
        idx = int(valid[0])
        delta_v = voltage_arr[idx + 1] - voltage_arr[idx]
        delta_i = current_arr[idx + 1] - current_arr[idx]

    if np.isclose(delta_i, 0.0):
        return float("nan")

    return float(np.abs(delta_v / delta_i))


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: tuple[int, ...],
    group_col: str = "battery_id",
) -> pd.DataFrame:
    """Add grouped lag features for a numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Target column name for lag computation.
    lags : tuple[int, ...]
        Lag periods to create.
    group_col : str, optional
        Grouping column (battery ID).

    Returns
    -------
    pandas.DataFrame
        DataFrame with lag feature columns appended.
    """

    out_df = df.copy()
    grouped = out_df.groupby(group_col)[column]
    for lag in lags:
        out_df[f"{column}_lag_{lag}"] = grouped.shift(lag)
    return out_df


def engineer_features_from_detailed(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Create requested engineered features from detailed cycle records.

    Parameters
    ----------
    detailed_df : pandas.DataFrame
        Detailed discharge DataFrame produced by preprocessing.

    Returns
    -------
    pandas.DataFrame
        Feature-enhanced dataset with one row per discharge cycle.
    """

    feature_df = (
        detailed_df.copy()
        .sort_values(["battery_id", "cycle_number"])
        .reset_index(drop=True)
    )

    feature_df["max_voltage"] = feature_df["discharge_voltage_profile"].map(
        lambda arr: float(np.nanmax(arr)) if np.asarray(arr).size else float("nan")
    )
    feature_df["min_voltage"] = feature_df["discharge_voltage_profile"].map(
        lambda arr: float(np.nanmin(arr)) if np.asarray(arr).size else float("nan")
    )
    feature_df["voltage_range"] = feature_df["max_voltage"] - feature_df["min_voltage"]

    feature_df["max_temperature"] = feature_df["discharge_temperature_profile"].map(
        lambda arr: float(np.nanmax(arr)) if np.asarray(arr).size else float("nan")
    )
    feature_df["temperature_rise"] = feature_df["discharge_temperature_profile"].map(
        lambda arr: (
            float(np.nanmax(arr) - np.asarray(arr, dtype=float).reshape(-1)[0])
            if np.asarray(arr).size
            else float("nan")
        )
    )

    feature_df["discharge_duration"] = feature_df["discharge_time"]

    feature_df["energy_discharged"] = feature_df.apply(
        lambda row: calculate_energy_wh(
            voltage=np.asarray(row["discharge_voltage_profile"], dtype=float),
            current=np.asarray(row["discharge_current_profile"], dtype=float),
            time_seconds=np.asarray(row["discharge_time_profile"], dtype=float),
        ),
        axis=1,
    )

    feature_df["internal_resistance_proxy"] = feature_df.apply(
        lambda row: compute_internal_resistance_proxy(
            voltage=np.asarray(row["discharge_voltage_profile"], dtype=float),
            current=np.asarray(row["discharge_current_profile"], dtype=float),
        ),
        axis=1,
    )

    feature_df["capacity_fade_rate"] = (
        feature_df.groupby("battery_id")["capacity_Ah"].diff(5) / 5.0
    )

    feature_df["cycle_efficiency"] = (
        feature_df["energy_discharged"] / feature_df["energy_charged_Wh"]
    )

    feature_df = add_lag_features(
        feature_df, column="capacity_Ah", lags=(1, 3, 5), group_col="battery_id"
    )
    feature_df = feature_df.rename(
        columns={
            "capacity_Ah_lag_1": "capacity_lag_1",
            "capacity_Ah_lag_3": "capacity_lag_3",
            "capacity_Ah_lag_5": "capacity_lag_5",
        }
    )

    feature_df = feature_df.drop(
        columns=[
            "discharge_voltage_profile",
            "discharge_current_profile",
            "discharge_temperature_profile",
            "discharge_time_profile",
        ]
    )

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_cols] = feature_df.groupby("battery_id")[numeric_cols].transform(
        lambda grp: grp.ffill().bfill()
    )

    return feature_df


def build_and_save_feature_dataset(
    raw_dir: Path,
    processed_dir: Path,
    battery_ids: tuple[str, ...] = DEFAULT_BATTERY_IDS,
) -> pd.DataFrame:
    """Run preprocessing + feature engineering and persist ``features_all.csv``.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Folder containing raw MAT files.
    processed_dir : pathlib.Path
        Output folder for processed CSV files.
    battery_ids : tuple[str, ...], optional
        Battery IDs to process.

    Returns
    -------
    pandas.DataFrame
        Feature dataset.
    """

    _, detailed_df = run_preprocessing_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        battery_ids=battery_ids,
    )
    feature_df = engineer_features_from_detailed(detailed_df)
    feature_df.to_csv(processed_dir / "features_all.csv", index=False)
    return feature_df
