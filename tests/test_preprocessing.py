"""Unit tests for the preprocessing and feature engineering pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.data_loader import parse_battery_cycles
from src.features import engineer_features_from_detailed
from src.preprocessing import (
    calculate_capacity_ah,
    compute_soh,
    preprocess_battery_file,
)


@pytest.fixture()
def mock_battery_mat(tmp_path: Path) -> Path:
    """Create a minimal NASA-like MAT file with charge/discharge cycles.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    pathlib.Path
        Path to generated MAT file.
    """

    cycle_dtype = np.dtype(
        [
            ("type", "O"),
            ("ambient_temperature", "O"),
            ("time", "O"),
            ("data", "O"),
        ]
    )

    charge_data_dtype = np.dtype(
        [
            ("Voltage_measured", "O"),
            ("Current_measured", "O"),
            ("Temperature_measured", "O"),
            ("Time", "O"),
        ]
    )
    discharge_data_dtype = np.dtype(
        [
            ("Voltage_measured", "O"),
            ("Current_measured", "O"),
            ("Temperature_measured", "O"),
            ("Time", "O"),
            ("Capacity", "O"),
        ]
    )

    cycles = np.empty((1, 4), dtype=cycle_dtype)

    for cycle_idx in range(4):
        is_charge = cycle_idx % 2 == 0
        if is_charge:
            data = np.empty((1, 1), dtype=charge_data_dtype)
            data[0, 0]["Voltage_measured"] = np.array([[4.2, 4.15, 4.1]])
            data[0, 0]["Current_measured"] = np.array([[1.0, 0.9, 0.2]])
            data[0, 0]["Temperature_measured"] = np.array([[24.0, 25.0, 26.0]])
            data[0, 0]["Time"] = np.array([[0.0, 10.0, 20.0]])
            cycle_type = "charge"
        else:
            data = np.empty((1, 1), dtype=discharge_data_dtype)
            data[0, 0]["Voltage_measured"] = np.array([[4.1, 3.9, 3.7]])
            data[0, 0]["Current_measured"] = np.array([[-2.0, -2.0, -2.0]])
            data[0, 0]["Temperature_measured"] = np.array([[25.0, 26.0, 28.0]])
            data[0, 0]["Time"] = np.array([[0.0, 1800.0, 3600.0]])
            data[0, 0]["Capacity"] = np.array([[2.0]])
            cycle_type = "discharge"

        cycles[0, cycle_idx]["type"] = np.array([cycle_type])
        cycles[0, cycle_idx]["ambient_temperature"] = np.array([[24]], dtype=np.uint8)
        cycles[0, cycle_idx]["time"] = np.array(
            [[2020, 1, 1, cycle_idx, 0, 0]], dtype=float
        )
        cycles[0, cycle_idx]["data"] = data

    battery = np.empty((1, 1), dtype=np.dtype([("cycle", "O")]))
    battery[0, 0]["cycle"] = cycles

    output_path = tmp_path / "BTEST.mat"
    savemat(output_path, {"BTEST": battery})
    return output_path


def test_mat_file_loading(mock_battery_mat: Path) -> None:
    """MAT parser should load all cycles and detect discharge cycles."""

    cycles = parse_battery_cycles(mock_battery_mat, battery_id="BTEST")
    assert len(cycles) == 4
    assert sum(c.cycle_type == "discharge" for c in cycles) == 2


def test_capacity_calculation() -> None:
    """Capacity integration should match analytical solution."""

    time_seconds = np.array([0.0, 1800.0, 3600.0])
    current = np.array([-2.0, -2.0, -2.0])
    capacity = calculate_capacity_ah(current=current, time_seconds=time_seconds)
    assert capacity == pytest.approx(2.0, abs=1e-6)


def test_soh_computation() -> None:
    """SOH should be computed as percent of nominal capacity."""

    soh = compute_soh(capacity_ah=1.8, nominal_capacity_ah=2.0)
    assert soh == pytest.approx(90.0, abs=1e-6)


def test_feature_engineering_columns(mock_battery_mat: Path, tmp_path: Path) -> None:
    """Feature engineering should produce requested feature columns."""

    _, detailed_df = preprocess_battery_file(
        mat_path=mock_battery_mat,
        processed_dir=tmp_path,
        nominal_capacity_ah=2.0,
        eol_capacity_ah=1.4,
    )
    features_df = engineer_features_from_detailed(detailed_df)

    expected_cols = {
        "max_voltage",
        "min_voltage",
        "voltage_range",
        "max_temperature",
        "temperature_rise",
        "discharge_duration",
        "energy_discharged",
        "internal_resistance_proxy",
        "capacity_fade_rate",
        "cycle_efficiency",
        "capacity_lag_1",
        "capacity_lag_3",
        "capacity_lag_5",
    }
    assert expected_cols.issubset(features_df.columns)


def test_preprocessed_output_shape(mock_battery_mat: Path, tmp_path: Path) -> None:
    """Processed output must include required columns and one row per discharge cycle."""

    summary_df, _ = preprocess_battery_file(
        mat_path=mock_battery_mat,
        processed_dir=tmp_path,
        nominal_capacity_ah=2.0,
        eol_capacity_ah=1.4,
    )

    required_columns = [
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

    assert list(summary_df.columns) == required_columns
    assert summary_df.shape[0] == 2
    assert summary_df["battery_id"].eq("BTEST").all()

    persisted = pd.read_csv(tmp_path / "BTEST_processed.csv")
    assert persisted.shape == summary_df.shape
