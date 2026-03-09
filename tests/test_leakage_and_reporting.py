"""Tests for leakage controls and reporting synchronization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models import prepare_cross_battery_data
from src.reporting import (
    AUTO_METRICS_END,
    AUTO_METRICS_START,
    build_results_summary,
    update_readme_metrics_block,
)


def _mock_feature_df() -> pd.DataFrame:
    """Build a minimal feature table for split and leakage tests.

    Returns
    -------
    pandas.DataFrame
        Synthetic feature dataset.
    """

    return pd.DataFrame(
        {
            "battery_id": ["B0005", "B0005", "B0018", "B0018"],
            "cycle_number": [1, 2, 1, 2],
            "capacity_Ah": [1.95, 1.90, 1.92, 1.86],
            "SOH": [97.5, 95.0, 96.0, 93.0],
            "RUL": [120, 119, 90, 89],
            "avg_voltage": [3.7, 3.6, 3.7, 3.5],
            "avg_current": [1.9, 1.9, 1.8, 1.8],
            "avg_temperature": [30.0, 31.0, 29.0, 30.0],
            "discharge_time": [3600, 3590, 3580, 3570],
        }
    )


def test_prepare_cross_battery_data_default_prevents_soh_capacity_leakage() -> None:
    """Default SOH split should exclude direct capacity leakage feature."""

    df = _mock_feature_df()
    prepared = prepare_cross_battery_data(df=df, target_col="SOH", test_battery="B0018")
    assert "capacity_Ah" not in prepared.feature_columns


def test_prepare_cross_battery_data_rejects_forbidden_soh_features() -> None:
    """Explicitly passing leakage-prone SOH features should raise an error."""

    df = _mock_feature_df()
    with pytest.raises(ValueError, match="Leakage risk detected"):
        prepare_cross_battery_data(
            df=df,
            target_col="SOH",
            test_battery="B0018",
            feature_columns=["capacity_Ah", "avg_voltage"],
        )


def test_readme_metrics_block_updates_from_metrics_source(tmp_path: Path) -> None:
    """README auto block should be replaced from metrics source table."""

    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Demo",
                AUTO_METRICS_START,
                "stale content",
                AUTO_METRICS_END,
            ]
        ),
        encoding="utf-8",
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest Regressor",
                "Task": "RUL",
                "MAE": 5.0,
                "RMSE": 6.0,
                "MAPE": 10.0,
                "R2": 0.95,
                "Train_Time_s": 0.4,
            }
        ]
    )

    update_readme_metrics_block(readme_path=readme_path, metrics_df=metrics_df)
    updated = readme_path.read_text(encoding="utf-8")

    assert "stale content" not in updated
    assert "Random Forest Regressor" in updated
    assert "6.0000" in updated


def test_results_summary_builds_from_metrics_source() -> None:
    """Results summary should include rendered values from metrics table."""

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Linear Regression",
                "Task": "SOH",
                "MAE": 0.1,
                "RMSE": 0.2,
                "MAPE": 0.3,
                "R2": 0.9,
                "Train_Time_s": 0.01,
            },
            {
                "Model": "Random Forest Regressor",
                "Task": "RUL",
                "MAE": 5.0,
                "RMSE": 6.0,
                "MAPE": 10.0,
                "R2": 0.95,
                "Train_Time_s": 0.4,
            },
        ]
    )

    summary = build_results_summary(metrics_df)
    assert "Single source of truth" in summary
    assert "Linear Regression" in summary
    assert "6.0000" in summary
