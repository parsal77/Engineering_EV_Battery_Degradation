"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth targets.
    y_pred : numpy.ndarray
        Predicted targets.

    Returns
    -------
    dict[str, float]
        Dictionary with RMSE, MAE, MAPE, and R2.
    """

    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))

    non_zero_mask = np.abs(y_true_arr) > 1e-8
    if np.any(non_zero_mask):
        mape = float(
            np.mean(
                np.abs(
                    (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask])
                    / y_true_arr[non_zero_mask]
                )
            )
            * 100.0
        )
    else:
        mape = float(np.mean(np.abs(y_true_arr - y_pred_arr)) * 100.0)
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def metrics_to_frame(records: list[dict[str, float | str]]) -> pd.DataFrame:
    """Convert metric dictionaries to a sorted DataFrame.

    Parameters
    ----------
    records : list[dict[str, float | str]]
        Metric records.

    Returns
    -------
    pandas.DataFrame
        Metrics DataFrame.
    """

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    return df.sort_values(["Task", "RMSE", "Model"]).reset_index(drop=True)


def save_metrics(
    records: list[dict[str, float | str]], output_path: Path
) -> pd.DataFrame:
    """Save metric records to CSV.

    Parameters
    ----------
    records : list[dict[str, float | str]]
        Metric records.
    output_path : pathlib.Path
        Output CSV path.

    Returns
    -------
    pandas.DataFrame
        Saved metrics table.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = metrics_to_frame(records)
    df.to_csv(output_path, index=False)
    return df
