"""Evaluation helpers for model predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import regression_metrics


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate one regression model output.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    dict[str, float]
        Metrics dictionary with RMSE, MAE, MAPE, and R2.
    """

    return regression_metrics(y_true=y_true, y_pred=y_pred)


def compare_models(records: list[dict[str, float | str]]) -> pd.DataFrame:
    """Create a benchmark table from metric records.

    Parameters
    ----------
    records : list[dict[str, float | str]]
        List of model result rows.

    Returns
    -------
    pandas.DataFrame
        Sorted metrics table.
    """

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    return df.sort_values(["Task", "RMSE", "Model"]).reset_index(drop=True)


def save_results_table(
    records: list[dict[str, float | str]], output_path: Path
) -> pd.DataFrame:
    """Persist benchmark results to CSV.

    Parameters
    ----------
    records : list[dict[str, float | str]]
        Model metric records.
    output_path : pathlib.Path
        Output CSV path.

    Returns
    -------
    pandas.DataFrame
        Saved benchmark table.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = compare_models(records)
    df.to_csv(output_path, index=False)
    return df
