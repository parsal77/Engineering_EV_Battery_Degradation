"""Training helpers for battery degradation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data_loader import project_root
from src.models import default_feature_columns, prepare_cross_battery_data


def load_data(features_path: Path | None = None) -> pd.DataFrame:
    """Load feature-engineered dataset from disk.

    Parameters
    ----------
    features_path : pathlib.Path | None, optional
        Path to ``features_all.csv``. Uses default project path when omitted.

    Returns
    -------
    pandas.DataFrame
        Feature table.
    """

    if features_path is None:
        features_path = project_root() / "data" / "processed" / "features_all.csv"
    return pd.read_csv(features_path)


def create_features(
    df: pd.DataFrame,
    target_col: str,
    drop_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build feature matrix and target vector.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target_col : str
        Target column name.
    drop_columns : set[str] | None, optional
        Additional columns to exclude from model features.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, list[str]]
        ``(X, y, feature_columns)``.
    """

    feature_columns = default_feature_columns(df, exclude=drop_columns or set())
    X = df.loc[:, feature_columns].copy()
    y = df.loc[:, target_col].copy()
    return X, y, feature_columns


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_battery: str = "B0018",
    drop_columns: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Split data by battery-level holdout with scaling.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target_col : str
        Target column name.
    test_battery : str, optional
        Holdout battery ID.
    drop_columns : set[str] | None, optional
        Extra columns to exclude from features.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list[str]]
        ``(X_train, X_test, y_train, y_test, feature_columns)``.
    """

    feature_columns = default_feature_columns(df, exclude=drop_columns or set())
    prepared = prepare_cross_battery_data(
        df=df,
        target_col=target_col,
        test_battery=test_battery,
        feature_columns=feature_columns,
    )
    return (
        prepared.X_train,
        prepared.X_test,
        prepared.y_train,
        prepared.y_test,
        prepared.feature_columns,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "random_forest",
    random_state: int = 42,
) -> Any:
    """Train a baseline regression model.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix.
    y_train : numpy.ndarray
        Training target vector.
    model_name : str, optional
        Model type. Supported: ``random_forest``, ``linear_regression``.
    random_state : int, optional
        Random seed.

    Returns
    -------
    Any
        Fitted model.
    """

    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "linear_regression":
        model = LinearRegression()
    else:
        raise ValueError(
            "Unsupported model_name. Use 'random_forest' or 'linear_regression'."
        )

    model.fit(X_train, y_train)
    return model
