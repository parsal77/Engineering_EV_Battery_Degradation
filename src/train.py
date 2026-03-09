"""Training utilities and CLI pipeline orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data_loader import (
    DEFAULT_BATTERY_IDS,
    download_battery_mat_files,
    project_root,
)
from src.features import build_and_save_feature_dataset
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


def build_feature_dataset(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    battery_ids: tuple[str, ...] = DEFAULT_BATTERY_IDS,
    skip_download: bool = False,
) -> Path:
    """Build the feature dataset from raw battery MAT files.

    Parameters
    ----------
    raw_dir : pathlib.Path | None, optional
        Directory containing raw MAT files.
    processed_dir : pathlib.Path | None, optional
        Directory for processed outputs.
    battery_ids : tuple[str, ...], optional
        Battery IDs to process.
    skip_download : bool, optional
        Skip MAT file download step.

    Returns
    -------
    pathlib.Path
        Path to generated ``features_all.csv``.
    """

    root = project_root()
    resolved_raw_dir = raw_dir or root / "data" / "raw"
    resolved_processed_dir = processed_dir or root / "data" / "processed"

    if not skip_download:
        download_battery_mat_files(output_dir=resolved_raw_dir, battery_ids=battery_ids)

    build_and_save_feature_dataset(
        raw_dir=resolved_raw_dir,
        processed_dir=resolved_processed_dir,
        battery_ids=battery_ids,
    )
    return resolved_processed_dir / "features_all.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for training CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        description="Build dataset and optionally run full benchmark evaluation."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw MAT directory (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading MAT files before preprocessing.",
    )
    parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help="Run full benchmark evaluation after feature generation.",
    )
    parser.add_argument(
        "--test-battery",
        type=str,
        default="B0018",
        help="Holdout battery ID for evaluation.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length for deep models.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    feature_path = build_feature_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        skip_download=args.skip_download,
    )
    print(f"Feature dataset generated: {feature_path}")

    if args.run_evaluation:
        from src.evaluate import run_evaluation_pipeline

        metrics_df = run_evaluation_pipeline(
            feature_path=feature_path,
            test_battery=args.test_battery,
            sequence_length=args.sequence_length,
        )
        print("Full evaluation completed.")
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
