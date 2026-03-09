"""Plotting utilities for EV battery degradation analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_SEED = 42


def apply_professional_style() -> None:
    """Apply a consistent publication-style plot theme."""

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titleweight"] = "bold"


def _save_current_figure(output_path: Path, dpi: int = 300) -> None:
    """Save and close current matplotlib figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_capacity_fade(
    df: pd.DataFrame, output_path: Path, eol_capacity: float = 1.4
) -> None:
    """Plot capacity fade curves for all batteries."""

    apply_professional_style()
    plt.figure()
    sns.lineplot(
        data=df, x="cycle_number", y="capacity_Ah", hue="battery_id", linewidth=2
    )
    plt.axhline(
        eol_capacity, color="red", linestyle="--", linewidth=2, label="EOL (1.4 Ah)"
    )
    plt.title("Capacity Fade Curves Across Batteries")
    plt.xlabel("Cycle Number")
    plt.ylabel("Capacity (Ah)")
    plt.legend()
    _save_current_figure(output_path)


def plot_soh_degradation(df: pd.DataFrame, output_path: Path) -> None:
    """Plot SOH degradation for all batteries."""

    apply_professional_style()
    plt.figure()
    sns.lineplot(data=df, x="cycle_number", y="SOH", hue="battery_id", linewidth=2)
    plt.title("SOH Degradation Over Cycles")
    plt.xlabel("Cycle Number")
    plt.ylabel("SOH (%)")
    plt.legend()
    _save_current_figure(output_path)


def plot_rul_ground_truth(df: pd.DataFrame, output_path: Path) -> None:
    """Plot RUL against cycle number."""

    apply_professional_style()
    plt.figure()
    sns.lineplot(data=df, x="cycle_number", y="RUL", hue="battery_id", linewidth=2)
    plt.title("Ground-Truth RUL Over Cycles")
    plt.xlabel("Cycle Number")
    plt.ylabel("RUL (cycles)")
    plt.legend()
    _save_current_figure(output_path)


def plot_voltage_profile_comparison(
    detailed_df: pd.DataFrame,
    battery_id: str,
    output_path: Path,
) -> None:
    """Overlay first/middle/last discharge voltage profiles for one battery."""

    subset = detailed_df[detailed_df["battery_id"] == battery_id].sort_values(
        "cycle_number"
    )
    if subset.empty:
        raise ValueError(f"No rows found for battery '{battery_id}'.")

    indices = [0, len(subset) // 2, len(subset) - 1]
    labels = ["First Cycle", "Middle Cycle", "Last Cycle"]

    apply_professional_style()
    plt.figure()

    for idx, label in zip(indices, labels):
        row = subset.iloc[idx]
        time_series = np.asarray(row["discharge_time_profile"], dtype=float)
        voltage_series = np.asarray(row["discharge_voltage_profile"], dtype=float)
        plt.plot(
            time_series,
            voltage_series,
            linewidth=2,
            label=f"{label} (#{int(row['cycle_number'])})",
        )

    plt.title(f"Voltage Profile Comparison for {battery_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    _save_current_figure(output_path)


def plot_temperature_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot temperature distribution using histogram + KDE."""

    apply_professional_style()
    plt.figure()
    sns.histplot(data=df, x="avg_temperature", bins=30, kde=True, color="#1f77b4")
    plt.title("Temperature Distribution Across Discharge Cycles")
    plt.xlabel("Average Temperature (degC)")
    plt.ylabel("Count")
    _save_current_figure(output_path)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot Pearson correlation heatmap for numeric features."""

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    apply_professional_style()
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0.0, annot=False)
    plt.title("Feature Correlation Heatmap")
    _save_current_figure(output_path)


def plot_pairplot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot pairplot of selected key features colored by battery."""

    cols = [
        "battery_id",
        "capacity_Ah",
        "SOH",
        "RUL",
        "avg_voltage",
        "avg_current",
        "avg_temperature",
    ]
    pair_df = df.loc[:, cols].copy()
    sns.set_theme(style="whitegrid")
    grid = sns.pairplot(pair_df, hue="battery_id", diag_kind="kde", corner=True)
    grid.fig.suptitle("Pairplot of Key Battery Features", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def plot_actual_vs_predicted(
    cycle_number: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Plot actual and predicted target values over cycle index."""

    apply_professional_style()
    plt.figure()
    plt.plot(cycle_number, y_true, label="Actual", linewidth=2)
    plt.plot(cycle_number, y_pred, label="Predicted", linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel("Cycle Number")
    plt.ylabel(y_label)
    plt.legend()
    _save_current_figure(output_path)


def plot_rul_error(
    cycle_number: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
) -> None:
    """Plot RUL prediction error over cycle number."""

    error = np.asarray(y_pred) - np.asarray(y_true)
    apply_professional_style()
    plt.figure()
    plt.plot(cycle_number, error, color="#d62728", linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--")
    plt.title("RUL Prediction Error Over Cycle Number")
    plt.xlabel("Cycle Number")
    plt.ylabel("Prediction Error (cycles)")
    _save_current_figure(output_path)


def plot_rmse_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Plot RMSE comparison across models and tasks."""

    apply_professional_style()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x="Model", y="RMSE", hue="Task")
    plt.xticks(rotation=30, ha="right")
    plt.title("RMSE Comparison Across SOH and RUL Models")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    _save_current_figure(output_path)


def plot_predicted_vs_actual_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Plot scatter of predicted vs actual targets."""

    apply_professional_style()
    plt.figure()
    sns.scatterplot(x=y_true, y=y_pred, s=40, alpha=0.7)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, color="red", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    _save_current_figure(output_path)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Plot top-n feature importances from a tree model."""

    top_df = importance_df.sort_values("importance", ascending=False).head(top_n)
    apply_professional_style()
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_df, x="importance", y="feature", orient="h")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    _save_current_figure(output_path)


def plot_learning_curve(
    history: dict[str, list[float]], title: str, output_path: Path
) -> None:
    """Plot train/validation loss curves for neural network training."""

    apply_professional_style()
    plt.figure()
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    _save_current_figure(output_path)
