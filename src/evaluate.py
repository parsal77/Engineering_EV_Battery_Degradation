"""Evaluation helpers and CLI pipeline for benchmark reproducibility."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import project_root
from src.evaluation import regression_metrics, save_metrics
from src.models import run_rul_benchmark, run_soh_benchmark
from src.reporting import update_readme_metrics_block, write_results_summary
from src.visualisation import plot_actual_vs_predicted


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


def run_evaluation_pipeline(
    feature_path: Path | None = None,
    models_dir: Path | None = None,
    results_dir: Path | None = None,
    reports_dir: Path | None = None,
    readme_path: Path | None = None,
    test_battery: str = "B0018",
    sequence_length: int = 10,
    include_transformer: bool = True,
    update_readme: bool = True,
) -> pd.DataFrame:
    """Run full benchmark evaluation from a feature dataset.

    Parameters
    ----------
    feature_path : pathlib.Path | None, optional
        Path to ``features_all.csv``.
    models_dir : pathlib.Path | None, optional
        Output directory for model artifacts.
    results_dir : pathlib.Path | None, optional
        Output directory for metrics/plots.
    reports_dir : pathlib.Path | None, optional
        Output directory for markdown reports.
    readme_path : pathlib.Path | None, optional
        README path for auto metrics block updates.
    test_battery : str, optional
        Battery ID for holdout testing.
    sequence_length : int, optional
        Sequence length for deep models.
    include_transformer : bool, optional
        Whether to include Transformer in RUL benchmark.
    update_readme : bool, optional
        Whether to refresh README auto-metrics block.

    Returns
    -------
    pandas.DataFrame
        Final metrics table.
    """

    root = project_root()
    resolved_feature_path = (
        feature_path or root / "data" / "processed" / "features_all.csv"
    )
    resolved_models_dir = models_dir or root / "models"
    resolved_results_dir = results_dir or root / "results"
    resolved_reports_dir = reports_dir or root / "reports"
    resolved_readme_path = readme_path or root / "README.md"

    resolved_models_dir.mkdir(parents=True, exist_ok=True)
    resolved_results_dir.mkdir(parents=True, exist_ok=True)
    resolved_reports_dir.mkdir(parents=True, exist_ok=True)

    if not resolved_feature_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {resolved_feature_path}. "
            "Run preprocessing/feature engineering first."
        )

    feature_df = pd.read_csv(resolved_feature_path)

    soh_metrics, _ = run_soh_benchmark(
        feature_df=feature_df,
        models_dir=resolved_models_dir,
        test_battery=test_battery,
        sequence_length=sequence_length,
    )
    rul_metrics, rul_results = run_rul_benchmark(
        feature_df=feature_df,
        models_dir=resolved_models_dir,
        test_battery=test_battery,
        sequence_length=sequence_length,
        include_transformer=include_transformer,
    )

    metrics_path = resolved_results_dir / "metrics.csv"
    metrics_df = save_metrics(soh_metrics + rul_metrics, metrics_path)

    write_results_summary(
        metrics_df=metrics_df, output_path=resolved_reports_dir / "results_summary.md"
    )

    best_rul = metrics_df[metrics_df["Task"] == "RUL"].sort_values("RMSE").iloc[0]
    best_rul_model_name = str(best_rul["Model"])
    best_payload = rul_results[best_rul_model_name]

    plot_actual_vs_predicted(
        cycle_number=np.asarray(best_payload["cycle_number"]),
        y_true=np.asarray(best_payload["y_true"]),
        y_pred=np.asarray(best_payload["predictions"]),
        title=f"Best RUL Model: {best_rul_model_name} on {test_battery}",
        y_label="RUL (cycles)",
        output_path=resolved_results_dir / "pred_vs_actual.png",
    )

    if update_readme:
        update_readme_metrics_block(resolved_readme_path, metrics_df=metrics_df)

    return metrics_df


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for evaluation CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        description="Run end-to-end benchmark evaluation pipeline."
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=None,
        help="Path to features_all.csv (default: data/processed/features_all.csv).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory for saved model artifacts (default: models/).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for metrics and generated plots (default: results/).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory for markdown reports (default: reports/).",
    )
    parser.add_argument(
        "--readme-path",
        type=Path,
        default=None,
        help="README path for auto-metrics refresh (default: README.md).",
    )
    parser.add_argument(
        "--test-battery",
        type=str,
        default="B0018",
        help="Holdout battery ID used for testing.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length for deep models.",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Disable Transformer model in RUL benchmark.",
    )
    parser.add_argument(
        "--skip-readme-update",
        action="store_true",
        help="Skip replacing README auto-metrics block.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    metrics_df = run_evaluation_pipeline(
        feature_path=args.feature_path,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        readme_path=args.readme_path,
        test_battery=args.test_battery,
        sequence_length=args.sequence_length,
        include_transformer=not args.no_transformer,
        update_readme=not args.skip_readme_update,
    )

    print("Saved metrics to results/metrics.csv and refreshed reports/README blocks.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
