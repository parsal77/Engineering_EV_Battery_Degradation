"""Reporting utilities driven by a single metrics source of truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AUTO_METRICS_START = "<!-- AUTO_METRICS_TABLE_START -->"
AUTO_METRICS_END = "<!-- AUTO_METRICS_TABLE_END -->"


def _format_float(value: float, digits: int = 4) -> str:
    """Format numeric values consistently for markdown output.

    Parameters
    ----------
    value : float
        Numeric value to format.
    digits : int, optional
        Decimal places.

    Returns
    -------
    str
        Rounded string representation.
    """

    return f"{float(value):.{digits}f}"


def metrics_markdown_table(metrics_df: pd.DataFrame) -> str:
    """Render metrics as a markdown table.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Metrics DataFrame containing model benchmark rows.

    Returns
    -------
    str
        Markdown table string.
    """

    view_cols = ["Model", "Task", "MAE", "RMSE", "R2"]
    table_df = metrics_df.loc[:, view_cols].copy()
    for col in ["MAE", "RMSE", "R2"]:
        table_df[col] = table_df[col].map(_format_float)

    lines = []
    lines.append("| Model | Task | MAE | RMSE | R² |")
    lines.append("|---|---|---:|---:|---:|")
    for _, row in table_df.iterrows():
        lines.append(
            "| "
            f"{row['Model']} | {row['Task']} | {row['MAE']} | {row['RMSE']} | {row['R2']} |"
        )
    return "\n".join(lines)


def build_results_summary(metrics_df: pd.DataFrame) -> str:
    """Build markdown content for results summary report.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Benchmark metrics table.

    Returns
    -------
    str
        Markdown report content.
    """

    sorted_df = metrics_df.sort_values(["Task", "RMSE", "Model"]).reset_index(drop=True)
    table_md = metrics_markdown_table(sorted_df)

    baseline_soh = sorted_df[
        (sorted_df["Task"] == "SOH") & (sorted_df["Model"] == "Linear Regression")
    ]
    baseline_rul = sorted_df[
        (sorted_df["Task"] == "RUL") & (sorted_df["Model"] == "Random Forest Regressor")
    ]

    lines = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append("_Single source of truth: `results/metrics.csv`._")
    lines.append("")
    lines.append("## Benchmark Table")
    lines.append("")
    lines.append(table_md)
    lines.append("")
    lines.append("## Baseline Comparison")
    lines.append("")

    if not baseline_soh.empty:
        row = baseline_soh.iloc[0]
        lines.append(
            "- SOH baseline (`Linear Regression`): "
            f"MAE {_format_float(row['MAE'])}, RMSE {_format_float(row['RMSE'])}, R² {_format_float(row['R2'])}"
        )

    if not baseline_rul.empty:
        row = baseline_rul.iloc[0]
        lines.append(
            "- RUL baseline (`Random Forest Regressor`): "
            f"MAE {_format_float(row['MAE'])}, RMSE {_format_float(row['RMSE'])}, R² {_format_float(row['R2'])}"
        )

    lines.append(
        "- This report is auto-generated from `results/metrics.csv` to keep documentation and metrics synchronized."
    )

    return "\n".join(lines)


def write_results_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Write results summary markdown file.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Benchmark metrics.
    output_path : pathlib.Path
        Destination report path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_results_summary(metrics_df), encoding="utf-8")


def build_readme_metrics_block(metrics_df: pd.DataFrame) -> str:
    """Build auto-generated README block from metrics.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Benchmark metrics table.

    Returns
    -------
    str
        Markdown block to place between README markers.
    """

    sorted_df = metrics_df.sort_values(["Task", "RMSE", "Model"]).reset_index(drop=True)
    table_md = metrics_markdown_table(sorted_df)
    return "\n".join(
        [
            "_Auto-generated from `results/metrics.csv`._",
            "",
            table_md,
        ]
    )


def update_readme_metrics_block(readme_path: Path, metrics_df: pd.DataFrame) -> None:
    """Replace README auto-metrics marker block using metrics source data.

    Parameters
    ----------
    readme_path : pathlib.Path
        README file path.
    metrics_df : pandas.DataFrame
        Benchmark metrics table.

    Raises
    ------
    ValueError
        If marker tags are missing from README.
    """

    content = readme_path.read_text(encoding="utf-8")
    if AUTO_METRICS_START not in content or AUTO_METRICS_END not in content:
        raise ValueError(
            "README is missing auto-metrics markers. "
            f"Expected markers: {AUTO_METRICS_START} / {AUTO_METRICS_END}"
        )

    start_idx = content.index(AUTO_METRICS_START) + len(AUTO_METRICS_START)
    end_idx = content.index(AUTO_METRICS_END)

    replacement = "\n" + build_readme_metrics_block(metrics_df) + "\n"
    updated = content[:start_idx] + replacement + content[end_idx:]
    readme_path.write_text(updated, encoding="utf-8")
