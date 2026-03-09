"""Streamlit app for battery RUL exploration."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_feature_data(csv_path: Path) -> pd.DataFrame:
    """Load feature dataset.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to feature CSV.

    Returns
    -------
    pandas.DataFrame
        Feature table.
    """

    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner=False)
def load_model_artifact(model_path: Path) -> dict:
    """Load serialized model artifact.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to model artifact.

    Returns
    -------
    dict
        Serialized model dictionary.
    """

    return joblib.load(model_path)


def main() -> None:
    """Run Streamlit UI."""

    root = Path(__file__).resolve().parents[1]
    features_path = root / "data" / "processed" / "features_all.csv"
    model_path = root / "models" / "rul_random_forest_regressor.pkl"

    st.set_page_config(page_title="Battery RUL Predictor", layout="wide")
    st.title("Engineering_EV_Battery_Degradation - RUL Explorer")
    st.caption(
        "Interactive remaining useful life prediction with a trained Random Forest model."
    )

    if not features_path.exists():
        st.error(
            "`data/processed/features_all.csv` is missing. Run preprocessing/model notebooks first."
        )
        return

    if not model_path.exists():
        st.error(
            "`models/rul_random_forest_regressor.pkl` is missing. Run model training first."
        )
        return

    df = load_feature_data(features_path)
    artifact = load_model_artifact(model_path)

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["features"]

    battery_ids = sorted(df["battery_id"].unique().tolist())
    selected_battery = st.selectbox(
        "Battery ID",
        battery_ids,
        index=battery_ids.index("B0018") if "B0018" in battery_ids else 0,
    )

    battery_df = (
        df[df["battery_id"] == selected_battery]
        .sort_values("cycle_number")
        .reset_index(drop=True)
    )
    min_cycle = int(battery_df["cycle_number"].min())
    max_cycle = int(battery_df["cycle_number"].max())

    cycle_range = st.slider(
        "Cycle range",
        min_value=min_cycle,
        max_value=max_cycle,
        value=(min_cycle, max_cycle),
    )

    view_df = battery_df[
        (battery_df["cycle_number"] >= cycle_range[0])
        & (battery_df["cycle_number"] <= cycle_range[1])
    ].copy()

    X_raw = view_df[feature_columns].copy().fillna(view_df[feature_columns].median())
    X_scaled = scaler.transform(X_raw)
    view_df["predicted_RUL"] = model.predict(X_scaled)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            view_df["cycle_number"], view_df["RUL"], label="Actual RUL", linewidth=2
        )
        ax.plot(
            view_df["cycle_number"],
            view_df["predicted_RUL"],
            label="Predicted RUL",
            linewidth=2,
            linestyle="--",
        )
        ax.set_title(f"RUL Prediction - {selected_battery}")
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("RUL (cycles)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        mae = (view_df["RUL"] - view_df["predicted_RUL"]).abs().mean()
        rmse = ((view_df["RUL"] - view_df["predicted_RUL"]) ** 2).mean() ** 0.5
        st.metric("Samples", f"{len(view_df)}")
        st.metric("MAE (range)", f"{mae:.2f}")
        st.metric("RMSE (range)", f"{rmse:.2f}")

    st.dataframe(
        view_df[["battery_id", "cycle_number", "RUL", "predicted_RUL"]].round(3),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
