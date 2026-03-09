"""Model training utilities for SOH and RUL prediction."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import regression_metrics

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:  # pragma: no cover - covered by runtime environment
    XGBRegressor = None

RANDOM_SEED = 42
LEAKAGE_FEATURE_EXCLUSIONS: dict[str, set[str]] = {
    "SOH": {"capacity_Ah", "SOH"},
    "RUL": {"RUL"},
}


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class PreparedData:
    """Prepared tabular dataset split for modeling."""

    feature_columns: list[str]
    scaler: MinMaxScaler
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def default_feature_columns(
    df: pd.DataFrame, exclude: set[str] | None = None
) -> list[str]:
    """Infer model feature columns from numeric fields.

    Parameters
    ----------
    df : pandas.DataFrame
        Input feature DataFrame.

    Returns
    -------
    list[str]
        Candidate numeric feature columns excluding identifiers/targets.
    """

    excluded = {"SOH", "RUL", "battery_id"}
    if exclude:
        excluded = excluded.union(exclude)
    return [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]


def leakage_exclusions_for_target(target_col: str) -> set[str]:
    """Return forbidden feature names for a target to avoid leakage.

    Parameters
    ----------
    target_col : str
        Target variable name.

    Returns
    -------
    set[str]
        Feature names that should not be used for the target.
    """

    return set(LEAKAGE_FEATURE_EXCLUSIONS.get(target_col, set()))


def validate_no_target_leakage(feature_columns: list[str], target_col: str) -> None:
    """Validate that selected features do not contain leakage-prone columns.

    Parameters
    ----------
    feature_columns : list[str]
        Candidate model feature names.
    target_col : str
        Target variable name.

    Raises
    ------
    ValueError
        If leakage-prone columns are present in selected features.
    """

    forbidden = leakage_exclusions_for_target(target_col=target_col)
    overlap = set(feature_columns).intersection(forbidden)
    if overlap:
        raise ValueError(
            "Leakage risk detected. Remove forbidden features for "
            f"target '{target_col}': {sorted(overlap)}"
        )


def soh_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return leakage-safe feature columns for SOH modeling.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature DataFrame.

    Returns
    -------
    list[str]
        SOH-safe feature list.
    """

    return default_feature_columns(
        df=df,
        exclude=leakage_exclusions_for_target("SOH"),
    )


def prepare_cross_battery_data(
    df: pd.DataFrame,
    target_col: str,
    test_battery: str = "B0018",
    feature_columns: list[str] | None = None,
) -> PreparedData:
    """Prepare cross-battery train/test split with MinMax scaling.

    Parameters
    ----------
    df : pandas.DataFrame
        Input feature DataFrame.
    target_col : str
        Target column name (``SOH`` or ``RUL``).
    test_battery : str, optional
        Battery ID reserved for testing.
    feature_columns : list[str] | None, optional
        Explicit feature set. When ``None``, inferred automatically.

    Returns
    -------
    PreparedData
        Processed split and arrays.
    """

    sorted_df = df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)
    if feature_columns is None:
        features = default_feature_columns(
            sorted_df,
            exclude=leakage_exclusions_for_target(target_col),
        )
    else:
        features = feature_columns

    validate_no_target_leakage(features, target_col=target_col)

    train_df = sorted_df[sorted_df["battery_id"] != test_battery].copy()
    test_df = sorted_df[sorted_df["battery_id"] == test_battery].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Check battery IDs in data.")

    train_medians = train_df[features].median()
    train_df[features] = train_df[features].astype(float).fillna(train_medians)
    test_df[features] = test_df[features].astype(float).fillna(train_medians)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    y_train = train_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df.loc[:, features] = pd.DataFrame(
        X_train, columns=features, index=train_df.index
    )
    test_df.loc[:, features] = pd.DataFrame(
        X_test, columns=features, index=test_df.index
    )

    return PreparedData(
        feature_columns=features,
        scaler=scaler,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def _fit_tabular_model(
    model: Any,
    model_name: str,
    task: str,
    prepared: PreparedData,
    models_dir: Path,
) -> tuple[dict[str, float | str], np.ndarray, float]:
    """Fit a tabular regressor and evaluate on test battery.

    Parameters
    ----------
    model : Any
        Estimator implementing ``fit`` and ``predict``.
    model_name : str
        Model label.
    task : str
        Task label (``SOH`` or ``RUL``).
    prepared : PreparedData
        Prepared split object.
    models_dir : pathlib.Path
        Directory to save serialized model files.

    Returns
    -------
    tuple[dict[str, float | str], numpy.ndarray, float]
        ``(metrics_record, predictions, elapsed_seconds)``.
    """

    models_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    model.fit(prepared.X_train, prepared.y_train)
    elapsed = time.time() - start

    predictions = model.predict(prepared.X_test)
    metric_values = regression_metrics(prepared.y_test, predictions)

    record: dict[str, float | str] = {
        "Model": model_name,
        "Task": task,
        **metric_values,
        "Train_Time_s": float(elapsed),
    }

    model_path = (
        models_dir / f"{task.lower()}_{model_name.lower().replace(' ', '_')}.pkl"
    )
    joblib.dump(
        {
            "model": model,
            "scaler": prepared.scaler,
            "features": prepared.feature_columns,
        },
        model_path,
    )

    return record, np.asarray(predictions, dtype=float), elapsed


def train_baseline_regressors(
    prepared: PreparedData,
    task: str,
    models_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Train baseline regression models.

    Parameters
    ----------
    prepared : PreparedData
        Prepared train/test split.
    task : str
        Task label.
    models_dir : pathlib.Path
        Output directory for models.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model results including metrics and predictions.
    """

    models: list[tuple[str, Any]] = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        (
            "Random Forest Regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ]

    outputs: dict[str, dict[str, Any]] = {}
    for name, model in models:
        record, preds, elapsed = _fit_tabular_model(
            model=model,
            model_name=name,
            task=task,
            prepared=prepared,
            models_dir=models_dir,
        )
        outputs[name] = {
            "metrics": record,
            "predictions": preds,
            "train_time": elapsed,
            "cycle_number": prepared.test_df["cycle_number"].to_numpy(dtype=float),
            "y_true": prepared.y_test,
            "model": model,
            "feature_columns": prepared.feature_columns,
        }
    return outputs


def train_xgboost_regressor(
    prepared: PreparedData,
    task: str,
    models_dir: Path,
) -> dict[str, Any]:
    """Train XGBoost regressor with GridSearchCV hyperparameter tuning.

    Parameters
    ----------
    prepared : PreparedData
        Prepared train/test split.
    task : str
        Task label.
    models_dir : pathlib.Path
        Output directory for models.

    Returns
    -------
    dict[str, Any]
        Result payload with metrics, predictions, and fitted model.

    Raises
    ------
    ModuleNotFoundError
        If xgboost is not installed.
    """

    if XGBRegressor is None:
        raise ModuleNotFoundError("xgboost is required for train_xgboost_regressor.")

    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    start = time.time()
    grid.fit(prepared.X_train, prepared.y_train)
    elapsed = time.time() - start

    best_model = grid.best_estimator_
    preds = best_model.predict(prepared.X_test)
    metric_values = regression_metrics(prepared.y_test, preds)

    model_path = models_dir / f"{task.lower()}_xgboost_regressor.pkl"
    joblib.dump(
        {
            "model": best_model,
            "scaler": prepared.scaler,
            "features": prepared.feature_columns,
            "best_params": grid.best_params_,
        },
        model_path,
    )

    return {
        "metrics": {
            "Model": "XGBoost Regressor",
            "Task": task,
            **metric_values,
            "Train_Time_s": float(elapsed),
        },
        "predictions": np.asarray(preds, dtype=float),
        "train_time": elapsed,
        "cycle_number": prepared.test_df["cycle_number"].to_numpy(dtype=float),
        "y_true": prepared.y_test,
        "model": best_model,
        "best_params": grid.best_params_,
        "feature_columns": prepared.feature_columns,
    }


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    sequence_length: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-length sequences grouped by battery.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame sorted by battery and cycle.
    feature_columns : list[str]
        Feature columns to include in sequence tensors.
    target_col : str
        Target column.
    sequence_length : int, optional
        Number of cycles per sequence.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(X_seq, y_seq, cycle_idx)``.
    """

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    cycle_numbers: list[float] = []

    for _, group in df.groupby("battery_id"):
        g = group.sort_values("cycle_number")
        x = g[feature_columns].to_numpy(dtype=np.float32)
        y = g[target_col].to_numpy(dtype=np.float32)
        cycles = g["cycle_number"].to_numpy(dtype=np.float32)

        if len(g) < sequence_length:
            continue

        for end_idx in range(sequence_length - 1, len(g)):
            start_idx = end_idx - sequence_length + 1
            sequences.append(x[start_idx : end_idx + 1])
            targets.append(float(y[end_idx]))
            cycle_numbers.append(float(cycles[end_idx]))

    return np.array(sequences), np.array(targets), np.array(cycle_numbers)


class LSTMRegressor(nn.Module):
    """Two-layer LSTM regressor."""

    def __init__(
        self, input_size: int, hidden_size: int = 64, dropout: float = 0.2
    ) -> None:
        """Initialize LSTM network.

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int, optional
            Hidden dimension.
        dropout : float, optional
            Dropout probability.
        """

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, features)``.

        Returns
        -------
        torch.Tensor
            Prediction tensor of shape ``(batch,)``.
        """

        output, _ = self.lstm(x)
        last_state = output[:, -1, :]
        pred = self.regressor(last_state)
        return pred.squeeze(-1)


class CNNBiLSTMRegressor(nn.Module):
    """1D-CNN + BiLSTM hybrid regressor."""

    def __init__(
        self,
        input_size: int,
        conv_channels: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ) -> None:
        """Initialize CNN-BiLSTM network.

        Parameters
        ----------
        input_size : int
            Number of input features.
        conv_channels : int, optional
            Number of convolution output channels.
        hidden_size : int, optional
            LSTM hidden size.
        dropout : float, optional
            Dropout probability.
        """

        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(batch, seq_len, features)``.

        Returns
        -------
        torch.Tensor
            Predictions ``(batch,)``.
        """

        x_conv = x.transpose(1, 2)
        x_conv = self.relu(self.conv1(x_conv))
        x_conv = x_conv.transpose(1, 2)
        seq_out, _ = self.bilstm(x_conv)
        pred = self.head(seq_out[:, -1, :])
        return pred.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        """Initialize positional encoding table.

        Parameters
        ----------
        d_model : int
            Embedding size.
        max_len : int, optional
            Maximum sequence length.
        """

        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings.

        Returns
        -------
        torch.Tensor
            Encoded embeddings.
        """

        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRegressor(nn.Module):
    """Transformer encoder based sequence regressor."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize Transformer regressor.

        Parameters
        ----------
        input_size : int
            Number of input features.
        d_model : int, optional
            Transformer model dimension.
        nhead : int, optional
            Number of attention heads.
        num_layers : int, optional
            Number of encoder layers.
        dropout : float, optional
            Dropout probability.
        """

        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.positional = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Predictions ``(batch,)``.
        """

        emb = self.proj(x)
        emb = self.positional(emb)
        encoded = self.encoder(emb)
        pooled = encoded[:, -1, :]
        return self.head(pooled).squeeze(-1)


def train_torch_sequence_model(
    model: nn.Module,
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    """Train a PyTorch sequence regressor with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance.
    X_train_seq : numpy.ndarray
        Training sequence inputs.
    y_train_seq : numpy.ndarray
        Training targets.
    epochs : int, optional
        Maximum number of epochs.
    batch_size : int, optional
        Batch size.
    learning_rate : float, optional
        Adam learning rate.
    weight_decay : float, optional
        Weight decay.
    patience : int, optional
        Early stopping patience.

    Returns
    -------
    tuple[torch.nn.Module, dict[str, list[float]], float]
        ``(best_model, history, elapsed_seconds)``.
    """

    set_random_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

    val_size = max(1, int(0.2 * len(X_tensor)))
    indices = torch.randperm(len(X_tensor))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    start = time.time()

    for _ in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    elapsed = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, elapsed


def predict_torch_model(
    model: nn.Module, X_seq: np.ndarray, batch_size: int = 128
) -> np.ndarray:
    """Generate predictions from a trained PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    X_seq : numpy.ndarray
        Sequence inputs.
    batch_size : int, optional
        Inference batch size.

    Returns
    -------
    numpy.ndarray
        Predictions.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(
        torch.tensor(X_seq, dtype=torch.float32), batch_size=batch_size, shuffle=False
    )
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).detach().cpu().numpy()
            preds.append(outputs)

    return np.concatenate(preds, axis=0)


def run_soh_benchmark(
    feature_df: pd.DataFrame,
    models_dir: Path,
    test_battery: str = "B0018",
    sequence_length: int = 10,
) -> tuple[list[dict[str, float | str]], dict[str, dict[str, Any]]]:
    """Train all required SOH models and collect benchmark metrics.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Feature dataset.
    models_dir : pathlib.Path
        Output directory for model artifacts.
    test_battery : str, optional
        Battery reserved for test set.
    sequence_length : int, optional
        Sequence length for deep models.

    Returns
    -------
    tuple[list[dict[str, float | str]], dict[str, dict[str, Any]]]
        Metrics records and model result payloads.
    """

    task = "SOH"
    feature_cols = soh_feature_columns(feature_df)
    prepared = prepare_cross_battery_data(
        feature_df,
        target_col="SOH",
        test_battery=test_battery,
        feature_columns=feature_cols,
    )

    metrics_records: list[dict[str, float | str]] = []
    results: dict[str, dict[str, Any]] = {}

    baseline_outputs = train_baseline_regressors(
        prepared=prepared, task=task, models_dir=models_dir
    )
    for model_name, payload in baseline_outputs.items():
        metrics_records.append(payload["metrics"])
        results[model_name] = payload

    xgb_output = train_xgboost_regressor(
        prepared=prepared, task=task, models_dir=models_dir
    )
    metrics_records.append(xgb_output["metrics"])
    results["XGBoost Regressor"] = xgb_output

    X_train_seq, y_train_seq, _ = create_sequences(
        df=prepared.train_df,
        feature_columns=prepared.feature_columns,
        target_col="SOH",
        sequence_length=sequence_length,
    )
    X_test_seq, y_test_seq, cycle_test_seq = create_sequences(
        df=prepared.test_df,
        feature_columns=prepared.feature_columns,
        target_col="SOH",
        sequence_length=sequence_length,
    )

    lstm_model = LSTMRegressor(
        input_size=len(prepared.feature_columns), hidden_size=64, dropout=0.2
    )
    lstm_model, lstm_history, lstm_time = train_torch_sequence_model(
        model=lstm_model,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
    )
    lstm_preds = predict_torch_model(lstm_model, X_test_seq)
    lstm_metrics = regression_metrics(y_test_seq, lstm_preds)
    torch.save(
        {
            "state_dict": lstm_model.state_dict(),
            "feature_columns": prepared.feature_columns,
            "sequence_length": sequence_length,
            "scaler": prepared.scaler,
        },
        models_dir / "soh_lstm.pt",
    )
    lstm_record: dict[str, float | str] = {
        "Model": "LSTM Neural Network",
        "Task": task,
        **lstm_metrics,
        "Train_Time_s": float(lstm_time),
    }
    metrics_records.append(lstm_record)
    results["LSTM Neural Network"] = {
        "metrics": lstm_record,
        "predictions": lstm_preds,
        "y_true": y_test_seq,
        "cycle_number": cycle_test_seq,
        "history": lstm_history,
        "model": lstm_model,
        "train_time": lstm_time,
        "feature_columns": prepared.feature_columns,
    }

    cnn_bilstm_model = CNNBiLSTMRegressor(
        input_size=len(prepared.feature_columns),
        conv_channels=32,
        hidden_size=64,
        dropout=0.2,
    )
    cnn_bilstm_model, cnn_history, cnn_time = train_torch_sequence_model(
        model=cnn_bilstm_model,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
    )
    cnn_preds = predict_torch_model(cnn_bilstm_model, X_test_seq)
    cnn_metrics = regression_metrics(y_test_seq, cnn_preds)
    torch.save(
        {
            "state_dict": cnn_bilstm_model.state_dict(),
            "feature_columns": prepared.feature_columns,
            "sequence_length": sequence_length,
            "scaler": prepared.scaler,
        },
        models_dir / "soh_cnn_bilstm.pt",
    )
    cnn_record: dict[str, float | str] = {
        "Model": "CNN-BiLSTM",
        "Task": task,
        **cnn_metrics,
        "Train_Time_s": float(cnn_time),
    }
    metrics_records.append(cnn_record)
    results["CNN-BiLSTM"] = {
        "metrics": cnn_record,
        "predictions": cnn_preds,
        "y_true": y_test_seq,
        "cycle_number": cycle_test_seq,
        "history": cnn_history,
        "model": cnn_bilstm_model,
        "train_time": cnn_time,
        "feature_columns": prepared.feature_columns,
    }

    return metrics_records, results


def run_rul_benchmark(
    feature_df: pd.DataFrame,
    models_dir: Path,
    test_battery: str = "B0018",
    sequence_length: int = 10,
    include_transformer: bool = True,
) -> tuple[list[dict[str, float | str]], dict[str, dict[str, Any]]]:
    """Train required RUL models and collect benchmark metrics.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Feature dataset.
    models_dir : pathlib.Path
        Output directory for model artifacts.
    test_battery : str, optional
        Battery reserved for testing.
    sequence_length : int, optional
        Sequence length for deep models.
    include_transformer : bool, optional
        Whether to train Transformer model.

    Returns
    -------
    tuple[list[dict[str, float | str]], dict[str, dict[str, Any]]]
        Metrics records and model result payloads.
    """

    task = "RUL"
    prepared = prepare_cross_battery_data(
        feature_df, target_col="RUL", test_battery=test_battery
    )

    metrics_records: list[dict[str, float | str]] = []
    results: dict[str, dict[str, Any]] = {}

    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_record, rf_preds, rf_time = _fit_tabular_model(
        model=rf_model,
        model_name="Random Forest Regressor",
        task=task,
        prepared=prepared,
        models_dir=models_dir,
    )
    metrics_records.append(rf_record)
    results["Random Forest Regressor"] = {
        "metrics": rf_record,
        "predictions": rf_preds,
        "y_true": prepared.y_test,
        "cycle_number": prepared.test_df["cycle_number"].to_numpy(dtype=float),
        "model": rf_model,
        "train_time": rf_time,
        "feature_columns": prepared.feature_columns,
    }

    xgb_output = train_xgboost_regressor(
        prepared=prepared, task=task, models_dir=models_dir
    )
    metrics_records.append(xgb_output["metrics"])
    results["XGBoost Regressor"] = xgb_output

    X_train_seq, y_train_seq, _ = create_sequences(
        df=prepared.train_df,
        feature_columns=prepared.feature_columns,
        target_col="RUL",
        sequence_length=sequence_length,
    )
    X_test_seq, y_test_seq, cycle_test_seq = create_sequences(
        df=prepared.test_df,
        feature_columns=prepared.feature_columns,
        target_col="RUL",
        sequence_length=sequence_length,
    )

    lstm_model = LSTMRegressor(
        input_size=len(prepared.feature_columns), hidden_size=64, dropout=0.2
    )
    lstm_model, lstm_history, lstm_time = train_torch_sequence_model(
        model=lstm_model,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
    )
    lstm_preds = predict_torch_model(lstm_model, X_test_seq)
    lstm_metrics = regression_metrics(y_test_seq, lstm_preds)
    torch.save(
        {
            "state_dict": lstm_model.state_dict(),
            "feature_columns": prepared.feature_columns,
            "sequence_length": sequence_length,
            "scaler": prepared.scaler,
        },
        models_dir / "rul_lstm.pt",
    )
    lstm_record: dict[str, float | str] = {
        "Model": "LSTM Neural Network",
        "Task": task,
        **lstm_metrics,
        "Train_Time_s": float(lstm_time),
    }
    metrics_records.append(lstm_record)
    results["LSTM Neural Network"] = {
        "metrics": lstm_record,
        "predictions": lstm_preds,
        "y_true": y_test_seq,
        "cycle_number": cycle_test_seq,
        "history": lstm_history,
        "model": lstm_model,
        "train_time": lstm_time,
        "feature_columns": prepared.feature_columns,
    }

    if include_transformer:
        transformer_model = TransformerRegressor(
            input_size=len(prepared.feature_columns),
            d_model=64,
            nhead=2,
            num_layers=2,
            dropout=0.2,
        )
        transformer_model, tr_history, tr_time = train_torch_sequence_model(
            model=transformer_model,
            X_train_seq=X_train_seq,
            y_train_seq=y_train_seq,
        )
        tr_preds = predict_torch_model(transformer_model, X_test_seq)
        tr_metrics = regression_metrics(y_test_seq, tr_preds)
        torch.save(
            {
                "state_dict": transformer_model.state_dict(),
                "feature_columns": prepared.feature_columns,
                "sequence_length": sequence_length,
                "scaler": prepared.scaler,
            },
            models_dir / "rul_transformer.pt",
        )
        tr_record: dict[str, float | str] = {
            "Model": "Transformer Encoder",
            "Task": task,
            **tr_metrics,
            "Train_Time_s": float(tr_time),
        }
        metrics_records.append(tr_record)
        results["Transformer Encoder"] = {
            "metrics": tr_record,
            "predictions": tr_preds,
            "y_true": y_test_seq,
            "cycle_number": cycle_test_seq,
            "history": tr_history,
            "model": transformer_model,
            "train_time": tr_time,
            "feature_columns": prepared.feature_columns,
        }

    return metrics_records, results
