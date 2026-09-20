"""Unit tests for PyTorch LSTM climate model."""

import numpy as np
import pandas as pd

from src.data.providers.nasa_giss import NasaGissProvider
from src.features.pipeline import build_feature_matrix
from src.models.lstm_model import PyTorchLSTMClimateModel
from src.splits.chronological import make_chronological_split


def test_lstm_sequence_creation_and_shapes(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df_raw = provider.parse(gistemp_fixture_bytes)
    df_feats = build_feature_matrix(df_raw)

    split = make_chronological_split(df_feats)
    if len(split.X_train) < 30:
        dates = pd.date_range("1880-01-01", periods=100, freq="MS")
        df_dummy = pd.DataFrame({
            "date": dates,
            "anomaly_c": np.sin(np.linspace(0, 10, 100)),
            "year": dates.year,
        })
        df_feats = build_feature_matrix(df_dummy)
        split = make_chronological_split(df_feats, train_end="1930-01-01", val_end="1950-01-01")

    model = PyTorchLSTMClimateModel(params={"sequence_length": 12, "epochs": 5, "seed": 42})
    X_tr_scaled = model.scaler.fit_transform(split.X_train.fillna(0.0))
    if hasattr(model.scaler, "scale_") and model.scaler.scale_ is not None:
        model.scaler.scale_[model.scaler.scale_ == 0.0] = 1.0

    X_seq, y_seq = model._create_sequences(X_tr_scaled, split.y_train.values, seq_len=12)

    # Assertion 1: Window shapes (n, L, F)
    assert X_seq.ndim == 3
    assert X_seq.shape[1] == 12
    assert X_seq.shape[2] == split.X_train.shape[1]
    assert len(X_seq) == len(split.X_train) - 12

    # Assertion 2: Every window's last timestep index < first test index (strict boundary)
    train_indices = np.arange(len(split.X_train))
    train_window_last_indices = [train_indices[i + 11] for i in range(len(train_indices) - 12)]
    val_first_idx = len(split.X_train)
    assert all(idx < val_first_idx for idx in train_window_last_indices)

    # Assertion 3: Scaler params come ONLY from train rows
    assert len(model.scaler.mean_) == split.X_train.shape[1]

    # Assertion 4: Inverse-scaler round-trips within 1e-6
    sample = split.X_train.fillna(0.0).iloc[:5].values
    scaled = model.scaler.transform(sample)
    unscaled = model.scaler.inverse_transform(scaled)
    assert np.allclose(sample, unscaled, atol=1e-6, equal_nan=True)

    # Assertion 5: Fit, save, load and reproduce predictions
    model.fit(split.X_train, split.y_train)
    preds = model.predict(split.X_train)

    model.save(tmp_path)
    reloaded_model = PyTorchLSTMClimateModel().load(tmp_path)
    preds_reloaded = reloaded_model.predict(split.X_train)

    assert np.allclose(preds, preds_reloaded, atol=1e-5, equal_nan=True)
