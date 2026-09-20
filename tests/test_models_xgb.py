"""Unit tests for XGBoost climate model."""

import numpy as np
import pandas as pd

from src.data.providers.nasa_giss import NasaGissProvider
from src.features.pipeline import build_feature_matrix
from src.models.xgboost_model import XGBoostClimateModel
from src.splits.chronological import make_chronological_split


def test_xgboost_fit_predict_save_load(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df_raw = provider.parse(gistemp_fixture_bytes)
    df_feats = build_feature_matrix(df_raw)

    split = make_chronological_split(df_feats)
    if split.X_train.empty:
        # Fallback dummy dataset if sample fixture is small
        dates = pd.date_range("1880-01-01", periods=100, freq="MS")
        df_dummy = pd.DataFrame({
            "date": dates,
            "anomaly_c": np.sin(np.linspace(0, 10, 100)),
            "year": dates.year,
        })
        df_feats = build_feature_matrix(df_dummy)
        split = make_chronological_split(df_feats, train_end="1930-01-01", val_end="1950-01-01")

    model = XGBoostClimateModel(params={"n_estimators": 10, "random_state": 42})
    model.fit(split.X_train, split.y_train)

    preds = model.predict(split.X_train)
    assert len(preds) == len(split.X_train)
    assert not np.isnan(preds).any()

    # Save and reload
    save_path = model.save(tmp_path)
    assert save_path.exists()

    model_reloaded = XGBoostClimateModel().load(tmp_path)
    preds_reloaded = model_reloaded.predict(split.X_train)
    assert np.allclose(preds, preds_reloaded)
