"""Unit tests for baseline climate models."""

import numpy as np
import pandas as pd

from src.models.baselines import ClimatologyModel, SeasonalNaiveModel


def test_seasonal_naive_model() -> None:
    X = pd.DataFrame({"anomaly_c_lag_12": [0.5, 0.6, 0.7]})
    y = pd.Series([0.5, 0.6, 0.7])

    model = SeasonalNaiveModel().fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 3
    assert np.allclose(preds, [0.5, 0.6, 0.7])


def test_climatology_model() -> None:
    X_train = pd.DataFrame({
        "month": [1, 1, 2, 2],
    })
    y_train = pd.Series([0.1, 0.3, 0.8, 1.0])  # Month 1 mean = 0.2, Month 2 mean = 0.9

    model = ClimatologyModel().fit(X_train, y_train)

    X_test = pd.DataFrame({"month": [1, 2]})
    preds = model.predict(X_test)

    assert abs(preds[0] - 0.2) < 1e-6
    assert abs(preds[1] - 0.9) < 1e-6
