"""Gate tests for prediction uncertainty estimation and quantile intervals per F8."""

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import XGBoostClimateModel


def test_uncertainty_band_width_varies_across_rows() -> None:
    """Assert prediction interval width varies across different input rows."""
    X_train = pd.DataFrame({
        "f1": np.linspace(0, 10, 100),
        "f2": np.random.randn(100),
    })
    y_train = pd.Series(X_train["f1"] * 0.5 + np.random.randn(100) * 0.2)

    model = XGBoostClimateModel({"n_estimators": 20, "max_depth": 3})
    model.fit(X_train, y_train)

    X_test = pd.DataFrame({
        "f1": [0.0, 5.0, 10.0],
        "f2": [-2.0, 0.0, 2.0],
    })
    lower, upper = model.predict_interval(X_test)
    widths = upper - lower

    assert len(widths) == 3
    # Assert width is non-constant across distinct rows
    assert not np.allclose(widths[0], widths[1]) or not np.allclose(widths[1], widths[2]), \
        f"Expected variable interval widths across rows, got constant widths: {widths}"


def test_missing_rmse_raises_exception() -> None:
    """Assert missing 'rmse' in metrics dictionary raises KeyError instead of substituting fallback."""
    from dashboard.pages.predictions import compute_uncertainty_interval

    metrics_missing_rmse = {"mae": 0.5131, "r2": -1.1}
    with pytest.raises(KeyError, match="rmse"):
        compute_uncertainty_interval(0.5, metrics_missing_rmse)
