"""Evaluation metrics computation for climate time-series regression."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    baseline_rmse: float | None = None,
) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, R2, and Skill Score vs Baseline."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Filter out NaNs
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    if np.sum(valid) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0, "skill_score": 0.0}

    yt_v = y_t[valid]
    yp_v = y_p[valid]

    mae = float(mean_absolute_error(yt_v, yp_v))
    rmse = float(np.sqrt(mean_squared_error(yt_v, yp_v)))

    # MAPE with epsilon to prevent zero division
    eps = 1e-5
    mape = float(np.mean(np.abs((yt_v - yp_v) / (np.abs(yt_v) + eps))) * 100.0)

    r2 = float(r2_score(yt_v, yp_v))

    skill_score = 0.0
    if baseline_rmse is not None and baseline_rmse > 0:
        skill_score = float(1.0 - (rmse / baseline_rmse))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "r2": round(r2, 4),
        "skill_score": round(skill_score, 4),
    }
