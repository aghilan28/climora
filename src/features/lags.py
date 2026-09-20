"""Lagged feature transformation module."""

from typing import List

import pandas as pd


def add_lag_features(
    df: pd.DataFrame, col: str = "anomaly_c", lags: List[int] | None = None
) -> pd.DataFrame:
    """Add strictly trailing lagged features.

    Note: Feature at row t is y(t-lag). For predicting y(t), lag-1 is y(t-1).
    """
    df_out = df.copy()
    lags = lags or [1, 3, 6, 12, 24, 60]

    if col not in df_out.columns:
        return df_out

    for lag in lags:
        df_out[f"{col}_lag_{lag}"] = df_out[col].shift(lag)

    return df_out
