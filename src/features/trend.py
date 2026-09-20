"""Trend and differencing feature transformation module."""

from typing import List

import pandas as pd


def add_trend_features(
    df: pd.DataFrame, col: str = "anomaly_c", diffs: List[int] | None = None
) -> pd.DataFrame:
    """Add strictly trailing first-difference and rate-of-change features."""
    df_out = df.copy()
    diffs = diffs or [1, 12]

    if col not in df_out.columns:
        return df_out

    # Shift by 1 first to avoid target leakage
    shifted = df_out[col].shift(1)

    for d in diffs:
        df_out[f"{col}_diff_{d}"] = shifted.diff(d)

    return df_out
