"""Rolling feature transformation module with strict anti-leakage controls."""

from typing import List

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    col: str = "anomaly_c",
    windows: List[int] | None = None,
    stats: List[str] | None = None,
) -> pd.DataFrame:
    """Add strictly trailing rolling statistics.

    Anti-leakage enforcement:
    1. df[col].shift(1) is taken first so row t depends strictly on trailing history [t-W, ..., t-1].
    2. center=False is strictly set so no future observations enter the window.
    """
    df_out = df.copy()
    windows = windows or [12, 24, 60]
    stats = stats or ["mean", "std", "min", "max"]

    if col not in df_out.columns:
        return df_out

    # Shift by 1 so current row t does NOT include y(t)
    shifted_series = df_out[col].shift(1)

    for w in windows:
        roll = shifted_series.rolling(window=w, center=False)
        if "mean" in stats:
            df_out[f"{col}_roll_mean_{w}"] = roll.mean()
        if "std" in stats:
            df_out[f"{col}_roll_std_{w}"] = roll.std()
        if "min" in stats:
            df_out[f"{col}_roll_min_{w}"] = roll.min()
        if "max" in stats:
            df_out[f"{col}_roll_max_{w}"] = roll.max()

    return df_out
