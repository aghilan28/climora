"""Temporal and calendar feature transformation module."""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add deterministic calendar features (month, quarter, sin/cos month)."""
    df_out = df.copy()

    if date_col in df_out.columns and pd.api.types.is_datetime64_any_dtype(df_out[date_col]):
        month = df_out[date_col].dt.month
        df_out["month"] = month
        df_out["quarter"] = df_out[date_col].dt.quarter
        df_out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df_out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    elif "month" in df_out.columns:
        month = df_out["month"]
        df_out["quarter"] = (month - 1) // 3 + 1
        df_out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df_out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    return df_out
