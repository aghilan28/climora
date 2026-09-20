"""Statistical outlier detection table generator (IQR + Z-Score)."""

import numpy as np
import pandas as pd


def detect_outliers(
    df: pd.DataFrame, col: str = "anomaly_c", z_threshold: float = 3.0
) -> pd.DataFrame:
    """Detect statistical outliers using IQR and Z-score methods.

    Note: Flagged outliers are statistical extremes and are NOT automatically removed,
    as climate extremes contain genuine physical signals.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing from DataFrame")

    series = df[col].dropna()
    if series.empty:
        return pd.DataFrame()

    # 1. IQR Method
    q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q75 - q25
    lower_iqr = q25 - 1.5 * iqr
    upper_iqr = q75 + 1.5 * iqr

    # 2. Z-Score Method
    mean, std = series.mean(), series.std()
    z_scores = (df[col] - mean) / std if std > 0 else pd.Series(0, index=df.index)

    outlier_mask = (
        (df[col] < lower_iqr)
        | (df[col] > upper_iqr)
        | (z_scores.abs() > z_threshold)
    )

    outliers_df = df[outlier_mask].copy()
    outliers_df["z_score"] = z_scores[outlier_mask]
    outliers_df["is_iqr_outlier"] = (df[col] < lower_iqr) | (df[col] > upper_iqr)
    outliers_df["is_zscore_outlier"] = z_scores.abs() > z_threshold
    outliers_df["note"] = "Flagged statistical extreme (preserved in dataset)"

    return outliers_df.sort_values(col, ascending=False).reset_index(drop=True)
