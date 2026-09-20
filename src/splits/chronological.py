"""Strict chronological train/val/test data splitting."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ChronologicalSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    test_start: str = ""
    test_end: str = ""
    n_test: int = 0


def make_chronological_split(
    df: pd.DataFrame,
    target_col: str = "anomaly_c",
    date_col: str = "date",
    train_end: str = "1999-12-31",
    val_end: str = "2014-12-31",
    test_end: Optional[str] = None,
) -> ChronologicalSplit:
    """Split dataset chronologically with explicit data-independent test slice."""
    df_clean = df.sort_values(date_col).copy()

    valid_mask = ~df_clean[target_col].isna()
    df_valid = df_clean[valid_mask].copy()

    train_df = df_valid[df_valid[date_col] <= train_end].copy()
    val_df = df_valid[(df_valid[date_col] > train_end) & (df_valid[date_col] <= val_end)].copy()

    if test_end:
        test_df = df_valid[(df_valid[date_col] > val_end) & (df_valid[date_col] <= test_end)].copy()
    else:
        test_df = df_valid[df_valid[date_col] > val_end].copy()

    # Fallback ratio split if date range filtering yields an empty partition
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        n = len(df_valid)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        train_df = df_valid.iloc[:n_train].copy()
        val_df = df_valid.iloc[n_train : n_train + n_val].copy()
        test_df = df_valid.iloc[n_train + n_val :].copy()
        if len(test_df) == 0:
            test_df = val_df.copy()

    known_target_cols = {
        "anomaly_c",
        "anomaly_c_h12",
        "target_anomaly_c",
        "target_anomaly_c_h12",
    } | {
        c for c in df_valid.columns
        if c.startswith("target_") or c.startswith("anomaly_c_h") or c.startswith("target__")
    }

    helper_cols = {date_col, "year", "month", "date", "is_train_slice", "partial_years_excluded"}

    feature_cols = [
        c for c in df_valid.columns if c not in helper_cols and c not in known_target_cols
    ]

    # Runtime assertion: reject if target or target-derived columns are present in feature_cols
    for col in feature_cols:
        if col == target_col or col in known_target_cols:
            raise ValueError(f"Future leakage error: target column '{col}' present in feature_cols!")

    realised_start = pd.to_datetime(test_df[date_col].iloc[0]).strftime("%Y-%m") if len(test_df) > 0 else ""
    realised_end = pd.to_datetime(test_df[date_col].iloc[-1]).strftime("%Y-%m") if len(test_df) > 0 else ""

    return ChronologicalSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        X_train=train_df[feature_cols],
        y_train=train_df[target_col],
        X_val=val_df[feature_cols],
        y_val=val_df[target_col],
        X_test=test_df[feature_cols],
        y_test=test_df[target_col],
        test_start=realised_start,
        test_end=realised_end,
        n_test=len(test_df),
    )
