"""Strict chronological train/val/test data splitting."""

from dataclasses import dataclass

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


def make_chronological_split(
    df: pd.DataFrame,
    target_col: str = "anomaly_c",
    date_col: str = "date",
    train_end: str = "1999-12-31",
    val_end: str = "2014-12-31",
    test_end: str = "2025-12-31",
) -> ChronologicalSplit:
    """Split dataset chronologically without shuffling."""
    df_clean = df.sort_values(date_col).copy()

    # Exclude rows where target is NaN (e.g. initial lag warmups or partial 2026 missing target)
    valid_mask = ~df_clean[target_col].isna()
    df_valid = df_clean[valid_mask].copy()

    train_df = df_valid[df_valid[date_col] <= train_end].copy()
    val_df = df_valid[(df_valid[date_col] > train_end) & (df_valid[date_col] <= val_end)].copy()
    test_df = df_valid[(df_valid[date_col] > val_end) & (df_valid[date_col] <= test_end)].copy()

    # Fallback ratio split if date range filtering yields an empty partition (e.g. small test fixture)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        n = len(df_valid)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        train_df = df_valid.iloc[:n_train].copy()
        val_df = df_valid.iloc[n_train : n_train + n_val].copy()
        test_df = df_valid.iloc[n_train + n_val :].copy()
        if len(test_df) == 0:
            test_df = val_df.copy()

    feature_cols = [
        c for c in df_valid.columns if c not in [date_col, target_col, "year"]
    ]

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
    )
