"""Dataset high-level summary overview computation."""

from typing import Any, Dict

import pandas as pd


def compute_dataset_overview(df: pd.DataFrame, target_col: str = "anomaly_c") -> Dict[str, Any]:
    """Compute summary KPIs and statistics for dataset overview."""
    if df.empty:
        return {
            "total_records": 0,
            "column_count": len(df.columns),
            "start_date": "N/A",
            "end_date": "N/A",
            "missing_pct": 0.0,
        }

    date_col = "date" if "date" in df.columns else None
    start_date = str(df[date_col].min().date()) if date_col and not df[date_col].isna().all() else "N/A"
    end_date = str(df[date_col].max().date()) if date_col and not df[date_col].isna().all() else "N/A"

    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    missing_pct = float((missing_cells / total_cells) * 100.0) if total_cells > 0 else 0.0

    stats: Dict[str, Any] = {
        "total_records": len(df),
        "column_count": len(df.columns),
        "start_date": start_date,
        "end_date": end_date,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
    }

    if target_col in df.columns:
        valid_series = df[target_col].dropna()
        if not valid_series.empty:
            stats.update({
                "mean": float(valid_series.mean()),
                "std": float(valid_series.std()),
                "min": float(valid_series.min()),
                "max": float(valid_series.max()),
                "latest_value": float(valid_series.iloc[-1]),
            })

    return stats
