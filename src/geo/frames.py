"""Spatial-temporal frame series generator for map visualisations."""

from typing import Any, Dict, List

import pandas as pd

from src.geo.risk_surface import compute_station_risk_surface


def build_temporal_map_frames(clean_df: pd.DataFrame, sample_interval: int = 12) -> List[Dict[str, Any]]:
    """Build a list of historical time frame snapshots containing spatial risk data for map time scrubbing."""
    if clean_df is None or "date" not in clean_df.columns or "anomaly_c" not in clean_df.columns:
        return []

    df_sub = clean_df.iloc[::sample_interval].copy()
    frames = []

    for _, row in df_sub.iterrows():
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m")
        anom = float(row["anomaly_c"]) if pd.notna(row["anomaly_c"]) else 0.0
        risk_df = compute_station_risk_surface(anom, methodology="A")

        frames.append({
            "date": date_str,
            "global_anomaly_c": round(anom, 2),
            "stations": risk_df.to_dict(orient="records"),
        })

    return frames
