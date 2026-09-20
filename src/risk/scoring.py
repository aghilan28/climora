"""Deterministic risk scoring (0-100) engine."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.risk.bands import RiskBand
from src.risk.methodology import (
    classify_risk_band,
    compute_methodology_a_thresholds,
    compute_methodology_b_thresholds,
)


def compute_risk_score(
    anomaly: float,
    rolling_exceedance_frac: float = 0.0,
    methodology: str = "A",
    thresholds: Dict[str, float] | None = None,
) -> Tuple[float, RiskBand]:
    """Compute deterministic risk score (0-100) and risk band.

    Score formulation:
    - Base term: monotone linear scaling of anomaly relative to low and extreme thresholds (0-80 pts).
    - Persistence term: rolling exceedance fraction * 20 pts (0-20 pts).
    Total score = min(100.0, max(0.0, Base + Persistence)).
    """
    if thresholds is None:
        if methodology.upper() == "B":
            thresholds = compute_methodology_b_thresholds()
        else:
            # Fallback default empirical bounds if train set not passed
            thresholds = {"low": 0.0, "moderate": 0.45, "high": 0.85, "extreme": 1.15}

    t_low = thresholds["low"]
    t_extreme = thresholds["extreme"]

    if t_extreme > t_low:
        raw_pct = (anomaly - t_low) / (t_extreme - t_low)
    else:
        raw_pct = 0.0

    base_score = float(np.clip(raw_pct, 0.0, 1.0) * 80.0)
    persistence_score = float(np.clip(rolling_exceedance_frac, 0.0, 1.0) * 20.0)

    total_score = float(np.clip(base_score + persistence_score, 0.0, 100.0))
    band = classify_risk_band(anomaly, thresholds)

    return round(total_score, 2), band


def score_dataframe_risk(
    df: pd.DataFrame,
    target_col: str = "anomaly_c",
    methodology: str = "A",
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score full DataFrame with risk scores, risk bands, and historical context."""
    df_out = df.copy()
    if target_col not in df_out.columns:
        return df_out

    if methodology.upper() == "A":
        ref_series = train_df[target_col] if train_df is not None else df_out[target_col]
        thresholds = compute_methodology_a_thresholds(ref_series)
    else:
        thresholds = compute_methodology_b_thresholds()

    # Compute rolling exceedance fraction over 12 months
    high_threshold = thresholds["high"]
    is_high = (df_out[target_col].shift(1) >= high_threshold).astype(float)
    rolling_exceedance = is_high.rolling(window=12, min_periods=1).mean().fillna(0.0)

    scores, bands = [], []
    for idx, row in df_out.iterrows():
        anom = row[target_col]
        if pd.isna(anom):
            scores.append(0.0)
            bands.append(RiskBand.LOW)
        else:
            p_frac = float(rolling_exceedance.loc[idx])
            s, b = compute_risk_score(float(anom), p_frac, methodology, thresholds)
            scores.append(s)
            bands.append(b)

    df_out["risk_score"] = scores
    df_out["risk_band"] = [b.value for b in bands]
    df_out["risk_methodology"] = methodology.upper()

    return df_out
