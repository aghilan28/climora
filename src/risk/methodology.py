"""Dual methodology risk threshold computation engine."""

from typing import Dict

import numpy as np
import pandas as pd

from config.settings import settings
from src.risk.bands import RiskBand


def compute_methodology_a_thresholds(y_train: pd.Series) -> Dict[str, float]:
    """Methodology A — Empirical Train-Set Quantiles.

    Uses train-set quantiles (50th, 75th, 90th, 97th percentiles) so thresholds
    are data-derived and leak-free. Strictly requires train set (<= 1999-12-31).
    """
    if isinstance(y_train.index, pd.DatetimeIndex):
        max_date = y_train.index.max()
        if max_date > pd.Timestamp("1999-12-31"):
            raise ValueError(f"Leakage Error: Methodology A thresholds must be computed strictly on training set (<= 1999-12-31). Got max date {max_date}")

    clean = y_train.dropna()
    return {
        "low": float(np.percentile(clean, 50)),
        "moderate": float(np.percentile(clean, 75)),
        "high": float(np.percentile(clean, 90)),
        "extreme": float(np.percentile(clean, 97)),
    }


def compute_methodology_b_thresholds(offset_c: float | None = None) -> Dict[str, float]:
    """Methodology B — Literature-Anchored Pre-Industrial Thresholds."""
    offset = offset_c if offset_c is not None else settings.pre_industrial_offset_c
    return {
        "low": 0.50 - offset,
        "moderate": 1.00 - offset,
        "high": 1.50 - offset,
        "extreme": 2.00 - offset,
    }


def classify_risk_band(anomaly: float, thresholds: Dict[str, float]) -> RiskBand:
    """Classify a single temperature anomaly into a RiskBand using given thresholds."""
    if anomaly >= thresholds["extreme"]:
        return RiskBand.EXTREME
    elif anomaly >= thresholds["high"]:
        return RiskBand.HIGH
    elif anomaly >= thresholds["moderate"]:
        return RiskBand.MODERATE
    else:
        return RiskBand.LOW
