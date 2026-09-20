"""Dual methodology risk threshold computation engine."""

from typing import Dict

import numpy as np
import pandas as pd

from config.settings import settings
from src.risk.bands import RiskBand


def compute_methodology_a_thresholds(y_train: pd.Series) -> Dict[str, float]:
    """Methodology A — Empirical Train-Set Quantiles.

    Uses train-set quantiles (50th, 75th, 90th, 97th percentiles) so thresholds
    are data-derived and leak-free.
    """
    clean = y_train.dropna()
    return {
        "low": float(np.percentile(clean, 50)),
        "moderate": float(np.percentile(clean, 75)),
        "high": float(np.percentile(clean, 90)),
        "extreme": float(np.percentile(clean, 97)),
    }


def compute_methodology_b_thresholds(offset_c: float | None = None) -> Dict[str, float]:
    """Methodology B — Literature-Anchored Pre-Industrial Thresholds.

    Thresholds: +0.5°C, +1.0°C, +1.5°C, and +2.0°C above pre-industrial baseline (1850–1900).
    Converts GISTEMP 1951–1980 reference to pre-industrial explicitly:
    GISTEMP_anomaly = PreIndustrial_anomaly - offset_c
    Where offset_c = 0.25 °C (IPCC AR6 estimated offset).
    """
    offset = offset_c if offset_c is not None else settings.pre_industrial_offset_c
    return {
        "low": 0.50 - offset,        # 0.25 °C in GISTEMP baseline
        "moderate": 1.00 - offset,   # 0.75 °C in GISTEMP baseline
        "high": 1.50 - offset,       # 1.25 °C in GISTEMP baseline (+1.5°C Paris Agreement limit)
        "extreme": 2.00 - offset,    # 1.75 °C in GISTEMP baseline (+2.0°C upper limit)
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
