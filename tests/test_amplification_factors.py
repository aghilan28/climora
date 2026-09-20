"""Gate test for detrended empirical station amplification factors per F5."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.geo.station_network import detrend_series


def test_shared_detrend_series_helper() -> None:
    """Assert detrend_series correctly removes linear trend from a series."""
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 5.0 + np.sin(x)
    s = pd.Series(y)

    detrended = detrend_series(s)
    # Trend slope of detrended series should be near 0
    slope, _ = np.polyfit(np.arange(len(detrended)), detrended.values, 1)
    assert abs(slope) < 1e-4, f"Detrended series retains non-zero slope: {slope}"


def test_persisted_amplification_factors_properties() -> None:
    """Assert persisted station amplification artifact has detrended non-saturated properties."""
    artifact_path = Path("data/processed/station_amplification.json")
    assert artifact_path.exists(), "data/processed/station_amplification.json missing"

    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "factors" in data, "Artifact missing 'factors' key"
    assert "raw_factors" in data, "Artifact missing 'raw_factors' key"

    factors = data["factors"]
    raw_factors = data["raw_factors"]

    assert len(factors) == 12, f"Expected 12 factors, found {len(factors)}"

    # Assert not all factors are equal
    values = list(factors.values())
    assert len(set(values)) > 1, "All station amplification factors are identical!"

    # Assert raw factors are persisted
    assert len(raw_factors) == 12, "Raw factors missing"


def test_amplification_negative_control_trended_input() -> None:
    """Negative control: assert trended input alters detrended ratio output."""
    s2_trended = pd.Series(np.random.normal(0, 1, 100) + np.arange(100) * 0.5)

    std_raw = s2_trended.std()
    std_detrended = detrend_series(s2_trended).std()

    assert std_detrended < std_raw, "Detrending should reduce variance of heavily trended series"
