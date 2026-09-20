"""Unit tests for climate risk classification engine."""

import pandas as pd

from src.data.providers.nasa_giss import NasaGissProvider
from src.risk.bands import RiskBand
from src.risk.methodology import (
    compute_methodology_a_thresholds,
    compute_methodology_b_thresholds,
)
from src.risk.scoring import compute_risk_score, score_dataframe_risk


def test_boundary_handling() -> None:
    thresholds = {"low": 0.0, "moderate": 0.5, "high": 1.0, "extreme": 1.5}

    # Exactly at extreme threshold
    score_ext, band_ext = compute_risk_score(1.5, thresholds=thresholds)
    assert band_ext == RiskBand.EXTREME
    assert score_ext >= 80.0

    # Exactly at high threshold
    score_h, band_h = compute_risk_score(1.0, thresholds=thresholds)
    assert band_h == RiskBand.HIGH

    # Below low threshold
    score_l, band_l = compute_risk_score(-0.5, thresholds=thresholds)
    assert band_l == RiskBand.LOW
    assert score_l == 0.0


def test_score_monotonicity() -> None:
    thresholds = {"low": 0.0, "moderate": 0.5, "high": 1.0, "extreme": 1.5}

    s1, _ = compute_risk_score(0.2, thresholds=thresholds)
    s2, _ = compute_risk_score(0.7, thresholds=thresholds)
    s3, _ = compute_risk_score(1.2, thresholds=thresholds)

    assert s1 <= s2 <= s3


def test_methodology_a_vs_b_difference(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    thresh_a = compute_methodology_a_thresholds(df["anomaly_c"])
    thresh_b = compute_methodology_b_thresholds(offset_c=0.25)

    # Assert thresholds differ between empirical distribution (A) and policy target (B)
    assert thresh_a != thresh_b


def test_risk_scoring_reproducibility(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    res1 = score_dataframe_risk(df, methodology="A")
    res2 = score_dataframe_risk(df, methodology="A")

    # Byte-identical reproducibility
    pd.testing.assert_frame_equal(res1, res2)
