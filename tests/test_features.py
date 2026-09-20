"""Unit tests for feature engineering and anti-leakage verification."""

import numpy as np
import pandas as pd

from src.data.providers.nasa_giss import NasaGissProvider
from src.features.lags import add_lag_features
from src.features.pipeline import build_feature_matrix
from src.features.registry import FEATURE_REGISTRY
from src.features.rolling import add_rolling_features
from src.features.trend import add_trend_features


def test_hand_computed_lag_and_rolling() -> None:
    """Verify hand-computed expected values for lag-12, rolling-12 mean/std, differencing.

    Test arithmetic rationale:
    Values: y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    For row 12 (value 13.0, 0-indexed index 12):
    - lag_12: y(12-12) = y(0) = 1.0
    - rolling_12_mean (shift 1): mean of y(0..11) = mean(1..12) = 6.5
    - diff_1 (shift 1): y(11) - y(10) = 12.0 - 11.0 = 1.0
    """
    values = [float(i) for i in range(1, 14)]
    dates = pd.date_range("2000-01-01", periods=13, freq="MS")
    df = pd.DataFrame({"date": dates, "anomaly_c": values})

    df_lags = add_lag_features(df, col="anomaly_c", lags=[12])
    assert df_lags.loc[12, "anomaly_c_lag_12"] == 1.0

    df_roll = add_rolling_features(df, col="anomaly_c", windows=[12], stats=["mean", "std"])
    assert abs(df_roll.loc[12, "anomaly_c_roll_mean_12"] - 6.5) < 1e-6
    expected_std = float(np.std(list(range(1, 13)), ddof=1))
    assert abs(df_roll.loc[12, "anomaly_c_roll_std_12"] - expected_std) < 1e-6

    df_trend = add_trend_features(df, col="anomaly_c", diffs=[1])
    assert df_trend.loc[12, "anomaly_c_diff_1"] == 1.0


def test_no_future_leakage(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    """Prove that perturbing a future target row cannot change any feature at a trailing row."""
    provider = NasaGissProvider(tmp_path)
    df_raw = provider.parse(gistemp_fixture_bytes)

    # Build feature matrix on unperturbed data
    df_feat_orig = build_feature_matrix(df_raw)

    # Corrupt a future row at index 50 (e.g. inject 99.0 °C anomaly)
    df_perturbed = df_raw.copy()
    target_idx = 50
    df_perturbed.loc[target_idx, "anomaly_c"] = 99.0

    df_feat_perturbed = build_feature_matrix(df_perturbed)

    # For all rows BEFORE target_idx (0 to target_idx - 1), all feature values must be IDENTICAL
    feat_cols = [c for c in df_feat_orig.columns if c not in ["date", "anomaly_c", "anomaly_c_h12", "year", "month"]]

    orig_sub = df_feat_orig.loc[: target_idx - 1, feat_cols].dropna(how="all")
    pert_sub = df_feat_perturbed.loc[: target_idx - 1, feat_cols].dropna(how="all")

    pd.testing.assert_frame_equal(orig_sub, pert_sub)


def test_feature_registry_completeness() -> None:
    """Verify feature registry entries have non-empty names, families, and leakage notes."""
    assert len(FEATURE_REGISTRY) > 0
    for feat in FEATURE_REGISTRY:
        assert feat.name
        assert feat.family
        assert feat.leakage_note
