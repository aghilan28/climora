"""Unit tests for Isolation Forest climate anomaly detection."""

import scipy.stats as stats

from src.anomaly.isolation_forest import ClimateAnomalyDetector
from src.data.providers.open_meteo import OpenMeteoProvider


def test_isolation_forest_anomaly_detection(openmeteo_fixture_bytes: bytes, tmp_path) -> None:
    provider = OpenMeteoProvider(tmp_path)
    df_raw = provider.parse(openmeteo_fixture_bytes)

    detector = ClimateAnomalyDetector(contamination=0.10, seed=42)
    df_results, summary = detector.fit_predict(df_raw)

    # 1. Labels in {0, 1}
    assert set(df_results["is_anomaly"].unique()).issubset({0, 1})

    # 2. Scores are finite and in range [0, 1]
    scores = df_results["anomaly_score"]
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0

    # 3. Counts match value_counts() exactly
    counts = df_results["is_anomaly"].value_counts().to_dict()
    assert summary["anomaly_count"] == counts.get(1, 0)
    assert summary["normal_count"] == counts.get(0, 0)

    # 4. Scores correlate monotonically with labels (higher score -> anomaly)
    spearman_corr, _ = stats.spearmanr(df_results["anomaly_score"], df_results["is_anomaly"])
    assert spearman_corr > 0.0

    # 5. Sensitivity table has 3 real computed rows
    sens_df = detector.compute_sensitivity_table(df_raw)
    assert len(sens_df) == 3
    assert list(sens_df["Contamination"]) == [0.01, 0.05, 0.10]
