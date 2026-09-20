"""Unit tests for model-grounded explainability and narrative builder."""

import numpy as np

from src.explain.narrative import build_explanation_narrative
from src.explain.shap_wrapper import compute_shap_explanation
from src.models.xgboost_model import XGBoostClimateModel


def test_narrative_derived_from_contributions() -> None:
    feature_names = ["anomaly_c_lag_1", "co2_ppm_lag_1", "month_sin"]
    contributions = np.array([0.25, -0.10, 0.05])

    narrative = build_explanation_narrative(feature_names, contributions, top_k=3)

    assert "anomaly_c_lag_1" in narrative
    assert "co2_ppm_lag_1" in narrative
    assert "0.250" in narrative
    assert "0.100" in narrative


def test_narrative_unavailable_on_empty_input() -> None:
    narrative = build_explanation_narrative([], np.array([]), reason_if_empty="Model unsupported")
    assert narrative.startswith("EXPLANATION UNAVAILABLE: Model unsupported")


def test_shap_explanation_consistency(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    from src.data.providers.nasa_giss import NasaGissProvider
    from src.features.pipeline import build_feature_matrix

    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)
    feat_df = build_feature_matrix(df)

    feature_cols = [c for c in feat_df.columns if c not in ["date", "year", "month", "anomaly_c"]]
    X = feat_df[feature_cols].fillna(0.0)
    y = feat_df["anomaly_c"].fillna(0.0)

    model = XGBoostClimateModel(params={"n_estimators": 5, "max_depth": 2})
    model.fit(X, y)

    res = compute_shap_explanation(model, X.iloc[[0]])
    assert isinstance(res, tuple)
    assert len(res) == 3
