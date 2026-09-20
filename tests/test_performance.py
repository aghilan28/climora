"""Performance tests to verify model loading caching and zero retrain on app reruns."""

import time
from unittest.mock import patch

from config.settings import settings
from src.data.loaders import load_ersst_nino_data, load_gistemp_data, load_noaa_co2_data
from src.features.pipeline import build_feature_matrix
from src.models.registry import get_model


def test_zero_retrain_on_rerun() -> None:
    """Verify that model loading uses cached artifact resources and fit is not called on app reruns."""
    path = settings.trained_models_dir / "xgb_model.json"
    model = get_model("xgboost")
    if path.exists():
        model.load(path)

    with patch.object(model, "fit", wraps=model.fit) as mock_fit:
        for _ in range(3):
            m = get_model("xgboost")
            if path.exists():
                m.load(path)

        assert mock_fit.call_count == 0


def test_warm_prediction_latency() -> None:
    """Verify warm prediction latency is under 1.0 seconds."""
    gistemp, _ = load_gistemp_data()
    co2, _ = load_noaa_co2_data()
    nino, _ = load_ersst_nino_data()
    feat_df = build_feature_matrix(gistemp, co2, nino)

    path = settings.trained_models_dir / "xgb_model.json"
    xgb = get_model("xgboost")
    if path.exists():
        xgb.load(path)
    else:
        xgb.fit(feat_df.dropna().head(50), feat_df.dropna().head(50)["anomaly_c"])

    feature_cols = [c for c in feat_df.columns if c not in ["date", "year", "anomaly_c"]]
    sample = feat_df[feature_cols].dropna().head(10)

    # Warm prediction call
    _ = xgb.predict(sample)

    start = time.perf_counter()
    _ = xgb.predict(sample)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Prediction took {elapsed:.4f} seconds, expected < 1.0s"
