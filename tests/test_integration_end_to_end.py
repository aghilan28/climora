"""End-to-end integration test verifying full pipeline execution and cross-module consistency."""

import time

from config.settings import settings
from src.anomaly.isolation_forest import ClimateAnomalyDetector
from src.data.cleaning import DataCleaner
from src.data.providers.nasa_giss import NasaGissProvider
from src.data.validation import DataValidator, ValidationStatus
from src.explain.narrative import build_explanation_narrative
from src.explain.tree_importance import get_tree_feature_importance
from src.features.pipeline import build_feature_matrix
from src.models.xgboost_model import XGBoostClimateModel
from src.risk.scoring import compute_risk_score
from src.splits.chronological import make_chronological_split


def test_end_to_end_pipeline_integration(gistemp_fixture_bytes: bytes) -> None:
    """Verify end-to-end pipeline execution from raw fixture bytes to prediction, risk, and explainability."""
    start_time = time.time()

    # 1. Provider Ingestion
    provider = NasaGissProvider(settings.raw_data_dir / "gistemp")
    raw_df = provider.parse(gistemp_fixture_bytes)
    assert len(raw_df) > 0
    assert "date" in raw_df.columns
    assert "anomaly_c" in raw_df.columns

    # 2. Validation
    validator = DataValidator()
    val_report = validator.validate(raw_df, dataset_name="GISTEMP Fixture")
    assert val_report.overall_status in [ValidationStatus.VALID, ValidationStatus.WARNING]

    # 3. Cleaning
    cleaner = DataCleaner()
    clean_df, clean_report = cleaner.clean(raw_df)
    assert len(clean_df) > 0
    assert clean_report.cleaned_rows == len(clean_df)

    # 4. Feature Matrix Construction
    feat_df = build_feature_matrix(clean_df)
    assert len(feat_df) > 0
    assert "anomaly_c_lag_1" in feat_df.columns
    assert "anomaly_c_roll_mean_12" in feat_df.columns

    # 5. Chronological Data Split
    splits = make_chronological_split(feat_df)
    assert len(splits.train_df) > 0
    assert len(splits.test_df) > 0

    # 6. Model Training & Inference (Fast XGBoost config)
    xgb_model = XGBoostClimateModel(params={"n_estimators": 10, "max_depth": 3})
    xgb_model.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
    assert xgb_model.name == "XGBoost"

    preds = xgb_model.predict(splits.X_test)
    assert len(preds) == len(splits.X_test)
    pred_val = float(preds[0])

    # 7. Risk Engine Scoring
    risk_score, risk_band = compute_risk_score(pred_val)
    assert 0.0 <= risk_score <= 100.0
    assert risk_band.value in ["Low", "Moderate", "High", "Severe", "Extreme"]

    # Cross-Module Consistency (§66): Recompute risk score from prediction and assert bitwise equality
    recomputed_score, recomputed_band = compute_risk_score(pred_val)
    assert risk_score == recomputed_score
    assert risk_band == recomputed_band

    # 8. Anomaly Detection
    detector = ClimateAnomalyDetector(contamination=0.05, seed=42)
    feature_cols = [c for c in feat_df.columns if c not in ["date", "year", "month", "anomaly_c"]]
    anom_df, sum_meta = detector.fit_predict(feat_df, feature_cols=feature_cols)
    assert "is_anomaly" in anom_df.columns
    assert sum_meta["total_observations"] == len(feat_df)

    # 9. Model-Grounded Explainability
    df_imp = get_tree_feature_importance(xgb_model)
    assert not df_imp.empty
    narrative = build_explanation_narrative(df_imp["feature"].tolist(), df_imp["importance"].values, top_k=3)
    assert not narrative.startswith("EXPLANATION UNAVAILABLE")

    elapsed = time.time() - start_time
    assert elapsed < 120.0, f"End-to-end integration test took too long: {elapsed:.2f} s"
