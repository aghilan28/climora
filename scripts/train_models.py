"""Headless training script for CLIMORA AI climate models."""
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.data.loaders import load_ersst_nino_data, load_gistemp_data, load_noaa_co2_data
from src.evaluation.metrics import compute_regression_metrics
from src.features.pipeline import build_feature_matrix
from src.models.artifacts import ArtifactManager
from src.models.baselines import ClimatologyModel, SeasonalNaiveModel
from src.models.xgboost_model import XGBoostClimateModel
from src.splits.chronological import make_chronological_split
from src.utils.logging import logger


def run_training(model_type: str = "all") -> Dict[str, Any]:
    """Train selected model(s) on chronological split and persist artifacts."""
    logger.info("=== Starting CLIMORA AI Model Training (model_type=%s) ===", model_type)

    # 1. Load Data & Build Feature Matrix
    df_gistemp, _ = load_gistemp_data(offline=True)
    df_co2, _ = load_noaa_co2_data(offline=True)
    df_nino, _ = load_ersst_nino_data(offline=True)

    df_feats = build_feature_matrix(df_gistemp, df_co2, df_nino)

    # 2. Chronological Split
    split = make_chronological_split(df_feats)
    logger.info(
        "Chronological Split: Train=%d, Val=%d, Test=%d",
        len(split.X_train),
        len(split.X_val),
        len(split.X_test),
    )

    artifact_mgr = ArtifactManager()
    metrics_summary = {}

    # 3. Fit Baselines First (Mandatory Benchmark)
    seasonal_naive = SeasonalNaiveModel().fit(split.X_train, split.y_train)
    val_preds_sn = seasonal_naive.predict(split.X_val)
    test_preds_sn = seasonal_naive.predict(split.X_test)

    sn_val_metrics = compute_regression_metrics(split.y_val, val_preds_sn)
    sn_test_metrics = compute_regression_metrics(split.y_test, test_preds_sn)
    baseline_test_rmse = sn_test_metrics["rmse"]

    sn_test_metrics["skill_score"] = 0.0
    artifact_mgr.save_metrics("seasonal_naive", sn_test_metrics)
    seasonal_naive.save(settings.models_dir)
    metrics_summary["SeasonalNaive"] = sn_test_metrics

    climatology = ClimatologyModel().fit(split.X_train, split.y_train)
    test_preds_clim = climatology.predict(split.X_test)
    clim_test_metrics = compute_regression_metrics(split.y_test, test_preds_clim, baseline_rmse=baseline_test_rmse)
    artifact_mgr.save_metrics("climatology", clim_test_metrics)
    climatology.save(settings.models_dir)
    metrics_summary["Climatology"] = clim_test_metrics

    # 4. Fit XGBoost
    if model_type.lower() in ["xgboost", "all"]:
        logger.info("Training XGBoost Climate Model...")
        xgb_model = XGBoostClimateModel()
        xgb_model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

        val_preds_xgb = xgb_model.predict(split.X_val)
        test_preds_xgb = xgb_model.predict(split.X_test)

        val_metrics_xgb = compute_regression_metrics(split.y_val, val_preds_xgb, baseline_rmse=sn_val_metrics["rmse"])
        test_metrics_xgb = compute_regression_metrics(split.y_test, test_preds_xgb, baseline_rmse=baseline_test_rmse)

        # Update metadata with metrics & save
        xgb_model.training_meta.update({
            "val_metrics": val_metrics_xgb,
            "test_metrics": test_metrics_xgb,
            "val_rmse": val_metrics_xgb["rmse"],
        })
        xgb_model.save(settings.models_dir)
        artifact_mgr.save_metrics("xgboost", test_metrics_xgb)
        metrics_summary["XGBoost"] = test_metrics_xgb

    # 5. Fit LightGBM
    if model_type.lower() in ["lightgbm", "all"]:
        from src.models.lightgbm_model import LightGBMClimateModel
        logger.info("Training LightGBM Climate Model...")
        lgbm_model = LightGBMClimateModel()
        lgbm_model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

        val_preds_lgbm = lgbm_model.predict(split.X_val)
        test_preds_lgbm = lgbm_model.predict(split.X_test)

        val_metrics_lgbm = compute_regression_metrics(split.y_val, val_preds_lgbm, baseline_rmse=sn_val_metrics["rmse"])
        test_metrics_lgbm = compute_regression_metrics(split.y_test, test_preds_lgbm, baseline_rmse=baseline_test_rmse)

        lgbm_model.training_meta.update({
            "val_metrics": val_metrics_lgbm,
            "test_metrics": test_metrics_lgbm,
            "val_rmse": val_metrics_lgbm["rmse"],
        })
        lgbm_model.save(settings.models_dir)
        artifact_mgr.save_metrics("lightgbm", test_metrics_lgbm)
        metrics_summary["LightGBM"] = test_metrics_lgbm

    # 6. Fit PyTorch LSTM
    if model_type.lower() in ["lstm", "all"]:
        from src.models.lstm_model import PyTorchLSTMClimateModel
        logger.info("Training PyTorch LSTM Climate Model...")
        lstm_model = PyTorchLSTMClimateModel(params={"epochs": 30, "patience": 10, "seed": settings.seed})
        lstm_model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

        val_preds_lstm = lstm_model.predict(split.X_val)
        test_preds_lstm = lstm_model.predict(split.X_test)

        val_metrics_lstm = compute_regression_metrics(split.y_val, val_preds_lstm, baseline_rmse=sn_val_metrics["rmse"])
        test_metrics_lstm = compute_regression_metrics(split.y_test, test_preds_lstm, baseline_rmse=baseline_test_rmse)

        lstm_model.training_meta.update({
            "val_metrics": val_metrics_lstm,
            "test_metrics": test_metrics_lstm,
            "val_rmse": val_metrics_lstm["rmse"],
        })
        lstm_model.save(settings.models_dir)
        artifact_mgr.save_metrics("lstm", test_metrics_lstm)
        metrics_summary["LSTM"] = test_metrics_lstm

    print("\n=======================================================")
    print("CLIMORA AI — MODEL TRAINING & EVALUATION REPORT")
    print("=======================================================")
    print(f"Train Slice: {len(split.X_train)} rows | Val Slice: {len(split.X_val)} rows | Test Slice: {len(split.X_test)} rows")
    print("-" * 55)
    print(f"{'Model':<15} | {'MAE':<7} | {'RMSE':<7} | {'R²':<7} | {'Skill Score':<10}")
    print("-" * 55)
    for m_name, m_vals in metrics_summary.items():
        print(f"{m_name:<15} | {m_vals['mae']:<7.4f} | {m_vals['rmse']:<7.4f} | {m_vals['r2']:<7.4f} | {m_vals['skill_score']:<10.4f}")
    print("=======================================================\n")

    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIMORA AI climate forecasting models.")
    parser.add_argument("--model", type=str, default="all", help="Model name: xgboost, lightgbm, lstm, baselines, all")
    args = parser.parse_args()

    run_training(model_type=args.model)


if __name__ == "__main__":
    main()
