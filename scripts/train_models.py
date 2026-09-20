"""Headless training script for CLIMORA AI climate models."""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from config.settings import settings
from src.data.loaders import load_ersst_nino_data, load_gistemp_data, load_noaa_co2_data
from src.evaluation.metrics import compute_regression_metrics
from src.features.pipeline import build_feature_matrix
from src.models.artifacts import ArtifactManager
from src.models.baselines import ClimatologyModel, SeasonalNaiveModel
from src.models.xgboost_model import XGBoostClimateModel
from src.splits.chronological import make_chronological_split
from src.utils.logging import logger


def run_training(model_type: str = "all", pin_check: bool = False) -> Dict[str, Any]:
    """Train selected model(s) on chronological split and persist artifacts."""
    logger.info("=== Starting CLIMORA AI Model Training (model_type=%s, pin_check=%s) ===", model_type, pin_check)

    # 1. Load Data & Build Feature Matrix
    try:
        df_gistemp, entry_gistemp = load_gistemp_data(offline=False)
        df_co2, _ = load_noaa_co2_data(offline=False)
        df_nino, _ = load_ersst_nino_data(offline=False)
    except Exception as e:
        logger.error("Failed to load raw datasets: %s", e)
        print("Data not available. Run: python scripts/download_data.py")
        sys.exit(2)

    # Pin check verification
    data_yaml_path = settings.base_dir / "config" / "data.yaml"
    pinned_sha = ""
    if data_yaml_path.exists():
        data_cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        pinned_sha = data_cfg.get("gistemp", {}).get("pinned_sha256", "")

    current_sha = entry_gistemp.get("sha256", "")
    if pin_check:
        if pinned_sha and current_sha != pinned_sha:
            print(f"PIN CHECK FAILED: disk GISTEMP sha256 ({current_sha}) != pinned sha256 ({pinned_sha})")
            sys.exit(1)
        print(f"PIN CHECK PASSED: disk GISTEMP sha256 matches pinned {current_sha}")

    df_feats = build_feature_matrix(df_gistemp, df_co2, df_nino)

    # 2. Chronological Split
    split = make_chronological_split(df_feats)
    logger.info(
        "Chronological Split: Train=%d, Val=%d, Test=%d (Window: %s .. %s)",
        len(split.X_train),
        len(split.X_val),
        len(split.X_test),
        split.test_start,
        split.test_end,
    )

    artifact_mgr = ArtifactManager()
    metrics_summary: Dict[str, Dict[str, Any]] = {}

    # 3. Fit Baselines First (Mandatory Benchmark)
    seasonal_naive = SeasonalNaiveModel().fit(split.X_train, split.y_train)
    val_preds_sn = seasonal_naive.predict(split.X_val)
    test_preds_sn = seasonal_naive.predict(split.X_test)

    sn_val_metrics: Dict[str, Any] = compute_regression_metrics(split.y_val, val_preds_sn)
    sn_test_metrics: Dict[str, Any] = compute_regression_metrics(split.y_test, test_preds_sn)
    baseline_test_rmse = float(sn_test_metrics["rmse"])

    sn_test_metrics["skill_score"] = 0.0
    sn_test_metrics["test_start"] = split.test_start
    sn_test_metrics["test_end"] = split.test_end
    sn_test_metrics["n_test"] = len(split.X_test)

    # h=12 baseline metrics
    test_df_h12 = split.test_df.dropna(subset=["anomaly_c_h12"])
    if len(test_df_h12) > 0:
        sn_h12_preds = test_df_h12["anomaly_c"].values
        h12_m: Dict[str, Any] = compute_regression_metrics(test_df_h12["anomaly_c_h12"], sn_h12_preds)
        h12_m["skill_score"] = 0.0
        sn_test_metrics["h12_metrics"] = h12_m

    artifact_mgr.save_metrics("seasonal_naive", sn_test_metrics)
    seasonal_naive.save(settings.models_dir)
    metrics_summary["SeasonalNaive"] = sn_test_metrics

    climatology = ClimatologyModel().fit(split.X_train, split.y_train)
    test_preds_clim = climatology.predict(split.X_test)
    clim_test_metrics: Dict[str, Any] = compute_regression_metrics(split.y_test, test_preds_clim, baseline_rmse=baseline_test_rmse)
    clim_test_metrics["test_start"] = split.test_start
    clim_test_metrics["test_end"] = split.test_end
    clim_test_metrics["n_test"] = len(split.X_test)

    if len(test_df_h12) > 0:
        clim_h12_preds = climatology.predict(test_df_h12[split.X_test.columns])
        sn_h12_rmse = float(sn_test_metrics.get("h12_metrics", {}).get("rmse", baseline_test_rmse))
        clim_test_metrics["h12_metrics"] = compute_regression_metrics(
            test_df_h12["anomaly_c_h12"], clim_h12_preds, baseline_rmse=sn_h12_rmse
        )

    artifact_mgr.save_metrics("climatology", clim_test_metrics)
    climatology.save(settings.models_dir)
    metrics_summary["Climatology"] = clim_test_metrics

    # Helper to train h=12 target model
    def train_h12_head(model_class: Any, **kwargs: Any) -> Dict[str, Any]:
        df_clean_h12 = df_feats.dropna(subset=["anomaly_c_h12"]).copy()
        split_h12 = make_chronological_split(df_clean_h12, target_col="anomaly_c_h12")
        m_h12 = model_class(**kwargs)
        m_h12.fit(split_h12.X_train, split_h12.y_train, split_h12.X_val, split_h12.y_val)
        h12_preds = m_h12.predict(split_h12.X_test)
        base_h12_dict = sn_test_metrics.get("h12_metrics", {})
        base_h12_rmse = float(base_h12_dict.get("rmse", baseline_test_rmse)) if isinstance(base_h12_dict, dict) else baseline_test_rmse
        return compute_regression_metrics(split_h12.y_test, h12_preds, baseline_rmse=base_h12_rmse)

    # 4. Fit XGBoost
    if model_type.lower() in ["xgboost", "all"]:
        logger.info("Training XGBoost Climate Model...")
        xgb_model = XGBoostClimateModel()
        xgb_model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

        val_preds_xgb = xgb_model.predict(split.X_val)
        test_preds_xgb = xgb_model.predict(split.X_test)

        val_metrics_xgb: Dict[str, Any] = compute_regression_metrics(split.y_val, val_preds_xgb, baseline_rmse=sn_val_metrics["rmse"])
        test_metrics_xgb: Dict[str, Any] = compute_regression_metrics(split.y_test, test_preds_xgb, baseline_rmse=baseline_test_rmse)

        test_metrics_xgb["test_start"] = split.test_start
        test_metrics_xgb["test_end"] = split.test_end
        test_metrics_xgb["n_test"] = len(split.X_test)
        test_metrics_xgb["h12_metrics"] = train_h12_head(XGBoostClimateModel)

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

        val_metrics_lgbm: Dict[str, Any] = compute_regression_metrics(split.y_val, val_preds_lgbm, baseline_rmse=sn_val_metrics["rmse"])
        test_metrics_lgbm: Dict[str, Any] = compute_regression_metrics(split.y_test, test_preds_lgbm, baseline_rmse=baseline_test_rmse)

        test_metrics_lgbm["test_start"] = split.test_start
        test_metrics_lgbm["test_end"] = split.test_end
        test_metrics_lgbm["n_test"] = len(split.X_test)
        test_metrics_lgbm["h12_metrics"] = train_h12_head(LightGBMClimateModel)

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

        val_metrics_lstm: Dict[str, Any] = compute_regression_metrics(split.y_val, val_preds_lstm, baseline_rmse=sn_val_metrics["rmse"])
        test_metrics_lstm: Dict[str, Any] = compute_regression_metrics(split.y_test, test_preds_lstm, baseline_rmse=baseline_test_rmse)

        test_metrics_lstm["test_start"] = split.test_start
        test_metrics_lstm["test_end"] = split.test_end
        test_metrics_lstm["n_test"] = len(split.X_test)
        test_metrics_lstm["h12_metrics"] = train_h12_head(PyTorchLSTMClimateModel, params={"epochs": 30, "patience": 10, "seed": settings.seed})

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
    print(f"GISTEMP Snapshot SHA: {current_sha}")
    print(f"Test Window: {split.test_start} .. {split.test_end} (n_test={len(split.X_test)} months)")
    print(f"Train Slice: {len(split.X_train)} rows | Val Slice: {len(split.X_val)} rows | Test Slice: {len(split.X_test)} rows")
    print("-" * 65)
    print(f"{'Model':<15} | {'MAE (h=1)':<9} | {'RMSE (h=1)':<10} | {'Skill (h=1)':<11} | {'Skill (h=12)':<11}")
    print("-" * 65)
    for m_name, m_vals in metrics_summary.items():
        h12_skill = m_vals.get("h12_metrics", {}).get("skill_score", 0.0)
        print(f"{m_name:<15} | {m_vals['mae']:<9.4f} | {m_vals['rmse']:<10.4f} | {m_vals['skill_score']:<11.4f} | {h12_skill:<11.4f}")
    print("=======================================================\n")

    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIMORA AI climate forecasting models.")
    parser.add_argument("--model", type=str, default="all", help="Model name: xgboost, lightgbm, lstm, baselines, all")
    parser.add_argument("--pin-check", action="store_true", help="Assert disk GISTEMP sha256 matches pinned_sha256.")
    args = parser.parse_args()

    run_training(model_type=args.model, pin_check=args.pin_check)


if __name__ == "__main__":
    main()
