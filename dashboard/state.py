"""Application state machine and session state bootstrap for CLIMORA AI."""

from enum import IntEnum
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from config.settings import settings
from src.anomaly.isolation_forest import ClimateAnomalyDetector
from src.data.cleaning import DataCleaner
from src.data.loaders import load_ersst_nino_data, load_gistemp_data, load_noaa_co2_data
from src.data.validation import DataValidator
from src.features.pipeline import build_feature_matrix
from src.models.artifacts import ArtifactManager
from src.models.registry import get_model
from src.utils.logging import setup_logger

logger = setup_logger("climora.dashboard.state")


class StateStage(IntEnum):
    """System lifecycle stages in strict dependency order."""

    NOT_LOADED = 0
    LOADED = 1
    VALIDATED = 2
    FEATURES_READY = 3
    MODELS_NOT_TRAINED = 4
    MODELS_TRAINED = 5
    PREDICTION_READY = 6
    ANOMALY_READY = 7
    FULL_ANALYSIS_READY = 8


class AppState:
    """Central state manager bound to Streamlit session state."""

    @classmethod
    def get_stage(cls) -> StateStage:
        """Get current system lifecycle stage."""
        if "stage" not in st.session_state:
            st.session_state["stage"] = StateStage.NOT_LOADED
        return StateStage(st.session_state["stage"])

    @classmethod
    def set_stage(cls, stage: StateStage) -> None:
        """Set current system lifecycle stage."""
        st.session_state["stage"] = stage
        logger.info(f"System stage transitioned to: {stage.name}")

    @classmethod
    def require_stage(cls, min_stage: StateStage) -> bool:
        """Check if current stage meets requirement. Display actionable message if not."""
        current = cls.get_stage()
        if current >= min_stage:
            return True

        stage_messages = {
            StateStage.LOADED: "Dataset is not loaded. Please initialize dataset from Overview or Dataset Explorer.",
            StateStage.VALIDATED: "Dataset validation incomplete.",
            StateStage.FEATURES_READY: "Feature engineering pipeline has not executed yet.",
            StateStage.MODELS_TRAINED: "Models have not been trained yet. Run `python scripts/train_models.py` or train models from Model Performance page.",
            StateStage.PREDICTION_READY: "Prediction state is not ready. Generate a forecast in Predictions page.",
            StateStage.ANOMALY_READY: "Anomaly detection has not been executed yet.",
            StateStage.FULL_ANALYSIS_READY: "Full analysis pipeline has not executed yet.",
        }

        msg = stage_messages.get(min_stage, f"Prerequisite state '{min_stage.name}' not reached.")
        st.warning(f"⚠️ **Prerequisite Unmet**: {msg}")
        return False

    @classmethod
    def bootstrap(cls, force_reload: bool = False) -> None:
        """Bootstrap the application state from cached data and artifacts."""
        if "bootstrapped" in st.session_state and not force_reload:
            return

        logger.info("Bootstrapping CLIMORA AI application state...")
        try:
            # 1. Load data
            gistemp_df, manifest_giss = load_gistemp_data(offline=True)
            co2_df, _ = load_noaa_co2_data(offline=True)
            nino_df, _ = load_ersst_nino_data(offline=True)

            st.session_state["raw_gistemp"] = gistemp_df
            st.session_state["raw_co2"] = co2_df
            st.session_state["raw_nino"] = nino_df
            cls.set_stage(StateStage.LOADED)

            # 2. Validate data
            validator = DataValidator()
            val_report = validator.validate(gistemp_df, "GISTEMP v4")
            st.session_state["validation_report"] = val_report
            cls.set_stage(StateStage.VALIDATED)

            # 3. Clean data
            cleaner = DataCleaner()
            clean_df, clean_report = cleaner.clean(gistemp_df)
            st.session_state["clean_gistemp"] = clean_df
            st.session_state["cleaning_report"] = clean_report

            # 4. Feature engineering
            feat_df = build_feature_matrix(clean_df, co2_df, nino_df)
            st.session_state["feature_df"] = feat_df
            cls.set_stage(StateStage.FEATURES_READY)

            # 5. Load models
            models = cls.load_models()
            if models:
                st.session_state["models"] = models
                cls.set_stage(StateStage.MODELS_TRAINED)
            else:
                cls.set_stage(StateStage.MODELS_NOT_TRAINED)

            # 6. Anomaly detection
            if cls.get_stage() >= StateStage.FEATURES_READY:
                detector = ClimateAnomalyDetector(contamination=0.05, seed=settings.seed)
                feature_cols = [
                    c for c in feat_df.columns if c not in ["date", "year", "month", "anomaly_c"]
                ]
                anomaly_df, anomaly_meta = detector.fit_predict(feat_df, feature_cols=feature_cols)
                st.session_state["anomaly_result"] = {
                    "df": anomaly_df,
                    "metadata": anomaly_meta,
                    "detector": detector,
                }
                cls.set_stage(StateStage.ANOMALY_READY)

            if cls.get_stage() >= StateStage.MODELS_TRAINED:
                cls.set_stage(StateStage.FULL_ANALYSIS_READY)

            st.session_state["bootstrapped"] = True
            logger.info("Application state bootstrap completed successfully.")

        except Exception as e:
            logger.error(f"Error during state bootstrap: {e}", exc_info=True)
            cls.set_stage(StateStage.NOT_LOADED)
            st.session_state["bootstrap_error"] = str(e)

    @classmethod
    def load_models(cls) -> Dict[str, Any]:
        """Load trained models from disk if present."""
        models: Dict[str, Any] = {}
        art_mgr = ArtifactManager()
        model_files = {
            "xgboost": "xgb_model.json",
            "lightgbm": "lgbm_model.txt",
            "lstm": "lstm_model.pt",
        }

        for model_name, filename in model_files.items():
            path = settings.trained_models_dir / filename
            if path.exists():
                try:
                    m = get_model(model_name)
                    m.load(path)
                    meta = art_mgr.load_metadata(model_name)
                    metrics = art_mgr.load_metrics(model_name)
                    models[model_name] = {
                        "model": m,
                        "metadata": meta,
                        "metrics": metrics,
                    }
                except Exception as e:
                    logger.debug(f"Model {model_name} could not be loaded: {e}")
        return models


def get_raw_gistemp() -> Optional[pd.DataFrame]:
    """Retrieve raw GISTEMP DataFrame from session state."""
    return st.session_state.get("raw_gistemp")


def get_clean_gistemp() -> Optional[pd.DataFrame]:
    """Retrieve clean GISTEMP DataFrame from session state."""
    return st.session_state.get("clean_gistemp")


def get_feature_df() -> Optional[pd.DataFrame]:
    """Retrieve engineered feature DataFrame from session state."""
    return st.session_state.get("feature_df")


def get_models() -> Dict[str, Any]:
    """Retrieve loaded model dictionary."""
    return st.session_state.get("models", {})
