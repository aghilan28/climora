"""XGBoost model implementation adhering to ClimateModel protocol."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config.settings import settings
from src.models.base import ClimateModel
from src.utils.logging import logger


class XGBoostClimateModel(ClimateModel):
    """XGBoost Gradient Boosted Trees Climate Forecasting Model."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": settings.seed,
        }
        self.scaler = StandardScaler()
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: List[str] = []
        self._is_fitted = False
        self.training_meta: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostClimateModel":
        self.feature_names = list(X_train.columns)

        # Fit scaler ONLY on train slice to prevent leakage
        X_tr_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        self.model = xgb.XGBRegressor(
            n_estimators=self.params.get("n_estimators", 300),
            learning_rate=self.params.get("learning_rate", 0.03),
            max_depth=self.params.get("max_depth", 5),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            random_state=self.params.get("random_state", settings.seed),
            early_stopping_rounds=self.params.get("early_stopping_rounds", 30),
            n_jobs=-1,
        )

        eval_set = [(X_tr_scaled, y_train.values)]
        if X_val_scaled is not None and y_val is not None:
            eval_set.append((X_val_scaled, y_val.values))

        self.model.fit(
            X_tr_scaled,
            y_train.values,
            eval_set=eval_set,
            verbose=False,
        )

        self._is_fitted = True
        self.training_meta = {
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "trained_at": datetime.utcnow().isoformat(),
            "best_iteration": int(getattr(self.model, "best_iteration", 0)),
        }
        logger.info("XGBoost climate model training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("XGBoost model must be fitted before predict()")

        # Ensure correct column ordering
        X_sub = X[self.feature_names]
        X_scaled = self.scaler.transform(X_sub)
        return self.model.predict(X_scaled)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """Predict prediction interval bounds using residual standard error estimate."""
        preds = self.predict(X)
        std_err = float(self.training_meta.get("val_rmse", 0.15))
        z = 1.645 if alpha == 0.10 else 1.96
        lower = preds - (z * std_err)
        upper = preds + (z * std_err)
        return lower, upper

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        trained_dir = settings.trained_models_dir
        metadata_dir = settings.metadata_dir
        trained_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        model_path = trained_dir / "xgb_model.json"
        scaler_path = trained_dir / "xgb_scaler.joblib"
        meta_path = metadata_dir / "xgb_metadata.json"

        if self.model is not None:
            self.model.save_model(str(model_path))
        joblib.dump(self.scaler, scaler_path)

        meta_data = self.get_metadata()
        meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
        logger.info("Saved XGBoost model artifacts to %s", trained_dir)
        return meta_path

    def load(self, model_dir: Path) -> "XGBoostClimateModel":
        trained_dir = settings.trained_models_dir
        metadata_dir = settings.metadata_dir

        model_path = trained_dir / "xgb_model.json"
        scaler_path = trained_dir / "xgb_scaler.joblib"
        meta_path = metadata_dir / "xgb_metadata.json"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"XGBoost model files missing in {trained_dir}")

        self.scaler = joblib.load(scaler_path)
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))

        if meta_path.exists():
            self.training_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.feature_names = self.training_meta.get("feature_names", [])

        self._is_fitted = True
        return self

    def get_metadata(self) -> Dict[str, Any]:
        return self.training_meta
