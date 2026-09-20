"""LightGBM model implementation adhering to ClimateModel protocol."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import settings
from src.models.base import ClimateModel
from src.utils.logging import logger


class LightGBMClimateModel(ClimateModel):
    """LightGBM Gradient Boosting Climate Forecasting Model."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": settings.seed,
            "verbose": -1,
        }
        self.scaler = StandardScaler()
        self.model: Any = None
        self.feature_names: List[str] = []
        self._is_fitted = False
        self.training_meta: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LightGBMClimateModel":
        self.feature_names = list(X_train.columns)

        # Fit scaler ONLY on train slice to prevent leakage (fillna 0.0 prevents NaNs in scaling)
        X_tr = X_train.fillna(0.0)
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        if hasattr(self.scaler, "scale_") and self.scaler.scale_ is not None:
            self.scaler.scale_[self.scaler.scale_ == 0.0] = 1.0

        X_val_scaled = self.scaler.transform(X_val.fillna(0.0)) if X_val is not None else None

        self.model = lgb.LGBMRegressor(
            n_estimators=self.params.get("n_estimators", 300),
            learning_rate=self.params.get("learning_rate", 0.03),
            max_depth=self.params.get("max_depth", 5),
            num_leaves=self.params.get("num_leaves", 31),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            random_state=self.params.get("random_state", settings.seed),
            n_jobs=-1,
            verbose=-1,
        )

        callbacks = []
        if X_val_scaled is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

        eval_set = [(X_tr_scaled, y_train.values)]
        if X_val_scaled is not None and y_val is not None:
            eval_set.append((X_val_scaled, y_val.values))

        self.model.fit(
            X_tr_scaled,
            y_train.values,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        self._is_fitted = True
        self.training_meta = {
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_iteration": int(getattr(self.model, "best_iteration_", 0)),
        }
        logger.info("LightGBM climate model training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("LightGBM model must be fitted before predict()")

        X_sub = X[self.feature_names].fillna(0.0)
        X_scaled = self.scaler.transform(X_sub)
        if hasattr(self.model, "predict"):
            try:
                return np.asarray(self.model.predict(X_scaled), dtype=float)
            except Exception:
                if hasattr(self.model, "booster_"):
                    return np.asarray(self.model.booster_.predict(X_scaled), dtype=float)
        return np.asarray(self.model.predict(X_scaled), dtype=float)

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

        model_path = trained_dir / "lgbm_model.txt"
        scaler_path = trained_dir / "lgbm_scaler.joblib"
        meta_path = metadata_dir / "lgbm_metadata.json"

        if self.model is not None and hasattr(self.model, "booster_"):
            self.model.booster_.save_model(str(model_path))
        joblib.dump(self.scaler, scaler_path)

        meta_data = self.get_metadata()
        meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
        logger.info("Saved LightGBM model artifacts to %s", trained_dir)
        return meta_path

    def load(self, model_dir: Path) -> "LightGBMClimateModel":
        trained_dir = settings.trained_models_dir
        metadata_dir = settings.metadata_dir

        model_path = trained_dir / "lgbm_model.txt"
        scaler_path = trained_dir / "lgbm_scaler.joblib"
        meta_path = metadata_dir / "lgbm_metadata.json"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"LightGBM model files missing in {trained_dir}")

        self.scaler = joblib.load(scaler_path)
        booster = lgb.Booster(model_file=str(model_path))
        self.model = booster

        if meta_path.exists():
            self.training_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.feature_names = self.training_meta.get("feature_names", [])

        self._is_fitted = True
        return self

    def get_metadata(self) -> Dict[str, Any]:
        return self.training_meta
