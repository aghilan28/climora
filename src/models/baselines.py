"""Mandatory baseline climate forecasting models: Seasonal Naive and Monthly Climatology."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.models.base import ClimateModel


class SeasonalNaiveModel(ClimateModel):
    """Seasonal Naive baseline forecasting model: predicts y(t) = y(t-12)."""

    def __init__(self) -> None:
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "SeasonalNaive"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "SeasonalNaiveModel":
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "anomaly_c_lag_12" in X.columns:
            return X["anomaly_c_lag_12"].fillna(0.0).values
        elif "anomaly_c_lag_1" in X.columns:
            return X["anomaly_c_lag_1"].fillna(0.0).values
        else:
            return np.zeros(len(X))

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        meta_file = model_dir / f"{self.name.lower()}_metadata.json"
        meta_file.write_text(json.dumps(self.get_metadata(), indent=2), encoding="utf-8")
        return meta_file

    def load(self, model_dir: Path) -> "SeasonalNaiveModel":
        self._is_fitted = True
        return self

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "baseline",
            "description": "Predicts next month anomaly equal to same month 1 year prior y(t-12)",
        }


class ClimatologyModel(ClimateModel):
    """Monthly Climatology baseline model: predicts historical mean for each calendar month."""

    def __init__(self) -> None:
        self.monthly_means: Dict[int, float] = {}
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "Climatology"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "ClimatologyModel":
        df_train = X_train.copy()
        df_train["target"] = y_train.values

        if "month" in df_train.columns:
            means = df_train.groupby("month")["target"].mean().to_dict()
            self.monthly_means = {int(k): float(v) for k, v in means.items()}
        else:
            overall_mean = float(y_train.mean())
            self.monthly_means = {m: overall_mean for m in range(1, 13)}

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("ClimatologyModel must be fitted before predict()")

        if "month" in X.columns:
            preds = [self.monthly_means.get(int(m), 0.0) for m in X["month"]]
            return np.array(preds, dtype=float)
        else:
            overall_mean = np.mean(list(self.monthly_means.values())) if self.monthly_means else 0.0
            return np.full(len(X), overall_mean, dtype=float)

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        meta_file = model_dir / f"{self.name.lower()}_metadata.json"
        data = self.get_metadata()
        meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return meta_file

    def load(self, model_dir: Path) -> "ClimatologyModel":
        meta_file = model_dir / f"{self.name.lower()}_metadata.json"
        if meta_file.exists():
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            self.monthly_means = {int(k): float(v) for k, v in data.get("monthly_means", {}).items()}
            self._is_fitted = True
        return self

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "baseline",
            "description": "Predicts historical mean temperature anomaly for each calendar month",
            "monthly_means": self.monthly_means,
        }
