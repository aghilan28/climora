"""Isolation Forest heat-extreme anomaly detection engine."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config.settings import settings
from src.utils.logging import logger


class ClimateAnomalyDetector:
    """Isolation Forest anomaly detector with sensitivity analysis across contamination levels."""

    def __init__(self, contamination: float = 0.05, seed: int = settings.seed) -> None:
        self.contamination = contamination
        self.seed = seed
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )

    def fit_predict(
        self, df: pd.DataFrame, feature_cols: List[str] | None = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_out = df.copy()
        numeric_df = df_out.select_dtypes(include=["number"])

        if feature_cols:
            valid_cols = [c for c in feature_cols if c in numeric_df.columns]
            feature_matrix = numeric_df[valid_cols].fillna(0.0)
        else:
            feature_matrix = numeric_df.fillna(0.0)

        # Fit Isolation Forest
        self.model.fit(feature_matrix)

        # Sklearn returns -1 for outliers and 1 for inliers
        raw_preds = self.model.predict(feature_matrix)
        is_anomaly = (raw_preds == -1).astype(int)

        # Decision function: lower values mean more anomalous
        raw_scores = self.model.decision_function(feature_matrix)
        # Normalize to 0-1 (1 = most anomalous, 0 = normal)
        score_min, score_max = raw_scores.min(), raw_scores.max()
        if score_max > score_min:
            norm_scores = 1.0 - ((raw_scores - score_min) / (score_max - score_min))
        else:
            norm_scores = np.zeros(len(df_out))

        df_out["is_anomaly"] = is_anomaly
        df_out["raw_anomaly_score"] = raw_scores
        df_out["anomaly_score"] = norm_scores
        df_out["anomaly_rank"] = df_out["anomaly_score"].rank(ascending=False, method="min").astype(int)

        total_obs = len(df_out)
        anomaly_count = int(is_anomaly.sum())
        anomaly_pct = float((anomaly_count / total_obs) * 100.0) if total_obs > 0 else 0.0

        summary = {
            "total_observations": total_obs,
            "normal_count": total_obs - anomaly_count,
            "anomaly_count": anomaly_count,
            "anomaly_pct": round(anomaly_pct, 2),
            "contamination": self.contamination,
        }

        logger.info(
            "Anomaly detection complete: %d anomalies detected (%.2f%% of %d records)",
            anomaly_count,
            anomaly_pct,
            total_obs,
        )
        return df_out, summary

    def compute_sensitivity_table(
        self, df: pd.DataFrame, feature_cols: List[str] | None = None
    ) -> pd.DataFrame:
        """Report sensitivity across contamination levels {0.01, 0.05, 0.10}."""
        levels = [0.01, 0.05, 0.10]
        results = []

        for c in levels:
            detector = ClimateAnomalyDetector(contamination=c, seed=self.seed)
            _, sum_dict = detector.fit_predict(df, feature_cols=feature_cols)
            results.append({
                "Contamination": c,
                "Normal Count": sum_dict["normal_count"],
                "Anomaly Count": sum_dict["anomaly_count"],
                "Anomaly %": sum_dict["anomaly_pct"],
            })

        return pd.DataFrame(results)
