"""Input perturbation sensitivity analysis for PyTorch LSTM models."""

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.models.base import ClimateModel
from src.utils.logging import logger


def compute_lstm_feature_sensitivity(
    model_obj: ClimateModel, X_sample: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    """Compute feature sensitivity vector for LSTM model via input perturbation.

    Returns (sensitivity_array, feature_names).
    """
    feat_names = getattr(model_obj, "feature_names", list(X_sample.columns))
    if not feat_names:
        return np.array([]), []

    try:
        baseline_pred = float(model_obj.predict(X_sample)[0])
        sensitivities = []

        for col in feat_names:
            if col not in X_sample.columns:
                sensitivities.append(0.0)
                continue

            X_perturbed = X_sample.copy()
            val = float(X_perturbed[col].iloc[0]) if not X_perturbed[col].isna().all() else 0.0
            delta = abs(val * 0.1) if val != 0 else 0.1
            X_perturbed[col] = val + delta

            pred_perturbed = float(model_obj.predict(X_perturbed)[0])
            diff = pred_perturbed - baseline_pred
            sensitivities.append(diff)

        return np.array(sensitivities, dtype=float), feat_names
    except Exception as e:
        logger.warning("LSTM sensitivity calculation failed: %s", e)
        return np.array([]), feat_names
