"""SHAP TreeExplainer wrapper for signed per-prediction contributions."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import ClimateModel
from src.utils.logging import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_shap_explanation(
    model_obj: ClimateModel, X_sample: pd.DataFrame
) -> Tuple[Optional[np.ndarray], Optional[float], List[str]]:
    """Compute SHAP values for a given sample frame.

    Returns:
    (shap_values_array, base_value, feature_names)
    or (None, None, []) if SHAP is unavailable or unsupported.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP package not available. Skipping SHAP explanation.")
        return None, None, []

    inner_model = getattr(model_obj, "model", None)
    feat_names = getattr(model_obj, "feature_names", list(X_sample.columns))

    if inner_model is None:
        return None, None, []

    try:
        scaler = getattr(model_obj, "scaler", None)
        X_eval = scaler.transform(X_sample[feat_names]) if scaler is not None else X_sample[feat_names].values

        explainer = shap.TreeExplainer(inner_model)
        shap_vals = explainer.shap_values(X_eval)

        base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0
        return np.asarray(shap_vals, dtype=float), base_val, feat_names
    except Exception as e:
        logger.warning("SHAP TreeExplainer computation failed: %s", e)
        return None, None, []
