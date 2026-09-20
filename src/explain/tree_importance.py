"""Tree model feature importance computation (gain, weight, cover)."""

import pandas as pd

from src.models.base import ClimateModel
from src.utils.logging import logger


def get_tree_feature_importance(model_obj: ClimateModel) -> pd.DataFrame:
    """Extract feature importance from tree model (XGBoost / LightGBM)."""
    inner_model = getattr(model_obj, "model", None)
    feat_names = getattr(model_obj, "feature_names", [])

    if inner_model is None or not hasattr(inner_model, "feature_importances_"):
        logger.warning("Model %s does not support feature_importances_", getattr(model_obj, "name", "Unknown"))
        return pd.DataFrame(columns=["feature", "importance"])

    importances = inner_model.feature_importances_
    if len(feat_names) != len(importances):
        feat_names = [f"feature_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp
