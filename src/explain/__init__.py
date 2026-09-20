"""CLIMORA AI Explainability package."""

from src.explain.lstm_sensitivity import compute_lstm_feature_sensitivity
from src.explain.narrative import build_explanation_narrative
from src.explain.shap_wrapper import compute_shap_explanation
from src.explain.tree_importance import get_tree_feature_importance

__all__ = [
    "build_explanation_narrative",
    "compute_shap_explanation",
    "get_tree_feature_importance",
    "compute_lstm_feature_sensitivity",
]
