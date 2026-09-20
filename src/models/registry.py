"""Model registry lookup system."""

from typing import Dict, Type

from src.models.base import ClimateModel
from src.models.baselines import ClimatologyModel, SeasonalNaiveModel
from src.models.lightgbm_model import LightGBMClimateModel
from src.models.lstm_model import PyTorchLSTMClimateModel
from src.models.xgboost_model import XGBoostClimateModel

MODEL_REGISTRY: Dict[str, Type[ClimateModel]] = {
    "seasonal_naive": SeasonalNaiveModel,
    "climatology": ClimatologyModel,
    "xgboost": XGBoostClimateModel,
    "lightgbm": LightGBMClimateModel,
    "lstm": PyTorchLSTMClimateModel,
}


def get_model(name: str) -> ClimateModel:
    """Instantiate a model instance by registered name."""
    lower_name = name.lower()
    if lower_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Registered models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[lower_name]()
