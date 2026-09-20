"""ClimateModel abstract base class protocol."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class ClimateModel(ABC):
    """Unified interface for all CLIMORA AI time-series models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name of the model."""
        pass

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "ClimateModel":
        """Fit model on training data slice with optional validation slice."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values for feature matrix X."""
        pass

    @abstractmethod
    def save(self, model_dir: Path) -> Path:
        """Save model weights and metadata artifacts to disk."""
        pass

    @abstractmethod
    def load(self, model_dir: Path) -> "ClimateModel":
        """Load model weights and metadata artifacts from disk."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata summary."""
        pass
