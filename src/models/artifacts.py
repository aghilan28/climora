"""Model artifact persistence and validation system."""

import json
from pathlib import Path
from typing import Any, Dict, cast

from config.settings import settings
from src.utils.logging import logger


class ArtifactManager:
    """Manages trained model artifacts, metrics JSON, and metadata persistence."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or settings.models_dir
        self.trained_dir = self.base_dir / "trained"
        self.metrics_dir = self.base_dir / "metrics"
        self.metadata_dir = self.base_dir / "metadata"

        for d in [self.trained_dir, self.metrics_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, model_name: str, metrics: Dict[str, float]) -> Path:
        """Save model evaluation metrics to models/metrics/{model_name}_metrics.json."""
        metrics_file = self.metrics_dir / f"{model_name.lower()}_metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Saved metrics for %s to %s", model_name, metrics_file)
        return metrics_file

    def load_metrics(self, model_name: str) -> Dict[str, float]:
        """Load model evaluation metrics from disk."""
        metrics_file = self.metrics_dir / f"{model_name.lower()}_metrics.json"
        if not metrics_file.exists():
            return {}
        res = json.loads(metrics_file.read_text(encoding="utf-8"))
        return cast(Dict[str, float], res)

    def save_metadata(self, model_name: str, metadata: Dict[str, Any]) -> Path:
        """Save metadata entry to models/metadata/{model_name}_metadata.json."""
        meta_file = self.metadata_dir / f"{model_name.lower()}_metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved metadata for %s to %s", model_name, meta_file)
        return meta_file

    def load_metadata(self, model_name: str) -> Dict[str, Any]:
        """Load model metadata JSON from disk."""
        meta_file = self.metadata_dir / f"{model_name.lower()}_metadata.json"
        if not meta_file.exists():
            return {}
        res = json.loads(meta_file.read_text(encoding="utf-8"))
        return cast(Dict[str, Any], res)
