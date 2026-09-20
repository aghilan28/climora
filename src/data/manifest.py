"""Manifest management module for logging raw dataset metadata, checksums, and row/column counts."""

import json
from pathlib import Path
from typing import Any, Dict, cast

from config.settings import settings
from src.utils.logging import logger


class ManifestManager:
    """Manager for data/manifest.json metadata state."""

    def __init__(self, manifest_path: Path | None = None) -> None:
        self.manifest_path = manifest_path or (settings.data_dir / "manifest.json")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """Load manifest JSON or return empty structure if absent."""
        if not self.manifest_path.exists():
            return {"datasets": {}}
        try:
            res = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return cast(Dict[str, Any], res)
        except Exception as e:
            logger.warning("Error reading manifest file %s: %s. Returning empty.", self.manifest_path, e)
            return {"datasets": {}}

    def save(self, data: Dict[str, Any]) -> None:
        """Atomically write manifest JSON."""
        temp_path = self.manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(self.manifest_path)
        logger.info("Manifest updated at %s", self.manifest_path)

    def update_entry(self, dataset_name: str, entry: Dict[str, Any]) -> None:
        """Update or insert a single dataset entry in manifest."""
        manifest = self.load()
        manifest["datasets"][dataset_name] = entry
        self.save(manifest)
