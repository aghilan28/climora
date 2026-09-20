"""Manifest management module for logging raw dataset metadata, checksums, and row/column counts."""

import json
from pathlib import Path
from typing import Any, Dict, cast

from config.settings import settings
from src.utils.logging import logger

VALID_PROVENANCE = {"live-fetch", "cache", "pinned-fixture"}


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

    def record_provenance(
        self,
        dataset_name: str,
        *,
        source_url: str,
        sha256: str,
        bytes: int = 0,
        raw_byte_count: int = 0,
        rows: int = 0,
        download_date: str = "",
        snapshot_version: str = "",
        provenance: str = "live-fetch",
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Record provenance metadata entry in manifest."""
        if provenance not in VALID_PROVENANCE:
            raise ValueError(f"Invalid provenance '{provenance}'. Must be one of {VALID_PROVENANCE}")

        byte_val = bytes if bytes > 0 else raw_byte_count
        if not download_date:
            from datetime import datetime, timezone
            download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not snapshot_version:
            snapshot_version = sha256[:8] if sha256 else "v1"

        entry = {
            "dataset_name": dataset_name,
            "source_url": source_url,
            "sha256": sha256,
            "raw_byte_count": byte_val,
            "rows": rows,
            "download_date": download_date,
            "snapshot_version": snapshot_version,
            "provenance": provenance,
        }
        if extra:
            entry.update(extra)

        self.update_entry(dataset_name, entry)
        return entry
