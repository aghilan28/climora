"""Base provider protocol and abstract class for dataset ingestion."""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class BaseProvider(ABC):
    """Abstract base class for all CLIMORA AI dataset providers."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset provider."""
        pass

    @property
    @abstractmethod
    def source_url(self) -> str:
        """Primary source URL for the dataset."""
        pass

    @abstractmethod
    def fetch_raw(self, offline: bool = False) -> bytes:
        """Fetch raw content from network or cache."""
        pass

    @abstractmethod
    def parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse raw bytes into a standardized DataFrame."""
        pass

    def compute_sha256(self, content: bytes) -> str:
        """Compute SHA256 checksum of raw bytes."""
        return hashlib.sha256(content).hexdigest()

    def get_manifest_entry(
        self, raw_bytes: bytes, df: pd.DataFrame, extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate standardized metadata manifest entry."""
        entry: Dict[str, Any] = {
            "dataset_name": self.name,
            "source_url": self.source_url,
            "sha256": self.compute_sha256(raw_bytes),
            "raw_byte_count": len(raw_bytes),
            "rows": len(df),
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
        }
        if extra:
            entry.update(extra)
        return entry
