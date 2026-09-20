"""GeoJSON boundaries data provider (Natural Earth country and state polygons)."""

import json
import ssl
import urllib.request
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config.settings import settings
from src.data.providers.base import BaseProvider
from src.utils.logging import logger


class GeoJSONProvider(BaseProvider):
    """Provider for Natural Earth boundary GeoJSON files."""

    def __init__(self, cache_dir: Path | None = None, scale: str = "110m") -> None:
        c_dir = cache_dir or (settings.raw_data_dir / "geo")
        super().__init__(c_dir)
        self.scale = scale

    @property
    def name(self) -> str:
        return "Natural Earth GeoJSON"

    @property
    def source_url(self) -> str:
        if self.scale == "50m":
            return settings.natural_earth_50m_url
        return settings.natural_earth_110m_url

    def fetch_raw(self, offline: bool = False) -> bytes:
        cache_file = self.cache_dir / f"natural_earth_{self.scale}.geojson"
        if cache_file.exists():
            logger.info("Reading GeoJSON %s from cache: %s", self.scale, cache_file)
            self.last_provenance = "cache"
            return cache_file.read_bytes()

        if offline:
            raise FileNotFoundError(f"Offline mode enabled and cache missing: {cache_file}")

        logger.info("Fetching GeoJSON %s from network: %s", self.scale, self.source_url)
        ctx = ssl.create_default_context()
        req = urllib.request.Request(self.source_url, headers={"User-Agent": "CLIMORA-AI/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            content = resp.read()
            self.last_provenance = "live-fetch"

        cache_file.write_bytes(content)
        return content

    def parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse GeoJSON bytes into a attribute metadata DataFrame."""
        geojson_obj: Dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))
        features = geojson_obj.get("features", [])

        records = []
        for feat in features:
            props = feat.get("properties", {})
            iso_a3 = props.get("ISO_A3", props.get("iso_a3", "-99"))
            if iso_a3 == "-99":
                iso_a3 = props.get("ADM0_A3", props.get("adm0_a3", "-99"))

            name = props.get("NAME", props.get("name", "Unknown"))
            geom = feat.get("geometry") or {}
            records.append({
                "iso_a3": iso_a3,
                "name": name,
                "type": props.get("TYPE", props.get("type", "Country")),
                "geometry_type": geom.get("type", "Unknown"),
            })

        return pd.DataFrame(records)
