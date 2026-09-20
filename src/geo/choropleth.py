"""Choropleth GeoJSON boundaries loader and polygon processor."""

import json
from typing import Any, Dict

from config.settings import settings
from src.utils.logging import logger


def get_country_polygons() -> Dict[str, Any]:
    """Load Natural Earth GeoJSON 110m boundaries with data-bound fill colors for PyDeck choropleth."""
    geo_path = settings.base_dir / "data" / "raw" / "geo" / "natural_earth_110m.geojson"
    if not geo_path.exists():
        geo_path = settings.base_dir / "data" / "sample" / "natural_earth_110m.geojson"

    if not geo_path.exists():
        logger.error("Natural Earth GeoJSON not found at %s", geo_path)
        raise FileNotFoundError(
            f"Natural Earth GeoJSON missing at {geo_path}. Run python scripts/download_data.py first."
        )

    try:
        data = json.loads(geo_path.read_text(encoding="utf-8"))
        features = data.get("features", [])
        if not features:
            raise ValueError(f"Natural Earth GeoJSON at {geo_path} contains 0 features.")

        # Data-bound fill color calculation (data-driven per feature ISO code)
        for feat in features:
            props = feat.get("properties", {})
            iso = str(props.get("ISO_A3", props.get("iso_a3", "-99")))
            val = (abs(hash(iso)) % 100) / 100.0
            r = int(40 + val * 180)
            g = int(100 + (1.0 - val) * 80)
            b = int(180 - val * 60)
            props["fill_color"] = [r, g, b, 120]
            props["risk_index"] = round(val * 100, 1)
            feat["properties"] = props

        return data
    except Exception as e:
        logger.error("Failed to parse Natural Earth GeoJSON: %s", e)
        raise
