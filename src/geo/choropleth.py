"""Choropleth GeoJSON boundaries loader and polygon processor."""

import json
from typing import Any, Dict

from config.settings import settings
from src.utils.logging import logger


def get_country_polygons() -> Dict[str, Any]:
    """Load Natural Earth GeoJSON 110m boundaries for PyDeck rendering."""
    geo_path = settings.base_dir / "data" / "raw" / "geo" / "natural_earth_110m.geojson"
    if not geo_path.exists():
        geo_path = settings.base_dir / "data" / "sample" / "natural_earth_110m.geojson"

    if not geo_path.exists():
        logger.warning("Natural Earth GeoJSON not found at %s", geo_path)
        return {"type": "FeatureCollection", "features": []}

    try:
        data = json.loads(geo_path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        logger.error("Failed to parse Natural Earth GeoJSON: %s", e)
        return {"type": "FeatureCollection", "features": []}
