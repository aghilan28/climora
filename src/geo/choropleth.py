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

        # Deterministic boundary styling based on Natural Earth geography metadata
        continent_colors = {
            "Asia": [239, 68, 68, 100],
            "Africa": [245, 158, 11, 100],
            "Europe": [59, 130, 246, 100],
            "North America": [16, 185, 129, 100],
            "South America": [139, 92, 246, 100],
            "Oceania": [236, 72, 153, 100],
            "Antarctica": [148, 163, 184, 100],
        }
        for feat in features:
            props = feat.get("properties", {})
            continent = str(props.get("CONTINENT", "Unknown"))
            props["fill_color"] = continent_colors.get(continent, [100, 116, 139, 100])
            feat["properties"] = props

        return data
    except Exception as e:
        logger.error("Failed to parse Natural Earth GeoJSON: %s", e)
        raise
