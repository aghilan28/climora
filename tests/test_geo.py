"""Gate tests for spatial geo package, choropleth layer, station network, and time scrubber."""

from pathlib import Path
import pytest

from src.geo.choropleth import get_country_polygons
from src.geo.station_network import get_indian_station_network


def test_station_network_no_literal_region_factors() -> None:
    """Ensure get_indian_station_network loads factors from computed artifact, not literal code table."""
    df_net = get_indian_station_network()
    assert "station" in df_net.columns
    assert "region_factor" in df_net.columns
    
    # Verify no float literal list is assigned directly in station_network.py source
    src_file = Path("src/geo/station_network.py")
    content = src_file.read_text(encoding="utf-8")
    assert '"region_factor":' not in content, "Literal region_factor found in station_network.py source code!"


def test_choropleth_features_count() -> None:
    """Ensure get_country_polygons loads Natural Earth 110m GeoJSON with 177 features."""
    geo_data = get_country_polygons()
    features = geo_data.get("features", [])
    assert len(features) == 177, f"Natural Earth GeoJSON expected 177 features, got {len(features)}"


def test_risk_page_contains_geojson_layer_and_scrubber() -> None:
    """Ensure dashboard/pages/risk.py contains GeoJsonLayer polygon choropleth and build_temporal_map_frames."""
    risk_page_file = Path("dashboard/pages/risk.py")
    content = risk_page_file.read_text(encoding="utf-8")
    assert "GeoJsonLayer" in content or "PolygonLayer" in content, "Risk page missing polygon choropleth layer!"
    assert "build_temporal_map_frames" in content, "Risk page missing temporal map scrubber logic!"


def test_temporal_map_frames_non_empty() -> None:
    """Ensure build_temporal_map_frames produces valid non-empty historical snapshot frames."""
    import pandas as pd
    import numpy as np
    from src.geo.frames import build_temporal_map_frames

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24, freq="ME"),
        "anomaly_c": np.random.randn(24),
    })
    frames = build_temporal_map_frames(df, sample_interval=6)
    assert len(frames) == 4
    assert len(frames[0]["stations"]) >= 12
