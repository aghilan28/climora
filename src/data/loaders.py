"""High-level dataset loaders with caching and validation support."""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from config.settings import settings
from src.data.providers.ersst_nino import ERSSTNinoProvider
from src.data.providers.geojson import GeoJSONProvider
from src.data.providers.nasa_giss import NasaGissProvider
from src.data.providers.noaa_gml import NOAACO2Provider
from src.data.providers.open_meteo import OpenMeteoProvider


def load_gistemp_data(
    raw_dir: Path | None = None, offline: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load NASA GISTEMP v4 monthly global anomaly DataFrame and raw manifest entry."""
    dir_path = raw_dir or (settings.raw_data_dir / "gistemp")
    provider = NasaGissProvider(dir_path)
    raw_bytes = provider.fetch_raw(offline=offline)
    df = provider.parse(raw_bytes)
    entry = provider.get_manifest_entry(raw_bytes, df)
    return df, entry


def load_noaa_co2_data(
    raw_dir: Path | None = None, offline: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load NOAA GML Mauna Loa CO2 DataFrame and raw manifest entry."""
    dir_path = raw_dir or (settings.raw_data_dir / "noaa")
    provider = NOAACO2Provider(dir_path)
    raw_bytes = provider.fetch_raw(offline=offline)
    df = provider.parse(raw_bytes)
    entry = provider.get_manifest_entry(raw_bytes, df)
    return df, entry


def load_ersst_nino_data(
    raw_dir: Path | None = None, offline: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load CPC ERSSTv5 Nino Indices DataFrame and raw manifest entry."""
    dir_path = raw_dir or (settings.raw_data_dir / "noaa")
    provider = ERSSTNinoProvider(dir_path)
    raw_bytes = provider.fetch_raw(offline=offline)
    df = provider.parse(raw_bytes)
    entry = provider.get_manifest_entry(raw_bytes, df)
    return df, entry


def load_open_meteo_station(
    station_name: str,
    lat: float,
    lon: float,
    start_date: str = "1940-01-01",
    end_date: str = "2025-12-31",
    raw_dir: Path | None = None,
    offline: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load Open-Meteo ERA5 daily data for a single station."""
    dir_path = raw_dir or (settings.raw_data_dir / "openmeteo")
    provider = OpenMeteoProvider(dir_path)
    raw_bytes = provider.fetch_raw_location(
        station_name, lat, lon, start_date, end_date, offline=offline
    )
    df = provider.parse(raw_bytes)
    df["station_name"] = station_name
    entry = provider.get_manifest_entry(raw_bytes, df, extra={"station_name": station_name})
    return df, entry


def load_geojson_boundaries(
    scale: str = "110m", raw_dir: Path | None = None, offline: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load GeoJSON boundary metadata DataFrame."""
    dir_path = raw_dir or (settings.raw_data_dir / "geo")
    provider = GeoJSONProvider(dir_path, scale=scale)
    raw_bytes = provider.fetch_raw(offline=offline)
    df = provider.parse(raw_bytes)
    entry = provider.get_manifest_entry(raw_bytes, df)
    return df, entry
