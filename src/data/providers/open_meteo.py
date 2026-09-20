"""Open-Meteo ERA5 historical reanalysis dataset provider."""

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from config.settings import settings
from src.data.providers.base import BaseProvider
from src.utils.logging import logger


class OpenMeteoProvider(BaseProvider):
    """Provider for Open-Meteo ERA5 Reanalysis daily spatial panel dataset."""

    def __init__(self, cache_dir: Path) -> None:
        super().__init__(cache_dir)
        self.verified_daily_vars = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "cloud_cover_mean",
            "soil_temperature_0_to_7cm_mean",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ]

    @property
    def name(self) -> str:
        return "Open-Meteo ERA5 Reanalysis"

    @property
    def source_url(self) -> str:
        return settings.open_meteo_url

    def fetch_raw_location(
        self,
        station_name: str,
        lat: float,
        lon: float,
        start_date: str = "1940-01-01",
        end_date: str = "2025-12-31",
        offline: bool = False,
    ) -> bytes:
        """Fetch raw JSON for a single location."""
        safe_name = station_name.lower().replace(" ", "_")
        cache_file = self.cache_dir / f"openmeteo_{safe_name}.json"

        if cache_file.exists():
            logger.info("Reading Open-Meteo station '%s' from cache: %s", station_name, cache_file)
            self.last_provenance = "cache"
            return cache_file.read_bytes()

        if offline:
            raise FileNotFoundError(f"Offline mode enabled and cache missing: {cache_file}")

        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(self.verified_daily_vars),
            "timezone": "auto",
        }
        url = f"{self.source_url}?{urllib.parse.urlencode(params)}"
        logger.info("Fetching Open-Meteo station '%s' from network: %s", station_name, url)

        import time
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "CLIMORA-AI/1.0"})

        content = None
        for attempt in range(5):
            try:
                time.sleep(1.2 * (attempt + 1))
                with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                    content = resp.read()
                    self.last_provenance = "live-fetch"
                break
            except Exception as e:
                logger.warning("Open-Meteo fetch attempt %d failed for %s: %s", attempt + 1, station_name, e)
                if attempt == 4:
                    raise

        if content is None:
            raise RuntimeError(f"Failed to fetch Open-Meteo data for {station_name}")

        part_file = cache_file.with_suffix(".part")
        part_file.write_bytes(content)
        part_file.replace(cache_file)
        return bytes(content)

    def fetch_raw(self, offline: bool = False) -> bytes:
        """Default fetch using primary station (Chennai)."""
        return self.fetch_raw_location("Chennai", 13.0827, 80.2707, "2020-01-01", "2020-12-31", offline)

    def parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse raw JSON bytes (single object or list) into a daily DataFrame."""
        data: Union[Dict[str, Any], List[Dict[str, Any]]] = json.loads(raw_bytes.decode("utf-8"))

        if isinstance(data, list):
            # Multi-location shape
            frames = []
            for item in data:
                frames.append(self._parse_single_json_obj(item))
            return pd.concat(frames, ignore_index=True)
        else:
            # Single-location shape
            return self._parse_single_json_obj(data)

    def _parse_single_json_obj(self, obj: Dict[str, Any]) -> pd.DataFrame:
        if "daily" not in obj:
            raise ValueError(f"Open-Meteo response missing 'daily' section: {list(obj.keys())}")

        daily = obj["daily"]
        df = pd.DataFrame(daily)
        df["date"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])

        # Capture snapped coordinates
        df["requested_lat"] = obj.get("latitude")
        df["requested_lon"] = obj.get("longitude")
        df["elevation"] = obj.get("elevation")

        # Compute derived mean temperature if max/min exist
        if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
            df["temperature_2m_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0

        return df.sort_values("date").reset_index(drop=True)
