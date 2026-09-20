"""NOAA GML Mauna Loa CO2 dataset provider."""

import io
import ssl
import urllib.request

import numpy as np
import pandas as pd

from config.settings import settings
from src.data.providers.base import BaseProvider
from src.utils.logging import logger


class NOAACO2Provider(BaseProvider):
    """Provider for NOAA GML Mauna Loa monthly atmospheric CO2 trend dataset."""

    @property
    def name(self) -> str:
        return "NOAA GML Mauna Loa CO2"

    @property
    def source_url(self) -> str:
        return settings.noaa_co2_url

    def fetch_raw(self, offline: bool = False) -> bytes:
        cache_file = self.cache_dir / "noaa_co2_raw.csv"
        if cache_file.exists():
            logger.info("Reading NOAA CO2 from cache: %s", cache_file)
            return cache_file.read_bytes()

        if offline:
            raise FileNotFoundError(f"Offline mode enabled and cache missing: {cache_file}")

        logger.info("Fetching NOAA CO2 from network: %s", self.source_url)
        ctx = ssl.create_default_context()
        req = urllib.request.Request(self.source_url, headers={"User-Agent": "CLIMORA-AI/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            content = resp.read()

        part_file = cache_file.with_suffix(".part")
        part_file.write_bytes(content)
        part_file.replace(cache_file)
        return bytes(content)

    def parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse raw NOAA CO2 CSV bytes, mapping sentinel values (-9.99, -1) to NaN."""
        text = raw_bytes.decode("utf-8")
        df = pd.read_csv(
            io.StringIO(text),
            comment="#",
            skipinitialspace=True,
        )

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Sentinel mapping
        if "ndays" in df.columns:
            df.loc[df["ndays"] < 0, ["average", "deseasonalized", "sdev", "unc"]] = np.nan
            df["ndays"] = df["ndays"].apply(lambda x: np.nan if x < 0 else x)

        for col in ["average", "deseasonalized", "sdev", "unc"]:
            if col in df.columns:
                df[col] = df[col].replace([-9.99, -99.99, -999.9], np.nan)

        # Construct datetime column
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        df["co2_ppm"] = df["average"]
        df["co2_deseasonalized_ppm"] = df["deseasonalized"]

        df = df.sort_values("date").reset_index(drop=True)
        return df
