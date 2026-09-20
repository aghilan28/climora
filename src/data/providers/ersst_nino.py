"""CPC ERSSTv5 Nino indices dataset provider."""

import io
import ssl
import urllib.request

import pandas as pd

from config.settings import settings
from src.data.providers.base import BaseProvider
from src.utils.logging import logger


class ERSSTNinoProvider(BaseProvider):
    """Provider for NOAA CPC ERSSTv5 El Nino SST Anomaly indices dataset."""

    @property
    def name(self) -> str:
        return "CPC ERSSTv5 Nino Indices"

    @property
    def source_url(self) -> str:
        return settings.ersst_nino_url

    def fetch_raw(self, offline: bool = False) -> bytes:
        cache_file = self.cache_dir / "ersst_nino_raw.ascii"
        if cache_file.exists():
            logger.info("Reading CPC ERSST Nino from cache: %s", cache_file)
            self.last_provenance = "cache"
            return cache_file.read_bytes()

        if offline:
            raise FileNotFoundError(f"Offline mode enabled and cache missing: {cache_file}")

        logger.info("Fetching CPC ERSST Nino from network: %s", self.source_url)
        ctx = ssl.create_default_context()
        req = urllib.request.Request(self.source_url, headers={"User-Agent": "CLIMORA-AI/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            content = resp.read()
            self.last_provenance = "live-fetch"

        part_file = cache_file.with_suffix(".part")
        part_file.write_bytes(content)
        part_file.replace(cache_file)
        return bytes(content)

    def parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse fixed-width ERSST Nino ASCII dataset."""
        text = raw_bytes.decode("utf-8")
        df = pd.read_csv(
            io.StringIO(text),
            sep=r"\s+",
            engine="python",
        )

        # Standardize column headers: YR, MON, NINO1+2, ANOM, NINO3, ANOM, NINO4, ANOM, NINO3.4, ANOM
        cols = list(df.columns)
        new_cols = [
            "year", "month",
            "nino12_sst", "nino12_anom",
            "nino3_sst", "nino3_anom",
            "nino4_sst", "nino4_anom",
            "nino34_sst", "nino34_anom"
        ]
        if len(cols) == len(new_cols):
            df.columns = new_cols

        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2) + "-01"
        )
        df = df.sort_values("date").reset_index(drop=True)
        return df
