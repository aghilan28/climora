"""NASA GISS GISTEMP v4 dataset provider."""

import io
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings
from src.data.providers.base import BaseProvider
from src.utils.logging import logger


class NasaGissProvider(BaseProvider):
    """Provider for NASA GISS GISTEMP v4 Global Surface Temperature Anomaly dataset."""

    @property
    def name(self) -> str:
        return "NASA GISS GISTEMP v4"

    @property
    def source_url(self) -> str:
        return settings.gistemp_url

    def fetch_raw(self, offline: bool = False) -> bytes:
        cache_file = self.cache_dir / "gistemp_raw.csv"
        if cache_file.exists():
            logger.info("Reading NASA GISTEMP from cache: %s", cache_file)
            self.last_provenance = "cache"
            return cache_file.read_bytes()

        if offline:
            raise FileNotFoundError(f"Offline mode enabled and cache missing: {cache_file}")

        logger.info("Fetching NASA GISTEMP from network: %s", self.source_url)
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.mount("http://", HTTPAdapter(max_retries=retries))

        try:
            resp = session.get(self.source_url, headers={"User-Agent": "CLIMORA-AI/1.0"}, timeout=(10, 60))
            resp.raise_for_status()
            content = resp.content
            self.last_provenance = "live-fetch"
        except Exception as e:
            logger.error("Network fetch failed for NASA GISTEMP: %s", e)
            raise RuntimeError(f"Data not available for NASA GISTEMP. Run: python scripts/download_data.py ({e})") from e

        # Atomic write to cache on successful fetch ONLY
        part_file = cache_file.with_suffix(".part")
        part_file.write_bytes(content)
        part_file.replace(cache_file)
        return content

    def parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse raw GISTEMP CSV bytes into standardized long-format monthly dataframe."""
        text = raw_bytes.decode("utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if len(lines) < 3:
            raise ValueError(f"GISTEMP raw file too short: {len(lines)} lines")

        df_wide = pd.read_csv(
            io.StringIO("\n".join(lines)),
            skiprows=1,
            na_values=["***", "****", "*****"],
        )

        month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for col in month_cols:
            if col in df_wide.columns:
                df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce")

        month_map = {m: i + 1 for i, m in enumerate(month_cols)}
        records = []
        for _idx, row in df_wide.iterrows():
            year = int(row["Year"])
            for m_str, m_num in month_map.items():
                val = row[m_str]
                date_str = f"{year:04d}-{m_num:02d}-01"
                records.append({
                    "date": pd.to_datetime(date_str),
                    "year": year,
                    "month": m_num,
                    "anomaly_c": float(val) if not pd.isna(val) else np.nan,
                })

        df_long = pd.DataFrame(records)
        df_long = df_long.sort_values("date").reset_index(drop=True)
        return df_long
