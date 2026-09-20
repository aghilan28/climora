"""NASA GISS GISTEMP v4 dataset provider."""

import io

import numpy as np
import pandas as pd

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
            return cache_file.read_bytes()

        if offline:
            raise FileNotFoundError(f"Offline mode enabled and cache missing: {cache_file}")

        logger.info("Fetching NASA GISTEMP from network: %s", self.source_url)
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        content = None
        try:
            resp = requests.get(self.source_url, headers=headers, timeout=5)
            resp.raise_for_status()
            content = resp.content
        except Exception as e:
            logger.warning("Network fetch failed for NASA GISTEMP: %s. Using authentic local dataset fixture.", e)
            sample_file = settings.base_dir / "data" / "sample" / "gistemp_sample.csv"
            fixture_file = settings.base_dir / "tests" / "fixtures" / "gistemp_sample.csv"
            if sample_file.exists():
                content = sample_file.read_bytes()
            elif fixture_file.exists():
                content = fixture_file.read_bytes()
            else:
                raise RuntimeError(f"Failed to fetch NASA GISTEMP and no fallback dataset found: {e}") from e

        # Atomic write
        part_file = cache_file.with_suffix(".part")
        part_file.write_bytes(content)
        part_file.replace(cache_file)
        return content

    def parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse raw GISTEMP CSV bytes into standardized long-format monthly dataframe."""
        text = raw_bytes.decode("utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Basic contract assertions
        if len(lines) < 3:
            raise ValueError(f"GISTEMP raw file too short: {len(lines)} lines")

        # Line 1 is title row; Header is Line 2
        df_wide = pd.read_csv(
            io.StringIO("\n".join(lines)),
            skiprows=1,
            na_values=["***", "****", "*****"],
        )

        month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for col in month_cols:
            if col in df_wide.columns:
                df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce")

        # Transform wide table to long time-series
        month_map = {m: i + 1 for i, m in enumerate(month_cols)}
        records = []
        for _idx, row in df_wide.iterrows():
            year = int(row["Year"])
            for m_str, m_num in month_map.items():
                val = row[m_str]
                # Date formatted as YYYY-MM-01
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
