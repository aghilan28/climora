"""Unit tests for dataset providers and loaders using committed offline fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.providers.ersst_nino import ERSSTNinoProvider
from src.data.providers.nasa_giss import NasaGissProvider
from src.data.providers.noaa_gml import NOAACO2Provider
from src.data.providers.open_meteo import OpenMeteoProvider


def test_nasa_giss_provider_parsing(tmp_path: Path, gistemp_fixture_bytes: bytes) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    assert not df.empty
    assert "date" in df.columns
    assert "anomaly_c" in df.columns
    assert "year" in df.columns
    assert "month" in df.columns

    # Verify data types
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["anomaly_c"])

    # Verify decimal parsing (-.19 -> -0.19)
    first_row = df.iloc[0]
    assert first_row["year"] == 1880
    assert first_row["month"] == 1
    assert abs(first_row["anomaly_c"] - (-0.19)) < 1e-5

    # Verify missing value sentinel conversion
    row_2026_sep = df[(df["year"] == 2026) & (df["month"] == 9)]
    assert len(row_2026_sep) == 1
    assert np.isnan(row_2026_sep["anomaly_c"].values[0])


def test_noaa_co2_provider_parsing(tmp_path: Path, noaa_co2_fixture_bytes: bytes) -> None:
    provider = NOAACO2Provider(tmp_path)
    df = provider.parse(noaa_co2_fixture_bytes)

    assert not df.empty
    assert "date" in df.columns
    assert "co2_ppm" in df.columns

    # Verify sentinel conversion (-1 ndays -> NaN for average/deseasonalized)
    row_1958 = df[df["year"] == 1958].iloc[0]
    assert np.isnan(row_1958["co2_ppm"])
    assert np.isnan(row_1958["co2_deseasonalized_ppm"])

    # Verify valid values
    row_2025 = df[df["year"] == 2025].iloc[0]
    assert abs(row_2025["co2_ppm"] - 425.10) < 1e-4


def test_open_meteo_provider_parsing(tmp_path: Path, openmeteo_fixture_bytes: bytes) -> None:
    provider = OpenMeteoProvider(tmp_path)
    df = provider.parse(openmeteo_fixture_bytes)

    assert len(df) == 3
    assert "date" in df.columns
    assert "temperature_2m_max" in df.columns
    assert "temperature_2m_min" in df.columns
    assert "temperature_2m_mean" in df.columns
    assert "precipitation_sum" in df.columns

    assert abs(df.iloc[0]["temperature_2m_mean"] - ((29.5 + 21.2) / 2.0)) < 1e-5


def test_ersst_nino_provider_parsing(tmp_path: Path, ersst_nino_fixture_bytes: bytes) -> None:
    provider = ERSSTNinoProvider(tmp_path)
    df = provider.parse(ersst_nino_fixture_bytes)

    assert not df.empty
    assert "date" in df.columns
    assert "nino34_anom" in df.columns

    first_row = df.iloc[0]
    assert first_row["year"] == 1950
    assert first_row["month"] == 1
    assert abs(first_row["nino34_anom"] - (-0.35)) < 1e-5
