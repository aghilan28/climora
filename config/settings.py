"""Application configuration settings managed via pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for CLIMORA AI."""

    model_config = SettingsConfigDict(
        env_prefix="CLIMORA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "production", "testing"] = "development"
    log_level: str = "INFO"
    seed: int = 42

    # Directories
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    sample_data_dir: Path = data_dir / "sample"
    models_dir: Path = base_dir / "models"
    trained_models_dir: Path = models_dir / "trained"
    metrics_dir: Path = models_dir / "metrics"
    metadata_dir: Path = models_dir / "metadata"

    # Dataset URLs
    gistemp_url: str = (
        "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    )
    open_meteo_url: str = "https://archive-api.open-meteo.com/v1/archive"
    noaa_co2_url: str = (
        "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    )
    owid_co2_url: str = (
        "https://ourworldindata.org/grapher/annual-co2-emissions-per-country.csv?v=1&csvType=full&columnKey=all"
    )
    natural_earth_110m_url: str = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    )
    natural_earth_50m_url: str = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson"
    )
    ersst_nino_url: str = (
        "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
    )
    ghcn_stations_url: str = (
        "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    )

    # Baselines & Offsets
    pre_industrial_offset_c: float = 0.25  # GISTEMP 1951-1980 baseline offset to 1850-1900 baseline


settings = Settings()
