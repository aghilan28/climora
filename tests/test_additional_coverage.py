"""Unit tests expanding coverage for metrics, manifest, risk context, logging, baselines, artifacts, providers, loaders, geo, and explainability."""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import (
    load_ersst_nino_data,
    load_geojson_boundaries,
    load_gistemp_data,
    load_noaa_co2_data,
    load_open_meteo_station,
)
from src.data.manifest import ManifestManager
from src.data.providers.ersst_nino import ERSSTNinoProvider
from src.data.providers.geojson import GeoJSONProvider
from src.data.providers.nasa_giss import NasaGissProvider
from src.data.providers.noaa_gml import NOAACO2Provider
from src.data.providers.open_meteo import OpenMeteoProvider
from src.evaluation.metrics import compute_regression_metrics
from src.explain.lstm_sensitivity import compute_lstm_feature_sensitivity
from src.geo.choropleth import get_country_polygons
from src.geo.frames import build_temporal_map_frames
from src.geo.risk_surface import compute_station_risk_surface
from src.geo.station_network import get_indian_station_network
from src.models.artifacts import ArtifactManager
from src.models.base import ClimateModel
from src.models.baselines import ClimatologyModel, SeasonalNaiveModel
from src.risk.context import compute_historical_context, stats_percentile_of_score
from src.utils.logging import setup_logger


def test_regression_metrics_computation() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    res = compute_regression_metrics(y_true, y_pred, baseline_rmse=0.5)

    assert "mae" in res
    assert "rmse" in res
    assert "mape" in res
    assert "r2" in res
    assert "skill_score" in res
    assert res["skill_score"] > 0.0

    nan_res = compute_regression_metrics(np.array([np.nan]), np.array([np.nan]))
    assert nan_res["mae"] == 0.0


def test_historical_risk_context() -> None:
    history = pd.Series([0.1, 0.5, 1.0, 1.5, 2.0])
    ctx = compute_historical_context(1.2, history)

    assert ctx["total_history"] == 5
    assert ctx["exceed_count"] == 2
    assert 0.0 <= ctx["percentile"] <= 100.0

    empty_ctx = compute_historical_context(1.0, pd.Series([], dtype=float))
    assert empty_ctx["percentile"] == 50.0
    assert stats_percentile_of_score(np.array([]), 1.0) == 50.0


def test_manifest_manager_file_ops(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    mgr = ManifestManager(manifest_path=manifest_file)

    initial = mgr.load()
    assert initial == {"datasets": {}}

    mgr.update_entry("test_dataset", {"rows": 100, "status": "ok"})
    loaded = mgr.load()
    assert "test_dataset" in loaded["datasets"]
    assert loaded["datasets"]["test_dataset"]["rows"] == 100


def test_artifact_manager_save_load(tmp_path: Path) -> None:
    art_mgr = ArtifactManager(base_dir=tmp_path)
    metrics = {"test_mae": 0.12, "test_rmse": 0.15}
    art_mgr.save_metrics("test_model", metrics)
    loaded_metrics = art_mgr.load_metrics("test_model")
    assert loaded_metrics["test_mae"] == 0.12

    meta = {"model_name": "test_model", "seed": 42}
    art_mgr.save_metadata("test_model", meta)
    loaded_meta = art_mgr.load_metadata("test_model")
    assert loaded_meta["seed"] == 42

    assert art_mgr.load_metrics("missing_model") == {}
    assert art_mgr.load_metadata("missing_model") == {}


def test_providers_manifest_and_parsing(
    gistemp_fixture_bytes: bytes,
    noaa_co2_fixture_bytes: bytes,
    ersst_nino_fixture_bytes: bytes,
    openmeteo_fixture_bytes: bytes,
    tmp_path: Path,
) -> None:
    nasa_prov = NasaGissProvider(tmp_path)
    assert nasa_prov.name == "NASA GISS GISTEMP v4"
    df_nasa = nasa_prov.parse(gistemp_fixture_bytes)
    e_nasa = nasa_prov.get_manifest_entry(gistemp_fixture_bytes, df_nasa)
    assert e_nasa["rows"] == len(df_nasa)

    noaa_prov = NOAACO2Provider(tmp_path)
    assert noaa_prov.name == "NOAA GML Mauna Loa CO2"
    df_noaa = noaa_prov.parse(noaa_co2_fixture_bytes)
    e_noaa = noaa_prov.get_manifest_entry(noaa_co2_fixture_bytes, df_noaa)
    assert e_noaa["rows"] == len(df_noaa)

    ersst_prov = ERSSTNinoProvider(tmp_path)
    assert ersst_prov.name == "CPC ERSSTv5 Nino Indices"
    df_ersst = ersst_prov.parse(ersst_nino_fixture_bytes)
    e_ersst = ersst_prov.get_manifest_entry(ersst_nino_fixture_bytes, df_ersst)
    assert e_ersst["rows"] == len(df_ersst)

    om_prov = OpenMeteoProvider(tmp_path)
    assert om_prov.name == "Open-Meteo ERA5 Reanalysis"
    df_om = om_prov.parse(openmeteo_fixture_bytes)
    e_om = om_prov.get_manifest_entry(openmeteo_fixture_bytes, df_om)
    assert e_om["rows"] == len(df_om)

    geo_prov_110 = GeoJSONProvider(tmp_path, scale="110m")
    geo_prov_50 = GeoJSONProvider(tmp_path, scale="50m")
    assert geo_prov_110.source_url != geo_prov_50.source_url
    sample_geo = b'{"type":"FeatureCollection","features":[{"properties":{"ISO_A3":"IND","NAME":"India"}}]}'
    df_geo = geo_prov_110.parse(sample_geo)
    assert len(df_geo) == 1
    assert "iso_a3" in df_geo.columns


def test_loaders_offline(
    tmp_path: Path,
    gistemp_fixture_bytes: bytes,
    noaa_co2_fixture_bytes: bytes,
    ersst_nino_fixture_bytes: bytes,
    openmeteo_fixture_bytes: bytes,
) -> None:
    giss_dir = tmp_path / "gistemp"
    giss_dir.mkdir(parents=True, exist_ok=True)
    (giss_dir / "gistemp_raw.csv").write_bytes(gistemp_fixture_bytes)
    df_giss, _ = load_gistemp_data(raw_dir=giss_dir, offline=True)
    assert len(df_giss) > 0

    noaa_dir = tmp_path / "noaa"
    noaa_dir.mkdir(parents=True, exist_ok=True)
    (noaa_dir / "noaa_co2_raw.csv").write_bytes(noaa_co2_fixture_bytes)
    df_co2, _ = load_noaa_co2_data(raw_dir=noaa_dir, offline=True)
    assert len(df_co2) > 0

    (noaa_dir / "ersst_nino_raw.ascii").write_bytes(ersst_nino_fixture_bytes)
    df_nino, _ = load_ersst_nino_data(raw_dir=noaa_dir, offline=True)
    assert len(df_nino) > 0

    om_dir = tmp_path / "openmeteo"
    om_dir.mkdir(parents=True, exist_ok=True)
    (om_dir / "openmeteo_chennai.json").write_bytes(openmeteo_fixture_bytes)
    df_om, entry_om = load_open_meteo_station("Chennai", 13.0827, 80.2707, raw_dir=om_dir, offline=True)
    assert len(df_om) > 0
    assert "Open-Meteo ERA5" in entry_om["dataset_name"]

    geo_dir = tmp_path / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)
    geojson_data = b'{"type":"FeatureCollection","features":[{"properties":{"ISO_A3":"IND","NAME":"India"},"geometry":null}]}'
    (geo_dir / "natural_earth_110m.geojson").write_bytes(geojson_data)
    (geo_dir / "ne_110m_admin_0_countries.geojson").write_bytes(geojson_data)

    df_geo, entry_geo = load_geojson_boundaries(scale="110m", raw_dir=geo_dir, offline=True)
    assert "iso_a3" in df_geo.columns
    assert entry_geo["dataset_name"] == "Natural Earth GeoJSON"


def test_manifest_provenance_validation(tmp_path: Path) -> None:
    mgr = ManifestManager(manifest_path=tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="Invalid provenance"):
        mgr.record_provenance(
            "TestDS",
            source_url="https://example.com",
            sha256="a" * 64,
            bytes=100,
            rows=10,
            download_date="2026-09-20",
            snapshot_version="v1",
            provenance="invalid-prov",
        )

    entry = mgr.record_provenance(
        "TestDS",
        source_url="https://example.com",
        sha256="a" * 64,
        bytes=100,
        rows=10,
        download_date="2026-09-20",
        snapshot_version="v1",
        provenance="live-fetch",
    )
    assert entry["provenance"] == "live-fetch"
    assert mgr.load()["datasets"]["TestDS"]["sha256"] == "a" * 64


def test_openmeteo_multi_location_parse(tmp_path: Path) -> None:
    prov = OpenMeteoProvider(tmp_path)
    multi_json = json.dumps([
        {
            "latitude": 13.08,
            "longitude": 80.27,
            "elevation": 6.0,
            "daily": {
                "time": ["2020-01-01"],
                "temperature_2m_max": [30.0],
                "temperature_2m_min": [20.0],
            },
        },
        {
            "latitude": 28.61,
            "longitude": 77.20,
            "elevation": 216.0,
            "daily": {
                "time": ["2020-01-01"],
                "temperature_2m_max": [25.0],
                "temperature_2m_min": [15.0],
            },
        },
    ]).encode("utf-8")

    df = prov.parse(multi_json)
    assert len(df) == 2
    assert "temperature_2m_mean" in df.columns


def test_provider_fetch_cache_hits(
    tmp_path: Path,
    gistemp_fixture_bytes: bytes,
    noaa_co2_fixture_bytes: bytes,
    ersst_nino_fixture_bytes: bytes,
    openmeteo_fixture_bytes: bytes,
) -> None:
    nasa_prov = NasaGissProvider(tmp_path)
    (tmp_path / "gistemp_raw.csv").write_bytes(gistemp_fixture_bytes)
    assert nasa_prov.fetch_raw(offline=True) == gistemp_fixture_bytes

    noaa_prov = NOAACO2Provider(tmp_path)
    (tmp_path / "noaa_co2_raw.csv").write_bytes(noaa_co2_fixture_bytes)
    assert noaa_prov.fetch_raw(offline=True) == noaa_co2_fixture_bytes

    ersst_prov = ERSSTNinoProvider(tmp_path)
    (tmp_path / "ersst_nino_raw.ascii").write_bytes(ersst_nino_fixture_bytes)
    assert ersst_prov.fetch_raw(offline=True) == ersst_nino_fixture_bytes

    om_prov = OpenMeteoProvider(tmp_path)
    (tmp_path / "openmeteo_chennai.json").write_bytes(openmeteo_fixture_bytes)
    assert om_prov.fetch_raw_location("Chennai", 13.0827, 80.2707, offline=True) == openmeteo_fixture_bytes

    geo_prov = GeoJSONProvider(tmp_path, scale="110m")
    (tmp_path / "natural_earth_110m.geojson").write_bytes(b'{"type":"FeatureCollection","features":[]}')
    assert geo_prov.fetch_raw(offline=True) == b'{"type":"FeatureCollection","features":[]}'


def test_nasa_provider_errors_and_fallbacks(tmp_path: Path) -> None:
    prov = NasaGissProvider(tmp_path)

    with pytest.raises(FileNotFoundError):
        prov.fetch_raw(offline=True)

    with pytest.raises(ValueError):
        prov.parse(b"single line")

    # Test network failure without cache raises RuntimeError per F2
    with patch("requests.Session.get", side_effect=Exception("Connection error")):
        with pytest.raises(RuntimeError, match="Data not available"):
            prov.fetch_raw(offline=False)


def test_geo_package_functions() -> None:
    df_net = get_indian_station_network()
    assert len(df_net) >= 12
    assert "station" in df_net.columns

    df_surf = compute_station_risk_surface(1.5, methodology="A")
    assert len(df_surf) == len(df_net)
    assert "risk_score" in df_surf.columns
    assert "band" in df_surf.columns

    geo_data = get_country_polygons()
    assert isinstance(geo_data, dict)
    assert "type" in geo_data

    clean_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24, freq="ME"),
        "anomaly_c": np.random.randn(24),
    })
    frames = build_temporal_map_frames(clean_df, sample_interval=6)
    assert len(frames) == 4
    assert "stations" in frames[0]


def test_lstm_sensitivity_calculation() -> None:
    class MockLSTMModel(ClimateModel):
        @property
        def name(self) -> str:
            return "MockLSTM"

        def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> ClimateModel:
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([float(X.iloc[0].sum())])

        def save(self, path: Path) -> Path:
            return path

        def load(self, path: Path) -> ClimateModel:
            return self

        def get_metadata(self) -> Dict[str, Any]:
            return {"name": "MockLSTM"}

    model = MockLSTMModel()
    model.feature_names = ["feat1", "feat2"]
    X_sample = pd.DataFrame({"feat1": [1.0], "feat2": [2.0]})

    sens, names = compute_lstm_feature_sensitivity(model, X_sample)
    assert len(sens) == 2
    assert names == ["feat1", "feat2"]
    assert sens[0] > 0.0


def test_logging_utilities() -> None:
    logger = setup_logger("test_custom_log")
    assert logger is not None


def test_openmeteo_multi_json_parse(tmp_path: Path) -> None:
    prov = OpenMeteoProvider(tmp_path)
    multi_json = b'[{"latitude":13.0,"longitude":80.0,"daily":{"time":["2020-01-01"],"temperature_2m_max":[30.0],"temperature_2m_min":[20.0]}}]'
    df = prov.parse(multi_json)
    assert len(df) == 1
    assert "temperature_2m_mean" in df.columns


def test_baselines_extra_methods(tmp_path: Path) -> None:
    s = SeasonalNaiveModel()
    assert s.get_metadata()["name"] == "SeasonalNaive"

    df_lag = pd.DataFrame({"anomaly_c_lag_12": [0.5, 0.6]})
    s.fit(df_lag, pd.Series([0.5, 0.6]))
    p1 = s.predict(df_lag)
    assert len(p1) == 2

    df_anom = pd.DataFrame({"anomaly_c": [0.3, 0.4]})
    p2 = s.predict(df_anom)
    assert len(p2) == 2

    c = ClimatologyModel()
    assert c.get_metadata()["name"] == "Climatology"
    df_month = pd.DataFrame({"month": [1, 2]})
    c.fit(df_month, pd.Series([0.1, 0.2]))
    p_c = c.predict(df_month)
    assert len(p_c) == 2
