"""Unit tests expanding coverage for metrics, manifest, risk context, logging, baselines, artifacts, providers, and loaders."""

from pathlib import Path
from typing import Any, Dict

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
from src.explain.shap_wrapper import compute_shap_explanation
from src.features.temporal import add_temporal_features
from src.models.artifacts import ArtifactManager
from src.models.base import ClimateModel
from src.models.baselines import ClimatologyModel, SeasonalNaiveModel
from src.models.lstm_model import PyTorchLSTMClimateModel
from src.models.registry import MODEL_REGISTRY, get_model
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

    # Test load missing files fallback
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


def test_temporal_features_edge_cases() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-15", "2024-06-15", "2024-10-15"])})
    tf = add_temporal_features(df, date_col="date")
    assert "month_sin" in tf.columns
    assert "month_cos" in tf.columns
    assert "quarter" in tf.columns


def test_model_registry_lookup() -> None:
    assert "xgboost" in MODEL_REGISTRY
    xgb = get_model("xgboost")
    assert xgb.name == "XGBoost"

    with pytest.raises(ValueError):
        get_model("nonexistent_model")


def test_lstm_multi_step(tmp_path: Path) -> None:
    lstm = PyTorchLSTMClimateModel(params={"sequence_length": 6, "epochs": 2})

    X = pd.DataFrame(np.random.randn(30, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(np.random.randn(30))

    lstm.fit(X, y)
    preds = lstm.predict(X)
    assert len(preds) == len(X)

    multi_preds = lstm.predict_multi_step(X, steps=12)
    assert len(multi_preds) == 12

    meta_file = lstm.save(tmp_path)
    assert meta_file.exists()


def test_baselines_methods(tmp_path: Path) -> None:
    naive = SeasonalNaiveModel()
    assert naive.name == "SeasonalNaive"

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24, freq="ME"),
        "anomaly_c": np.random.randn(24),
        "anomaly_c_lag_12": np.random.randn(24),
    })
    naive.fit(df, df["anomaly_c"])
    preds = naive.predict(df)
    assert len(preds) == 24

    save_path = naive.save(tmp_path)
    assert save_path.exists()
    assert naive.load(save_path).name == "SeasonalNaive"

    clim = ClimatologyModel()
    assert clim.name == "Climatology"
    df["month"] = df["date"].dt.month
    clim.fit(df, df["anomaly_c"])
    c_preds = clim.predict(df)
    assert len(c_preds) == 24

    c_save_path = clim.save(tmp_path)
    assert c_save_path.exists()
    assert clim.load(c_save_path).name == "Climatology"


def test_logging_setup() -> None:
    log = setup_logger("test_logger_unique")
    assert log is not None
    log.info("Test log message execution")


def test_shap_wrapper_fallback() -> None:
    class DummyModel(ClimateModel):
        @property
        def name(self) -> str:
            return "Dummy"

        def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> ClimateModel:
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X))

        def save(self, path: Path) -> Path:
            return path

        def load(self, path: Path) -> ClimateModel:
            return self

        def get_metadata(self) -> Dict[str, Any]:
            return {"name": "Dummy"}

    vals, base, names = compute_shap_explanation(DummyModel(), pd.DataFrame({"a": [1]}))
    assert names == []
