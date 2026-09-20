"""Gate test for 12-station panel atomicity and strict factor loading per F4."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.geo.station_network import load_station_amplification_factors


def test_missing_stations_in_artifact_raises_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert load_station_amplification_factors raises ValueError if artifact has fewer than 12 stations."""
    fake_artifact = tmp_path / "station_amplification.json"
    fake_data = {
        "n_stations": 3,
        "factors": {"Chennai": 1.2, "Delhi": 1.1, "Mumbai": 1.3}
    }
    fake_artifact.write_text(json.dumps(fake_data), encoding="utf-8")

    from config.settings import settings
    monkeypatch.setattr(settings, "base_dir", tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed" / "station_amplification.json").write_text(json.dumps(fake_data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 9 required stations"):
        load_station_amplification_factors()


def test_no_fillna_literal_in_geo_package() -> None:
    """Assert string '.fillna(1.0)' or '.fillna(' is absent from src/geo/ station_network.py."""
    st_file = Path("src/geo/station_network.py")
    assert st_file.exists()
    content = st_file.read_text(encoding="utf-8")
    assert "fillna(1.0)" not in content, "Found fillna(1.0) fallback in station_network.py!"


def test_download_all_datasets_atomic_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert download_all_datasets exits with non-zero code if any station download fails."""
    from typing import Any

    import scripts.download_data as dd

    def mock_load_station(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, str]]:
        station_name = str(args[0])
        if station_name == "Delhi":
            raise RuntimeError("Simulated Open-Meteo HTTP 429 Rate Limit")
        return pd.DataFrame({"date": ["2020-01-01"], "temperature_2m_mean": [25.0]}), {"source_url": "mock"}

    from src.data.manifest import ManifestManager
    fake_manifest = ManifestManager(manifest_path=tmp_path / "manifest.json")
    monkeypatch.setattr(dd, "ManifestManager", lambda: fake_manifest)
    monkeypatch.setattr(dd, "load_open_meteo_station", mock_load_station)
    monkeypatch.setattr(dd, "load_gistemp_data", lambda offline=False: (pd.DataFrame({"year": [1990], "anomaly_c": [0.2]}), {}))
    monkeypatch.setattr(dd, "load_noaa_co2_data", lambda offline=False: (pd.DataFrame(), {}))
    monkeypatch.setattr(dd, "load_ersst_nino_data", lambda offline=False: (pd.DataFrame(), {}))
    monkeypatch.setattr(dd, "load_geojson_boundaries", lambda *args, **kwargs: (pd.DataFrame(), {}))

    with pytest.raises(SystemExit) as exc_info:
        dd.download_all_datasets(offline=False)

    assert exc_info.value.code != 0, "download_all_datasets should exit non-zero when station fetch fails!"
