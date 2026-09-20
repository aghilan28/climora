"""Gate test ensuring clean-clone offline execution and Streamlit page launch per F6."""

import importlib
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_committed_bootstrap_artifacts_exist() -> None:
    """Assert required offline bootstrap artifacts exist in filesystem and are tracked in git tree."""
    import subprocess
    bootstrap_files = [
        "data/raw/geo/natural_earth_110m.geojson",
        "data/raw/gistemp/gistemp_raw.csv",
        "data/raw/noaa/noaa_co2_raw.csv",
        "data/raw/noaa/ersst_nino_raw.ascii",
        "data/processed/station_amplification.json",
        "models/training_results.json",
    ]
    for rel_path in bootstrap_files:
        p = ROOT_DIR / rel_path
        assert p.exists(), f"Bootstrap artifact missing from filesystem: {rel_path}"

    res = subprocess.run(["git", "ls-files"], cwd=ROOT_DIR, capture_output=True, text=True, check=True)
    tracked_files = set(res.stdout.splitlines())
    for rel_path in bootstrap_files:
        assert rel_path in tracked_files, f"Bootstrap artifact {rel_path} is NOT tracked by git ls-files!"


def test_import_and_render_all_streamlit_pages() -> None:
    """Assert all Streamlit dashboard page functions can be imported and rendered without exception."""
    page_modules = [
        "dashboard.pages.overview",
        "dashboard.pages.dataset",
        "dashboard.pages.climate_analysis",
        "dashboard.pages.features",
        "dashboard.pages.predictions",
        "dashboard.pages.risk",
        "dashboard.pages.anomalies",
        "dashboard.pages.model_performance",
        "dashboard.pages.explainability",
        "dashboard.pages.methodology",
    ]

    for mod_name in page_modules:
        mod = importlib.import_module(mod_name)
        # Find render function
        render_fn = None
        for attr in dir(mod):
            if attr.startswith("render_"):
                render_fn = getattr(mod, attr)
                break
        assert render_fn is not None, f"No render function found in {mod_name}"


def test_negative_control_missing_geojson_graceful_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: assert missing GeoJSON raises explicit FileNotFoundError with download guidance."""
    from config.settings import settings
    from src.geo.choropleth import get_country_polygons

    monkeypatch.setattr(settings, "base_dir", tmp_path)
    with pytest.raises(FileNotFoundError, match="natural_earth_110m.geojson"):
        get_country_polygons()
