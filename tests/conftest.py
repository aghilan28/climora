"""Pytest configuration and shared test fixtures."""

import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest  # noqa: E402


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def gistemp_fixture_bytes(fixtures_dir: Path) -> bytes:
    return (fixtures_dir / "gistemp_sample.csv").read_bytes()

@pytest.fixture
def noaa_co2_fixture_bytes(fixtures_dir: Path) -> bytes:
    return (fixtures_dir / "noaa_co2_sample.csv").read_bytes()

@pytest.fixture
def ersst_nino_fixture_bytes(fixtures_dir: Path) -> bytes:
    return (fixtures_dir / "ersst_nino_sample.ascii").read_bytes()

@pytest.fixture
def openmeteo_fixture_bytes(fixtures_dir: Path) -> bytes:
    return (fixtures_dir / "openmeteo_sample.json").read_bytes()
