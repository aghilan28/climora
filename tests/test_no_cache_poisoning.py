"""Gate test ensuring no cache poisoning on network fetch failures."""

from pathlib import Path
from unittest.mock import patch
import pytest
import requests

from src.data.providers.nasa_giss import NasaGissProvider


def test_no_cache_poisoning_on_fetch_failure(tmp_path: Path) -> None:
    """Ensure fetch_raw raises when network fails without creating or poisoning cache file."""
    prov = NasaGissProvider(tmp_path)
    cache_file = tmp_path / "gistemp_raw.csv"

    # Ensure no cache file exists initially
    assert not cache_file.exists()

    with patch("requests.get", side_effect=requests.RequestException("Simulated network failure")):
        with pytest.raises((RuntimeError, requests.RequestException)):
            prov.fetch_raw(offline=False)

    # Cache file MUST NOT be written on fetch failure
    assert not cache_file.exists(), "Cache file was written despite network fetch failure!"


def test_no_authentic_disguise_wording() -> None:
    """Ensure no misleading 'Using authentic' fallback log wording exists in src/."""
    src_dir = Path("src")
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "Using authentic" not in content, f"Misleading fallback wording found in {py_file}"
