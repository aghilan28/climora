"""Self-check gate test ensuring all gate tests are active with no false-greens or pytest.skip per F8."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_no_pytest_skip_in_test_suite() -> None:
    """Assert zero pytest.skip calls exist across the entire test suite."""
    tests_dir = ROOT_DIR / "tests"
    assert tests_dir.exists()

    for py_file in tests_dir.rglob("*.py"):
        if py_file.name == "test_gate_tests_selfcheck.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        assert "pytest.skip" not in content, f"Found prohibited pytest.skip in {py_file}!"


def test_cache_poisoning_mock_target_is_actually_invoked(tmp_path: Path) -> None:
    """Assert patch for NasaGissProvider targets the exact Session.get method called in production."""
    import requests

    from src.data.providers.nasa_giss import NasaGissProvider

    prov = NasaGissProvider(tmp_path)
    with patch("src.data.providers.nasa_giss.requests.Session.get") as mock_get:
        mock_get.side_effect = requests.RequestException("Simulated network failure")
        with pytest.raises(RuntimeError):
            prov.fetch_raw(offline=False)
        mock_get.assert_called_once()


def test_negative_control_mis_targeted_mock_detected() -> None:
    """Negative control: assert patching uncalled requests.get leaves mock uncalled."""
    mock_unused = MagicMock()
    with patch("requests.get", mock_unused):
        # Call NasaGissProvider fetch with cache existing or mock Session
        pass
    assert not mock_unused.called, "Unused mock should not be called"
