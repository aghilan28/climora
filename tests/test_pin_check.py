"""Gate test for --pin-check functionality per F1."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_config_pinned_sha_resolves() -> None:
    """Assert dataset.pinned_sha256 in config/data.yaml resolves to a valid 64-hex SHA256 string."""
    data_yaml = ROOT_DIR / "config" / "data.yaml"
    assert data_yaml.exists(), "config/data.yaml missing"

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    pinned_sha = cfg.get("dataset", {}).get("pinned_sha256", "")
    assert isinstance(pinned_sha, str) and len(pinned_sha) == 64, f"Invalid pinned_sha256: {pinned_sha}"


def test_pin_check_valid_exits_zero_without_training() -> None:
    """Assert train_models.py --pin-check exits 0 in check-only mode."""
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / "train_models.py"), "--pin-check"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
    assert res.returncode == 0, f"--pin-check failed: {res.stdout}\n{res.stderr}"
    assert "PIN CHECK PASSED" in res.stdout
    assert "Training" not in res.stdout, "Training executed during --pin-check check-only mode!"


def test_pin_check_negative_control_mismatched_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: assert --pin-check fails with exit code 1 when hash is mismatched."""
    import src.data.providers.nasa_giss as nasa_mod

    # Create fake GISTEMP data with bad SHA
    fake_prov_class = nasa_mod.NasaGissProvider

    def mock_fetch(self: nasa_mod.NasaGissProvider, offline: bool = False) -> bytes:
        return (
            b"Land-Ocean: Global Means\n"
            b"Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            b"1880,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20\n"
            b"1881,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20,-.20\n"
        )

    monkeypatch.setattr(fake_prov_class, "fetch_raw", mock_fetch)

    from scripts.train_models import main as train_main

    monkeypatch.setattr(sys, "argv", ["train_models.py", "--pin-check"])
    with pytest.raises(SystemExit) as exc_info:
        train_main()

    assert exc_info.value.code == 1, "Expected SystemExit(1) on hash mismatch in --pin-check"
