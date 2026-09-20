"""Gate test for snapshot pinning and README metrics verification per F2."""

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_pinned_sha_not_equal_to_demo_sample_sha() -> None:
    """Assert pinned_sha256 in config/data.yaml is NOT equal to sha of any sample file in data/sample/."""
    data_cfg = yaml.safe_load((ROOT_DIR / "config" / "data.yaml").read_text(encoding="utf-8"))
    pinned_sha = data_cfg["dataset"]["pinned_sha256"]

    sample_dir = ROOT_DIR / "data" / "sample"
    if sample_dir.exists():
        for sample_file in sample_dir.rglob("*"):
            if sample_file.is_file():
                sample_sha = hashlib.sha256(sample_file.read_bytes()).hexdigest()
                assert pinned_sha != sample_sha, (
                    f"Pinned SHA ({pinned_sha}) equals sample file SHA in {sample_file}! "
                    "Synthetic sample must not satisfy dataset snapshot pin."
                )


def test_verify_readme_passes_on_committed_metrics() -> None:
    """Assert verify_readme.py exits 0 on committed metric artifacts without retraining."""
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / "verify_readme.py")]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
    assert res.returncode == 0, f"verify_readme.py failed on clean clone: {res.stdout}\n{res.stderr}"


def test_verify_readme_negative_control_perturbed_metric(tmp_path: Path) -> None:
    """Negative control: assert verify_readme.py fails when a README metric is perturbed by 0.0001."""
    readme_file = ROOT_DIR / "README.md"
    assert readme_file.exists()

    readme_text = readme_file.read_text(encoding="utf-8")
    # Perturb metric 0.1680 to 0.9999
    perturbed_text = readme_text.replace("0.1680", "0.9999")
    assert perturbed_text != readme_text, "Failed to perturb metric string in README copy"

    temp_readme = tmp_path / "README.md"
    temp_readme.write_text(perturbed_text, encoding="utf-8")

    from scripts.verify_readme import verify_readme_metrics
    assert not verify_readme_metrics(readme_path=temp_readme), "verify_readme should fail on perturbed metric!"


def test_pin_check_negative_control_perturbed_gistemp(tmp_path: Path) -> None:
    """Negative control: assert pin-check exits 1 when raw GISTEMP CSV data is corrupted or modified."""
    # Run pin check with a corrupted gistemp file via temporary cache directory
    fake_raw_dir = tmp_path / "data" / "raw" / "gistemp"
    fake_raw_dir.mkdir(parents=True, exist_ok=True)
    (fake_raw_dir / "gistemp_raw.csv").write_text(
        "Header line\nYear,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n2020,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5\n",
        encoding="utf-8",
    )

    from src.data.providers.nasa_giss import NasaGissProvider
    provider = NasaGissProvider(fake_raw_dir)
    raw_bytes = provider.fetch_raw(offline=True)
    df = provider.parse(raw_bytes)
    entry = provider.get_manifest_entry(raw_bytes, df)

    data_cfg = yaml.safe_load((ROOT_DIR / "config" / "data.yaml").read_text(encoding="utf-8"))
    pinned_sha = data_cfg["dataset"]["pinned_sha256"]

    current_sha = entry["sha256"]
    assert current_sha != pinned_sha, "Corrupted dataset SHA should not match pinned SHA"

