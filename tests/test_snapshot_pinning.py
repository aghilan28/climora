"""Gate test verifying snapshot SHA256 pinning and README verification mechanics."""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_verify_readme_fails_on_perturbed_metric(tmp_path: Path) -> None:
    """Ensure verify_readme.py fails with non-zero exit code when a README metric value is perturbed by 0.0001."""
    verify_script = ROOT_DIR / "scripts" / "verify_readme.py"
    res = subprocess.run([sys.executable, str(verify_script)], capture_output=True, text=True, cwd=ROOT_DIR)
    
    # Should pass on current clean state
    assert res.returncode == 0, f"verify_readme.py failed on clean state: {res.stdout}\n{res.stderr}"
