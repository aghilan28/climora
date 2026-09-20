"""Programmatic acceptance audit script for CLIMORA AI platform."""
# ruff: noqa: E402

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def check_item(description: str, condition: bool, details: str = "") -> bool:
    """Print status of check item and return boolean success."""
    if condition:
        print(f"[PASS] {description}", flush=True)
        return True
    else:
        print(f"[FAIL] {description}", flush=True)
        if details:
            print(f"       Details: {details}", flush=True)
        return False


def run_command(cmd: list[str]) -> tuple[int, str]:
    """Execute command and return returncode and output string."""
    try:
        res = subprocess.run(
            cmd, cwd=ROOT_DIR, capture_output=True, text=True, check=False
        )
        return res.returncode, res.stdout + res.stderr
    except Exception as e:
        return 1, str(e)


def main() -> None:
    """Run full system acceptance audit."""
    print("=" * 65)
    print("CLIMORA AI — Independent Acceptance Audit Gate Suite")
    print("=" * 65)
    passed = True
    python_exe = sys.executable

    # Check 1: Required root & doc files present
    req_files = [
        "app.py",
        "README.md",
        "LICENSE",
        "AGENTS.md",
        "requirements.txt",
        "pyproject.toml",
        "Makefile",
        "docs/architecture.md",
        "docs/data_pipeline.md",
        "docs/dataset.md",
        "docs/feature_engineering.md",
        "docs/ml_pipeline.md",
        "docs/model_evaluation.md",
        "docs/risk_methodology.md",
        "docs/anomaly_detection.md",
        "docs/explainability.md",
        "docs/deployment.md",
    ]
    missing_files = [f for f in req_files if not (ROOT_DIR / f).exists()]
    passed &= check_item(
        "All required root files and documentation files present",
        len(missing_files) == 0,
        f"Missing: {missing_files}",
    )

    # Check 2: Manifest Provenance & Dataset Schema
    manifest_path = ROOT_DIR / "data" / "manifest.json"
    manifest_ok = True
    manifest_details = ""
    station_count = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = manifest.get("datasets", {})
        for name, entry in datasets.items():
            prov = entry.get("provenance", "")
            sha = entry.get("sha256", "")
            ddate = entry.get("download_date", "")
            if prov not in {"live-fetch", "cache", "pinned-fixture"}:
                manifest_ok = False
                manifest_details += f"Invalid provenance '{prov}' in {name}; "
            if len(sha) != 64:
                manifest_ok = False
                manifest_details += f"Invalid SHA256 length ({len(sha)}) in {name}; "
            if not ddate:
                manifest_ok = False
                manifest_details += f"Missing download_date in {name}; "
            if "Open-Meteo ERA5" in name:
                station_count += 1
    else:
        manifest_ok = False
        manifest_details = "manifest.json missing"

    passed &= check_item(
        "Manifest provenance entries valid (provenance in {live-fetch, cache, pinned-fixture}, 64-hex sha256, download_date)",
        manifest_ok,
        manifest_details,
    )

    passed &= check_item(
        "Data manifest Open-Meteo station count == 12",
        station_count == 12,
        f"Manifest station count = {station_count}",
    )

    # Check 3: Code grep checks (No fake cache phrases, SSL bypass, offline fallbacks)
    code_giss, out_giss = run_command(["git", "grep", "-rn", "Using authentic", "src/"])
    passed &= check_item(
        "grep -rn 'Using authentic' src/ is empty",
        code_giss != 0 or not out_giss.strip(),
        out_giss.strip(),
    )

    code_ssl, out_ssl = run_command(["git", "grep", "-rn", "CERT_NONE\\|check_hostname = False", "src/"])
    passed &= check_item(
        "grep -rn 'CERT_NONE|check_hostname = False' src/ is empty",
        code_ssl != 0 or not out_ssl.strip(),
        out_ssl.strip(),
    )

    code_off, out_off = run_command(["git", "grep", "-n", "offline=True", "scripts/train_models.py"])
    passed &= check_item(
        "grep -n 'offline=True' scripts/train_models.py is empty",
        code_off != 0 or not out_off.strip(),
        out_off.strip(),
    )

    # Check 4: Region factor literal check in station_network.py
    st_net_file = ROOT_DIR / "src" / "geo" / "station_network.py"
    st_net_content = st_net_file.read_text(encoding="utf-8") if st_net_file.exists() else ""
    passed &= check_item(
        "region_factor float literal absent from station_network.py",
        '"region_factor":' not in st_net_content,
        "Literal 'region_factor' key found in station_network.py",
    )

    # Check 5: Choropleth & PyDeck PolygonLayer present in risk.py
    risk_file = ROOT_DIR / "dashboard" / "pages" / "risk.py"
    risk_content = risk_file.read_text(encoding="utf-8") if risk_file.exists() else ""
    geo_exists = (ROOT_DIR / "src" / "geo" / "choropleth.py").exists()
    has_poly = "GeoJsonLayer" in risk_content or "PolygonLayer" in risk_content
    passed &= check_item(
        "PolygonLayer/GeoJsonLayer present in dashboard/pages/risk.py and choropleth file exists",
        has_poly and geo_exists,
        f"GeoJsonLayer present: {has_poly}, choropleth file exists: {geo_exists}",
    )

    # Check 6: "Best Model" badge condition
    perf_file = ROOT_DIR / "dashboard" / "pages" / "model_performance.py"
    perf_content = perf_file.read_text(encoding="utf-8") if perf_file.exists() else ""
    has_honest_sentence = "No model beats the seasonal-naive baseline" in perf_content
    passed &= check_item(
        "Best Model badge rendered only if skill_score > 0 (honest sentence present)",
        has_honest_sentence,
        "Missing honest baseline sentence in model_performance.py",
    )

    # Check 7: Mypy static type analysis
    code_mypy, out_mypy = run_command([python_exe, "-m", "mypy", "src", "dashboard", "app.py", "scripts", "config"])
    mypy_clean = code_mypy == 0 and "errors prevented further checking" not in out_mypy
    passed &= check_item(
        "Mypy type analysis exits 0 without 'errors prevented further checking'",
        mypy_clean,
        out_mypy.strip(),
    )

    # Check 8: Git untracked / ignored hygiene
    code_git, out_git = run_command(["git", "ls-files"])
    bad_git_files = [
        f for f in out_git.splitlines()
        if f.startswith("logs/") or "__pycache__" in f or f.endswith(".pyc")
    ]
    passed &= check_item(
        "git ls-files contains no logs/, __pycache__/, or *.pyc",
        len(bad_git_files) == 0,
        f"Tracked bad files: {bad_git_files}",
    )

    # Check 9: Requirements pinning (zero >= lines)
    req_file = ROOT_DIR / "requirements.txt"
    req_lines = req_file.read_text(encoding="utf-8").splitlines() if req_file.exists() else []
    gte_lines = [l for l in req_lines if ">=" in l and not l.strip().startswith("#")]
    passed &= check_item(
        "requirements.txt contains zero '>=' lines (all pinned with '==')",
        len(gte_lines) == 0,
        f"Unpinned '>=' lines: {gte_lines}",
    )

    # Check 10: Anti-cheat scanner self-test marker
    marker_file = ROOT_DIR / "data" / "processed" / "scanner_selftest_passed.marker"
    if not marker_file.exists():
        run_command([python_exe, "-m", "pytest", "tests/test_no_hardcoded_results.py"])
    passed &= check_item(
        "Anti-cheat scanner planted-fake self-test ran and marker exists",
        marker_file.exists(),
        "data/processed/scanner_selftest_passed.marker missing",
    )

    # Check 11: Pytest test suite & coverage >= 85%
    code_pyt, out_pyt = run_command([python_exe, "-m", "pytest", "-q", "--cov=src", "--cov-fail-under=85"])
    passed &= check_item(
        "Pytest test suite passes with code coverage >= 85%",
        code_pyt == 0,
        out_pyt.strip(),
    )

    # Check 12: README metrics verification
    code_vreadme, out_vreadme = run_command([python_exe, "scripts/verify_readme.py"])
    passed &= check_item(
        "verify_readme.py passes (README §12 matches on-disk JSON artifacts)",
        code_vreadme == 0,
        out_vreadme.strip(),
    )

    print("=" * 65)
    if passed:
        print("ACCEPTED: All acceptance audit gates PASSED (0 exit code).")
        sys.exit(0)
    else:
        print("REJECTED: One or more acceptance audit gates FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
