"""Programmatic final audit script for CLIMORA AI platform (PART 9 checklist)."""

import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def check_item(description: str, condition: bool, details: str = "") -> bool:
    """Print status of check item and return boolean success."""
    if condition:
        print(f"[PASS] {description}")
        return True
    else:
        print(f"[FAIL] {description}")
        if details:
            print(f"       Details: {details}")
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


def audit_grep_patterns() -> bool:
    """Check forbidden literal patterns in source code and docs."""
    all_ok = True

    # 1. No hardcoded metric literals in st.metric
    metric_pattern = re.compile(r"st\.metric\([^)]*,\s*-?[0-9]+\.[0-9]")
    metric_matches = []
    for search_dir in [ROOT_DIR / "app.py", ROOT_DIR / "dashboard", ROOT_DIR / "src"]:
        if search_dir.is_file():
            files = [search_dir]
        else:
            files = list(search_dir.rglob("*.py"))
        for f in files:
            content = f.read_text(encoding="utf-8")
            if metric_pattern.search(content):
                metric_matches.append(str(f.relative_to(ROOT_DIR)))

    all_ok &= check_item(
        "No literal metric values in st.metric calls",
        len(metric_matches) == 0,
        f"Matches in: {metric_matches}",
    )

    # 2. No placeholder strings in shipped code and docs
    ph_pattern = re.compile(
        r"TODO|FIXME|coming soon|demo result|sample prediction|placeholder|not implemented",
        re.IGNORECASE,
    )
    ph_matches = []
    scan_paths = [
        ROOT_DIR / "app.py",
        ROOT_DIR / "dashboard",
        ROOT_DIR / "src",
        ROOT_DIR / "README.md",
        ROOT_DIR / "docs",
    ]
    for p in scan_paths:
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = list(p.rglob("*"))
        else:
            continue
        for f in files:
            if f.is_file() and not f.name.startswith("."):
                try:
                    content = f.read_text(encoding="utf-8")
                    if ph_pattern.search(content):
                        ph_matches.append(str(f.relative_to(ROOT_DIR)))
                except Exception:
                    pass

    all_ok &= check_item(
        "No placeholder strings in code or documentation",
        len(ph_matches) == 0,
        f"Matches in: {ph_matches}",
    )

    # 3. No secrets or API keys hardcoded
    secret_pattern = re.compile(r"(api[_-]?key|token|secret)\s*=\s*['\"][^'\"]{12,}", re.IGNORECASE)
    secret_matches = []
    for f in ROOT_DIR.rglob("*.py"):
        if ".git" in f.parts or "venv" in f.parts:
            continue
        try:
            content = f.read_text(encoding="utf-8")
            if secret_pattern.search(content):
                secret_matches.append(str(f.relative_to(ROOT_DIR)))
        except Exception:
            pass

    all_ok &= check_item(
        "No hardcoded API keys or secrets in codebase",
        len(secret_matches) == 0,
        f"Matches in: {secret_matches}",
    )

    return all_ok


def main() -> None:
    """Run full system audit."""
    print("=" * 60)
    print("CLIMORA AI — Programmatic System Acceptance Audit")
    print("=" * 60)
    passed = True

    # Check 1: Required files existence
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
        "All required root files and 10 documentation files present",
        len(missing_files) == 0,
        f"Missing: {missing_files}",
    )

    # Check 2: Code linters and typecheckers
    python_exe = sys.executable
    code, out = run_command([python_exe, "-m", "ruff", "check", "."])
    passed &= check_item("Ruff linter exits 0 with zero errors", code == 0, out.strip())

    code, out = run_command([python_exe, "-m", "mypy", "src", "dashboard", "app.py"])
    passed &= check_item("Mypy type checker exits 0 with zero errors", code == 0, out.strip())

    # Check 3: Unit tests and code coverage
    code, out = run_command(
        [python_exe, "-m", "pytest", "-q", "--cov=src", "--cov-fail-under=85"]
    )
    passed &= check_item("Pytest coverage >= 85% and test suite green", code == 0, out.strip())

    # Check 4: Anti-cheat grep checks
    passed &= audit_grep_patterns()

    # Check 5: Data and Model Artifacts
    data_manifest = ROOT_DIR / "data" / "manifest.json"
    passed &= check_item("Data manifest.json exists", data_manifest.exists())

    model_files = ["xgb_model.json", "lgbm_model.txt", "lstm_weights.pt"]
    missing_models = [m for m in model_files if not (ROOT_DIR / "models" / "trained" / m).exists()]
    passed &= check_item(
        "All trained model artifacts present in models/trained/",
        len(missing_models) == 0,
        f"Missing: {missing_models}",
    )

    print("=" * 60)
    if passed:
        print("SYSTEM AUDIT PASSED: All PART 9 acceptance criteria satisfied.")
        sys.exit(0)
    else:
        print("SYSTEM AUDIT FAILED: One or more acceptance criteria failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
