"""Anti-cheat verification scanning codebase for hardcoded metrics or static results."""

import ast
import re
from pathlib import Path

import pytest


def test_no_hardcoded_metrics_in_dashboard() -> None:
    """Ensure st.metric calls in dashboard pages use variable references, not hardcoded numeric string literals."""
    dashboard_dir = Path("dashboard")
    python_files = list(dashboard_dir.glob("**/*.py")) + [Path("app.py")]

    # Match pattern: st.metric("Title", "0.5") or st.metric("Title", f"0.5") with literal float
    # We permit metric labels, but value string must not be a plain literal numeric constant like "1.23 °C" hardcoded without variable formatting
    metric_literal_pattern = re.compile(r'st\.metric\([^,]+,\s*["\'](?:\+|-)?\d+\.\d+\s*(?:°C|%|ppm)?["\']\)')

    for filepath in python_files:
        content = filepath.read_text(encoding="utf-8")
        matches = metric_literal_pattern.findall(content)
        assert not matches, f"Hardcoded metric literal found in {filepath}: {matches}"


def test_no_hardcoded_risk_bands_outside_src_risk() -> None:
    """Ensure risk band names are referenced from RiskBand enum rather than raw hardcoded strings in logic."""
    dashboard_dir = Path("dashboard")
    python_files = list(dashboard_dir.glob("**/*.py"))

    # Check that strings like "SEVERE_RISK" or "EXTREME_RISK" aren't hardcoded as raw variable values outside bands.py
    hardcoded_band_pattern = re.compile(r'["\'](?:SEVERE|EXTREME)_RISK["\']')

    for filepath in python_files:
        content = filepath.read_text(encoding="utf-8")
        matches = hardcoded_band_pattern.findall(content)
        assert not matches, f"Hardcoded risk band string found in {filepath}: {matches}"


def test_ast_parsing_validity() -> None:
    """Parse all shipped python files with AST to verify valid python syntax."""
    shipped_dirs = [Path("src"), Path("dashboard"), Path("config"), Path("scripts")]
    for path in shipped_dirs:
        for filepath in path.glob("**/*.py"):
            try:
                ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {filepath}: {e}")
