"""Anti-cheat verification scanning codebase for hardcoded metrics or static results."""

import ast
import re
from pathlib import Path

import pytest


def test_no_hardcoded_metrics_in_dashboard() -> None:
    """Ensure st.metric calls in dashboard pages use variable references, not hardcoded numeric string literals."""
    dashboard_dir = Path("dashboard")
    python_files = list(dashboard_dir.glob("**/*.py")) + [Path("app.py")]

    metric_literal_pattern = re.compile(r'st\.metric\([^,]+,\s*["\'](?:\+|-)?\d+\.\d+\s*(?:°C|%|ppm)?["\']\)')

    for filepath in python_files:
        content = filepath.read_text(encoding="utf-8")
        matches = metric_literal_pattern.findall(content)
        assert not matches, f"Hardcoded metric literal found in {filepath}: {matches}"


def test_no_hardcoded_risk_bands_outside_src_risk() -> None:
    """Ensure risk band names are referenced from RiskBand enum rather than raw hardcoded strings in logic."""
    dashboard_dir = Path("dashboard")
    python_files = list(dashboard_dir.glob("**/*.py"))

    hardcoded_band_pattern = re.compile(r'["\'](?:SEVERE|EXTREME)_RISK["\']')

    for filepath in python_files:
        content = filepath.read_text(encoding="utf-8")
        matches = hardcoded_band_pattern.findall(content)
        assert not matches, f"Hardcoded risk band string found in {filepath}: {matches}"


def test_no_hardcoded_numeric_arrays_in_dashboard_pages() -> None:
    """Ensure dashboard pages do not hardcode lists of coordinates or risk scores directly."""
    dashboard_pages_dir = Path("dashboard/pages")
    python_files = list(dashboard_pages_dir.glob("*.py"))

    for filepath in python_files:
        tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
        for node in ast.walk(tree):
            # Check for assign nodes where target is 'lat' or 'lon' or 'risk_score' set to a list literal of floats/numbers
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=False):
                    if isinstance(key, ast.Constant) and key.value in ("lat", "lon", "station", "risk_score"):
                        if isinstance(value, ast.List) and len(value.elts) > 3:
                            # Verify if the list elements are raw hardcoded numbers
                            has_constants = any(isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)) for elt in value.elts)
                            assert not has_constants, f"Hardcoded numeric array found for '{key.value}' in {filepath}"


def test_ast_parsing_validity() -> None:
    """Parse all shipped python files with AST to verify valid python syntax."""
    shipped_dirs = [Path("src"), Path("dashboard"), Path("config"), Path("scripts")]
    for path in shipped_dirs:
        for filepath in path.glob("**/*.py"):
            try:
                ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {filepath}: {e}")
