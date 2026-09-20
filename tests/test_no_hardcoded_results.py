"""Anti-cheat scanner gate with AST inspection and self-testing verification."""

import ast
from pathlib import Path
from typing import List

import pytest


def scan_file_for_anti_cheat_violations(filepath: Path) -> List[str]:
    """Scan a single Python file for hardcoded metrics, numeric array literals, and fake generators."""
    violations: List[str] = []
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))
    except Exception as e:
        return [f"Failed to parse AST for {filepath}: {e}"]

    rel_path = str(filepath).replace("\\", "/")

    for node in ast.walk(tree):
        # 1. ast.List or ast.Tuple with >= 3 numeric constant elements (specifically float literals or metric arrays)
        if isinstance(node, (ast.List, ast.Tuple)):
            float_constants = [
                elt.value for elt in node.elts
                if isinstance(elt, ast.Constant) and type(elt.value) is float
            ]
            if len(float_constants) >= 3:
                # Exclude reference geography data and isolation forest contamination levels
                if "station_network.py" not in rel_path and "isolation_forest.py" not in rel_path:
                    violations.append(
                        f"Hardcoded float array literal (>= 3 float constants) found in {filepath}: {float_constants}"
                    )

        # 2. st.metric calls with constant or static f-string 2nd arg
        if isinstance(node, ast.Call):
            is_st_metric = False
            if isinstance(node.func, ast.Attribute) and node.func.attr == "metric":
                is_st_metric = True
            elif isinstance(node.func, ast.Name) and node.func.id == "metric":
                is_st_metric = True

            if is_st_metric and len(node.args) >= 2:
                val_arg = node.args[1]

                # Hardcoded constant (number or string literal)
                if isinstance(val_arg, ast.Constant):
                    violations.append(
                        f"st.metric 2nd argument is hardcoded constant ({val_arg.value!r}) in {filepath}"
                    )

                # JoinedStr (f-string) without formatted variable references
                elif isinstance(val_arg, ast.JoinedStr):
                    formatted_vars = [
                        v for v in val_arg.values if isinstance(v, ast.FormattedValue)
                    ]
                    if not formatted_vars:
                        violations.append(
                            f"st.metric 2nd argument is static f-string without variable reference in {filepath}"
                        )

        # 3. ast.ListComp inside src/geo/** or dashboard/** whose elt is BinOp operating on range() iterator
        if isinstance(node, ast.ListComp):
            if "src/geo" in rel_path or "dashboard" in rel_path:
                if isinstance(node.elt, ast.BinOp):
                    # Check if iterating over range()
                    for generator in node.generators:
                        if isinstance(generator.iter, ast.Call):
                            func_name = getattr(generator.iter.func, "id", getattr(generator.iter.func, "attr", ""))
                            if func_name in ("range", "arange"):
                                violations.append(
                                    f"ListComp deriving synthetic values from range() BinOp found in {filepath}"
                                )

    return violations


def run_anti_cheat_scanner(paths: List[Path]) -> List[str]:
    """Scan all python files in given paths for anti-cheat violations."""
    all_violations: List[str] = []
    for root_path in paths:
        if root_path.is_file():
            if root_path.suffix == ".py":
                all_violations.extend(scan_file_for_anti_cheat_violations(root_path))
        elif root_path.is_dir():
            for p in root_path.rglob("*.py"):
                if "__pycache__" not in p.parts and ".venv" not in p.parts:
                    all_violations.extend(scan_file_for_anti_cheat_violations(p))
    return all_violations


def test_codebase_anti_cheat_scan() -> None:
    """Scan src, dashboard, app.py, and scripts for hardcoded anti-cheat violations."""
    paths_to_scan = [Path("src"), Path("dashboard"), Path("app.py"), Path("scripts")]
    violations = run_anti_cheat_scanner(paths_to_scan)
    assert not violations, f"Anti-cheat violations found in codebase:\n" + "\n".join(violations)


def test_scanner_detects_planted_fakes(tmp_path: Path) -> None:
    """Self-test: Write 4 planted bad snippets and assert scanner catches each one."""
    snippet1 = tmp_path / "fake1.py"
    snippet1.write_text("vals = [10.5, 12.0, 14.5, 16.0]\n")

    snippet2 = tmp_path / "fake2.py"
    snippet2.write_text("import streamlit as st\nst.metric('Risk', 42.0)\n")

    snippet3 = tmp_path / "fake3.py"
    snippet3.write_text("import streamlit as st\nst.metric('Anomaly', f'0.55 °C')\n")

    snippet4 = tmp_path / "src" / "geo" / "fake4.py"
    snippet4.parent.mkdir(parents=True, exist_ok=True)
    snippet4.write_text("risk_score = 50.0\nscores = [risk_score + (i * 2 - 5) for i in range(8)]\n")

    violations1 = scan_file_for_anti_cheat_violations(snippet1)
    violations2 = scan_file_for_anti_cheat_violations(snippet2)
    violations3 = scan_file_for_anti_cheat_violations(snippet3)
    violations4 = scan_file_for_anti_cheat_violations(snippet4)

    assert len(violations1) >= 1, f"Scanner failed to detect planted fake 1 (numeric array literal): {violations1}"
    assert len(violations2) >= 1, f"Scanner failed to detect planted fake 2 (constant metric): {violations2}"
    assert len(violations3) >= 1, f"Scanner failed to detect planted fake 3 (static f-string metric): {violations3}"
    assert len(violations4) >= 1, f"Scanner failed to detect planted fake 4 (single-scalar ListComp): {violations4}"

    # Write marker artifact to prove self-test execution
    marker_file = Path("data/processed/scanner_selftest_passed.marker")
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_text("Scanner self-test passed successfully on 4 planted fakes.\n")


def test_ast_parsing_validity() -> None:
    """Parse all shipped python files with AST to verify valid python syntax."""
    shipped_dirs = [Path("src"), Path("dashboard"), Path("config"), Path("scripts")]
    for path in shipped_dirs:
        for filepath in path.rglob("*.py"):
            try:
                ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {filepath}: {e}")
