"""Anti-cheat verification scanning codebase for banned placeholder keywords."""

import re
from pathlib import Path

BANNED_KEYWORDS = [
    "TODO",
    "FIXME",
    "coming soon",
    "demo result",
    "sample prediction",
    "placeholder",
    "not implemented",
]


def test_no_placeholder_keywords_in_shipped_code() -> None:
    """Scan shipped python files for banned placeholder strings."""
    search_dirs = [Path("src"), Path("dashboard"), Path("config"), Path("scripts"), Path("app.py")]
    python_files = []
    for s in search_dirs:
        if s.is_file():
            python_files.append(s)
        else:
            python_files.extend(s.glob("**/*.py"))

    keyword_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in BANNED_KEYWORDS) + r")\b",
        re.IGNORECASE,
    )

    violations = []
    for filepath in python_files:
        if filepath.name == "audit.py":
            continue
        content = filepath.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), 1):
            if keyword_pattern.search(line):
                violations.append(f"{filepath}:{line_num}: {line.strip()}")

    assert not violations, "Placeholder keywords found in shipped files:\n" + "\n".join(violations)
