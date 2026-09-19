"""Compatibility gates for English-only test and source conventions."""
from __future__ import annotations

import re
from pathlib import Path

_JP = re.compile(
    f"[{chr(0x3040)}-{chr(0x30FF)}{chr(0x3400)}-{chr(0x9FFF)}]"
)
_MARKER = "Japanese text intentionally " + "kept for CJK processing test"
_SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib"}
_TEXT_SUFFIXES = {".py", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".md", ".toml"}


def _is_marked(lines: list[str], idx: int) -> bool:
    """True if this line or a nearby preceding comment carries the CJK marker."""
    for back in range(0, 6):
        j = idx - back
        if j < 0:
            break
        if _MARKER in lines[j]:
            return True
    return False


def test_tests_directory_is_english() -> None:
    root = Path(__file__).resolve().parent
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        if path.suffix and path.suffix not in _TEXT_SUFFIXES:
            continue
        if path.name == "test_english_only.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not _JP.search(line):
                continue
            rel = path.relative_to(root.parent)
            violations.append(f"{rel}:{i + 1}: {line.strip()[:120]}")
    assert not violations, (
        "CJK text found in tests/:\n" + "\n".join(violations[:80])
    )


def test_src_directory_is_english_except_marked_cjk() -> None:
    """Gate: src/ must be English except intentionally marked CJK literals (#3342)."""
    root = Path(__file__).resolve().parent.parent / "src"
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            file_text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = file_text.splitlines()
        for i, line in enumerate(lines):
            if not _JP.search(line):
                continue
            if _is_marked(lines, i):
                continue
            rel = path.relative_to(root.parent)
            violations.append(f"{rel}:{i + 1}: {line.strip()[:120]}")
    assert not violations, (
        "Unmarked Japanese in src/ (add English translation or "
        f"`# {_MARKER}` near the data):\n" + "\n".join(violations[:80])
    )
