"""CI gate: CJK characters must be absent from designated test files.

Checks each registered path for:
- Literal CJK characters in the ranges U+3000-U+9FFF and U+FF00-U+FFEF
- \\uXXXX escape sequences where the codepoint falls in those ranges

To expand coverage as more files are cleaned, add their paths to _CJK_CLEAN_FILES.
"""
from __future__ import annotations

import re
from pathlib import Path

_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")

_CJK_LOW_MIN = 0x3000
_CJK_LOW_MAX = 0x9FFF
_CJK_HIGH_MIN = 0xFF00
_CJK_HIGH_MAX = 0xFFEF

# Files confirmed CJK-clean; expand this list as more files are migrated.
_TESTS_ROOT = Path(__file__).parent
_CJK_CLEAN_FILES = [
    _TESTS_ROOT / "conftest.py",
    _TESTS_ROOT / "agent" / "test_deterministic_gate.py",
    _TESTS_ROOT / "agent" / "test_dispatch_decision.py",
    _TESTS_ROOT / "persona" / "test_compress.py",
    _TESTS_ROOT / "persona" / "test_formatter.py",
    _TESTS_ROOT / "test_channel_router.py",
]


def _is_cjk(cp: int) -> bool:
    return (_CJK_LOW_MIN <= cp <= _CJK_LOW_MAX) or (_CJK_HIGH_MIN <= cp <= _CJK_HIGH_MAX)


def _violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    found: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(_is_cjk(ord(ch)) for ch in line):
            found.append(f"  {path.name}:{lineno}: literal CJK character")
        for m in _ESCAPE_RE.finditer(line):
            if _is_cjk(int(m.group(1), 16)):
                found.append(f"  {path.name}:{lineno}: CJK escape \\u{m.group(1)}")
    return found


def test_no_cjk_in_clean_files() -> None:
    violations: list[str] = []
    for py_file in _CJK_CLEAN_FILES:
        violations.extend(_violations(py_file))

    assert not violations, (
        f"CJK found in {len(violations)} location(s):\n" + "\n".join(violations)
    )
