"""CI gate preventing CJK data from being committed under ``tests/``."""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).parent
_SRC_ROOT = _TESTS_ROOT.parent / "src" / "mltgnt"
_SOURCE_EXCEPTIONS: set[Path] = set()
_CJK_RANGES = (
    (0x3040, 0x30FF),
    (0x3400, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9F),
)
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_BYTE_ESCAPE_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
_NAMED_ESCAPE_RE = re.compile(r"\\N\{([^}]+)\}")
_FIXTURE_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


def _is_cjk(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def _decoded_byte_escape(match: re.Match[str]) -> str | None:
    raw = bytes.fromhex(match.group().replace(r"\x", ""))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_cjk_named_escape(match: re.Match[str]) -> bool:
    try:
        character = unicodedata.lookup(match.group(1))
    except KeyError:
        return False
    return _is_cjk(ord(character))


def _violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    found: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(_is_cjk(ord(character)) for character in line):
            found.append(f"{path}:{lineno}: literal CJK character")
        if any(_is_cjk(int(match.group(1), 16)) for match in _UNICODE_ESCAPE_RE.finditer(line)):
            found.append(f"{path}:{lineno}: CJK Unicode escape")
        for match in _BYTE_ESCAPE_RE.finditer(line):
            decoded = _decoded_byte_escape(match)
            if decoded is not None and any(_is_cjk(ord(character)) for character in decoded):
                found.append(f"{path}:{lineno}: CJK UTF-8 byte escape")
        if any(_is_cjk_named_escape(match) for match in _NAMED_ESCAPE_RE.finditer(line)):
            found.append(f"{path}:{lineno}: CJK named escape")
    return found


def _scanned_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (path.suffix == ".py" or path.suffix in _FIXTURE_SUFFIXES)
    )


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ("prefix " + chr(0x3042), "literal CJK character"),
        ("prefix " + "\\" + "u3042", "CJK Unicode escape"),
        (
            "prefix " + "\\" + "xe3" + "\\" + "x81" + "\\" + "x82",
            "CJK UTF-8 byte escape",
        ),
        ("prefix " + "\\" + "N{CJK UNIFIED IDEOGRAPH-65E5}", "CJK named escape"),
    ],
)
def test_violation_variants_are_detected(tmp_path: Path, payload: str, kind: str) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(payload, encoding="utf-8")

    assert any(kind in violation for violation in _violations(candidate))


def test_unknown_named_escape_is_not_a_violation(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("prefix " + "\\" + "N{NOT A REAL NAME}", encoding="utf-8")

    assert _violations(candidate) == []


def test_no_cjk_in_tests() -> None:
    violations = [violation for path in _scanned_files(_TESTS_ROOT) for violation in _violations(path)]

    assert not violations, "CJK test data found:\n" + "\n".join(violations)


def test_no_cjk_in_source_modules() -> None:
    source_files = set(_SRC_ROOT.rglob("*.py"))
    violations = [
        violation
        for path in sorted(source_files - _SOURCE_EXCEPTIONS)
        for violation in _violations(path)
    ]

    assert not violations, "CJK source data found:\n" + "\n".join(violations)


def test_source_cjk_exceptions_are_empty() -> None:
    assert set() == _SOURCE_EXCEPTIONS


def test_persona_section_keys_are_canonical_english() -> None:
    from mltgnt.config import DEFAULT_WEIGHT_MAP
    from mltgnt.persona.schema import REQUIRED_SECTIONS

    assert all(key.isascii() for key in DEFAULT_WEIGHT_MAP)
    assert all(key.isascii() for key in REQUIRED_SECTIONS)


_DUMMY_ALIASES = {
    "Legacy heavy": "Heavy",
    "Legacy background": "Background",
    "Legacy values": "Values",
    "Legacy reactions": "Reaction patterns",
    "Legacy tone": "Tone",
    "Legacy output": "Output format",
    "Legacy light": "Light",
    "Legacy reference": "Reference",
    "Legacy triage": "Triage",
}


@pytest.fixture
def dummy_aliases(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    from mltgnt.config.language import EN

    pack = dataclasses.replace(EN, persona_section_aliases=dict(_DUMMY_ALIASES))
    monkeypatch.setattr("mltgnt.config.language._current", pack)
    return pack.persona_section_aliases


def test_legacy_persona_headings_normalize_to_english(dummy_aliases: dict[str, str]) -> None:
    from mltgnt.persona.loader import _parse_sections

    legacy_heavy = next(legacy for legacy, canonical in dummy_aliases.items() if canonical == "Heavy")
    legacy_background = next(legacy for legacy, canonical in dummy_aliases.items() if canonical == "Background")
    body = f"## {legacy_heavy}\n\n### {legacy_background}\n\nlegacy content"

    assert _parse_sections(body) == {"Background": "legacy content"}


def test_legacy_required_sections_remain_valid(dummy_aliases: dict[str, str]) -> None:
    from mltgnt.persona.schema import PersonaFM, REQUIRED_SECTIONS, validate_sections

    headings = []
    for canonical in REQUIRED_SECTIONS:
        legacy = next(alias for alias, mapped in dummy_aliases.items() if mapped == canonical)
        headings.append(f"## {legacy}\n\ncontent")

    result = validate_sections("\n\n".join(headings), PersonaFM(name="legacy"))
    assert result.warnings == []
