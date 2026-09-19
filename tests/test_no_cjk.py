"""CI gate preventing CJK data from being committed under ``tests/``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).parent
_SRC_ROOT = _TESTS_ROOT.parent / "src" / "mltgnt"
_SOURCE_FILES = (
    _SRC_ROOT / "config" / "__init__.py",
    _SRC_ROOT / "persona" / "schema.py",
    _SRC_ROOT / "persona" / "extractor.py",
    _SRC_ROOT / "persona" / "loader.py",
    _SRC_ROOT / "memory" / "_format.py",
    _SRC_ROOT / "memory" / "compaction.py",
    _SRC_ROOT / "memory" / "dream" / "synthesizer.py",
    _SRC_ROOT / "memory" / "dream" / "api.py",
    _SRC_ROOT / "routing" / "triage.py",
)
_CJK_RANGES = ((0x3000, 0x9FFF), (0xFF00, 0xFFEF))
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_BYTE_ESCAPE_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
_FIXTURE_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


def _is_cjk(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def _decoded_byte_escape(match: re.Match[str]) -> str | None:
    raw = bytes.fromhex(match.group().replace(r"\x", ""))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


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
    ],
)
def test_violation_variants_are_detected(tmp_path: Path, payload: str, kind: str) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(payload, encoding="utf-8")

    assert any(kind in violation for violation in _violations(candidate))


def test_no_cjk_in_tests() -> None:
    violations = [violation for path in _scanned_files(_TESTS_ROOT) for violation in _violations(path)]

    assert not violations, "CJK test data found:\n" + "\n".join(violations)


def test_no_cjk_in_source_modules() -> None:
    violations = [violation for path in _SOURCE_FILES for violation in _violations(path)]

    assert not violations, "CJK source data found:\n" + "\n".join(violations)


def test_persona_section_keys_are_canonical_english() -> None:
    from mltgnt.config import DEFAULT_WEIGHT_MAP
    from mltgnt.persona.schema import REQUIRED_SECTIONS

    assert all(key.isascii() for key in DEFAULT_WEIGHT_MAP)
    assert all(key.isascii() for key in REQUIRED_SECTIONS)


def test_legacy_persona_headings_normalize_to_english() -> None:
    from mltgnt.config import PERSONA_SECTION_ALIASES
    from mltgnt.persona.loader import _parse_sections

    legacy_heavy = next(legacy for legacy, canonical in PERSONA_SECTION_ALIASES.items() if canonical == "Heavy")
    legacy_background = next(
        legacy for legacy, canonical in PERSONA_SECTION_ALIASES.items() if canonical == "Background"
    )
    body = f"## {legacy_heavy}\n\n### {legacy_background}\n\nlegacy content"

    assert _parse_sections(body) == {"Background": "legacy content"}


def test_legacy_required_sections_remain_valid() -> None:
    from mltgnt.config import PERSONA_SECTION_ALIASES
    from mltgnt.persona.schema import PersonaFM, REQUIRED_SECTIONS, validate_sections

    headings = []
    for canonical in REQUIRED_SECTIONS:
        legacy = next(alias for alias, mapped in PERSONA_SECTION_ALIASES.items() if mapped == canonical)
        headings.append(f"## {legacy}\n\ncontent")

    result = validate_sections("\n\n".join(headings), PersonaFM(name="legacy"))
    assert result.warnings == []
