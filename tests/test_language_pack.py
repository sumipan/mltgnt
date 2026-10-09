"""LanguagePack locale keys for legacy persona headings and Phase 1 meta lines (#5040)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from mltgnt.config.language import EN, LanguagePack

_DUMMY_ALIASES = {
    "LEGACY-LIGHT": "Light",
    "LEGACY-HEAVY": "Heavy",
    "LEGACY-TRIAGE": "Triage",
}


def _install(monkeypatch: pytest.MonkeyPatch, **changes: object) -> LanguagePack:
    pack = dataclasses.replace(EN, **changes)
    monkeypatch.setattr("mltgnt.config.language._current", pack)
    return pack


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, re.Pattern):
        return [value.pattern]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(value, (tuple, frozenset)):
        return [s for item in value for s in _strings(item)]
    return []


def test_en_new_keys_are_empty() -> None:
    assert EN.persona_section_aliases == {}
    assert EN.phase1_meta_prefixes == ()
    assert EN.phase1_meta_markers == ()


def test_en_strings_are_ascii() -> None:
    for f in dataclasses.fields(EN):
        for text in _strings(getattr(EN, f.name)):
            assert text.isascii(), f"{f.name}: {text!r}"


def test_config_reads_aliases_from_current_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltgnt.config import DEFAULT_WEIGHT_MAP, PersonaConfig

    _install(monkeypatch, persona_section_aliases=dict(_DUMMY_ALIASES))

    assert DEFAULT_WEIGHT_MAP.get("LEGACY-LIGHT") == "light"
    assert PersonaConfig().section_aliases == _DUMMY_ALIASES


def test_sanitize_phase1_output_uses_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltgnt.memory.compaction import _sanitize_phase1_output

    _install(monkeypatch, phase1_meta_prefixes=("META-ACK",), phase1_meta_markers=("**META-STATS**",))

    assert _sanitize_phase1_output("META-ACK done\n- likes tea\nx **META-STATS** y") == "- likes tea"


def test_extract_uses_pack_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltgnt.persona.extractor import extract

    _install(monkeypatch, persona_section_aliases={"LEGACY-LIGHT": "Light"})
    assert extract({"LEGACY-LIGHT": "light text"}, "light") == "light text"

    _install(monkeypatch, persona_section_aliases={"LEGACY-HEAVY": "Heavy"})
    assert extract({"LEGACY-HEAVY": "heavy text"}, "heavy") == "heavy text"


def test_regenerate_light_block_uses_pack_aliases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mltgnt.persona import compress

    _install(monkeypatch, persona_section_aliases=dict(_DUMMY_ALIASES))
    monkeypatch.setattr(compress, "compress_heavy_to_light", lambda heavy_text, **kwargs: "new light")
    monkeypatch.setattr(compress, "_validate_v21_light_block", lambda text, **kwargs: None)
    persona = tmp_path / "dummy.md"
    persona.write_text("## LEGACY-HEAVY\n\nheavy text\n\n## LEGACY-LIGHT\n\nold light\n", encoding="utf-8")

    result = compress.regenerate_light_block(persona)

    assert result.light_text == "new light"
    content = persona.read_text(encoding="utf-8")
    assert "## LEGACY-LIGHT\n\nnew light" in content
    assert "old light" not in content
    assert "heavy text" in content


def test_extract_triage_section_uses_pack_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltgnt.routing.triage import extract_triage_section

    _install(monkeypatch, persona_section_aliases={"LEGACY-TRIAGE": "Triage"})

    assert extract_triage_section("## LEGACY-TRIAGE\n\nbody\n\n## Other\n\nx") == "body"


def test_pack_argument_overrides_current_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltgnt.persona.extractor import extract
    from mltgnt.routing.triage import extract_triage_section

    _install(monkeypatch, persona_section_aliases={"OTHER": "Triage"})
    explicit = dataclasses.replace(EN, persona_section_aliases={"LEGACY-TRIAGE": "Triage"})

    assert extract_triage_section("## LEGACY-TRIAGE\n\nbody", pack=explicit) == "body"
    assert extract_triage_section("## LEGACY-TRIAGE\n\nbody") is None
    explicit_light = dataclasses.replace(EN, persona_section_aliases={"LEGACY-LIGHT": "Light"})
    assert extract({"LEGACY-LIGHT": "light text"}, "light", pack=explicit_light) == "light text"


def test_en_does_not_recognize_legacy_headings() -> None:
    from mltgnt.routing.triage import extract_triage_section

    assert extract_triage_section("## LEGACY-TRIAGE\n\nbody") is None
    assert extract_triage_section("## Light\n\nlight body") == "light body"
    assert extract_triage_section("## Triage\n\ntriage body") == "triage body"
