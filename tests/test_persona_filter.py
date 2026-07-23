"""Tests for Persona.register_prompt_filter API."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from freezegun import freeze_time

from mltgnt.persona import load_persona
from mltgnt.persona.loader import Persona, PromptFilter
from mltgnt.interfaces.persona import PersonaProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: タチコマ
      aliases:
        - tachikoma
    ops:
      engine: claude
    ---

    ## 基本情報

    タチコマはGHSの多脚戦車型AIロボット。

    ## 価値観

    好奇心旺盛で仲間を大切にする。

    ## 反応パターン

    質問には積極的に答える。

    ## 口調

    フレンドリーで明るい。
""")


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    return d


@pytest.fixture
def tachikoma_persona(agents_dir: Path) -> Persona:
    (agents_dir / "タチコマ.md").write_text(VALID_PERSONA_CONTENT, encoding="utf-8")
    return load_persona("タチコマ", persona_dir=agents_dir)


# ---------------------------------------------------------------------------
# AC: register_prompt_filter でフィルタ登録 → format_prompt に反映
# ---------------------------------------------------------------------------


def test_register_custom_filter_output_appears(tachikoma_persona: Persona) -> None:
    """カスタムフィルタの返り値が format_prompt 出力に含まれること。"""
    def custom_fn(accumulated: str, ctx: dict[str, Any]) -> str:
        return accumulated + "CUSTOM_PREFIX_LINE\n\n"

    tachikoma_persona.register_prompt_filter("custom", custom_fn)
    result = tachikoma_persona.format_prompt("test")
    assert "CUSTOM_PREFIX_LINE" in result


def test_replace_datetime_filter(tachikoma_persona: Persona) -> None:
    """datetime フィルタを置換すると旧デフォルトの '現在日時:' 行が消えること。"""
    def new_fn(accumulated: str, ctx: dict[str, Any]) -> str:
        return accumulated + "REPLACED_DATETIME\n\n"

    tachikoma_persona.register_prompt_filter("datetime", new_fn)
    result = tachikoma_persona.format_prompt("test")
    assert "REPLACED_DATETIME" in result
    assert "現在日時:" not in result


@freeze_time("2026-04-23T10:00:00+09:00")
def test_default_datetime_filter_backward_compat(tachikoma_persona: Persona) -> None:
    """register_prompt_filter を一度も呼ばない場合、既存と同一の '現在日時:' 行が含まれること。"""
    result = tachikoma_persona.format_prompt("test")
    assert "現在日時: 2026-04-23 10:00:00 (JST)" in result


def test_multiple_filters_ordered_accumulation(tachikoma_persona: Persona) -> None:
    """複数フィルタが登録順に accumulated_prefix を積み上げること。"""
    calls: list[str] = []

    def filter_a(accumulated: str, ctx: dict[str, Any]) -> str:
        calls.append("a")
        return accumulated + "AAA\n\n"

    def filter_b(accumulated: str, ctx: dict[str, Any]) -> str:
        calls.append("b")
        assert "AAA" in accumulated  # a の出力が b に渡される
        return accumulated + "BBB\n\n"

    # datetime を差し替えてシンプルにする
    tachikoma_persona.register_prompt_filter("datetime", filter_a)
    tachikoma_persona.register_prompt_filter("extra", filter_b)
    result = tachikoma_persona.format_prompt("test")

    assert calls == ["a", "b"]
    assert "AAA" in result
    assert "BBB" in result
    aaa_pos = result.index("AAA")
    bbb_pos = result.index("BBB")
    assert aaa_pos < bbb_pos


def test_protocol_has_register_prompt_filter() -> None:
    """PersonaProtocol に register_prompt_filter が定義されていること。"""
    assert hasattr(PersonaProtocol, "register_prompt_filter")


def test_persona_satisfies_protocol(tachikoma_persona: Persona) -> None:
    """Persona が PersonaProtocol を満たすこと。"""
    assert isinstance(tachikoma_persona, PersonaProtocol)
