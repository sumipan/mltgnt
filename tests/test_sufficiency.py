"""
tests/test_sufficiency.py — _sufficiency.py のユニットテスト

TC1: SUFFICIENT 応答のパース
TC2: INSUFFICIENT + MEMORY 応答のパース
TC3: INSUFFICIENT + SKILL 応答のパース
TC4: INSUFFICIENT + 行不足（フェイルセーフ）
TC5: 空応答（フェイルセーフ）
TC6: 不明ソース（フェイルセーフ）
TC7: 不明フォーマット（フェイルセーフ）
TC8: LLM 例外の伝播
TC9: rewritten_query 互換プロパティ（INSUFFICIENT/MEMORY）
TC10: rewritten_query 互換プロパティ（SUFFICIENT）
"""
from __future__ import annotations

import logging
import pytest
from mltgnt.memory._sufficiency import (
    judge_sufficiency,
)


def _llm(response: str):
    """固定応答を返す llm_call ファクトリ"""
    def call(_prompt: str) -> str:
        return response
    return call


# ---------------------------------------------------------------------------
# TC1: SUFFICIENT
# ---------------------------------------------------------------------------


def test_tc1_sufficient():
    result = judge_sufficiency("質問", "情報", _llm("SUFFICIENT"))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC2: INSUFFICIENT + MEMORY
# ---------------------------------------------------------------------------


def test_tc2_insufficient_memory():
    response = "INSUFFICIENT\nMEMORY\nプロジェクト進捗"
    result = judge_sufficiency("質問", "情報", _llm(response))
    assert result.sufficient is False
    assert result.action is not None
    assert result.action.source == "memory"
    assert result.action.query == "プロジェクト進捗"


# ---------------------------------------------------------------------------
# TC3: INSUFFICIENT + SKILL
# ---------------------------------------------------------------------------


def test_tc3_insufficient_skill():
    response = "INSUFFICIENT\nSKILL\nデプロイ手順"
    result = judge_sufficiency("質問", "情報", _llm(response))
    assert result.sufficient is False
    assert result.action is not None
    assert result.action.source == "skill"
    assert result.action.query == "デプロイ手順"


# ---------------------------------------------------------------------------
# TC4: INSUFFICIENT + 行不足（フェイルセーフ）
# ---------------------------------------------------------------------------


def test_tc4_insufficient_missing_lines(caplog):
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("質問", "情報", _llm("INSUFFICIENT\nMEMORY"))
    assert result.sufficient is True
    assert result.action is None
    assert any("missing" in r.message or "SUFFICIENT" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TC5: 空応答（フェイルセーフ）
# ---------------------------------------------------------------------------


def test_tc5_empty_response(caplog):
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("質問", "情報", _llm(""))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC6: 不明ソース（フェイルセーフ）
# ---------------------------------------------------------------------------


def test_tc6_unknown_source(caplog):
    response = "INSUFFICIENT\nWEB\n検索クエリ"
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("質問", "情報", _llm(response))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC7: 不明フォーマット（フェイルセーフ）
# ---------------------------------------------------------------------------


def test_tc7_unexpected_format(caplog):
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("質問", "情報", _llm("MAYBE"))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC8: LLM 例外の伝播
# ---------------------------------------------------------------------------


def test_tc8_llm_exception():
    def failing_llm(_prompt: str) -> str:
        raise RuntimeError("API error")

    with pytest.raises(RuntimeError, match="API error"):
        judge_sufficiency("質問", "情報", failing_llm)


# ---------------------------------------------------------------------------
# TC9: rewritten_query 互換プロパティ（INSUFFICIENT/MEMORY）
# ---------------------------------------------------------------------------


def test_tc9_rewritten_query_insufficient():
    response = "INSUFFICIENT\nMEMORY\n先週の進捗"
    result = judge_sufficiency("質問", "情報", _llm(response))
    assert result.rewritten_query == "先週の進捗"


# ---------------------------------------------------------------------------
# TC10: rewritten_query 互換プロパティ（SUFFICIENT）
# ---------------------------------------------------------------------------


def test_tc10_rewritten_query_sufficient():
    result = judge_sufficiency("質問", "情報", _llm("SUFFICIENT"))
    assert result.rewritten_query is None
