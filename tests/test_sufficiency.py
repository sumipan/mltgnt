"""
tests/test_sufficiency.py — judge_sufficiency() の単体テスト

TC6: LLM 応答のパース失敗 → フェイルセーフ（sufficient=True）
TC7: llm_call が例外を送出 → 例外が伝播する
TC8: rewritten_query が空文字 → フェイルセーフ（sufficient=True）
"""
from mltgnt.memory._sufficiency import judge_sufficiency
import pytest


def test_tc6_parse_failure_returns_sufficient():
    result = judge_sufficiency("query", "excerpt", lambda p: "よくわかりません")
    assert result.sufficient is True
    assert result.rewritten_query is None


def test_tc6_sufficient_response():
    result = judge_sufficiency("query", "excerpt", lambda p: "SUFFICIENT")
    assert result.sufficient is True
    assert result.rewritten_query is None


def test_tc6_insufficient_with_query():
    response = "INSUFFICIENT\nDB接続のホスト名とポート設定"
    result = judge_sufficiency("query", "excerpt", lambda p: response)
    assert result.sufficient is False
    assert result.rewritten_query == "DB接続のホスト名とポート設定"


def test_tc7_llm_raises_propagates():
    def failing_llm(prompt):
        raise RuntimeError("LLM unavailable")
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        judge_sufficiency("query", "excerpt", failing_llm)


def test_tc8_insufficient_empty_rewritten_query():
    response = "INSUFFICIENT\n"
    result = judge_sufficiency("query", "excerpt", lambda p: response)
    assert result.sufficient is True
    assert result.rewritten_query is None


def test_tc8_insufficient_only_whitespace():
    response = "INSUFFICIENT\n   "
    result = judge_sufficiency("query", "excerpt", lambda p: response)
    assert result.sufficient is True
    assert result.rewritten_query is None


def test_tc8_insufficient_no_newline():
    result = judge_sufficiency("query", "excerpt", lambda p: "INSUFFICIENT")
    assert result.sufficient is True
    assert result.rewritten_query is None


def test_tc6_empty_response_returns_sufficient():
    result = judge_sufficiency("query", "excerpt", lambda p: "")
    assert result.sufficient is True
    assert result.rewritten_query is None
