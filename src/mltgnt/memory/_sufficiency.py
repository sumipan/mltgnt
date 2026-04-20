"""
mltgnt.memory._sufficiency — LLM による十分性判定。

設計: Issue #197 Phase 2/3
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

_log = logging.getLogger(__name__)

SUFFICIENT_TOKEN = "SUFFICIENT"
INSUFFICIENT_TOKEN = "INSUFFICIENT"
MEMORY_TOKEN = "MEMORY"
SKILL_TOKEN = "SKILL"

__all__ = [
    "SearchAction",
    "SufficiencyResult",
    "judge_sufficiency",
]


@dataclass(frozen=True)
class SearchAction:
    """LLM が決定した次の検索アクション"""

    source: Literal["memory", "skill"]
    query: str


@dataclass(frozen=True)
class SufficiencyResult:
    """十分性判定の結果"""

    sufficient: bool
    action: SearchAction | None  # sufficient=True なら None

    @property
    def rewritten_query(self) -> str | None:
        """Phase 2 互換プロパティ。action.query へ委譲する。"""
        if self.action is None:
            return None
        return self.action.query


def _build_prompt(query: str, collected_text: str) -> str:
    return f"""あなたは情報収集の十分性を判断するアシスタントです。

ユーザーの質問: {query}

収集済みの情報:
{collected_text}

上記の情報がユーザーの質問に答えるのに十分かどうか判断してください。

十分な場合は1行目に「SUFFICIENT」とだけ出力してください。
不十分な場合は以下の形式で出力してください:
1行目: 「INSUFFICIENT」
2行目: 検索ソース（「MEMORY」または「SKILL」）
3行目: 不足情報を補うための検索クエリ"""


def judge_sufficiency(
    query: str,
    collected_text: str,
    llm_call: Callable[[str], str],
) -> SufficiencyResult:
    """LLM を使って収集済み情報の十分性を判断する。

    LLM 応答のパースに失敗した場合は sufficient=True として扱う（フェイルセーフ）。
    LLM 呼び出し自体が例外を投げた場合は、その例外を呼び出し元に伝播させる。

    Args:
        query: ユーザーの質問
        collected_text: 収集済み情報のテキスト
        llm_call: LLM を呼び出す関数

    Returns:
        SufficiencyResult
    """
    prompt = _build_prompt(query, collected_text)
    response = llm_call(prompt)

    lines = [line.strip() for line in response.strip().splitlines()]
    if not lines:
        _log.warning("judge_sufficiency: empty response, treating as SUFFICIENT")
        return SufficiencyResult(sufficient=True, action=None)

    first = lines[0]

    if first == SUFFICIENT_TOKEN:
        return SufficiencyResult(sufficient=True, action=None)

    if first == INSUFFICIENT_TOKEN:
        if len(lines) < 3:
            _log.warning(
                "judge_sufficiency: INSUFFICIENT response missing source/query lines, "
                "treating as SUFFICIENT"
            )
            return SufficiencyResult(sufficient=True, action=None)
        source_raw = lines[1].upper()
        requery = lines[2]
        if source_raw == MEMORY_TOKEN:
            source: Literal["memory", "skill"] = "memory"
        elif source_raw == SKILL_TOKEN:
            source = "skill"
        else:
            _log.warning(
                "judge_sufficiency: unknown source '%s', treating as SUFFICIENT",
                source_raw,
            )
            return SufficiencyResult(sufficient=True, action=None)
        return SufficiencyResult(
            sufficient=False,
            action=SearchAction(source=source, query=requery),
        )

    # 不明なフォーマット → フェイルセーフ
    _log.warning(
        "judge_sufficiency: unexpected response format '%s', treating as SUFFICIENT",
        first,
    )
    return SufficiencyResult(sufficient=True, action=None)
