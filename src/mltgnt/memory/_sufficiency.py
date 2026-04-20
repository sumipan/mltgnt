"""十分性判定モジュール."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

LlmCall = Callable[[str], str]

_PROMPT_TEMPLATE = """\
あなたは検索結果の十分性を判定するアシスタントです。

## ユーザーの質問
{query}

## 検索結果
{memory_excerpt}

## 指示
上記の検索結果だけでユーザーの質問に適切に回答できるか判定してください。

- 回答に十分な情報がある場合: 1行目に「SUFFICIENT」とだけ出力
- 不十分な場合: 1行目に「INSUFFICIENT」、2行目に不足情報を補うための検索クエリを出力

出力例（十分な場合）:
SUFFICIENT

出力例（不十分な場合）:
INSUFFICIENT
〇〇に関する具体的な設定値と手順"""


@dataclass(frozen=True)
class SufficiencyResult:
    """十分性判定の結果."""
    sufficient: bool
    rewritten_query: str | None  # sufficient=False の場合のみ非 None


def judge_sufficiency(
    query: str,
    memory_excerpt: str,
    llm_call: LlmCall,
) -> SufficiencyResult:
    """検索結果が質問に対して十分かを LLM で判定する.

    llm_call が例外を raise した場合は伝播させる（キャッチしない）。
    """
    prompt = _PROMPT_TEMPLATE.format(query=query, memory_excerpt=memory_excerpt)
    response = llm_call(prompt)

    lines = response.splitlines()
    if not lines:
        return SufficiencyResult(sufficient=True, rewritten_query=None)

    first_line = lines[0].strip()

    if first_line == "SUFFICIENT":
        return SufficiencyResult(sufficient=True, rewritten_query=None)

    if first_line == "INSUFFICIENT":
        rest = "\n".join(lines[1:]).strip()
        if not rest:
            # フェイルセーフ: 再検索クエリがない場合は十分とみなす
            return SufficiencyResult(sufficient=True, rewritten_query=None)
        return SufficiencyResult(sufficient=False, rewritten_query=rest)

    # parse failure → フェイルセーフ
    return SufficiencyResult(sufficient=True, rewritten_query=None)
