"""mltgnt.agent._parse — LLM レスポンスから JSON をパースする。"""
from __future__ import annotations

import json
import re


def _parse_json_response(raw: str) -> dict | None:
    """LLM の生テキストから {"tool": str, "args": dict} を抽出する。

    パース優先順位:
      1. ```json {...} ``` コードブロック内の JSON
      2. 最初の { から最後の } までの部分文字列

    "args" キーがない場合、"tool" 以外のキーをまとめて args として扱う（後方互換）。
    """
    # 1. コードブロック内 JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        # 2. 最初の { から最後の } まで
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        candidate = raw[start : end + 1]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    if "tool" not in data:
        return None

    # args キー後方互換
    if "args" not in data:
        args = {k: v for k, v in data.items() if k != "tool"}
        data = {"tool": data["tool"], "args": args}

    return data
