"""mltgnt.agent._parse — LLM レスポンスから JSON をパースする。"""
from __future__ import annotations

import json
import logging
import re

_logger = logging.getLogger(__name__)


def _validate_tool_entry(item: dict) -> dict | None:
    if "tool" not in item or "args" not in item:
        return None
    if not isinstance(item["args"], dict):
        return None
    if "thought" not in item:
        _logger.warning("thought key missing in LLM response; proceeding without thought")
    return item


def _parse_json_response(raw: str) -> dict | list[dict] | None:
    """LLM の生テキストからツール呼び出し JSON を抽出する。

    パース優先順位:
      1. ```json {...} ``` コードブロック内の JSON
      2. 最初の { から最後の } までの部分文字列

    単一形式: {"tool": str, "args": dict} → dict
    複数形式: {"tools": [{"tool": str, "args": dict}, ...]} → list[dict]
    "args" キーは必須。"thought" キーは省略可能（欠落時は WARN ログ）。
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

    if "tools" in data:
        tools = data["tools"]
        if not isinstance(tools, list):
            return None
        validated: list[dict] = []
        for item in tools:
            if not isinstance(item, dict):
                return None
            entry = _validate_tool_entry(item)
            if entry is None:
                return None
            validated.append(entry)
        return validated

    return _validate_tool_entry(data)
