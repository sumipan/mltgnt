"""mltgnt.agent._parse — parse JSON from an LLM response."""
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
    """Extract tool-call JSON from raw LLM text.

    Parse priority:
      1. JSON inside a ```json {...} ``` code block
      2. Substring from the first { to the last }

    Single form: {"tool": str, "args": dict} → dict
    Multi form: {"tools": [{"tool": str, "args": dict}, ...]} → list[dict]
    "args" is required. "thought" is optional (WARN log when missing).
    """
    # 1. JSON inside a code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        # 2. From the first { to the last }
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
