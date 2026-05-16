"""mltgnt.routing.triage

Slack トリアージ前処理ユーティリティ。
persona/triage.py から移動（Issue #911）。
"""
from __future__ import annotations

import json
import re

TRIAGE_PROFILE_MAX_CHARS = 6000


def extract_triage_section(markdown: str) -> str | None:
    """人物像 Markdown 内のトリアージ用セクション本文を返す。無ければ None。

    v2 形式の `## 軽量` を優先し、フォールバックとして v1 形式の `## トリアージ用` も探す。
    """
    m = re.search(r"^##\s+軽量\s*$", markdown, re.MULTILINE)
    if not m:
        m = re.search(r"^##\s+トリアージ用\s*$", markdown, re.MULTILINE)
    if not m:
        return None
    after = markdown[m.end():].lstrip("\n")
    m2 = re.search(r"^##\s+", after, re.MULTILINE)
    if m2:
        body = after[: m2.start()].rstrip()
    else:
        body = after.rstrip()
    return body if body else None


def prepare_profile_for_triage(profile_content: str | None, logger) -> str | None:
    """トリアージ用に人物像を短縮する。"""
    if not profile_content or not profile_content.strip():
        return None
    raw = profile_content.strip()
    section = extract_triage_section(raw)
    if section and section.strip():
        text = section.strip()
        source = "triage_section"
    else:
        text = raw
        source = "full_persona"
    orig_len = len(text)
    truncated = 0
    if len(text) > TRIAGE_PROFILE_MAX_CHARS:
        text = (
            text[:TRIAGE_PROFILE_MAX_CHARS].rstrip()
            + "\n…（以降省略。`## トリアージ用` セクションで要約を置くと安定します）"
        )
        truncated = 1
    logger.info(
        "[slack-triage] triage_profile source=%s original_chars=%d embedded_chars=%d truncated=%d",
        source,
        orig_len,
        len(text),
        truncated,
    )
    return text


def extract_json_object(text: str) -> dict | None:
    """LLM の stdout から JSON オブジェクトを1つ取り出す。

    - 空文字なら None
    - ``` で囲まれていればフェンス行を除去
    - 最初の { から最後の } までを json.loads でパース
    """
    s = text.strip()
    if not s:
        return None
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start: end + 1])
    except json.JSONDecodeError:
        return None
