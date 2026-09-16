"""mltgnt.persona.formatter — 口調整形・ペルソナ接頭辞（#3318）。

媒体・タスクパス非依存。SDK / エンジン / 成果物パス語彙を持たない。
"""
from __future__ import annotations

import re


def extract_persona_block_after_meta_headers(s: str) -> str:
    """「〇〇としての応答（標準出力相当）」メタ見出しの直後だけを本文として採用する。"""
    text = s.strip()
    if not text:
        return text
    needles = (
        "としての応答（標準出力相当）",
        "としての応答（stdout相当）",
        "としての応答",
    )
    start_content = -1
    for n in needles:
        i = text.find(n)
        if i == -1:
            continue
        line_start = text.rfind("\n", 0, i) + 1
        line = text[line_start : i + len(n)]
        if n == "としての応答" and len(line.strip()) > 48:
            continue
        if "としての応答" not in line:
            continue
        nl = text.find("\n", i + len(n))
        if nl == -1:
            start_content = len(text)
        else:
            start_content = nl + 1
        break
    if start_content == -1:
        return text
    after = text[start_content:].lstrip("\n")
    if not after:
        return text
    end_m = re.search(r"\n-{3,}\s*\n\s*（以上）", after)
    if end_m:
        after = after[: end_m.start()].rstrip()
    return after


def dedupe_persona_prefix(body: str) -> str:
    """同一 result に応答と追記が連結され、同趣旨の本文が二重になる場合の除去。"""
    body = body.strip()
    if not body:
        return body
    opener = re.compile(r"^今週（[^）]{1,80}）の計画[、,]", re.MULTILINE)
    matches = list(opener.finditer(body))
    if len(matches) >= 2:
        return body[matches[-1].start() :].strip()
    return body


_PERSONA_CUT_RE = re.compile(r"\n\n\S+口調の本文は")


def format_persona_body(text: str) -> str:
    """ペルソナ向け整形: メタ見出し抽出 → 接頭辞 dedupe → 口調メタ切断。"""
    s = text.strip()
    if not s:
        return ""
    s = extract_persona_block_after_meta_headers(s)
    m = _PERSONA_CUT_RE.search(s)
    if m:
        s = s[: m.start()].rstrip()
    s = dedupe_persona_prefix(s)
    return s.strip()
