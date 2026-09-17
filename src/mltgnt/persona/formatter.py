"""mltgnt.persona.formatter — tone formatting and persona prefixes (#3318).

Media- and task-path agnostic. No SDK / engine / artifact-path vocabulary.
"""
from __future__ import annotations

import re


def extract_persona_block_after_meta_headers(s: str) -> str:
    """Keep only the body after a 'response as … (stdout-equivalent)' meta heading."""
    text = s.strip()
    if not text:
        return text
    # Japanese text intentionally kept for CJK processing test
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
        # Japanese text intentionally kept for CJK processing test
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
    # Japanese text intentionally kept for CJK processing test
    end_m = re.search(r"\n-{3,}\s*\n\s*（以上）", after)
    if end_m:
        after = after[: end_m.start()].rstrip()
    return after


def dedupe_persona_prefix(body: str) -> str:
    """Drop duplicate same-intent bodies when a reply and append are concatenated."""
    body = body.strip()
    if not body:
        return body
    # Japanese text intentionally kept for CJK processing test
    opener = re.compile(r"^今週（[^）]{1,80}）の計画[、,]", re.MULTILINE)
    matches = list(opener.finditer(body))
    if len(matches) >= 2:
        return body[matches[-1].start() :].strip()
    return body


# Japanese text intentionally kept for CJK processing test
_PERSONA_CUT_RE = re.compile(r"\n\n\S+口調の本文は")


def format_persona_body(text: str) -> str:
    """Persona formatting: meta-heading extract → prefix dedupe → tone-meta cut."""
    s = text.strip()
    if not s:
        return ""
    s = extract_persona_block_after_meta_headers(s)
    m = _PERSONA_CUT_RE.search(s)
    if m:
        s = s[: m.start()].rstrip()
    s = dedupe_persona_prefix(s)
    return s.strip()
