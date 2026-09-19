"""mltgnt.routing.triage

Slack triage preprocessing utilities.
Moved from persona/triage.py (Issue #911).
"""

from __future__ import annotations

import json
import re

TRIAGE_PROFILE_MAX_CHARS = 6000


def extract_triage_section(markdown: str) -> str | None:
    """Return the triage section body from persona Markdown, or None.

    Prefer v2 ``## Light``; fall back to v1 ``## Triage``. Legacy localized
    headings remain readable for existing persona files.
    """
    m = re.search(r"^##\s+Light\s*$", markdown, re.MULTILINE)
    if not m:
        m = re.search(r"^##\s+Triage\s*$", markdown, re.MULTILINE)
    if not m:
        m = re.search(
            r"^##\s+\N{CJK UNIFIED IDEOGRAPH-8EFD}\N{CJK UNIFIED IDEOGRAPH-91CF}\s*$",
            markdown,
            re.MULTILINE,
        )
    if not m:
        m = re.search(
            r"^##\s+\N{KATAKANA LETTER TO}\N{KATAKANA LETTER RI}\N{KATAKANA LETTER A}\N{KATAKANA-HIRAGANA PROLONGED SOUND MARK}\N{KATAKANA LETTER ZI}\N{CJK UNIFIED IDEOGRAPH-7528}\s*$",
            markdown,
            re.MULTILINE,
        )
    if not m:
        return None
    after = markdown[m.end() :].lstrip("\n")
    m2 = re.search(r"^##\s+", after, re.MULTILINE)
    if m2:
        body = after[: m2.start()].rstrip()
    else:
        body = after.rstrip()
    return body if body else None


def prepare_profile_for_triage(profile_content: str | None, logger) -> str | None:
    """Shorten a persona profile for triage."""
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
            text[:TRIAGE_PROFILE_MAX_CHARS].rstrip() + "\n…(truncated. Place a summary under `## Triage` for stability)"
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
    """Extract one JSON object from LLM stdout.

    - None on empty string
    - Strip fence lines if wrapped in ```
    - Parse from the first { to the last } via json.loads
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
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
