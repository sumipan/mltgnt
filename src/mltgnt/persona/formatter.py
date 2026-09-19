"""mltgnt.persona.formatter — tone formatting and persona prefixes (#3318).

Media- and task-path agnostic. No SDK / engine / artifact-path vocabulary.
"""
from __future__ import annotations

from mltgnt.config.language import JA, LanguagePack


def extract_persona_block_after_meta_headers(s: str, pack: LanguagePack | None = None) -> str:
    """Keep only the body after a 'response as … (stdout-equivalent)' meta heading."""
    _pack = pack or JA
    text = s.strip()
    if not text:
        return text
    needles = _pack.meta_header_needles
    # The last (most general) needle needs a line-length guard to avoid false positives.
    short_needle = needles[-1] if needles else None
    start_content = -1
    for n in needles:
        i = text.find(n)
        if i == -1:
            continue
        line_start = text.rfind("\n", 0, i) + 1
        line = text[line_start : i + len(n)]
        if n == short_needle and len(line.strip()) > 48:
            continue
        if short_needle and short_needle not in line:
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
    end_m = _pack.persona_end_re.search(after)
    if end_m:
        after = after[: end_m.start()].rstrip()
    return after


def dedupe_persona_prefix(body: str, pack: LanguagePack | None = None) -> str:
    """Drop duplicate same-intent bodies when a reply and append are concatenated."""
    body = body.strip()
    if not body:
        return body
    opener = (pack or JA).dedupe_opener_re
    matches = list(opener.finditer(body))
    if len(matches) >= 2:
        return body[matches[-1].start() :].strip()
    return body


def format_persona_body(text: str, pack: LanguagePack | None = None) -> str:
    """Persona formatting: meta-heading extract → prefix dedupe → tone-meta cut."""
    _pack = pack or JA
    s = text.strip()
    if not s:
        return ""
    s = extract_persona_block_after_meta_headers(s, pack=_pack)
    m = _pack.persona_cut_re.search(s)
    if m:
        s = s[: m.start()].rstrip()
    s = dedupe_persona_prefix(s, pack=_pack)
    return s.strip()
