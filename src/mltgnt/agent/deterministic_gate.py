"""Deterministic gate (no side effects, medium-agnostic) (#3318).

Pure functions for co-occurrence of request phrasing with artifact
references, and for detecting async work promises in direct replies.
"""
from __future__ import annotations

import re
import unicodedata

# Japanese text intentionally kept for CJK processing test
_WORK_REQUEST_MARKERS: tuple[str, ...] = (
    "して",
    "お願い",
    "もらえる",
    "ブラッシュアップ",
    # Japanese text intentionally kept for CJK processing test
    "修正",
    "更新",
    "作成",
    "つくって",
    "作って",
    # Japanese text intentionally kept for CJK processing test
    "新規",
    "作りたい",
)

# Japanese text intentionally kept for CJK processing test
_CREATE_REQUEST_MARKERS: tuple[str, ...] = (
    "つくって",
    "作って",
    "新規作成",
    "新しく",
    # Japanese text intentionally kept for CJK processing test
    "作りたい",
    "ファイルつくって",
)

_ARTIFACT_EXTS = ("md", "txt", "docx", "xlsx", "pptx", "pdf", "csv", "tsv")
_EXT_ALT = "|".join(_ARTIFACT_EXTS)

# Japanese text intentionally kept for CJK processing test
_URL_RE = re.compile(r"https?://[^\s<>\[\]()（）]+")
_FILE_RE = re.compile(
    rf"(?:[^\s/]+/)*[^\s/]+\.(?:{_EXT_ALT})",
    re.IGNORECASE,
)

# Japanese text intentionally kept for CJK processing test
_DEFERRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"待ってて"),
    re.compile(r"ちょっと待(?!ってる)"),
    re.compile(r"やっておく"),
    re.compile(r"後で"),
    # Japanese text intentionally kept for CJK processing test
    re.compile(r"いま.{0,30}ている"),
    re.compile(r"進めておく"),
    re.compile(r"[^\s、。]{1,12}(?:に|へ)お願いして(?!くれた|おいた|もらった)"),
    re.compile(r"[^\s、。]{1,12}(?:に|へ)(?:依頼|頼んで)(?!くれた|おいた|もらった)"),
    # Japanese text intentionally kept for CJK processing test
    re.compile(r"[^\s、。]{1,12}の担当だから"),
    re.compile(r"——+\s*[^\s]{1,12}への依頼\s*——+"),
)


def extract_artifact_references(text: str) -> tuple[str, ...]:
    """Return URL and artifact file refs in order, without duplicates."""
    if not text:
        return ()

    spans: list[tuple[int, int, str]] = []
    for m in _URL_RE.finditer(text):
        # Japanese text intentionally kept for CJK processing test
        spans.append((m.start(), m.end(), m.group(0).rstrip(".,;:、。")))

    url_ranges = [(s, e) for s, e, _ in spans]
    for m in _FILE_RE.finditer(text):
        start, end = m.start(), m.end()
        if any(us <= start < ue or us < end <= ue for us, ue in url_ranges):
            continue
        spans.append((start, end, m.group(0)))

    spans.sort(key=lambda t: t[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, _, value in spans:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def has_work_request(text: str) -> bool:
    """Whether text contains request phrasing (co-occurrence is caller's job)."""
    if not text:
        return False
    return any(marker in text for marker in _WORK_REQUEST_MARKERS)


def is_create_request(text: str) -> bool:
    """Whether text contains create-request phrasing (after NFKC normalize)."""
    if not text:
        return False
    normalized = unicodedata.normalize("NFKC", text)
    return any(marker in normalized for marker in _CREATE_REQUEST_MARKERS)


def should_force_delegate(text: str) -> bool:
    """True when request phrasing co-occurs with an artifact reference."""
    return has_work_request(text) and bool(extract_artifact_references(text))


should_preempt_delegate = should_force_delegate


def match_deferred_promise(reply: str) -> str | None:
    """Return the matched async-work-promise substring, or None."""
    if not reply or not reply.strip():
        return None
    normalized = unicodedata.normalize("NFKC", reply)
    for pat in _DEFERRED_PATTERNS:
        m = pat.search(normalized)
        if m is not None:
            return m.group(0)
    return None
