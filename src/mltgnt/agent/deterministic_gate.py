"""決定論ゲート（副作用なし・媒体非依存）（#3318）。

依頼表現と成果物参照の共起判定、および直接応答における
非同期作業約束の検出を純関数で行う。
"""
from __future__ import annotations

import re
import unicodedata

_WORK_REQUEST_MARKERS: tuple[str, ...] = (
    "して",
    "お願い",
    "もらえる",
    "ブラッシュアップ",
    "修正",
    "更新",
    "作成",
    "つくって",
    "作って",
    "新規",
    "作りたい",
)

_CREATE_REQUEST_MARKERS: tuple[str, ...] = (
    "つくって",
    "作って",
    "新規作成",
    "新しく",
    "作りたい",
    "ファイルつくって",
)

_ARTIFACT_EXTS = ("md", "txt", "docx", "xlsx", "pptx", "pdf", "csv", "tsv")
_EXT_ALT = "|".join(_ARTIFACT_EXTS)

_URL_RE = re.compile(r"https?://[^\s<>\[\]()（）]+")
_FILE_RE = re.compile(
    rf"(?:[^\s/]+/)*[^\s/]+\.(?:{_EXT_ALT})",
    re.IGNORECASE,
)

_DEFERRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"待ってて"),
    re.compile(r"ちょっと待(?!ってる)"),
    re.compile(r"やっておく"),
    re.compile(r"後で"),
    re.compile(r"いま.{0,30}ている"),
    re.compile(r"進めておく"),
    re.compile(r"[^\s、。]{1,12}(?:に|へ)お願いして(?!くれた|おいた|もらった)"),
    re.compile(r"[^\s、。]{1,12}(?:に|へ)(?:依頼|頼んで)(?!くれた|おいた|もらった)"),
    re.compile(r"[^\s、。]{1,12}の担当だから"),
    re.compile(r"——+\s*[^\s]{1,12}への依頼\s*——+"),
)


def extract_artifact_references(text: str) -> tuple[str, ...]:
    """URL と成果物ファイル参照を出現順・重複なしで返す。"""
    if not text:
        return ()

    spans: list[tuple[int, int, str]] = []
    for m in _URL_RE.finditer(text):
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
    """依頼表現を含むか（成果物との共起判定は呼び出し側）。"""
    if not text:
        return False
    return any(marker in text for marker in _WORK_REQUEST_MARKERS)


def is_create_request(text: str) -> bool:
    """新規作成依頼表現を含むか（NFKC 正規化後に判定）。"""
    if not text:
        return False
    normalized = unicodedata.normalize("NFKC", text)
    return any(marker in normalized for marker in _CREATE_REQUEST_MARKERS)


def should_force_delegate(text: str) -> bool:
    """依頼表現と成果物参照が共起すれば True。"""
    return has_work_request(text) and bool(extract_artifact_references(text))


should_preempt_delegate = should_force_delegate


def match_deferred_promise(reply: str) -> str | None:
    """非同期作業約束に一致した部分文字列を返す。非一致は None。"""
    if not reply or not reply.strip():
        return None
    normalized = unicodedata.normalize("NFKC", reply)
    for pat in _DEFERRED_PATTERNS:
        m = pat.search(normalized)
        if m is not None:
            return m.group(0)
    return None
