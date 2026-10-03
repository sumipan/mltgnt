"""mltgnt.config.language — language-pack dataclass for locale-specific vocabulary."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["LanguagePack", "EN", "JA", "get_language_pack", "set_language_pack"]


@dataclass(frozen=True)
class LanguagePack:
    """Locale-specific vocabulary injected into processing functions.

    All fields that previously held hardcoded Japanese strings are collected
    here so callers can substitute a different locale by passing a custom pack.
    None-default arguments on each consuming function fall back to the module-level
    JA constant, keeping existing call sites compatible.
    """

    work_request_markers: tuple[str, ...]
    create_request_markers: tuple[str, ...]
    deferred_patterns: tuple[re.Pattern[str], ...]
    compress_prompt_template: str
    v21_required_sections: tuple[str, ...]
    v21_example_section: str
    meta_header_needles: tuple[str, ...]
    dedupe_opener_re: re.Pattern[str]
    persona_cut_re: re.Pattern[str]
    # Persona stems to exclude from listing by default (e.g. sample/template files)
    exclude_stems: frozenset[str]
    # Japanese text intentionally kept for CJK processing test
    persona_end_re: re.Pattern[str] = re.compile(r"\n-{3,}\s*\n\s*（以上）")
    cancel_words: frozenset[str] = frozenset({"キャンセル", "止めて", "cancel", "stop"})
    composite_header: str = "処理中に以下の発言がありました。これらを踏まえて対応してください。"
    composite_cancel_suffix: str = (
        "※ 中止指示が含まれています。現在の作業を中止し、中止した旨を報告してください。"
    )
    # Media-layer vocabulary (mltgnt.media)
    approval_words: frozenset[str] = frozenset({"OK", "ok", "yes", "approve"})
    # Status value (mltgnt.interfaces.media.Status) -> display label
    status_labels: dict[str, str] = field(
        hash=False,
        default_factory=lambda: {
            "received": "Received",
            "working": "Working",
            "done": "Done",
            "failed": "Failed",
            "cancelled": "Cancelled",
        }
    )
    enqueue_failed_text: str = "Failed to enqueue the request. Please try again later."
    progress_line_pattern: re.Pattern[str] = re.compile(r"^\s*\[progress\]:?\s*(?P<text>.+)$", re.MULTILINE)
    # Memory-tool trigger words (mltgnt.memory.tools); empty by default, injected by callers
    remember_trigger_words: frozenset[str] = frozenset()
    forget_trigger_words: frozenset[str] = frozenset()


# Japanese text intentionally kept for CJK processing test
_JA_WORK_REQUEST_MARKERS: tuple[str, ...] = (
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
_JA_CREATE_REQUEST_MARKERS: tuple[str, ...] = (
    "つくって",
    "作って",
    "新規作成",
    "新しく",
    # Japanese text intentionally kept for CJK processing test
    "作りたい",
    "ファイルつくって",
)

# Japanese text intentionally kept for CJK processing test
_JA_DEFERRED_PATTERNS: tuple[re.Pattern[str], ...] = (
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

# Japanese text intentionally kept for CJK processing test
_JA_COMPRESS_PROMPT_TEMPLATE = (
    "以下のペルソナの重量ブロックから、v2.1 形式の軽量ブロックを生成してください。\n\n"
    "## 出力形式\n\n"
    # Japanese text intentionally kept for CJK processing test
    "1. リード文（1〜2文で人物の本質を要約）\n"
    "2. 必須サブセクション（太字見出し）:\n"
    "   - **口調** — 話し方の特徴\n"
    # Japanese text intentionally kept for CJK processing test
    "   - **価値観** — 大切にしていること\n"
    "   - **好意的反応** — どんなとき喜ぶか\n"
    "   - **引っかかる** — どんなとき不快になるか\n"
    # Japanese text intentionally kept for CJK processing test
    "3. 推奨（任意）:\n"
    "   - **発言例** — 引用ブロック（> ）形式で1〜3例\n\n"
    "## 制約\n"
    # Japanese text intentionally kept for CJK processing test
    "- 1500文字以内（厳守）\n"
    "- 太字見出しは上記の名前をそのまま使う\n"
    # Japanese text intentionally kept for CJK processing test
    "- 発言例がある場合は必ず引用ブロック形式にする\n\n"
    "## 重量ブロック:\n"
    "{heavy_text}"
)

# Japanese text intentionally kept for CJK processing test
_JA_V21_REQUIRED_SECTIONS: tuple[str, ...] = (
    "**口調**",
    "**価値観**",
    "**好意的反応**",
    "**引っかかる**",
)

# Japanese text intentionally kept for CJK processing test
_JA_META_HEADER_NEEDLES: tuple[str, ...] = (
    "としての応答（標準出力相当）",
    "としての応答（stdout相当）",
    "としての応答",
)

# deprecated: to be removed in #4349. Migrate to EN + set_language_pack.
JA = LanguagePack(
    work_request_markers=_JA_WORK_REQUEST_MARKERS,
    create_request_markers=_JA_CREATE_REQUEST_MARKERS,
    deferred_patterns=_JA_DEFERRED_PATTERNS,
    compress_prompt_template=_JA_COMPRESS_PROMPT_TEMPLATE,
    v21_required_sections=_JA_V21_REQUIRED_SECTIONS,
    # Japanese text intentionally kept for CJK processing test
    v21_example_section="**発言例**",
    meta_header_needles=_JA_META_HEADER_NEEDLES,
    # Japanese text intentionally kept for CJK processing test
    dedupe_opener_re=re.compile(r"^今週（[^）]{1,80}）の計画[、,]", re.MULTILINE),
    # Japanese text intentionally kept for CJK processing test
    persona_cut_re=re.compile(r"\n\n\S+口調の本文は"),
    # Japanese text intentionally kept for CJK processing test
    exclude_stems=frozenset({"\u30b5\u30f3\u30d7\u30eb"}),
)

_EN_WORK_REQUEST_MARKERS: tuple[str, ...] = (
    "please",
    "could you",
    "can you",
    "fix",
    "update",
    "create",
    "make",
    "write",
    "brush up",
)

_EN_CREATE_REQUEST_MARKERS: tuple[str, ...] = (
    "create",
    "make a new",
    "new file",
    "write a new",
)

_EN_DEFERRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"will do (it )?later", re.IGNORECASE),
    re.compile(r"\bhold on\b", re.IGNORECASE),
    re.compile(r"\blater\b", re.IGNORECASE),
    re.compile(r"\bI'll (get|work) on it\b", re.IGNORECASE),
    re.compile(r"\basked [^\s]{1,24} to\b", re.IGNORECASE),
)

_EN_COMPRESS_PROMPT_TEMPLATE = (
    "Generate a v2.1 light block from the persona's heavy block below.\n\n"
    "## Output format\n\n"
    "1. Lead sentence (summarize the essence of the persona in 1-2 sentences)\n"
    "2. Required subsections (bold headings):\n"
    "   - **Tone** - characteristics of how they speak\n"
    "   - **Values** - what they care about\n"
    "   - **Positive reactions** - when they are pleased\n"
    "   - **Friction** - when they are bothered\n"
    "3. Recommended (optional):\n"
    "   - **Examples** - 1-3 examples as block quotes (> )\n\n"
    "## Constraints\n"
    "- At most 1500 characters (strict)\n"
    "- Use the bold heading names above exactly as written\n"
    "- If examples are included, always format them as block quotes\n\n"
    "## Heavy block:\n"
    "{heavy_text}"
)

_EN_V21_REQUIRED_SECTIONS: tuple[str, ...] = (
    "**Tone**",
    "**Values**",
    "**Positive reactions**",
    "**Friction**",
)

_EN_META_HEADER_NEEDLES: tuple[str, ...] = (
    "response (stdout)",
    "'s response",
)

EN = LanguagePack(
    work_request_markers=_EN_WORK_REQUEST_MARKERS,
    create_request_markers=_EN_CREATE_REQUEST_MARKERS,
    deferred_patterns=_EN_DEFERRED_PATTERNS,
    compress_prompt_template=_EN_COMPRESS_PROMPT_TEMPLATE,
    v21_required_sections=_EN_V21_REQUIRED_SECTIONS,
    v21_example_section="**Examples**",
    meta_header_needles=_EN_META_HEADER_NEEDLES,
    dedupe_opener_re=re.compile(r"^This week's plan \([^)]{1,80}\)[,:]", re.MULTILINE),
    persona_cut_re=re.compile(r"\n\n\S+ tone body:"),
    exclude_stems=frozenset({"sample"}),
    persona_end_re=re.compile(r"\n-{3,}\s*\n\s*\(end\)"),
    cancel_words=frozenset({"cancel", "stop"}),
    composite_header=(
        "The following messages arrived while you were working. Take them into account."
    ),
    composite_cancel_suffix=(
        "Note: a stop request is included. Stop the current work and report that you stopped."
    ),
    approval_words=frozenset({"OK", "ok", "yes", "approve"}),
    status_labels={
        "received": "Received",
        "working": "Working",
        "done": "Done",
        "failed": "Failed",
        "cancelled": "Cancelled",
    },
    enqueue_failed_text="Failed to enqueue the request. Please try again later.",
    progress_line_pattern=re.compile(r"^\s*\[progress\]:?\s*(?P<text>.+)$", re.MULTILINE),
    remember_trigger_words=frozenset(),
    forget_trigger_words=frozenset(),
)

_current: LanguagePack = EN


def get_language_pack() -> LanguagePack:
    """Return the current language pack (EN until set_language_pack is called)."""
    return _current


def set_language_pack(pack: LanguagePack) -> None:
    """Replace the current language pack. Raises TypeError for non-LanguagePack values."""
    global _current
    if not isinstance(pack, LanguagePack):
        raise TypeError(f"pack must be a LanguagePack, got {type(pack).__name__}")
    _current = pack
