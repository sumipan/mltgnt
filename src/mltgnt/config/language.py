"""mltgnt.config.language — language-pack dataclass for locale-specific vocabulary."""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["LanguagePack", "JA"]


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
    exclude_stems=frozenset({"サンプル"}),
)
