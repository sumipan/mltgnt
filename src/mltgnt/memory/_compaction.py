"""
mltgnt.memory._compaction — メモリコンパクション。

設計: Issue #123, #137 (per-section LLM calls)
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

from mltgnt.memory._format import MemorySections, parse_memory, assemble_memory

_log = logging.getLogger(__name__)

LlmCall = Callable[[str], str]

__all__ = [
    "LlmCallError",
    "CompactionResult",
    "needs_compaction",
    "compact",
]


class LlmCallError(RuntimeError):
    """llm_call の実行中に発生したエラーをラップする例外。"""


@dataclass(frozen=True)
class CompactionResult:
    before_bytes: int
    after_bytes: int
    summary: str
    warnings: list[str] = field(default_factory=list)


def needs_compaction(config: "MemoryConfig", persona_stem: str) -> bool:
    """メモリファイルがコンパクション閾値を超えているか判定する。"""
    from mltgnt.memory import memory_file_path
    path = memory_file_path(config, persona_stem)
    if not path.exists():
        return False
    return path.stat().st_size >= config.compact_threshold_bytes


def _build_section_prompt(section_text: str, target_bytes: int) -> str:
    """個別セクション用のコンパクションプロンプトを生成する。"""
    return (
        "以下の文章を要約・圧縮してください。"
        "見出しは不要です。本文のみ出力してください。"
        f"目標サイズ: {target_bytes}バイト以内\n\n"
        f"{section_text}"
    )


def _strip_heading(section_text: str) -> str:
    """セクションテキストから先頭の ``## ...`` 見出し行を除去して本文だけ返す。"""
    import re
    return re.sub(r"^##\s+[^\n]*\n*", "", section_text, count=1).strip()


def _compact_section(
    section_name: str,
    section_text: str,
    target_bytes: int,
    llm_call: LlmCall,
) -> tuple[str, str | None]:
    """1 セクションをコンパクションする。

    Returns:
        (compacted_body, warning_or_none)
        失敗時は元の本文をそのまま返し、warning に理由を入れる。
    """
    body = _strip_heading(section_text)
    if not body:
        return body, None

    MIN_RATIO = 0.05
    original_size = len(body.encode("utf-8"))

    try:
        prompt = _build_section_prompt(body, target_bytes)
        result = llm_call(prompt)
    except Exception as e:
        warning = f"{section_name}: LLM call failed ({e}), using original text"
        _log.warning(warning)
        return body, warning

    result_size = len(result.encode("utf-8"))
    if result_size < original_size * MIN_RATIO:
        warning = (
            f"{section_name}: result too small "
            f"({result_size}B < {original_size}B * {MIN_RATIO}), "
            f"using original text"
        )
        _log.warning(warning)
        return body, warning

    return result.strip(), None


def compact(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    llm_call: LlmCall,
    dry_run: bool = False,
) -> CompactionResult:
    """メモリファイルをコンパクションする（per-section LLM 呼び出し）。

    llm_call はプロンプト文字列を受け取り、コンパクション後のテキストを返す callable。
    dry_run=True のときはファイル書き込みを行わない。

    preferences セクションは LLM に送らず、元テキストをそのまま保持する。
    各セクションの LLM 呼び出しが失敗した場合、そのセクションは元テキストにフォールバックする。
    """
    from mltgnt.memory import memory_file_path, persona_memory_lock

    path = memory_file_path(config, persona_stem)
    if not path.exists():
        raise FileNotFoundError(f"Memory file not found: {path}")

    with persona_memory_lock(config, persona_stem) as ok:
        if not ok:
            raise TimeoutError(f"Failed to acquire memory lock for {persona_stem}")

        original_text = path.read_text(encoding="utf-8")
        before_bytes = len(original_text.encode("utf-8"))

        sections = parse_memory(
            original_text, preferences_heading=config.preferences_section_name
        )

        # preferences はそのまま保持 (LLM に送らない)
        prefs_body = _strip_heading(sections.preferences)

        # 残り 3 セクションのターゲットサイズ配分
        # preferences を除いた残りを 3 等分
        prefs_size = len(prefs_body.encode("utf-8")) if prefs_body else 0
        remaining_target = max(config.compact_target_bytes - prefs_size, 1024)
        section_target = remaining_target // 3

        warnings: list[str] = []

        # 各セクションを個別にコンパクション
        section_defs = [
            ("long_term", sections.long_term),
            ("mid_term", sections.mid_term),
            ("recent", sections.recent),
        ]

        compacted: dict[str, str] = {}
        for name, text in section_defs:
            body, warning = _compact_section(name, text, section_target, llm_call)
            compacted[name] = body
            if warning:
                warnings.append(warning)

        # assemble_memory で見出し付きテキストを組み立て
        new_text = assemble_memory(
            preferences=prefs_body,
            long_term=compacted["long_term"],
            mid_term=compacted["mid_term"],
            recent=compacted["recent"],
            preamble=sections.preamble,
            preferences_heading=config.preferences_section_name,
        )
        after_bytes = len(new_text.encode("utf-8"))

        # サイズ上限チェック
        if after_bytes > config.compact_target_bytes * 1.3:
            raise ValueError(
                f"Compaction result too large for {persona_stem}: "
                f"{after_bytes}B exceeds limit {int(config.compact_target_bytes * 1.3)}B "
                f"— aborting"
            )

        # サイズ下限チェック: 元サイズの 5% 未満は異常
        MIN_RATIO = 0.05
        if after_bytes < before_bytes * MIN_RATIO:
            raise ValueError(
                f"Compaction produced near-empty result for {persona_stem}: "
                f"{before_bytes}B -> {after_bytes}B "
                f"(ratio {after_bytes / before_bytes:.3f} < {MIN_RATIO}) "
                f"— aborting to prevent data loss"
            )

        if not dry_run:
            path.write_text(new_text, encoding="utf-8")

        return CompactionResult(
            before_bytes=before_bytes,
            after_bytes=after_bytes,
            summary=f"compacted {persona_stem}: {before_bytes}B -> {after_bytes}B",
            warnings=warnings,
        )
