"""
mltgnt.memory._compaction — メモリコンパクション（JSONL + MemoryEntry ベース）。

設計: Issue #123, #137, #823
"""
from __future__ import annotations

import datetime
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

from mltgnt.memory._format import MemoryEntry, parse_jsonl, serialize_entry

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
    from mltgnt.memory import _ensure_jsonl
    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return False
    return path.stat().st_size >= config.compact_threshold_bytes


def _parse_timestamp(ts: str) -> datetime.datetime | None:
    """ISO 8601 タイムスタンプをパース。失敗時は None。"""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.datetime.strptime(ts, fmt)
        except (ValueError, TypeError):
            pass
    try:
        return datetime.datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _compact_entries(
    section_name: str,
    entries: list[MemoryEntry],
    target_bytes: int,
    llm_call: LlmCall,
) -> tuple[str, str | None]:
    """複数エントリを結合してコンパクションする。

    Returns:
        (compacted_text, warning_or_none)
    """
    if not entries:
        return "", None

    combined = "\n\n".join(e.content for e in entries if e.content.strip())
    if not combined.strip():
        return "", None

    MIN_RATIO = 0.05
    original_size = len(combined.encode("utf-8"))

    prompt = (
        "以下の文章を要約・圧縮してください。"
        "見出しは不要です。本文のみ出力してください。"
        f"目標サイズ: {target_bytes}バイト以内\n\n"
        f"{combined}"
    )

    try:
        result = llm_call(prompt)
    except Exception as e:
        warning = f"{section_name}: LLM call failed ({e}), using original text"
        _log.warning(warning)
        return combined, warning

    result_size = len(result.encode("utf-8"))
    if result_size < original_size * MIN_RATIO:
        warning = (
            f"{section_name}: result too small "
            f"({result_size}B < {original_size}B * {MIN_RATIO}), "
            f"using original text"
        )
        _log.warning(warning)
        return combined, warning

    return result.strip(), None


def compact(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    llm_call: LlmCall,
    dry_run: bool = False,
) -> CompactionResult:
    """JSONL メモリファイルをコンパクションする。

    - protected_layers（デフォルト: caveat）に属するエントリはコンパクション対象外
    - source_tag="preferences" のエントリはそのまま保持
    - 残りを期間（raw_days / mid_weeks）でグループ化して LLM 要約
    """
    from mltgnt.memory import persona_memory_lock, _ensure_jsonl

    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        raise FileNotFoundError(f"Memory file not found: {path}")

    with persona_memory_lock(config, persona_stem) as ok:
        if not ok:
            raise TimeoutError(f"Failed to acquire memory lock for {persona_stem}")

        original_text = path.read_text(encoding="utf-8")
        before_bytes = len(original_text.encode("utf-8"))

        entries = parse_jsonl(path)

        now = datetime.datetime.now(datetime.timezone.utc)
        raw_cutoff = now - datetime.timedelta(days=config.raw_days)
        mid_cutoff = now - datetime.timedelta(weeks=config.mid_weeks)

        # 分類
        protected_entries: list[MemoryEntry] = []
        prefs_entries: list[MemoryEntry] = []
        recent_entries: list[MemoryEntry] = []
        mid_entries: list[MemoryEntry] = []
        long_entries: list[MemoryEntry] = []

        for entry in entries:
            if entry.layer is not None and entry.layer in config.protected_layers:
                protected_entries.append(entry)
                continue
            if entry.source_tag == "preferences":
                prefs_entries.append(entry)
                continue
            ts = _parse_timestamp(entry.timestamp)
            if ts is None:
                # タイムスタンプ不明は長期扱い
                long_entries.append(entry)
            elif ts >= raw_cutoff:
                recent_entries.append(entry)
            elif ts >= mid_cutoff:
                mid_entries.append(entry)
            else:
                long_entries.append(entry)

        prefs_size = sum(len(e.content.encode("utf-8")) for e in prefs_entries)
        protected_size = sum(len(e.content.encode("utf-8")) for e in protected_entries)
        remaining_target = max(config.compact_target_bytes - prefs_size - protected_size, 1024)
        section_target = remaining_target // 3

        warnings: list[str] = []
        compacted_ts = now.astimezone(
            datetime.timezone(datetime.timedelta(hours=9))
        ).isoformat(timespec="seconds")

        summary_entries: list[MemoryEntry] = []

        for section_name, group in [
            ("long_term", long_entries),
            ("mid_term", mid_entries),
        ]:
            if not group:
                continue
            body, warning = _compact_entries(section_name, group, section_target, llm_call)
            if warning:
                warnings.append(warning)
            if body.strip():
                summary_entries.append(MemoryEntry(
                    timestamp=compacted_ts,
                    role="assistant",
                    content=body,
                    source_tag="compaction",
                ))

        new_entries = protected_entries + prefs_entries + summary_entries + recent_entries

        # 書き戻し
        lines = [serialize_entry(e) + "\n" for e in new_entries]
        new_text = "".join(lines)
        after_bytes = len(new_text.encode("utf-8"))

        # サイズ上限チェック
        if after_bytes > config.compact_target_bytes * 1.3:
            raise ValueError(
                f"Compaction result too large for {persona_stem}: "
                f"{after_bytes}B exceeds limit {int(config.compact_target_bytes * 1.3)}B "
                f"— aborting"
            )

        MIN_RATIO = 0.05
        if before_bytes > 0 and after_bytes < before_bytes * MIN_RATIO:
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
