"""
mltgnt.memory._compaction — メモリコンパクション。

設計: Issue #123
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

from mltgnt.memory._format import MemorySections, parse_memory, format_memory

_log = logging.getLogger(__name__)

LlmCall = Callable[[str], str]

__all__ = [
    "LlmCallError",
    "CompactionResult",
    "needs_compaction",
    "build_compaction_prompt",
    "compact",
]


class LlmCallError(RuntimeError):
    """llm_call の実行中に発生したエラーをラップする例外。"""


@dataclass(frozen=True)
class CompactionResult:
    before_bytes: int
    after_bytes: int
    summary: str


def needs_compaction(config: "MemoryConfig", persona_stem: str) -> bool:
    """メモリファイルがコンパクション閾値を超えているか判定する。"""
    from mltgnt.memory import memory_file_path
    path = memory_file_path(config, persona_stem)
    if not path.exists():
        return False
    return path.stat().st_size >= config.compact_threshold_bytes


def build_compaction_prompt(config: "MemoryConfig", sections: MemorySections) -> str:
    """コンパクション用プロンプトを生成する。"""
    target_kb = config.compact_target_bytes // 1024
    min_kb = int(config.compact_target_bytes * 0.7 // 1024)
    max_kb = int(config.compact_target_bytes * 1.3 // 1024)
    prefs_kb = config.preferences_max_bytes // 1024
    return f"""メモリファイルのコンパクション（圧縮・要約）を行ってください。

## 目標サイズ
コンパクション後: {target_kb}KB 以内（許容範囲: {min_kb}〜{max_kb}KB）

## 入力メモリ（4 層構造）

### 好みセクション（保持必須）
{sections.preferences or "(なし)"}

### 長期要約
{sections.long_term or "(なし)"}

### 中期要約
{sections.mid_term or "(なし)"}

### 直近ログ（{config.raw_days}日以内は生保持）
{sections.recent or "(なし)"}

## 段階的忘却ルール（Asia/Tokyo 日付基準）

| 期間 | 処理 |
|------|------|
| 0〜{config.raw_days}日以内 | 生ログをそのまま保持 |
| {config.raw_days + 1}〜{config.mid_weeks * 7}日前 | 週単位に要約（主要話題 + 感情トーン） |
| {config.mid_weeks * 7 + 1}日以上前 | 月単位に要約（月あたり 3〜5 行） |

## 好みセクションのルール
- `## {config.preferences_section_name}` セクションは必ず保持・更新する
- {prefs_kb}KB 以内に収める
- 削除しない

## 出力形式
以下の 4 層構造で出力してください。各セクションは `---` で区切る。

## {config.preferences_section_name}
（好みセクション内容）

---

## 長期要約（1ヶ月超）
（月次要約）

---

## 中期要約（1〜{config.mid_weeks}週間前）
（週次要約）

---

## 直近ログ（{config.raw_days}日以内）
（生ログ）
"""


def compact(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    llm_call: LlmCall,
    compaction_prompt: str | None = None,
    dry_run: bool = False,
) -> CompactionResult:
    """メモリファイルをコンパクションする。

    llm_call はプロンプト文字列を受け取り、コンパクション後のテキストを返す callable。
    dry_run=True のときはファイル書き込みを行わない。
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

        prompt = compaction_prompt or build_compaction_prompt(config, sections)

        try:
            llm_response = llm_call(prompt)
        except Exception as e:
            raise LlmCallError(f"llm_call failed: {e}") from e

        # Parse the LLM response as memory sections
        new_sections = parse_memory(
            llm_response, preferences_heading=config.preferences_section_name
        )
        # Preserve preamble from original
        new_sections_with_preamble = MemorySections(
            preferences=new_sections.preferences,
            long_term=new_sections.long_term,
            mid_term=new_sections.mid_term,
            recent=new_sections.recent,
            preamble=sections.preamble,
        )
        new_text = format_memory(new_sections_with_preamble)
        after_bytes = len(new_text.encode("utf-8"))

        if after_bytes > config.compact_target_bytes * 1.3:
            _log.warning(
                "compact: after_bytes=%d exceeds compact_target_bytes*1.3=%d",
                after_bytes,
                int(config.compact_target_bytes * 1.3),
            )

        if not dry_run:
            path.write_text(new_text, encoding="utf-8")

        return CompactionResult(
            before_bytes=before_bytes,
            after_bytes=after_bytes,
            summary=f"compacted {persona_stem}: {before_bytes}B -> {after_bytes}B",
        )
