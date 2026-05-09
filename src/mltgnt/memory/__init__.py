"""
mltgnt.memory — 人物像メモリのパス解決・追記・末尾読込・ロック。

設計: Issue #118 §3 (T3)、Issue #823 (JSONL 統一)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig
    from mltgnt.memory._scoring import ScoredEntry

from mltgnt.memory._format import (
    MemoryEntry,
    assemble_entries_text,
    migrate_markdown_to_jsonl,
    parse_jsonl,
    serialize_entry,
)

__all__ = [
    "persona_memory_lock",
    "append_memory_entry",
    "read_memory_preferences",
    "read_memory_tail_text",
    "read_memory_by_relevance",
    "read_memory_with_sufficiency_check",
    "read_memory_agentic",
    "memory_file_path",
    "normalize_source_prefix",
    "compact",
    "needs_compaction",
    "extract_and_append",
    "LlmCallError",
    "CompactionResult",
]

LlmCall = Callable[[str], str]

_log = logging.getLogger(__name__)

MEMORY_DEDUPE_SCAN_BYTES = 32 * 1024
MEMORY_DEDUPE_SCAN_LINES = 200
MEMORY_CORRUPT_THRESHOLD_BYTES = 10


def _resolve_memory_dir(config: "MemoryConfig") -> Path:
    if config.chat_memory_dir is not None:
        return config.chat_memory_dir
    return config.chat_dir / "memory"


def memory_file_path(config: "MemoryConfig", persona_stem: str) -> Path:
    """`_resolve_memory_dir(config) / f\"{persona_stem}.jsonl\"`"""
    return _resolve_memory_dir(config) / f"{persona_stem}.jsonl"


def normalize_source_prefix(body: str) -> str:
    """先頭行のソースタグを正規化する（後方互換）。"""
    lines = body.splitlines()
    if not lines:
        return body
    if lines[0].strip() == "[file-chat]":
        lines[0] = "[file]"
        return "\n".join(lines)
    return body


def _tail_utf8_bytes(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    cut = b[-max_bytes:]
    return cut.decode("utf-8", errors="replace")


@contextmanager
def persona_memory_lock(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    timeout_sec: float | None = None,
) -> Iterator[bool]:
    if timeout_sec is None:
        timeout_sec = config.lock_timeout_sec
    lock_path = config.chat_dir / f".lock-memory-{persona_stem}"
    config.chat_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    acquired = False
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL, 0o644)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.05)
        except OSError:
            time.sleep(0.05)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def _ensure_jsonl(config: "MemoryConfig", persona_stem: str) -> Path:
    """JSONL ファイルパスを返す。

    - .jsonl 不在かつ .md 在: .md を JSONL にマイグレーション
    - .jsonl 在だが有効 JSON 行が 0 行で先頭が Markdown らしい: Markdown として in-place マイグレーション
    """
    jsonl_path = memory_file_path(config, persona_stem)
    if not jsonl_path.exists():
        md_path = jsonl_path.with_suffix(".md")
        if md_path.exists():
            migrate_markdown_to_jsonl(md_path, jsonl_path)
        return jsonl_path
    # ファイルが存在するが有効 JSON 行がなければ Markdown の可能性がある
    try:
        content = jsonl_path.read_text(encoding="utf-8")
    except OSError:
        return jsonl_path
    if content.strip() and not any(
        line.strip().startswith("{") for line in content.splitlines() if line.strip()
    ):
        # 有効 JSON が 1 行も無い → Markdown として in-place マイグレーション
        import tempfile, os
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(content)
                tf_name = tf.name
            migrate_markdown_to_jsonl(Path(tf_name), jsonl_path)
        finally:
            try:
                os.unlink(tf_name)
            except OSError:
                pass
    return jsonl_path


def _scan_tail_for_dedupe_key(path: Path, dedupe_key: str) -> bool:
    """JSONL ファイルの末尾付近に dedupe_key が含まれているか確認。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = text.splitlines()
    tail_lines = lines[-MEMORY_DEDUPE_SCAN_LINES:] if len(lines) > MEMORY_DEDUPE_SCAN_LINES else lines
    for line in tail_lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("dedupe_key") == dedupe_key:
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


def append_memory_entry(
    config: "MemoryConfig",
    persona_stem: str,
    role: str,
    content: str,
    timestamp: str,
    *,
    source_tag: str,
    dedupe_key: str | None = None,
    under_lock: bool = False,
    layer: str | None = None,
) -> bool:
    """JSONL 形式でエントリを追記。

    layer が指定された場合、エントリに layer フィールドを付与する。
    dedupe_key が非空で末尾付近に同一キーがあれば追記しない（冪等、True）。
    ロック取得失敗（under_lock False 時）は False。
    """
    def _write() -> None:
        _resolve_memory_dir(config).mkdir(parents=True, exist_ok=True)
        mp = memory_file_path(config, persona_stem)
        entry = MemoryEntry(
            timestamp=timestamp,
            role=role,
            content=content.strip(),
            source_tag=source_tag,
            layer=layer,
            dedupe_key=dedupe_key,
        )
        line = serialize_entry(entry) + "\n"
        with mp.open("a", encoding="utf-8") as f:
            f.write(line)

    mp = memory_file_path(config, persona_stem)

    if mp.exists():
        try:
            sz = mp.stat().st_size
        except OSError:
            sz = -1
        if 0 < sz <= MEMORY_CORRUPT_THRESHOLD_BYTES:
            _log.error(
                "append_memory_entry: %s is only %d bytes — likely corrupted, "
                "refusing to append (threshold=%d)",
                mp, sz, MEMORY_CORRUPT_THRESHOLD_BYTES,
            )
            return False

    if dedupe_key and mp.exists() and _scan_tail_for_dedupe_key(mp, dedupe_key):
        return True

    if under_lock:
        _write()
        return True

    with persona_memory_lock(config, persona_stem) as ok:
        if not ok:
            return False
        if dedupe_key and mp.exists() and _scan_tail_for_dedupe_key(mp, dedupe_key):
            return True
        _write()
    return True


def read_memory_preferences(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    max_bytes: int | None = None,
) -> str:
    """JSONL から source_tag="preferences" エントリを抽出して返す。"""
    if max_bytes is None:
        max_bytes = config.preferences_max_bytes
    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return ""
    entries = parse_jsonl(path)
    prefs_entries = [e for e in entries if e.source_tag == "preferences"]
    if not prefs_entries:
        return ""
    text = assemble_entries_text(
        prefs_entries,
        preferences_heading=config.preferences_section_name,
    )
    text = text.strip()
    if not text:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


def read_memory_tail_text(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    max_bytes: int,
    max_entries: int,
    layers: list[str] | None = None,
) -> str:
    """JSONL ファイル末尾から最大 max_entries エントリを返す。

    layers 指定時は layer がリストに含まれるエントリのみ対象。
    """
    if max_bytes == 0:
        return ""
    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return ""
    entries = parse_jsonl(path)
    if not entries:
        return ""
    if layers is not None:
        entries = [e for e in entries if e.layer in layers]
    selected = entries[-max_entries:] if len(entries) > max_entries else entries
    text = assemble_entries_text(
        selected,
        preferences_heading=config.preferences_section_name,
    )
    if not text.strip():
        return ""
    return _tail_utf8_bytes(text, max_bytes)


def _search_and_score(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_entries: int,
) -> "list[ScoredEntry]":
    """JSONL から preferences 以外のエントリをスコアリングして返す。"""
    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return []
    entries = parse_jsonl(path)
    if not entries:
        return []
    non_prefs = [e for e in entries if e.source_tag != "preferences"]
    if not non_prefs:
        return []
    # スコアリングは content テキスト単位
    entry_texts = [
        assemble_entries_text([e], preferences_heading=config.preferences_section_name).strip()
        for e in non_prefs
    ]
    from mltgnt.memory._scoring import score_entries
    scored = score_entries(query, entry_texts)
    return scored[:max_entries]


def read_memory_by_relevance(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    layers: list[str] | None = None,
) -> str:
    """クエリとの関連度が高いエントリを選択して返す。

    layers 指定時は layer がリストに含まれるエントリのみをスコアリング対象とする。
    """
    if not query:
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries, layers=layers
        )

    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return ""

    entries = parse_jsonl(path)
    if not entries:
        return ""

    prefs_entries = [e for e in entries if e.source_tag == "preferences"]
    non_prefs = [e for e in entries if e.source_tag != "preferences"]

    if layers is not None:
        non_prefs = [e for e in non_prefs if e.layer in layers]

    if not non_prefs:
        text = assemble_entries_text(
            prefs_entries,
            preferences_heading=config.preferences_section_name,
        )
        return _tail_utf8_bytes(text, max_bytes)

    try:
        entry_texts = [
            assemble_entries_text([e], preferences_heading=config.preferences_section_name).strip()
            for e in non_prefs
        ]
        from mltgnt.memory._scoring import score_entries
        scored = score_entries(query, entry_texts)
        top_texts = [s.text for s in scored[:max_entries]]
    except Exception as e:
        _log.warning(
            "read_memory_by_relevance: tfidf scoring error, fallback to tail text: %s", e
        )
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries, layers=layers
        )

    prefs_text = assemble_entries_text(
        prefs_entries,
        preferences_heading=config.preferences_section_name,
    ).rstrip("\n")
    if prefs_text and top_texts:
        text = prefs_text + "\n\n---\n\n" + "\n\n---\n\n".join(top_texts) + "\n"
    elif prefs_text:
        text = prefs_text + "\n"
    else:
        text = "\n\n---\n\n".join(top_texts) + "\n"

    return _tail_utf8_bytes(text, max_bytes)


def read_memory_with_sufficiency_check(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    llm_call: "Callable[[str], str] | None" = None,
) -> str:
    """十分性判定付き memory 検索。"""
    if llm_call is None:
        return read_memory_by_relevance(
            config, persona_stem, query, max_bytes=max_bytes, max_entries=max_entries
        )

    if not query:
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries
        )

    try:
        initial_scored = _search_and_score(config, persona_stem, query, max_entries=max_entries)
    except Exception as e:
        _log.warning(
            "read_memory_with_sufficiency_check: initial search error, fallback: %s", e
        )
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries
        )

    top_scored = list(initial_scored)

    if initial_scored:
        excerpt = "\n\n---\n\n".join(s.text for s in initial_scored)
        try:
            from mltgnt.memory._sufficiency import judge_sufficiency
            result = judge_sufficiency(query, excerpt, llm_call)
            if not result.sufficient and result.rewritten_query:
                try:
                    rewritten_scored = _search_and_score(
                        config, persona_stem, result.rewritten_query, max_entries=max_entries
                    )
                    seen_texts: set[str] = set()
                    merged: list[ScoredEntry] = []  # type: ignore[type-arg]
                    for entry in initial_scored:
                        if entry.text not in seen_texts:
                            seen_texts.add(entry.text)
                            merged.append(entry)
                    for entry in rewritten_scored:
                        if entry.text not in seen_texts:
                            seen_texts.add(entry.text)
                            merged.append(entry)
                    merged.sort(key=lambda e: e.score, reverse=True)
                    top_scored = merged[:max_entries]
                except Exception as e2:
                    _log.warning(
                        "read_memory_with_sufficiency_check: re-search error, using initial: %s", e2
                    )
        except Exception as e:
            _log.warning(
                "read_memory_with_sufficiency_check: sufficiency check error, using initial results: %s", e
            )

    path = _ensure_jsonl(config, persona_stem)
    prefs_entries: list[MemoryEntry] = []
    if path.exists():
        all_entries = parse_jsonl(path)
        prefs_entries = [e for e in all_entries if e.source_tag == "preferences"]

    prefs_text = assemble_entries_text(
        prefs_entries,
        preferences_heading=config.preferences_section_name,
    ).rstrip("\n")
    top_texts = [s.text for s in top_scored]

    if prefs_text and top_texts:
        text = prefs_text + "\n\n---\n\n" + "\n\n---\n\n".join(top_texts) + "\n"
    elif prefs_text:
        text = prefs_text + "\n"
    elif top_texts:
        text = "\n\n---\n\n".join(top_texts) + "\n"
    else:
        return ""

    return _tail_utf8_bytes(text, max_bytes)


def read_memory_agentic(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    llm_call: "Callable[[str], str]",
    skill_paths: "list[Path] | None" = None,
    max_iterations: int = 3,
) -> str:
    """Phase 3: Agentic RAG による情報収集。"""
    from mltgnt.memory._agentic import AgenticRetriever

    retriever = AgenticRetriever(
        config=config,
        persona_stem=persona_stem,
        skill_paths=skill_paths or [],
        llm_call=llm_call,
        max_iterations=max_iterations,
    )
    return retriever.retrieve(query, max_bytes=max_bytes, max_entries=max_entries)


def extract_and_append(
    config: "MemoryConfig",
    persona_stem: str,
    session_text: str,
    *,
    llm_call: LlmCall,
    target_layers: list[str] | None = None,
) -> list[MemoryEntry]:
    """セッションテキストから教訓・学びを抽出してメモリに追記する。

    LLM エラー時は空リスト [] を返し、例外を投げない。
    """
    if target_layers is None:
        target_layers = ["caveat", "learning"]

    layers_str = "・".join(target_layers)
    prompt = (
        f"以下のセッションテキストから、{layers_str} に分類できる教訓・禁止事項・学びを抽出してください。\n"
        "各項目を JSON 形式（1行1件）で出力してください:\n"
        '{"layer": "<caveat|learning>", "content": "<抽出テキスト>"}\n\n'
        f"セッションテキスト:\n{session_text}"
    )

    try:
        response = llm_call(prompt)
    except Exception as e:
        _log.warning("extract_and_append: llm_call failed: %s", e)
        return []

    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).astimezone(
        datetime.timezone(datetime.timedelta(hours=9))
    ).isoformat(timespec="seconds")

    appended: list[MemoryEntry] = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            layer = data.get("layer", "")
            content = data.get("content", "").strip()
            if not content or layer not in target_layers:
                continue
            entry = MemoryEntry(
                timestamp=ts,
                role="assistant",
                content=content,
                source_tag="auto_extract",
                layer=layer,
            )
            append_memory_entry(
                config,
                persona_stem,
                entry.role,
                entry.content,
                entry.timestamp,
                source_tag=entry.source_tag,
                layer=entry.layer,
            )
            appended.append(entry)
        except (json.JSONDecodeError, TypeError):
            pass

    return appended


# Lazy imports for compaction
from mltgnt.memory._compaction import (  # noqa: E402
    compact,
    needs_compaction,
    LlmCallError,
    CompactionResult,
)
