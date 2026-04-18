"""
tests/test_mltgnt_memory.py — mltgnt.memory のユニットテスト（AC-1）

設計: Issue #118 §7 AC-1
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from mltgnt.config import MemoryConfig  # noqa: E402
from mltgnt.memory import (  # noqa: E402
    append_memory_entry,
    read_memory_preferences,
    read_memory_tail_text,
    memory_file_path,
    persona_memory_lock,
    compact,
    needs_compaction,
    LlmCallError,
    CompactionResult,
)
from mltgnt.memory._format import parse_memory, format_memory, assemble_memory  # noqa: E402


def make_config(tmp_path: Path) -> MemoryConfig:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=mem_dir,
    )


# ---------------------------------------------------------------------------
# AC-1: 正常系 - append_memory_entry
# ---------------------------------------------------------------------------

def test_append_memory_entry_creates_file(tmp_path: Path) -> None:
    """append_memory_entry が指定ペルソナのファイルにエントリを追記する。"""
    config = make_config(tmp_path)
    result = append_memory_entry(
        config, "test_persona", "user", "hello", "2026-04-17T10:00:00+09:00",
        source_tag="[file]"
    )
    assert result is True
    mp = memory_file_path(config, "test_persona")
    assert mp.exists()
    text = mp.read_text(encoding="utf-8")
    assert "## 2026-04-17T10:00:00+09:00 — user" in text
    assert "[file]" in text
    assert "hello" in text
    assert "---" in text


def test_append_memory_entry_format(tmp_path: Path) -> None:
    """追記エントリのフォーマットが正しい。"""
    config = make_config(tmp_path)
    append_memory_entry(
        config, "test_persona", "user", "hello", "2026-04-17T10:00:00+09:00",
        source_tag="[file]"
    )
    mp = memory_file_path(config, "test_persona")
    text = mp.read_text(encoding="utf-8")
    # フォーマット: ## {timestamp} — {role}\n\n{source_tag}\n{body}\n\n---\n\n
    assert "## 2026-04-17T10:00:00+09:00 — user\n\n[file]\nhello\n\n---" in text


# ---------------------------------------------------------------------------
# AC-1: 正常系 - read_memory_tail_text
# ---------------------------------------------------------------------------

def test_read_memory_tail_text_returns_entries(tmp_path: Path) -> None:
    """read_memory_tail_text が末尾エントリを返す。"""
    config = make_config(tmp_path)
    for i in range(10):
        append_memory_entry(
            config, "test_persona", "user", f"entry {i}", f"2026-04-17T1{i:01d}:00:00+09:00",
            source_tag="[file]"
        )
    result = read_memory_tail_text(config, "test_persona", max_bytes=1024, max_entries=5)
    assert result
    # 最大5エントリに収まる
    assert result.count("## 20") <= 5


def test_read_memory_tail_text_nonexistent(tmp_path: Path) -> None:
    """存在しないファイルへの read_memory_tail_text は空文字列を返す。"""
    config = make_config(tmp_path)
    result = read_memory_tail_text(config, "nonexistent", max_bytes=1024, max_entries=5)
    assert result == ""


def test_read_memory_tail_text_max_bytes_zero(tmp_path: Path) -> None:
    """max_bytes=0 での read_memory_tail_text は空文字列を返す。"""
    config = make_config(tmp_path)
    append_memory_entry(
        config, "test_persona", "user", "hello", "2026-04-17T10:00:00+09:00",
        source_tag="[file]"
    )
    result = read_memory_tail_text(config, "test_persona", max_bytes=0, max_entries=5)
    assert result == ""


# ---------------------------------------------------------------------------
# AC-1: 正常系 - read_memory_preferences
# ---------------------------------------------------------------------------

def test_read_memory_preferences_extracts_section(tmp_path: Path) -> None:
    """read_memory_preferences が ## ユーザーの好み・傾向 セクションを抽出する。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "test_persona")
    mp.write_text(
        "## ユーザーの好み・傾向\n\n好みのテキストです。\n\n---\n\n## 2026-04-17 — user\n\nentryです。\n\n---\n\n",
        encoding="utf-8",
    )
    result = read_memory_preferences(config, "test_persona")
    assert "ユーザーの好み・傾向" in result
    assert "好みのテキストです。" in result


def test_read_memory_preferences_nonexistent(tmp_path: Path) -> None:
    """存在しないファイルへの read_memory_preferences は空文字列を返す（例外なし）。"""
    config = make_config(tmp_path)
    result = read_memory_preferences(config, "nonexistent")
    assert result == ""


def test_read_memory_preferences_no_section(tmp_path: Path) -> None:
    """好みセクションがないファイルは空文字列を返す。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "test_persona")
    mp.write_text("## 2026-04-17 — user\n\nentryです。\n\n---\n\n", encoding="utf-8")
    result = read_memory_preferences(config, "test_persona")
    assert result == ""


# ---------------------------------------------------------------------------
# AC-1: 重複排除
# ---------------------------------------------------------------------------

def test_dedupe_key_prevents_second_write(tmp_path: Path) -> None:
    """dedupe_key で2回 append → 2回目は追記せず True を返す。"""
    config = make_config(tmp_path)
    r1 = append_memory_entry(
        config, "test_persona", "user", "hi", "2026-04-17 10:00",
        source_tag="[file]", dedupe_key="key1"
    )
    r2 = append_memory_entry(
        config, "test_persona", "user", "hi", "2026-04-17 10:00",
        source_tag="[file]", dedupe_key="key1"
    )
    assert r1 is True
    assert r2 is True
    mp = memory_file_path(config, "test_persona")
    text = mp.read_text(encoding="utf-8")
    assert text.count("2026-04-17 10:00 — user") == 1


def test_no_dedupe_key_allows_duplicate(tmp_path: Path) -> None:
    """dedupe_key=None で2回 append → 2回とも追記される。"""
    config = make_config(tmp_path)
    append_memory_entry(
        config, "test_persona", "user", "hi", "2026-04-17 10:00",
        source_tag="[file]", dedupe_key=None
    )
    append_memory_entry(
        config, "test_persona", "user", "hi", "2026-04-17 10:00",
        source_tag="[file]", dedupe_key=None
    )
    mp = memory_file_path(config, "test_persona")
    text = mp.read_text(encoding="utf-8")
    assert text.count("2026-04-17 10:00 — user") == 2


# ---------------------------------------------------------------------------
# AC-1: ロック
# ---------------------------------------------------------------------------

def test_persona_memory_lock_blocks_second_acquire(tmp_path: Path) -> None:
    """ロック取得中に別スレッドが同じ stem でロック取得 → タイムアウトまでブロック。"""
    config = make_config(tmp_path)
    ready = threading.Event()

    def hold():
        with persona_memory_lock(config, "lock_test", timeout_sec=5.0) as ok:
            assert ok
            ready.set()
            time.sleep(0.5)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert ready.wait(timeout=2.0)
    with persona_memory_lock(config, "lock_test", timeout_sec=0.05) as ok2:
        assert not ok2
    t.join(timeout=3.0)


# ---------------------------------------------------------------------------
# AC-1: memory_file_path
# ---------------------------------------------------------------------------

def test_memory_file_path(tmp_path: Path) -> None:
    """memory_file_path が正しいパスを返す。"""
    config = make_config(tmp_path)
    path = memory_file_path(config, "test_persona")
    assert path == config.chat_memory_dir / "test_persona.md"


# ---------------------------------------------------------------------------
# AC-1: MemoryConfig の新フィールド・chat_memory_dir デフォルト
# ---------------------------------------------------------------------------

def test_memory_config_default_chat_memory_dir(tmp_path: Path) -> None:
    """chat_memory_dir 省略時は chat_dir / 'memory' が使われる。"""
    config = MemoryConfig(chat_dir=tmp_path)
    path = memory_file_path(config, "persona")
    assert path == tmp_path / "memory" / "persona.md"


def test_memory_config_explicit_chat_memory_dir(tmp_path: Path) -> None:
    """chat_memory_dir を明示指定すると、そのパスが使われる（後方互換）。"""
    other = tmp_path / "other"
    config = MemoryConfig(chat_dir=tmp_path, chat_memory_dir=other)
    path = memory_file_path(config, "persona")
    assert path == other / "persona.md"


def test_memory_config_new_fields() -> None:
    """MemoryConfig の新フィールドがデフォルト値を持つ。"""
    config = MemoryConfig(chat_dir=Path("/tmp/chat"))
    assert config.raw_days == 7
    assert config.mid_weeks == 3
    assert config.compact_threshold_bytes == 40_960
    assert config.compact_target_bytes == 25_600
    assert config.preferences_section_name == "ユーザーの好み・傾向"


def test_memory_config_new_fields_custom() -> None:
    """MemoryConfig の新フィールドをカスタマイズできる。"""
    config = MemoryConfig(
        chat_dir=Path("/tmp/chat"),
        raw_days=14,
        mid_weeks=4,
        compact_threshold_bytes=81_920,
        compact_target_bytes=51_200,
        preferences_section_name="User Preferences",
    )
    assert config.raw_days == 14
    assert config.mid_weeks == 4
    assert config.compact_threshold_bytes == 81_920
    assert config.compact_target_bytes == 51_200
    assert config.preferences_section_name == "User Preferences"


# ---------------------------------------------------------------------------
# _format: parse_memory / format_memory
# ---------------------------------------------------------------------------

def test_parse_memory_all_sections() -> None:
    """parse_memory が 4 層すべてを正しく抽出する。"""
    text = (
        "## ユーザーの好み・傾向\n\n好み内容\n\n---\n\n"
        "## 長期要約\n\n長期内容\n\n---\n\n"
        "## 中期要約\n\n中期内容\n\n---\n\n"
        "## 直近ログ\n\n直近内容\n"
    )
    sections = parse_memory(text)
    assert "好み内容" in sections.preferences
    assert "長期内容" in sections.long_term
    assert "中期内容" in sections.mid_term
    assert "直近内容" in sections.recent


def test_parse_memory_missing_sections() -> None:
    """セクションが一部欠落しても正常にパースできる。"""
    text = "## ユーザーの好み・傾向\n\n好み内容\n"
    sections = parse_memory(text)
    assert "好み内容" in sections.preferences
    assert sections.long_term == ""
    assert sections.mid_term == ""
    assert sections.recent == ""


def test_parse_memory_no_known_sections() -> None:
    """既知セクションなしのテキストはすべて preamble になる。"""
    text = "なんらかのテキスト"
    sections = parse_memory(text)
    assert sections.preamble == "なんらかのテキスト"
    assert sections.preferences == ""


def test_parse_memory_custom_heading() -> None:
    """preferences_heading をカスタマイズできる。"""
    text = "## User Preferences\n\nsome prefs\n"
    sections = parse_memory(text, preferences_heading="User Preferences")
    assert "some prefs" in sections.preferences


def test_format_memory_roundtrip() -> None:
    """parse_memory → format_memory のラウンドトリップで内容が保持される。"""
    text = (
        "## ユーザーの好み・傾向\n\n好み内容\n\n---\n\n"
        "## 長期要約\n\n長期内容\n\n---\n\n"
        "## 中期要約\n\n中期内容\n\n---\n\n"
        "## 直近ログ\n\n直近内容\n"
    )
    sections = parse_memory(text)
    result = format_memory(sections)
    assert "好み内容" in result
    assert "長期内容" in result
    assert "中期内容" in result
    assert "直近内容" in result
    assert "---" in result


# ---------------------------------------------------------------------------
# needs_compaction
# ---------------------------------------------------------------------------

def test_needs_compaction_false_when_file_missing(tmp_path: Path) -> None:
    """ファイルが存在しない場合は False。"""
    config = make_config(tmp_path)
    assert needs_compaction(config, "nonexistent") is False


def test_needs_compaction_false_when_below_threshold(tmp_path: Path) -> None:
    """閾値未満は False。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "test")
    mp.write_text("x" * 100, encoding="utf-8")
    assert needs_compaction(config, "test") is False


def test_needs_compaction_true_when_above_threshold(tmp_path: Path) -> None:
    """閾値以上は True。"""
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
        compact_threshold_bytes=100,
    )
    mp = memory_file_path(config, "test")
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text("x" * 200, encoding="utf-8")
    assert needs_compaction(config, "test") is True


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------

VALID_MEMORY = """\
## ユーザーの好み・傾向

好みの内容

---

## 長期要約（1ヶ月超）

長期要約の内容

---

## 中期要約（1〜3週間前）

中期要約の内容

---

## 直近ログ（7日以内）

直近ログの内容
"""


def _setup_memory(tmp_path: Path, content: str = VALID_MEMORY) -> tuple:
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
        compact_threshold_bytes=10,
        compact_target_bytes=25_600,
    )
    (tmp_path / "memory").mkdir(exist_ok=True)
    (tmp_path / "memory" / "persona.md").write_text(content, encoding="utf-8")
    return config, "persona"


def _identity_llm(prompt: str) -> str:
    """LLM mock: prompt 中の本文をそのまま返す。"""
    parts = prompt.split("\n\n", 1)
    return parts[-1] if len(parts) > 1 else prompt


def test_compact_calls_llm_per_section_and_writes_result(tmp_path: Path) -> None:
    """compact がセクションごとに llm_call を呼び出し、結果をファイルに書き込む。"""
    config, persona = _setup_memory(tmp_path)

    calls: list[str] = []

    def mock_llm(prompt: str) -> str:
        calls.append(prompt)
        parts = prompt.split("\n\n", 1)
        body = parts[-1] if len(parts) > 1 else prompt
        return body + "（コンパクト）"

    result = compact(config, persona, llm_call=mock_llm)

    # 3 セクション (long_term, mid_term, recent) で呼ばれる（preferences は除外）
    assert len(calls) == 3
    assert isinstance(result, CompactionResult)
    assert result.before_bytes > 0
    assert result.after_bytes > 0

    written = (tmp_path / "memory" / "persona.md").read_text(encoding="utf-8")
    assert "コンパクト" in written
    # preferences は元テキストのまま保持
    assert "好みの内容" in written


def test_compact_dry_run_does_not_write(tmp_path: Path) -> None:
    """dry_run=True のときはファイルを書き込まない。"""
    config, persona = _setup_memory(tmp_path)
    original = (tmp_path / "memory" / "persona.md").read_text(encoding="utf-8")

    compact(config, persona, llm_call=_identity_llm, dry_run=True)

    assert (tmp_path / "memory" / "persona.md").read_text(encoding="utf-8") == original


def test_compact_all_llm_failures_fallback(tmp_path: Path) -> None:
    """全 LLM 呼び出しが失敗しても、元テキストにフォールバックして書き込みが行われる。"""
    config, persona = _setup_memory(tmp_path)

    call_count = 0

    def failing_llm(_: str) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("LLM unavailable")

    result = compact(config, persona, llm_call=failing_llm)

    assert call_count == 3
    assert len(result.warnings) == 3
    assert result.after_bytes > 0


def test_compact_raises_file_not_found(tmp_path: Path) -> None:
    """メモリファイルが存在しない場合は FileNotFoundError。"""
    config = make_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        compact(config, "nonexistent", llm_call=lambda p: p)


# ---------------------------------------------------------------------------
# TC6: 1 セクション LLM 失敗でも他のセクションは正常処理される
# ---------------------------------------------------------------------------

def test_one_section_failure_others_succeed(tmp_path: Path) -> None:
    """1 セクションの LLM が失敗しても、他のセクションは正常にコンパクションされる。"""
    config, persona = _setup_memory(tmp_path)

    call_count = 0

    def fail_on_second(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("mid_term LLM failure")
        parts = prompt.split("\n\n", 1)
        body = parts[-1] if len(parts) > 1 else prompt
        return body + "（要約済み）"

    result = compact(config, persona, llm_call=fail_on_second)

    assert call_count == 3
    assert len(result.warnings) == 1
    assert "LLM call failed" in result.warnings[0]

    content = (tmp_path / "memory" / "persona.md").read_text(encoding="utf-8")
    assert "要約済み" in content
    assert "好みの内容" in content


# ---------------------------------------------------------------------------
# TC7: preferences セクションは LLM に送られない
# ---------------------------------------------------------------------------

def test_preferences_not_sent_to_llm(tmp_path: Path) -> None:
    """preferences セクションは LLM に送信されず、元テキストがそのまま保持される。"""
    config, persona = _setup_memory(tmp_path)

    prompts_received: list[str] = []

    def tracking_llm(prompt: str) -> str:
        prompts_received.append(prompt)
        parts = prompt.split("\n\n", 1)
        return parts[-1] if len(parts) > 1 else prompt

    compact(config, persona, llm_call=tracking_llm)

    assert len(prompts_received) == 3

    for prompt in prompts_received:
        assert "好みの内容" not in prompt

    content = (tmp_path / "memory" / "persona.md").read_text(encoding="utf-8")
    assert "好みの内容" in content
    assert "## ユーザーの好み・傾向" in content


# ---------------------------------------------------------------------------
# TC8: assemble_memory が正しい構造を生成する
# ---------------------------------------------------------------------------

def test_assemble_memory_produces_correct_structure() -> None:
    """assemble_memory が見出し・セパレータ付きの正しい構造を返す。"""
    result = assemble_memory(
        preferences="好みテキスト",
        long_term="長期テキスト",
        mid_term="中期テキスト",
        recent="直近テキスト",
    )

    assert "## ユーザーの好み・傾向" in result
    assert "## 長期要約（1ヶ月超）" in result
    assert "## 中期要約（1〜3週間前）" in result
    assert "## 直近ログ（7日以内）" in result
    assert result.count("---") == 3
    assert "好みテキスト" in result
    assert result.endswith("\n")


def test_assemble_memory_with_preamble() -> None:
    """assemble_memory が preamble を先頭に配置する。"""
    result = assemble_memory(
        preferences="好み",
        long_term="長期",
        mid_term="中期",
        recent="直近",
        preamble="# メモリファイル",
    )

    assert result.startswith("# メモリファイル")
    assert result.count("---") == 4


def test_assemble_memory_empty_sections() -> None:
    """assemble_memory が空セクションでも見出しを出力する。"""
    result = assemble_memory(
        preferences="",
        long_term="",
        mid_term="",
        recent="",
    )

    assert "## ユーザーの好み・傾向" in result
    assert "## 長期要約（1ヶ月超）" in result
    assert "## 中期要約（1〜3週間前）" in result
    assert "## 直近ログ（7日以内）" in result
