"""
tests/test_memory_relevance.py — read_memory_by_relevance() の結合テスト

TC1: スコア順選択
TC2: preferences 常時包含
TC3: max_entries 制限
TC4: max_bytes 制限
TC5: 日本語テキスト対応
TC6: スコアリングエラーフォールバック
TC7: 空 memory
TC8: preferences のみ
TC9: 空クエリ
TC10: エントリが1件のみ

Note: Issue #198 で embedding ベースから TF-IDF ベースに変更。
embedding_call パラメータを削除し、TF-IDF をローカルで使用する。
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from mltgnt.config import MemoryConfig
from mltgnt.memory import read_memory_by_relevance, read_memory_tail_text, memory_file_path, read_memory_with_sufficiency_check
from mltgnt.memory._scoring import ScoredEntry


def make_config(tmp_path: Path) -> MemoryConfig:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=mem_dir,
    )


def _write_memory(config: MemoryConfig, persona: str, content: str) -> None:
    # Markdown 形式で .md に書き込む。_ensure_jsonl が読み込み時に自動マイグレーションを実行する。
    md_path = memory_file_path(config, persona).with_suffix(".md")
    md_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# TC1: スコア順選択
# ---------------------------------------------------------------------------


MEMORY_THREE_ENTRIES = """\
## 2026-01-01T10:00:00+09:00 — user

[file]
料理のレシピについて話した。おいしいパスタの作り方を学んだ。

---

## 2026-01-02T10:00:00+09:00 — user

[file]
Python のデコレータについて調べた。コードの再利用性が高まる。

---

## 2026-01-03T10:00:00+09:00 — user

[file]
今日の天気は晴れだった。気温が上がってきた。

---

"""


def test_tc1_score_ordering(tmp_path: Path) -> None:
    """TC1: Python クエリに対し、プログラミングエントリが最上位で返る。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python デコレータ",
        max_bytes=4096,
        max_entries=3,
    )

    # プログラミングエントリが料理・天気より先に来ていること
    prog_idx = result.find("Python のデコレータ")
    cook_idx = result.find("料理のレシピ")
    assert prog_idx != -1
    assert cook_idx != -1
    assert prog_idx < cook_idx


# ---------------------------------------------------------------------------
# TC2: preferences 常時包含
# ---------------------------------------------------------------------------


MEMORY_WITH_PREFERENCES = """\
## ユーザーの好み・傾向

プログラミングが得意。Python を主に使う。

---

## 2026-01-01T10:00:00+09:00 — user

[file]
料理のレシピについて話した。

---

## 2026-01-02T10:00:00+09:00 — user

[file]
天気の話をした。

---

"""


def test_tc2_preferences_always_included(tmp_path: Path) -> None:
    """TC2: preferences セクションはスコアリング結果に関わらず出力に含まれる。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_WITH_PREFERENCES)

    result = read_memory_by_relevance(
        config,
        "persona",
        "今日の天気",
        max_bytes=4096,
        max_entries=1,
    )

    assert "ユーザーの好み・傾向" in result
    assert "Python を主に使う" in result


# ---------------------------------------------------------------------------
# TC3: max_entries 制限
# ---------------------------------------------------------------------------


def _make_10_entries_memory() -> str:
    entries = []
    for i in range(10):
        entries.append(
            f"## 2026-01-{i+1:02d}T10:00:00+09:00 — user\n\n[file]\nentry {i}\n\n---\n\n"
        )
    return "".join(entries)


def test_tc3_max_entries_limit(tmp_path: Path) -> None:
    """TC3: 10 件エントリに max_entries=3 → 3 件のみ返る。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", _make_10_entries_memory())

    result = read_memory_by_relevance(
        config,
        "persona",
        "entry query",
        max_bytes=65536,
        max_entries=3,
    )

    # "entry N" の出現回数 ≤ 3
    entry_count = sum(1 for i in range(10) if f"entry {i}" in result)
    assert entry_count <= 3


# ---------------------------------------------------------------------------
# TC4: max_bytes 制限
# ---------------------------------------------------------------------------


def test_tc4_max_bytes_limit(tmp_path: Path) -> None:
    """TC4: 上位エントリの合計が max_bytes を超える場合、バイト数以内に収まる。"""
    config = make_config(tmp_path)
    big_entries = "".join(
        f"## 2026-01-{i+1:02d}T10:00:00+09:00 — user\n\n[file]\n{'x' * 500}\n\n---\n\n"
        for i in range(10)
    )
    _write_memory(config, "persona", big_entries)

    max_bytes = 800
    result = read_memory_by_relevance(
        config,
        "persona",
        "test",
        max_bytes=max_bytes,
        max_entries=10,
    )

    assert len(result.encode("utf-8")) <= max_bytes


# ---------------------------------------------------------------------------
# TC5: 日本語テキスト対応
# ---------------------------------------------------------------------------


def test_tc5_japanese_text(tmp_path: Path) -> None:
    """TC5: 日本語テキストに対して TF-IDF ベクトル化が正常に動作し、スコアが返る。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python デコレータ コード",
        max_bytes=4096,
        max_entries=3,
    )

    # 結果が空でなく、エントリが含まれること
    assert result
    assert "Python のデコレータ" in result


# ---------------------------------------------------------------------------
# TC6: スコアリングエラーフォールバック
# ---------------------------------------------------------------------------


def test_tc6_scoring_error_fallback(tmp_path: Path, caplog) -> None:
    """TC6: score_entries() が例外を送出 → read_memory_tail_text() と同等の結果が返る。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_tail_text(
        config, "persona", max_bytes=4096, max_entries=5
    )

    with caplog.at_level(logging.WARNING, logger="mltgnt.memory"):
        with patch("mltgnt.memory._scoring.score_entries", side_effect=RuntimeError("TF-IDF error")):
            result = read_memory_by_relevance(
                config,
                "persona",
                "何か",
                max_bytes=4096,
                max_entries=5,
            )

    assert result == expected
    assert any(
        "tfidf" in r.message.lower()
        or "error" in r.message.lower()
        or "fallback" in r.message.lower()
        or "scoring" in r.message.lower()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# TC7: 空 memory
# ---------------------------------------------------------------------------


def test_tc7_empty_memory(tmp_path: Path) -> None:
    """TC7: memory ファイルが存在しない場合、空文字列が返る。"""
    config = make_config(tmp_path)

    result = read_memory_by_relevance(
        config,
        "nonexistent",
        "何か質問",
        max_bytes=4096,
        max_entries=5,
    )

    assert result == ""


# ---------------------------------------------------------------------------
# TC8: preferences のみ
# ---------------------------------------------------------------------------


MEMORY_PREFERENCES_ONLY = """\
## ユーザーの好み・傾向

プログラミングが好き。

---

"""


def test_tc8_preferences_only(tmp_path: Path) -> None:
    """TC8: preferences のみの memory → preferences のみ返る。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PREFERENCES_ONLY)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python について",
        max_bytes=4096,
        max_entries=5,
    )

    assert "ユーザーの好み・傾向" in result
    assert "プログラミングが好き" in result


# ---------------------------------------------------------------------------
# TC9: 空クエリ
# ---------------------------------------------------------------------------


def test_tc9_empty_query_fallback(tmp_path: Path) -> None:
    """TC9: クエリが空文字列 → read_memory_tail_text() にフォールバック。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_tail_text(
        config, "persona", max_bytes=4096, max_entries=5
    )

    result = read_memory_by_relevance(
        config,
        "persona",
        "",
        max_bytes=4096,
        max_entries=5,
    )

    assert result == expected


# ---------------------------------------------------------------------------
# TC10: エントリが1件のみ
# ---------------------------------------------------------------------------


MEMORY_SINGLE_ENTRY = """\
## 2026-01-01T10:00:00+09:00 — user

[file]
Python のデコレータについて調べた。

---

"""


def test_tc10_single_entry(tmp_path: Path) -> None:
    """TC10: エントリが1件のみでも TF-IDF が正常に動作する。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SINGLE_ENTRY)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python",
        max_bytes=4096,
        max_entries=5,
    )

    assert "Python のデコレータ" in result


# ---------------------------------------------------------------------------
# Phase 2 tests — read_memory_with_sufficiency_check()
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# TC1: llm_call returns SUFFICIENT → same result as read_memory_by_relevance
# ---------------------------------------------------------------------------


def test_suf_tc1_sufficient_same_as_relevance(tmp_path: Path) -> None:
    """TC1: llm_call が SUFFICIENT を返す → read_memory_by_relevance と同一結果。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_by_relevance(
        config, "persona", "Python デコレータ", max_bytes=4096, max_entries=3
    )

    result = read_memory_with_sufficiency_check(
        config,
        "persona",
        "Python デコレータ",
        max_bytes=4096,
        max_entries=3,
        llm_call=lambda p: "SUFFICIENT",
    )

    assert result == expected


# ---------------------------------------------------------------------------
# TC2: llm_call returns INSUFFICIENT → re-search, results merged
# ---------------------------------------------------------------------------


def test_suf_tc2_insufficient_merges_results(tmp_path: Path) -> None:
    """TC2: INSUFFICIENT → 再検索し3エントリすべてが結果に含まれる。"""
    config = make_config(tmp_path)
    # memory ファイルを作成しておく（preferences 読み込みのため）
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    entry_a = ScoredEntry("エントリA: プロジェクト進捗", 0.9)
    entry_b = ScoredEntry("エントリB: DB接続設定", 0.8)
    entry_c = ScoredEntry("エントリC: 料理", 0.3)

    call_count = [0]

    def mock_search(cfg, persona, q, *, max_entries):
        call_count[0] += 1
        if call_count[0] == 1:
            return [entry_a, entry_c]
        else:
            return [entry_b, entry_a]

    with patch("mltgnt.memory._search_and_score", side_effect=mock_search):
        result = read_memory_with_sufficiency_check(
            config,
            "persona",
            "プロジェクト",
            max_bytes=4096,
            max_entries=10,
            llm_call=lambda p: "INSUFFICIENT\nMEMORY\nDB接続の詳細設定",
        )

    assert "エントリA" in result
    assert "エントリB" in result
    assert "エントリC" in result
    assert call_count[0] == 2


# ---------------------------------------------------------------------------
# TC3: duplicate deduplication
# ---------------------------------------------------------------------------


def test_suf_tc3_deduplication(tmp_path: Path) -> None:
    """TC3: 初回と再検索が同一エントリを返す → 重複なし。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    entries = [
        ScoredEntry("エントリA", 0.9),
        ScoredEntry("エントリB", 0.8),
    ]

    with patch("mltgnt.memory._search_and_score", return_value=entries):
        result = read_memory_with_sufficiency_check(
            config,
            "persona",
            "テスト",
            max_bytes=4096,
            max_entries=10,
            llm_call=lambda p: "INSUFFICIENT\nMEMORY\n追加クエリ",
        )

    # エントリAが2回以上現れないことを確認
    assert result.count("エントリA") == 1
    assert result.count("エントリB") == 1


# ---------------------------------------------------------------------------
# TC4: max_entries limit after merge
# ---------------------------------------------------------------------------


def test_suf_tc4_max_entries_after_merge(tmp_path: Path) -> None:
    """TC4: マージ後も max_entries を超えない。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    first_entries = [ScoredEntry(f"エントリ{i}", float(10 - i) / 10) for i in range(8)]
    second_entries = [ScoredEntry(f"エントリ{i}", float(10 - i) / 10) for i in range(4, 10)]

    call_count = [0]

    def mock_search(cfg, persona, q, *, max_entries):
        call_count[0] += 1
        if call_count[0] == 1:
            return first_entries
        else:
            return second_entries

    with patch("mltgnt.memory._search_and_score", side_effect=mock_search):
        result = read_memory_with_sufficiency_check(
            config,
            "persona",
            "テスト",
            max_bytes=65536,
            max_entries=10,
            llm_call=lambda p: "INSUFFICIENT\nMEMORY\n追加クエリ",
        )

    # 結果に含まれるエントリ数を数える（エントリ0〜エントリ9）
    entry_count = sum(1 for i in range(10) if f"エントリ{i}" in result)
    assert entry_count <= 10


# ---------------------------------------------------------------------------
# TC5: llm_call=None → same as read_memory_by_relevance
# ---------------------------------------------------------------------------


def test_suf_tc5_no_llm_call_same_as_relevance(tmp_path: Path) -> None:
    """TC5: llm_call=None → read_memory_by_relevance() と同一結果。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_by_relevance(
        config, "persona", "Python デコレータ", max_bytes=4096, max_entries=3
    )

    result = read_memory_with_sufficiency_check(
        config,
        "persona",
        "Python デコレータ",
        max_bytes=4096,
        max_entries=3,
        llm_call=None,
    )

    assert result == expected


# ---------------------------------------------------------------------------
# TC7 integration: llm_call raises → initial result returned with warning
# ---------------------------------------------------------------------------


def test_suf_tc7_llm_raises_returns_initial(tmp_path: Path, caplog) -> None:
    """TC7 統合: judge_sufficiency が例外 → 初回結果を返してログに警告。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    initial_entries = [ScoredEntry("Python のデコレータについて調べた。コードの再利用性が高まる。", 0.9)]

    def mock_search(cfg, persona, q, *, max_entries):
        return initial_entries

    with patch("mltgnt.memory._search_and_score", side_effect=mock_search):
        with patch("mltgnt.memory._sufficiency.judge_sufficiency", side_effect=RuntimeError("LLM error")):
            with caplog.at_level(logging.WARNING, logger="mltgnt.memory"):
                result = read_memory_with_sufficiency_check(
                    config,
                    "persona",
                    "Python",
                    max_bytes=4096,
                    max_entries=5,
                    llm_call=lambda p: "SUFFICIENT",
                )

    assert "Python のデコレータ" in result
    assert any(
        "sufficiency" in r.message.lower() or "error" in r.message.lower()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# layers フィルタ: read_memory_by_relevance
# ---------------------------------------------------------------------------


def test_read_memory_by_relevance_layers_filter(tmp_path: Path) -> None:
    """`layers=["learning"]` 指定時、layer="learning" のエントリのみ返す。"""
    from mltgnt.memory._format import MemoryEntry, serialize_entry

    config = make_config(tmp_path)
    mp = memory_file_path(config, "persona")
    mp.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        MemoryEntry("2030-01-01T00:00:00+09:00", "user", "学びエントリ", "file", layer="learning"),
        MemoryEntry("2030-01-02T00:00:00+09:00", "user", "caveatエントリ", "file", layer="caveat"),
        MemoryEntry("2030-01-03T00:00:00+09:00", "user", "通常エントリ", "file"),
    ]
    with mp.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(serialize_entry(e) + "\n")

    result = read_memory_by_relevance(
        config, "persona", "学び",
        max_bytes=4096, max_entries=10, layers=["learning"],
    )
    assert "学びエントリ" in result
    assert "caveatエントリ" not in result
    assert "通常エントリ" not in result
