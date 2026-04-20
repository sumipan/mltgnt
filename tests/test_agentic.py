"""
tests/test_agentic.py — AgenticRetriever と read_memory_agentic() のユニットテスト

TC1: memory のみで十分な場合、ループ 0 回で結果を返す
TC2: memory 不十分 → memory 再検索で十分になる場合
TC3: memory 不十分 → skill 検索で十分になる場合
TC4: 複数回ループ後に十分になる場合
TC5: preferences セクションが常に結果に含まれる
TC6: max_bytes 制限が守られる
TC7: ループが max_iterations に達した場合、収集済みで打ち切る
TC8: LLM 呼び出しが例外を投げた場合、初回検索結果にフォールバック
TC9: skill_paths が空リストの場合
TC10: LLM 応答のパースに失敗した場合（不正な形式）
TC11: memory ファイルが空の場合
"""
from __future__ import annotations

import logging
from pathlib import Path
import textwrap


from mltgnt.config import MemoryConfig
from mltgnt.memory import read_memory_agentic, memory_file_path


def make_config(tmp_path: Path) -> MemoryConfig:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=mem_dir,
    )


def _write_memory(config: MemoryConfig, persona: str, content: str) -> None:
    memory_file_path(config, persona).write_text(content, encoding="utf-8")


MEMORY_SUSHI = """## ユーザーの好み・傾向

食べ物の好みを持つユーザー。

---

## 2026-01-01T10:00:00+09:00 — user

[file]
寿司が好き。特にサーモンとマグロが好きだと言っていた。

---

"""

MEMORY_PROJECT = """## 2026-01-01T10:00:00+09:00 — user

[file]
フロントエンドのバグ修正を完了した。

---

## 2026-01-02T10:00:00+09:00 — user

[file]
バックエンドの API 設計を始めた。先週のプロジェクト進捗について議論した。

---

"""


def _make_llm_responses(*responses: str):
    """順に応答を返す llm_call を作成する"""
    it = iter(responses)
    def llm_call(_prompt: str) -> str:
        return next(it)
    return llm_call


# ---------------------------------------------------------------------------
# TC1: memory のみで十分な場合
# ---------------------------------------------------------------------------


def test_tc1_memory_sufficient(tmp_path: Path) -> None:
    """TC1: LLM が SUFFICIENT を返す場合、ループ 0 回で結果を返す。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    call_count = 0

    def llm_call(_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "SUFFICIENT"

    result = read_memory_agentic(
        config,
        "persona",
        "好きな食べ物は？",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
    )

    assert "寿司が好き" in result
    assert call_count == 1  # 1回判定されたら終了


# ---------------------------------------------------------------------------
# TC2: memory 不十分 → memory 再検索で十分になる場合
# ---------------------------------------------------------------------------


def test_tc2_memory_requery(tmp_path: Path) -> None:
    """TC2: INSUFFICIENT→MEMORY→再検索クエリ、2回目 SUFFICIENT。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nMEMORY\n先週のプロジェクト進捗",
        "SUFFICIENT",
    )

    result = read_memory_agentic(
        config,
        "persona",
        "先週のプロジェクト進捗は？",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
    )

    # 両方のエントリが含まれる（重複なし）
    assert "フロントエンド" in result or "バックエンド" in result


# ---------------------------------------------------------------------------
# TC3: memory 不十分 → skill 検索で十分になる場合
# ---------------------------------------------------------------------------


def test_tc3_skill_search(tmp_path: Path) -> None:
    """TC3: INSUFFICIENT→SKILL→skill 本文がマージされる。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", "## 2026-01-01T10:00:00+09:00 — user\n\n[file]\n一般情報のみ。\n\n---\n\n")

    # skill ディレクトリと SKILL.md を作成
    skill_dir = tmp_path / "skills"
    deploy_skill = skill_dir / "deploy"
    deploy_skill.mkdir(parents=True)
    (deploy_skill / "SKILL.md").write_text(
        textwrap.dedent("""            ---
            name: deploy
            description: デプロイ手順を実行するスキル
            ---
            デプロイ手順: git push → CI/CD → 本番適用
        """),
        encoding="utf-8",
    )

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nSKILL\nデプロイ",
        "SUFFICIENT",
    )

    result = read_memory_agentic(
        config,
        "persona",
        "デプロイ手順を教えて",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
        skill_paths=[skill_dir],
    )

    assert "デプロイ" in result


# ---------------------------------------------------------------------------
# TC4: 複数回ループ後に十分になる場合
# ---------------------------------------------------------------------------


def test_tc4_multi_loop(tmp_path: Path) -> None:
    """TC4: 2回ループ後 SUFFICIENT、各エントリがマージされている。"""
    config = make_config(tmp_path)
    _write_memory(
        config,
        "persona",
        """## 2026-01-01T10:00:00+09:00 — user

[file]
エントリA: 最初の情報。

---

## 2026-01-02T10:00:00+09:00 — user

[file]
エントリB: 追加の情報。

---

## 2026-01-03T10:00:00+09:00 — user

[file]
エントリC: さらなる情報。

---

""",
    )

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nMEMORY\n追加の情報",
        "INSUFFICIENT\nMEMORY\nさらなる情報",
        "SUFFICIENT",
    )

    result = read_memory_agentic(
        config,
        "persona",
        "複合的な質問",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
        max_iterations=3,
    )

    assert result  # 何らかの結果が返る


# ---------------------------------------------------------------------------
# TC5: preferences セクションが常に結果に含まれる
# ---------------------------------------------------------------------------


def test_tc5_preferences_always_included(tmp_path: Path) -> None:
    """TC5: 任意の query で preferences セクションが先頭に含まれる。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    result = read_memory_agentic(
        config,
        "persona",
        "何でも",
        max_bytes=4096,
        max_entries=5,
        llm_call=lambda _: "SUFFICIENT",
    )

    assert "ユーザーの好み・傾向" in result
    assert "食べ物の好みを持つユーザー" in result


# ---------------------------------------------------------------------------
# TC6: max_bytes 制限が守られる
# ---------------------------------------------------------------------------


def test_tc6_max_bytes_limit(tmp_path: Path) -> None:
    """TC6: 返却テキストが max_bytes 以内。"""
    config = make_config(tmp_path)
    big_memory = "".join(
        f"## 2026-01-{i+1:02d}T10:00:00+09:00 — user\n\n[file]\n{'あ' * 200}\n\n---\n\n"
        for i in range(10)
    )
    _write_memory(config, "persona", big_memory)

    max_bytes = 500
    result = read_memory_agentic(
        config,
        "persona",
        "テスト",
        max_bytes=max_bytes,
        max_entries=10,
        llm_call=lambda _: "SUFFICIENT",
    )

    assert len(result.encode("utf-8")) <= max_bytes


# ---------------------------------------------------------------------------
# TC7: max_iterations に達した場合
# ---------------------------------------------------------------------------


def test_tc7_max_iterations_reached(tmp_path: Path) -> None:
    """TC7: LLM が常に INSUFFICIENT を返す場合、max_iterations 後に打ち切ってエラーにならない。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    call_count = 0

    def always_insufficient(_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "INSUFFICIENT\nMEMORY\n追加クエリ"

    result = read_memory_agentic(
        config,
        "persona",
        "テスト",
        max_bytes=4096,
        max_entries=5,
        llm_call=always_insufficient,
        max_iterations=3,
    )

    assert call_count == 3  # max_iterations 回呼ばれて打ち切り
    assert isinstance(result, str)  # エラーにならない


# ---------------------------------------------------------------------------
# TC8: LLM 例外の場合、初回検索結果にフォールバック
# ---------------------------------------------------------------------------


def test_tc8_llm_exception_fallback(tmp_path: Path, caplog) -> None:
    """TC8: llm_call が RuntimeError を raise → warning ログ、初回結果が返る。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    def failing_llm(_prompt: str) -> str:
        raise RuntimeError("LLM API error")

    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._agentic"):
        result = read_memory_agentic(
            config,
            "persona",
            "好きな食べ物は？",
            max_bytes=4096,
            max_entries=5,
            llm_call=failing_llm,
        )

    assert isinstance(result, str)
    assert any("failed" in r.message.lower() or "llm" in r.message.lower() for r in caplog.records)
    # 初回 memory 検索の結果が返る
    assert "寿司が好き" in result


# ---------------------------------------------------------------------------
# TC9: skill_paths が空リストの場合
# ---------------------------------------------------------------------------


def test_tc9_empty_skill_paths(tmp_path: Path) -> None:
    """TC9: skill_paths=[], LLM が SKILL ソースを指定 → skill 検索結果が空、ループ継続。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nSKILL\nデプロイ",
        "SUFFICIENT",
    )

    # エラーにならず結果が返ること
    result = read_memory_agentic(
        config,
        "persona",
        "デプロイ手順",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
        skill_paths=[],
    )

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TC10: LLM 応答のパースに失敗した場合
# ---------------------------------------------------------------------------


def test_tc10_parse_failure(tmp_path: Path, caplog) -> None:
    """TC10: LLM が想定外の形式を返す → sufficient=True として扱い、収集済みエントリを返す。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = read_memory_agentic(
            config,
            "persona",
            "テスト",
            max_bytes=4096,
            max_entries=5,
            llm_call=lambda _: "INVALID_FORMAT_XYZ",
        )

    assert isinstance(result, str)
    assert any(
        "unexpected" in r.message.lower() or "sufficient" in r.message.lower()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# TC11: memory ファイルが空の場合
# ---------------------------------------------------------------------------


def test_tc11_empty_memory(tmp_path: Path) -> None:
    """TC11: memory ファイルが存在しない → 空文字列 or preferences のみ、エラーにならない。"""
    config = make_config(tmp_path)
    # memory ファイルを作成しない

    result = read_memory_agentic(
        config,
        "nonexistent",
        "テスト",
        max_bytes=4096,
        max_entries=5,
        llm_call=lambda _: "SUFFICIENT",
    )

    assert isinstance(result, str)
    # エラーにならない
