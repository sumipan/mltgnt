"""Tests for mltgnt.memory.archive (Issue #4037)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mltgnt.memory.archive import archive_episodes


def _line(ts: str, content: str) -> str:
    return json.dumps({"timestamp": ts, "role": "user", "content": content, "source_tag": "chat"})


def test_archive_moves_old_lines_by_month(tmp_path: Path) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    lines = [
        _line("2026-05-10 09:00", "may"),
        _line("2026-06-27 23:59", "june old"),
        "not json",
        _line("garbage", "bad ts"),
        _line("2026-06-28 00:00", "june kept"),
        _line("2026-09-20 12:00", "recent"),
    ]
    episodes = persona_dir / "episodes.jsonl"
    episodes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (persona_dir / "episodes").mkdir()
    (persona_dir / "episodes" / "2026-05.jsonl").write_text(_line("2026-05-01 00:00", "pre") + "\n", encoding="utf-8")

    moved = archive_episodes(persona_dir, now=datetime(2026, 9, 26), keep_days=90)

    assert moved == 2
    may = (persona_dir / "episodes" / "2026-05.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["content"] for line in may] == ["pre", "may"]
    june = (persona_dir / "episodes" / "2026-06.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["content"] for line in june] == ["june old"]
    remaining = episodes.read_text(encoding="utf-8").splitlines()
    assert remaining == [lines[2], lines[3], lines[4], lines[5]]
    assert not (persona_dir / "episodes.jsonl.tmp").exists()


def test_archive_nothing_to_move(tmp_path: Path) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    episodes = persona_dir / "episodes.jsonl"
    original = _line("2026-09-20 12:00", "recent") + "\n"
    episodes.write_text(original, encoding="utf-8")
    assert archive_episodes(persona_dir, now=datetime(2026, 9, 26)) == 0
    assert episodes.read_text(encoding="utf-8") == original
    assert not (persona_dir / "episodes").exists()


def test_archive_missing_file(tmp_path: Path) -> None:
    assert archive_episodes(tmp_path / "none", now=datetime(2026, 9, 26)) == 0
