"""tests/test_skill/test_context.py — unit tests for build_extra_context (Issue #3030 / #3173)."""
from __future__ import annotations

import json
from pathlib import Path

from mltgnt.skill.context import build_extra_context
from mltgnt.skill.models import SkillMeta


def _meta(tmp_path: Path, *, knowledge_paths: list[Path] | None = None) -> SkillMeta:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: demo\ndescription: demo\n---\n\nbody",
        encoding="utf-8",
    )
    return SkillMeta(
        name="demo",
        description="demo",
        argument_hint="",
        model=None,
        path=skill_file,
        knowledge_paths=list(knowledge_paths or []),
    )


def _write_memory(repo_root: Path, persona_name: str, text: str) -> None:
    mem_dir = repo_root / "chat" / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / f"{persona_name}.jsonl").write_text(text, encoding="utf-8")


def _jsonl_line(
    *,
    timestamp: str = "2026-04-21 14:25",
    role: str = "user",
    content: str,
    source_tag: str | None = None,
) -> str:
    record: dict = {
        "timestamp": timestamp,
        "role": role,
        "content": content,
    }
    if source_tag is not None:
        record["source_tag"] = source_tag
    return json.dumps(record, ensure_ascii=False)


class TestBuildExtraContext:
    def test_neither_returns_none(self, tmp_path: Path) -> None:
        meta = _meta(tmp_path)
        assert build_extra_context(meta, tmp_path, "persona-a") is None

    def test_defaults_are_zero_even_if_files_exist(self, tmp_path: Path) -> None:
        """AC-1: default knowledge_count/memory_max_bytes are 0 and inject nothing."""
        knowledge = tmp_path / "skills" / "demo" / "knowledge.md"
        knowledge.parent.mkdir(parents=True, exist_ok=True)
        knowledge.write_text("p1\n\np2", encoding="utf-8")
        meta = _meta(tmp_path, knowledge_paths=[knowledge])
        _write_memory(
            tmp_path,
            "persona-a",
            _jsonl_line(content="last night's talk") + "\n",
        )
        assert build_extra_context(meta, tmp_path, "persona-a") is None

    def test_knowledge_only(self, tmp_path: Path) -> None:
        knowledge = tmp_path / "skills" / "demo" / "knowledge.md"
        knowledge.parent.mkdir(parents=True, exist_ok=True)
        knowledge.write_text("p1\n\np2\n\np3", encoding="utf-8")
        meta = _meta(tmp_path, knowledge_paths=[knowledge])

        result = build_extra_context(meta, tmp_path, "persona-a", knowledge_count=2)

        assert result is not None
        assert "### knowledge (last 2)" in result
        assert "p2" in result and "p3" in result
        assert "p1" not in result
        assert "### Memory (tail)" not in result

    def test_memory_only_formatted(self, tmp_path: Path) -> None:
        """AC-2: JSONL is formatted as bullet list and must not include raw JSON."""
        meta = _meta(tmp_path)
        line = _jsonl_line(
            timestamp="2026-04-21 14:25",
            role="user",
            content="Is persona-b around?",
            source_tag="slack",
        )
        _write_memory(tmp_path, "persona-a", line + "\n")

        result = build_extra_context(meta, tmp_path, "persona-a", memory_max_bytes=4096)

        assert result is not None
        assert "### Memory (tail)" in result
        assert "- [2026-04-21 14:25] user: Is persona-b around?" in result
        assert '{"timestamp"' not in result
        assert "### knowledge" not in result

    def test_both(self, tmp_path: Path) -> None:
        knowledge = tmp_path / "skills" / "demo" / "knowledge.md"
        knowledge.parent.mkdir(parents=True, exist_ok=True)
        knowledge.write_text("k1\n\nk2", encoding="utf-8")
        meta = _meta(tmp_path, knowledge_paths=[knowledge])
        _write_memory(
            tmp_path,
            "persona-a",
            _jsonl_line(content="last night's talk") + "\n",
        )

        result = build_extra_context(
            meta,
            tmp_path,
            "persona-a",
            knowledge_count=1,
            memory_max_bytes=4096,
        )

        assert result is not None
        assert "### knowledge (last 1)" in result
        assert "k2" in result and "k1" not in result
        assert "### Memory (tail)" in result
        assert "last night's talk" in result
        assert "- [" in result

    def test_knowledge_subdir_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "skills" / "demo" / "knowledge"
        sub.mkdir(parents=True, exist_ok=True)
        a = sub / "a.md"
        b = sub / "b.md"
        a.write_text("from-a\n\nmid-a", encoding="utf-8")
        b.write_text("from-b", encoding="utf-8")
        meta = _meta(tmp_path, knowledge_paths=sorted([a, b]))

        result = build_extra_context(meta, tmp_path, "persona-a", knowledge_count=2)

        assert result is not None
        assert "mid-a" in result
        assert "from-b" in result
        assert "from-a" not in result

    def test_memory_exclude_source_tags(self, tmp_path: Path) -> None:
        """AC-3: memory_exclude_source_tags excludes matching source_tag entries."""
        meta = _meta(tmp_path)
        lines = "\n".join(
            [
                _jsonl_line(content="keep-slack", source_tag="slack"),
                _jsonl_line(content="drop-observe", source_tag="slack-observe"),
                _jsonl_line(content="keep-bracket", source_tag="[slack-observe]"),
                _jsonl_line(content="keep-none"),
            ]
        )
        _write_memory(tmp_path, "persona-a", lines + "\n")

        result = build_extra_context(
            meta,
            tmp_path,
            "persona-a",
            memory_max_bytes=8192,
            memory_exclude_source_tags=["slack-observe"],
        )

        assert result is not None
        assert "keep-slack" in result
        assert "drop-observe" not in result
        assert "keep-bracket" in result
        assert "keep-none" in result

    def test_memory_skips_invalid_and_empty_content(self, tmp_path: Path) -> None:
        meta = _meta(tmp_path)
        lines = "\n".join(
            [
                "not-json",
                _jsonl_line(content=""),
                _jsonl_line(content="valid"),
            ]
        )
        _write_memory(tmp_path, "persona-a", lines + "\n")

        result = build_extra_context(meta, tmp_path, "persona-a", memory_max_bytes=4096)

        assert result is not None
        assert "valid" in result
        assert "not-json" not in result
