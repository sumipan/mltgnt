"""Context injection tests for run_skill_action."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler.actions.skill import run_skill_action
from mltgnt.skill.models import SkillMeta

from .skill_helpers import (
    _ENQUEUE,
    _ENQUEUE_DAG,
    _FANOUT_RESPONSE,
    _make_persona,
    _make_skill_meta,
    _skill_job,
)

def _write_knowledge(skill_meta: SkillMeta, text: str) -> None:
    path = skill_meta.path.parent / "knowledge.md"
    path.write_text(text, encoding="utf-8")
    skill_meta.knowledge_paths = [path]


def _write_memory(repo_root: Path, persona_name: str, text: str) -> None:
    mem_dir = repo_root / "chat" / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / f"{persona_name}.jsonl").write_text(text, encoding="utf-8")


def _context_injection_records(audit_path: Path) -> list[dict]:
    if not audit_path.exists():
        return []
    return [
        json.loads(line)
        for line in audit_path.read_text().splitlines()
        if line.strip() and json.loads(line).get("event_type") == "context_injection"
    ]


class TestContextInjection:
    """Issue #3021 / #3173: four combinations of knowledge.md × memory file plus audit."""

    def _capture_prompt(
        self,
        tmp_path: Path,
        *,
        with_knowledge: bool,
        with_memory: bool,
        knowledge_count: int | None = 5,
        memory_max_bytes: int | None = 4096,
        memory_exclude_source_tags: list[str] | None = None,
        omit_injection_args: bool = False,
    ) -> tuple[str, list[dict]]:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        if with_knowledge:
            _write_knowledge(
                meta,
                "k1\n\nk2\n\nk3\n\nk4\n\nk5\n\nk6",
            )
        if with_memory:
            mem = (
                '{"timestamp":"2026-09-09 10:00","role":"user",'
                '"content":"last night\'s talk","source_tag":"slack"}\n'
                '{"timestamp":"2026-09-09 10:01","role":"assistant",'
                '"content":"observe-only","source_tag":"slack-observe"}\n'
            )
            _write_memory(tmp_path, "persona-a", mem)
        (tmp_path / "jobs").mkdir(exist_ok=True)
        action_args: dict = {
            "skill": "test-skill",
            "persona": "persona-a",
        }
        if not omit_injection_args:
            if knowledge_count is not None:
                action_args["knowledge_count"] = knowledge_count
            if memory_max_bytes is not None:
                action_args["memory_max_bytes"] = memory_max_bytes
            if memory_exclude_source_tags is not None:
                action_args["memory_exclude_source_tags"] = memory_exclude_source_tags
        job = _skill_job(action_args=action_args)
        captured: dict = {}

        def capture_enqueue(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return True, "ok"

        with patch(_ENQUEUE, side_effect=capture_enqueue):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )
        assert ok is True
        records = _context_injection_records(tmp_path / "jobs" / "audit.jsonl")
        return captured["prompt"], records

    def test_defaults_skip_injection_and_audit(self, tmp_path: Path) -> None:
        """AC-1 / AC-5: unset action_args injects nothing and writes no audit."""
        prompt, records = self._capture_prompt(
            tmp_path,
            with_knowledge=True,
            with_memory=True,
            omit_injection_args=True,
        )
        assert "## Context" not in prompt
        assert records == []

    def test_neither_knowledge_nor_memory(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=False, with_memory=False
        )
        assert "## Context" not in prompt
        # Even when explicitly set, skip audit if real data is empty (#3173)
        assert records == []

    def test_knowledge_only(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=True, with_memory=False, knowledge_count=3
        )
        assert "## Context" in prompt
        assert "### knowledge (last 3)" in prompt
        assert "k4" in prompt and "k5" in prompt and "k6" in prompt
        assert "k3" not in prompt
        assert "### Memory (tail)" not in prompt
        assert len(records) == 1
        assert records[0]["knowledge_count"] == 3
        assert records[0]["memory_bytes"] == 0

    def test_memory_only(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=False, with_memory=True
        )
        assert "## Context" in prompt
        assert "### knowledge" not in prompt
        assert "### Memory (tail)" in prompt
        assert "- [2026-09-09 10:00] user: last night's talk" in prompt
        assert '{"timestamp"' not in prompt
        assert len(records) == 1
        assert records[0]["knowledge_count"] == 0
        assert records[0]["memory_bytes"] > 0

    def test_both_knowledge_and_memory(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=True, with_memory=True, knowledge_count=2
        )
        assert "## Context" in prompt
        assert "### knowledge (last 2)" in prompt
        assert "### Memory (tail)" in prompt
        assert "k5" in prompt and "k6" in prompt
        assert "last night's talk" in prompt
        assert records[0]["event_type"] == "context_injection"
        assert records[0]["skill_name"] == "test-skill"
        assert records[0]["job_id"] == "skill_job"
        assert records[0]["schema_version"] == 1
        assert records[0]["knowledge_count"] == 2
        assert records[0]["memory_bytes"] > 0
        assert "timestamp" in records[0]

    def test_memory_exclude_source_tags_via_action_args(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path,
            with_knowledge=False,
            with_memory=True,
            memory_exclude_source_tags=["slack-observe"],
        )
        assert "## Context" in prompt
        assert "last night's talk" in prompt
        assert "observe-only" not in prompt
        assert len(records) == 1
        assert records[0]["memory_bytes"] > 0
