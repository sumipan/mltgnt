"""tests/test_fanout_integration.py — integration tests for the fanout path."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler import PersonaScheduler, ScheduleJob
from mltgnt.skill.models import SkillMeta

from tests.scheduler.actions.skill_helpers import _FANOUT_RESPONSE

_ENQUEUE = "mltgnt.bridges.ghdag_bridge.enqueue_and_wait"
_ENQUEUE_DAG = "mltgnt.bridges.ghdag_bridge.enqueue_dag"


def _make_skill_scheduler(tmp_path: Path) -> PersonaScheduler:
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: test-skill\ndescription: test\n---\n\nskill body",
        encoding="utf-8",
    )
    meta = SkillMeta(
        name="test-skill",
        description="test",
        argument_hint="",
        model=None,
        path=skill_file,
    )
    persona_dir = tmp_path / "agents"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona-a.md").write_text(
        "---\npersona:\n  name: persona-a\n"
        "ops:\n  engine: claude\n  model: claude-sonnet-4-6\n"
        "---\n\npersona body",
        encoding="utf-8",
    )
    sch = PersonaScheduler(
        slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path
    )
    sch._skill_registry = {"test-skill": meta}
    return sch


def test_reply_block_children_are_left_to_ghdag(tmp_path: Path) -> None:
    """Issue #7554 AC1: ghdag owns fan-out; mltgnt only waits once and audits."""
    sch = _make_skill_scheduler(tmp_path)
    (tmp_path / "jobs").mkdir()
    job = ScheduleJob.from_dict(
        {
            "id": "fanout_job",
            "mode": "scheduled",
            "action": "skill",
            "notify": "silent",
            "every_day_at": "10:00",
            "action_args": {
                "skill": "test-skill",
                "persona": "persona-a",
                "enable_fanout": True,
            },
        }
    )

    enqueue_calls: list = []

    def _fake_enqueue(**kwargs):
        enqueue_calls.append(kwargs)
        return True, _FANOUT_RESPONSE

    with (
        patch(_ENQUEUE, side_effect=_fake_enqueue) as mock_wait,
        patch(_ENQUEUE_DAG) as mock_dag,
    ):
        ok, msg = sch.execute_action(job)

    assert len(enqueue_calls) == 1
    mock_wait.assert_called_once()
    mock_dag.assert_not_called()
    assert ok is True
    assert msg == _FANOUT_RESPONSE

    audit_lines = (tmp_path / "jobs" / "audit.jsonl").read_text().splitlines()
    fanout_records = [
        json.loads(line)
        for line in audit_lines
        if line.strip() and json.loads(line).get("event_type") == "skill_fanout"
    ]
    assert len(fanout_records) == 1
    assert fanout_records[0]["children"] == 2
    assert fanout_records[0]["ok"] is True
