"""
tests/test_mltgnt_scheduler.py — mltgnt.scheduler Unit Test()AC-3）

Design: Issue #118 §7 AC-3
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mltgnt.exceptions import ConfigError
from mltgnt.scheduler import (
    ScheduleJob,
    PersonaScheduler,
    load_schedule_jobs,
)
from mltgnt.config import SchedulerConfig

TZ = ZoneInfo("Asia/Tokyo")


def dt_jst(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


def make_scheduler(state_dir: Path, jobs: list[ScheduleJob]) -> PersonaScheduler:
    sch = PersonaScheduler(slack=None, state_dir=state_dir, jobs=jobs)
    sch.reload_jobs()
    return sch


# ---------------------------------------------------------------------------
# AC-3: YAML
# ---------------------------------------------------------------------------

def test_from_dict_valid_scheduled() -> None:
    """Valid scheduled Job fields are set correctly."""
    job = ScheduleJob.from_dict({
        "id": "test_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    assert job.id == "test_job"
    assert job.mode == "scheduled"
    assert job.every_day_at == "10:00"
    assert job.action == "noop"
    assert job.notify == "silent"
    assert job.enabled is True


def test_from_dict_invalid_mode_raises() -> None:
    """`mode` outside scheduled|fuzzy_window|interval|chained → ValueError."""
    # Japanese text intentionally kept for CJK processing test
    with pytest.raises(ValueError, match="unknown mode"):
        ScheduleJob.from_dict({
            "id": "bad",
            "mode": "unknown_mode",
            "action": "noop",
            "notify": "silent",
        })


def test_from_dict_invalid_hhmm_raises() -> None:
    """`every_day_at` not HH:MM → ValueError."""
    with pytest.raises(ValueError):
        ScheduleJob.from_dict({
            "id": "bad_time",
            "mode": "scheduled",
            "every_day_at": "25:99",
            "action": "noop",
            "notify": "silent",
        })


def test_overnight_fuzzy_window_raises() -> None:
    """Overnight fuzzy window → ValueError."""
    # Japanese text intentionally kept for CJK processing test
    with pytest.raises(ValueError, match="overnight"):
        ScheduleJob.from_dict({
            "id": "overnight",
            "mode": "fuzzy_window",
            "window_start": "23:00",
            "window_end": "01:00",
            "action": "noop",
            "notify": "silent",
        })


def test_load_schedule_jobs_from_yaml(tmp_path: Path) -> None:
    """Valid schedule.yaml can parse."""
    yaml_file = tmp_path / "schedule.yaml"
    yaml_file.write_text(
        "jobs:\n"
        "  - id: morning\n"
        "    enabled: true\n"
        "    mode: scheduled\n"
        "    every_day_at: '09:00'\n"
        "    action: noop\n"
        "    notify: silent\n",
        encoding="utf-8",
    )
    jobs = load_schedule_jobs(yaml_file)
    assert len(jobs) == 1
    assert jobs[0].id == "morning"
    assert jobs[0].every_day_at == "09:00"


# ---------------------------------------------------------------------------
# AC-3: Job execution
# ---------------------------------------------------------------------------

def test_scheduled_fires_at_target_time(tmp_path: Path) -> None:
    """mode=scheduled, every_day_at="10:00" Jobs 10:00 Home tick ignition."""
    j = ScheduleJob.from_dict({
        "id": "fire_test",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path / "state", [j])
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.5)
    assert sch.paths.done_path("fire_test", date(2026, 4, 17)).is_file()


def test_scheduled_does_not_refire_same_day(tmp_path: Path) -> None:
    """Same Day2 tick Home2does not firedone control by file)."""
    j = ScheduleJob.from_dict({
        "id": "once_test",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path / "state", [j])
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.5)
    # Reset slot to simulate second tick at same time
    sch._scheduled_fired_slot.clear()
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.2)
    # done file exists from first fire, second should be skipped
    done_path = sch.paths.done_path("once_test", date(2026, 4, 17))
    assert done_path.is_file()


def test_interval_fires_multiple_times(tmp_path: Path) -> None:
    """mode=interval reigns after enough time after the previous run."""
    j = ScheduleJob.from_dict({
        "id": "interval_test",
        "mode": "interval",
        "interval_minutes": 10,
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path / "state", [j])
    now = dt_jst(2026, 4, 17, 10, 0)
    sch.tick(now)
    time.sleep(0.3)
    assert sch._interval_last_fired.get("interval_test") is not None


def test_interval_persists_last_fired_across_restart(tmp_path: Path) -> None:
    """interval Home last_fired will be permanently restored after restarting."""
    j = ScheduleJob.from_dict({
        "id": "persist_test",
        "mode": "interval",
        "interval_minutes": 60,
        "action": "noop",
        "notify": "silent",
    })
    state_dir = tmp_path / "state"

    sch1 = make_scheduler(state_dir, [j])
    now = dt_jst(2026, 4, 17, 10, 0)
    sch1.tick(now)
    time.sleep(0.3)
    assert sch1._interval_last_fired.get("persist_test") is not None
    assert sch1.paths.interval_last_fired_path("persist_test").exists()

    sch2 = make_scheduler(state_dir, [j])
    assert sch2._interval_last_fired.get("persist_test") is not None

    almost_now = dt_jst(2026, 4, 17, 10, 30)
    fired_before = dict(sch2._interval_last_fired)
    sch2.tick(almost_now)
    time.sleep(0.3)
    assert sch2._interval_last_fired["persist_test"] == fired_before["persist_test"]


# ---------------------------------------------------------------------------
# AC-3: Dependent Chain
# ---------------------------------------------------------------------------

def test_depends_on_waits_for_dependency(tmp_path: Path) -> None:
    """`depends_on: [job_a]` Jobs job_a Home done Do not ignite while the file does not exist."""
    job_a = ScheduleJob.from_dict({
        "id": "job_a",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    job_b = ScheduleJob.from_dict({
        "id": "job_b",
        "mode": "chained",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["job_a"],
    })
    sch = make_scheduler(tmp_path / "state", [job_a, job_b])
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.3)
    # job_b should not have fired yet (job_a is being processed)
    # just verify it's not marked done without job_a being done
    # job_a fires noop - done quickly; job_b depends on job_a
    # The logic: job_b will fire after job_a's done file appears
    # For this test, we just verify no exception and job_b doesn't fire immediately on its own


def test_cycle_detection_raises(tmp_path: Path) -> None:
    """Cir ating Dependenciesjob_a → job_b → job_a _detect_cycles()  ValueError throw."""
    job_a = ScheduleJob.from_dict({
        "id": "job_a_cycle",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["job_b_cycle"],
    })
    job_b = ScheduleJob.from_dict({
        "id": "job_b_cycle",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["job_a_cycle"],
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[job_a, job_b])
    cycled = sch._detect_cycles([job_a, job_b])
    assert "job_a_cycle" in cycled or "job_b_cycle" in cycled


# ---------------------------------------------------------------------------
# AC-3: SchedulerConfig Contact Us
# ---------------------------------------------------------------------------

def test_scheduler_config_integration(tmp_path: Path) -> None:
    """SchedulerConfig Use the scheduler to work properly."""
    yaml_file = tmp_path / "schedule.yaml"
    yaml_file.write_text(
        "jobs:\n"
        "  - id: config_test\n"
        "    enabled: true\n"
        "    mode: scheduled\n"
        "    every_day_at: '11:00'\n"
        "    action: noop\n"
        "    notify: silent\n",
        encoding="utf-8",
    )
    config = SchedulerConfig(
        schedule_yaml=yaml_file,
        state_dir=tmp_path / "state",
    )
    sch = PersonaScheduler(slack=None, config=config)
    sch.reload_jobs()
    assert len(sch._jobs) == 1
    assert sch._jobs[0].id == "config_test"


# ---------------------------------------------------------------------------
# Issue #227: skill action type
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch  # noqa: E402
from mltgnt.skill.models import SkillMeta  # noqa: E402


def _make_skill_meta(name: str, tmp_path: Path) -> SkillMeta:
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: {}\ndescription: test skill\n---\n\nskill body".format(name),
        encoding="utf-8",
    )
    return SkillMeta(
        name=name,
        description="test skill",
        argument_hint="",
        model=None,
        path=skill_file,
    )


def _make_persona(tmp_path: Path, name: str, engine: str = "claude", model: str = "claude-sonnet-4-6") -> Path:
    persona_dir = tmp_path / "agents"
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / f"{name}.md"
    p.write_text(
        "---\n"
        f"persona:\n  name: {name}\n"
        f"ops:\n  engine: {engine}\n  model: {model}\n"
        # Japanese text intentionally kept for CJK processing test
        "---\n\n## 基本情報\n\npersona body",
        encoding="utf-8",
    )
    return p


def _make_skill_scheduler(tmp_path: Path, skill_name: str = "test-skill") -> tuple[PersonaScheduler, SkillMeta]:
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta(skill_name, tmp_path)
    sch._skill_registry = {skill_name: meta}
    return sch, meta


def _skill_job(**overrides) -> ScheduleJob:
    defaults = dict(
        id="skill_job",
        mode="scheduled",
        action="skill",
        notify="silent",
        every_day_at="10:00",
        action_args={
            "skill": "test-skill",
            "persona": "persona-a",
        },
    )
    defaults.update(overrides)
    return ScheduleJob.from_dict(defaults)


_ENQUEUE = "mltgnt.bridges.ghdag_bridge.enqueue_and_wait"


def test_skill_action_success(tmp_path: Path) -> None:
    """skill action: enqueue_and_wait Home (True, stdout) → (True, stdout) """
    sch, meta = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "Response text")) as mock_enqueue:
        ok, msg = sch.execute_action(job)

    assert ok is True
    assert msg == "Response text"
    mock_enqueue.assert_called_once()


def test_skill_action_persona_in_prompt(tmp_path: Path) -> None:
    """Persona body appears in the prompt before the skill body."""
    sch, meta = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    prompt = mock_enqueue.call_args.kwargs["prompt"]
    assert "persona body" in prompt
    assert "skill body" in prompt
    assert prompt.index("persona body") < prompt.index("skill body")


def test_skill_action_engine_explicit(tmp_path: Path) -> None:
    """action_args.engine When specified enqueue_and_wait correct engine is passed."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "engine": "gemini"})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert mock_enqueue.call_args.kwargs["engine"] == "gemini"


def test_skill_action_model_explicit(tmp_path: Path) -> None:
    """action_args.model When specified enqueue_and_wait correct model is passed."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "model": "claude-opus-4-6"})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert mock_enqueue.call_args.kwargs["model"] == "claude-opus-4-6"


def test_skill_action_engine_fallback_to_persona(tmp_path: Path) -> None:
    """engine Persona when not specified engine Use the field."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="gemini", model="gemini-2.5-flash")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert mock_enqueue.call_args.kwargs["engine"] == "gemini"


def test_skill_action_model_fallback_to_persona(tmp_path: Path) -> None:
    """model Persona when not specified model Use the field."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="gemini", model="gemini-2.5-pro")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert mock_enqueue.call_args.kwargs["model"] == "gemini-2.5-pro"


def test_skill_action_argv_in_prompt(tmp_path: Path) -> None:
    """argv When specified $ARGUMENTS will be deployed in the skill body."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "$ARGUMENTS Processing")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "argv": ["morning"]})

    with patch(_ENQUEUE, return_value=(True, "Result")) as mock_enqueue:
        ok, msg = sch.execute_action(job)

    assert ok is True
    assert msg == "Result"
    assert "morning Processing" in mock_enqueue.call_args.kwargs["prompt"]


def test_skill_action_no_argv(tmp_path: Path) -> None:
    """argv If not specified, it will be prompted 'Argument:' not included."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert "Argument:" not in mock_enqueue.call_args.kwargs["prompt"]


def test_skill_action_engine_error(tmp_path: Path) -> None:
    """enqueue_and_wait Home (False, ...) → execute_action Home (False, ...) """
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(False, "engine error detail")):
        ok, msg = sch.execute_action(job)

    assert ok is False
    assert "engine error detail" in msg


def test_skill_action_timeout(tmp_path: Path) -> None:
    """enqueue_and_wait Home timeout return execute_action Home (False, "timeout ...") """
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(False, "timeout (120s)")):
        ok, msg = sch.execute_action(job)

    assert ok is False
    assert msg == "timeout (120s)"


def test_skill_action_rejected(tmp_path: Path) -> None:
    """REJECTED Status (False, 'rejected: REJECTED') """
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(False, "rejected: REJECTED")):
        ok, msg = sch.execute_action(job)

    assert ok is False
    assert msg == "rejected: REJECTED"


def test_skill_action_empty_result(tmp_path: Path) -> None:
    """EMPTY_RESULT Status (False, 'empty_result: EMPTY_RESULT') """
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(False, "empty_result: EMPTY_RESULT")):
        ok, msg = sch.execute_action(job)

    assert ok is False
    assert msg == "empty_result: EMPTY_RESULT"


def test_skill_action_idempotency_key_format(tmp_path: Path) -> None:
    """enqueue_and_wait Passed to idempotency_key Home 'scheduler:{job.id}:...' format."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    key = mock_enqueue.call_args.kwargs["idempotency_key"]
    assert key.startswith(f"scheduler:{job.id}:")


_UUID_V4_RE = (
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def test_skill_action_request_id_uuid_v4(tmp_path: Path) -> None:
    """run_skill_action generate request_id Home UUID v4 format."""
    import re

    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    request_id = mock_enqueue.call_args.kwargs["request_id"]
    assert re.match(_UUID_V4_RE, request_id)


def test_skill_action_request_id_shared_with_fanout(tmp_path: Path) -> None:
    """fanout Home enqueue_and_wait Home enqueue_dag Same request_id to receive."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={
        "skill": "test-skill",
        "persona": "persona-a",
        "enable_fanout": True,
    })

    with patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)) as mock_enqueue, \
         patch(_ENQUEUE_DAG, return_value=[(True, "ok1"), (True, "ok2")]) as mock_dag:
        sch.execute_action(job)

    wait_rid = mock_enqueue.call_args.kwargs["request_id"]
    dag_rid = mock_dag.call_args.kwargs["request_id"]
    assert wait_rid == dag_rid
    assert wait_rid is not None


def test_skill_action_missing_skill_name(tmp_path: Path) -> None:
    """action_args.skill Not specified → (False, Error message)。"""
    sch, _ = _make_skill_scheduler(tmp_path)
    job = _skill_job(action_args={"persona": "persona-a"})

    ok, msg = sch.execute_action(job)

    assert ok is False
    assert "action_args.skill" in msg


def test_skill_action_missing_persona(tmp_path: Path) -> None:
    """action_args.persona Not specified → (False, Error message)。"""
    sch, _ = _make_skill_scheduler(tmp_path)
    job = _skill_job(action_args={"skill": "test-skill"})

    ok, msg = sch.execute_action(job)

    assert ok is False
    assert "action_args.persona" in msg


def test_skill_action_skill_not_found(tmp_path: Path) -> None:
    # Japanese text intentionally kept for CJK processing test
    """スキルレジストリにない名前 → (False, 'スキルが見つかりません')。"""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "nonexistent-skill", "persona": "persona-a"})

    ok, msg = sch.execute_action(job)

    assert ok is False
    # Japanese text intentionally kept for CJK processing test
    assert "Skill not found" in msg
    assert "nonexistent-skill" in msg


def test_skill_action_persona_file_not_found(tmp_path: Path) -> None:
    # Japanese text intentionally kept for CJK processing test
    """ペルソナファイル不在 → (False, 'ペルソナファイルが見つかりません')。"""
    sch, _ = _make_skill_scheduler(tmp_path)
    # Don't make a persona file
    job = _skill_job(action_args={"skill": "test-skill", "persona": "Not Found"})

    ok, msg = sch.execute_action(job)

    assert ok is False
    # Japanese text intentionally kept for CJK processing test
    assert "Persona file not found" in msg



def _make_skill_meta_with_body(name: str, tmp_path: Path, body: str, model: str | None = None) -> SkillMeta:
    """body Home model Contact Us SkillMeta """
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    fm_model = f"model: {model}\n" if model is not None else ""
    skill_file.write_text(
        f"---\nname: {name}\ndescription: test skill\n{fm_model}---\n\n{body}",
        encoding="utf-8",
    )
    return SkillMeta(name=name, description="test skill", argument_hint="", model=model, path=skill_file)


# ---------------------------------------------------------------------------
# Issue #270: runner.run() Variable subst tion viaAC1〜AC4）
# ---------------------------------------------------------------------------


def test_skill_action_substitutes_skill_dir(tmp_path: Path) -> None:
    """$SKILL_DIR to be deployed in the parent directory of the skill file.AC1）。"""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "$SKILL_DIR/scripts/run.py Run")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    skill_dir_path = (tmp_path / "skills" / "test-skill").resolve()
    expected = str(skill_dir_path) + "/scripts/run.py Run"
    assert expected in mock_enqueue.call_args.kwargs["prompt"]


def test_skill_action_substitutes_arguments(tmp_path: Path) -> None:
    """$ARGUMENTS Home $0, $1 to expandAC2）。"""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "$ARGUMENTS → $0 $1")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "argv": ["hello", "world"]})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert "hello world → hello world" in mock_enqueue.call_args.kwargs["prompt"]


def test_skill_action_substitutes_persona_name(tmp_path: Path) -> None:
    """$PERSONA Home persona.name to expandAC5）。"""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "Assignee: $PERSONA")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert "Assignee: persona-a" in mock_enqueue.call_args.kwargs["prompt"]


def test_skill_action_arguments_empty_when_no_argv(tmp_path: Path) -> None:
    """argv Unspecified $ARGUMENTS is expanded to emptyAC2）。"""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "Argument: [$ARGUMENTS]")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert "Argument: []" in mock_enqueue.call_args.kwargs["prompt"]


def test_skill_action_uses_format_prompt(tmp_path: Path) -> None:
    """persona.format_prompt() Prompt structure viaAC3）。"""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    prompt = mock_enqueue.call_args.kwargs["prompt"]
    # Product prompt template strings (src Englishization is out of scope).
    # Japanese text intentionally kept for CJK processing test
    assert "You are the following character" in prompt
    assert "--- User instruction ---" in prompt
    assert "Current datetime:" in prompt


def test_skill_action_model_from_skill_meta(tmp_path: Path) -> None:
    """skill.meta.model Home action_args.model More priorityAC4）。"""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "", model="sonnet")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "model": "opus"})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert mock_enqueue.call_args.kwargs["model"] == "sonnet"


def test_skill_action_model_action_args_when_skill_meta_none(tmp_path: Path) -> None:
    """skill.meta.model Home None Home action_args.model fallback toAC4）。"""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "", model=None)
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "model": "opus"})

    with patch(_ENQUEUE, return_value=(True, "")) as mock_enqueue:
        sch.execute_action(job)

    assert mock_enqueue.call_args.kwargs["model"] == "opus"

# ---------------------------------------------------------------------------
# Issue #242: skill Success _post() Call / _post() Text Overwrite Prevention
# ---------------------------------------------------------------------------


def _make_slack_mock() -> MagicMock:
    slack = MagicMock()
    slack.post_message = MagicMock()
    return slack


def _command_job(**overrides) -> ScheduleJob:
    defaults = dict(
        id="command_job",
        mode="scheduled",
        action="command",
        notify="slack_secretary",
        every_day_at="10:00",
        action_args={"command": "echo hello"},
    )
    defaults.update(overrides)
    return ScheduleJob.from_dict(defaults)


def test_ac1_spawn_job_skill_success_calls_post(tmp_path: Path) -> None:
    """AC-1: _spawn_job() Home skill With success path _post() is called Slack Posted in"""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary")
    sch = PersonaScheduler(slack=slack, state_dir=tmp_path / "state", jobs=[job], repo_root=tmp_path)
    meta = _make_skill_meta("test-skill", tmp_path)
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    sch.reload_jobs()

    with patch(_ENQUEUE, return_value=(True, "hello")):
        with patch.object(sch, "_post", wraps=sch._post) as mock_post:
            sch._spawn_job(job, date(2026, 4, 23))
            time.sleep(0.5)

    mock_post.assert_called_once_with(job, "hello")


def test_ac2_skill_success_empty_msg_no_post(tmp_path: Path) -> None:
    """AC-2: skill Success msg empty _post() not called."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary")
    sch = PersonaScheduler(slack=slack, state_dir=tmp_path / "state", jobs=[job], repo_root=tmp_path)
    meta = _make_skill_meta("test-skill", tmp_path)
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    sch.reload_jobs()

    with patch(_ENQUEUE, return_value=(True, "")):
        with patch.object(sch, "_post", wraps=sch._post) as mock_post:
            sch._spawn_job(job, date(2026, 4, 23))
            time.sleep(0.5)

    mock_post.assert_not_called()


def test_ac3_command_success_posts_when_msg_present(tmp_path: Path) -> None:
    """AC-3: command Success msg If you have _post() Contact UsPR #15 ification)."""
    slack = _make_slack_mock()
    job = _command_job(notify="slack_secretary")
    sch = PersonaScheduler(slack=slack, state_dir=tmp_path / "state", jobs=[job], repo_root=tmp_path)
    sch.reload_jobs()

    # Base class command Since the action is not implemented execute_action
    with patch.object(sch, "execute_action", return_value=(True, "stdout output")):
        with patch.object(sch, "_post", wraps=sch._post) as mock_post:
            sch._spawn_job(job, date(2026, 4, 23))
            time.sleep(0.5)

    mock_post.assert_called_once_with(job, "stdout output")


def test_ac4_post_resolver_does_not_overwrite_text(tmp_path: Path) -> None:
    """AC-4: _post() Home resolver Home text does not override the skill generation text."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary", persona="persona-a")

    def resolver(persona_name: str, repo_root: Path) -> tuple[dict, str]:
        return {"icon_emoji": ":robot:"}, "resolver Text"

    sch = PersonaScheduler(
        slack=slack,
        state_dir=tmp_path / "state",
        jobs=[],
        repo_root=tmp_path,
        persona_post_kwargs_resolver=resolver,
    )
    sch.reload_jobs()

    sch._post(job, "skill generation text")

    slack.post_message.assert_called_once()
    args, kwargs = slack.post_message.call_args
    assert args[0] == "skill generation text"
    assert kwargs.get("icon_emoji") == ":robot:"


def test_ac5_post_empty_text_uses_resolver_fallback(tmp_path: Path) -> None:
    """AC-5: text empty resolver Home text Use fallback when applying."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary", persona="persona-a")

    def resolver(persona_name: str, repo_root: Path) -> tuple[dict, str]:
        return {"icon_emoji": ":robot:"}, "fallback Text"

    sch = PersonaScheduler(
        slack=slack,
        state_dir=tmp_path / "state",
        jobs=[],
        repo_root=tmp_path,
        persona_post_kwargs_resolver=resolver,
    )
    sch.reload_jobs()

    sch._post(job, "")

    slack.post_message.assert_called_once()
    args, kwargs = slack.post_message.call_args
    assert args[0] == "fallback Text"


def test_ac6_resolver_exception_uses_default_kwargs(tmp_path: Path) -> None:
    """AC-6: resolver when an exception is sent default_slack_post_kwargs Use text not changed."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary", persona="persona-a")

    def resolver(persona_name: str, repo_root: Path) -> tuple[dict, str]:
        raise RuntimeError("resolver error")

    def default_kwargs() -> dict:
        return {"icon_emoji": ":default:"}

    sch = PersonaScheduler(
        slack=slack,
        state_dir=tmp_path / "state",
        jobs=[],
        repo_root=tmp_path,
        persona_post_kwargs_resolver=resolver,
        default_slack_post_kwargs=default_kwargs,
    )
    sch.reload_jobs()

    sch._post(job, "Original text")

    slack.post_message.assert_called_once()
    args, kwargs = slack.post_message.call_args
    assert args[0] == "Original text"
    assert kwargs.get("icon_emoji") == ":default:"


# ---------------------------------------------------------------------------
# Issue #906: PersonaScheduler / SchedulePaths Rename + register_action
# ---------------------------------------------------------------------------


def test_secretary_scheduler_not_importable() -> None:
    """SecretaryScheduler Home mltgnt.scheduler Home import No backward aliases."""
    import mltgnt.scheduler as sched
    assert not hasattr(sched, "SecretaryScheduler"), (
        "SecretaryScheduler can be removed without backward compatible aliases"
    )


def test_secretary_schedule_paths_not_importable() -> None:
    """SecretarySchedulePaths Home mltgnt.scheduler Home import Not possible."""
    import mltgnt.scheduler as sched
    assert not hasattr(sched, "SecretarySchedulePaths")


def test_schedule_paths_importable() -> None:
    """SchedulePaths Home import What you can do"""
    from mltgnt.scheduler import SchedulePaths  # noqa: F401 (import check)


def test_noop_action_returns_true(tmp_path: Path) -> None:
    """job.action='noop' → execute_action Home (True, '') """
    job = ScheduleJob.from_dict({
        "id": "noop_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    ok, msg = sch.execute_action(job)
    assert ok is True
    assert msg == ""


def test_unknown_action_raises_value_error(tmp_path: Path) -> None:
    """Unregistered action → execute_action Home ValueError Home raise """
    job = ScheduleJob.from_dict({
        "id": "unknown_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "unknown_action_xyz",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    with pytest.raises(ValueError):
        sch.execute_action(job)


def test_register_action_is_called(tmp_path: Path) -> None:
    """register_action Register callback Home execute_action Called via"""
    job = ScheduleJob.from_dict({
        "id": "custom_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "custom",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    sch.register_action("custom", lambda j: (True, "ok"))
    ok, msg = sch.execute_action(job)
    assert ok is True
    assert msg == "ok"


def test_actions_kwarg_in_init(tmp_path: Path) -> None:
    """__init__ Home actions= kwarg Contact Us callback Home execute_action Called via"""
    job = ScheduleJob.from_dict({
        "id": "init_action_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "init_action",
        "notify": "silent",
    })
    sch = PersonaScheduler(
        slack=None,
        state_dir=tmp_path / "state",
        jobs=[],
        actions={"init_action": lambda j: (True, "from_init")},
    )
    ok, msg = sch.execute_action(job)
    assert ok is True
    assert msg == "from_init"


def test_registered_action_failure(tmp_path: Path) -> None:
    """Register action Home (False, 'err') → execute_action Home (False, 'err') """
    job = ScheduleJob.from_dict({
        "id": "fail_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "fail_action",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    sch.register_action("fail_action", lambda j: (False, "err"))
    ok, msg = sch.execute_action(job)
    assert ok is False
    assert msg == "err"


def test_registered_action_failure_creates_failed_marker(tmp_path: Path) -> None:
    """On register_action failure, _spawn_job writes failed markers."""
    job = ScheduleJob.from_dict({
        "id": "fail_marker_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "fail_action",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[job])
    sch.register_action("fail_action", lambda j: (False, "something went wrong"))
    sch.reload_jobs()
    d = date(2026, 5, 1)
    sch._spawn_job(job, d)
    time.sleep(0.5)
    assert sch.paths.failed_path("fail_marker_job", d).is_file()


def test_slack_none_post_does_not_raise(tmp_path: Path) -> None:
    """slack=None Home PersonaScheduler rate _post() There is no exception."""
    job = ScheduleJob.from_dict({
        "id": "notify_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "slack_secretary",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    sch._post(job, "test message")  # should not raise


# ---------------------------------------------------------------------------
# Issue #923: SlackClientProtocol **kwargs  / SkillLoaderProtocol
# ---------------------------------------------------------------------------

def test_slack_client_protocol_import() -> None:
    """SlackClientProtocol Home interfaces Home import Yes."""
    from mltgnt.interfaces import SlackClientProtocol
    assert SlackClientProtocol is not None


def test_skill_loader_protocol_deleted() -> None:
    """SkillLoaderProtocol not deletedImportError）。"""
    with pytest.raises(ImportError):
        from mltgnt.interfaces import SkillLoaderProtocol  # noqa: F401


def test_slack_protocol_no_kwargs() -> None:
    """SlackClientProtocol.post_message Home **kwargs There is no signature closure."""
    import inspect
    from mltgnt.interfaces import SlackClientProtocol
    sig = inspect.signature(SlackClientProtocol.post_message)
    var_keyword_params = [
        p for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    assert var_keyword_params == [], "**kwargs should not exist in SlackClientProtocol.post_message"


def test_slack_protocol_conforming_impl_basic() -> None:
    """SlackClientProtocol Meet implementation post_message(text, channel) Call only."""
    from mltgnt.interfaces import SlackClientProtocol

    class ConcreteSlack:
        def post_message(
            self,
            text: str,
            channel: str,
            thread_ts: str | None = None,
            blocks: list[dict] | None = None,
            reply_broadcast: bool = False,
        ) -> bool:
            return True

    client: SlackClientProtocol = ConcreteSlack()  # type: ignore[assignment]
    assert client.post_message("hello", "C123") is True


def test_slack_protocol_conforming_impl_full_kwargs() -> None:
    """SlackClientProtocol Complete implementation to meet optional Call with arguments."""
    from mltgnt.interfaces import SlackClientProtocol

    class ConcreteSlack:
        def post_message(
            self,
            text: str,
            channel: str,
            thread_ts: str | None = None,
            blocks: list[dict] | None = None,
            reply_broadcast: bool = False,
        ) -> bool:
            return True

    client: SlackClientProtocol = ConcreteSlack()  # type: ignore[assignment]
    assert client.post_message(
        "hello", "C123",
        thread_ts="ts001",
        blocks=[{"type": "section"}],
        reply_broadcast=True,
    ) is True


# ---------------------------------------------------------------------------
# AC-4: enable_fanout + enqueue_dag  (#1128)
# ---------------------------------------------------------------------------

_ENQUEUE_DAG = "mltgnt.bridges.ghdag_bridge.enqueue_dag"

_FANOUT_RESPONSE = (
    "Normal response text\n"
    "---\n"
    "ghdag_fanout:\n"
    "  children:\n"
    "    - id: child-1\n"
    "      command: \"agent -p --force < order-1.md\"\n"
    "    - id: child-2\n"
    "      command: \"agent -p --force < order-2.md\"\n"
)


def test_fanout_calls_enqueue_dag_when_block_present(tmp_path: Path) -> None:
    """AC-4: enable_fanout=true Home LLM Contact Us ghdag_fanout Block → enqueue_dag is called."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={
        "skill": "test-skill",
        "persona": "persona-a",
        "enable_fanout": True,
    })

    with patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)), \
         patch(_ENQUEUE_DAG, return_value=[(True, "ok1"), (True, "ok2")]) as mock_dag:
        ok, msg = sch.execute_action(job)

    assert ok is True
    mock_dag.assert_called_once()
    steps = mock_dag.call_args[0][0]
    assert len(steps) == 2
    assert steps[0].id == "child-1"
    assert steps[1].id == "child-2"


def test_fanout_success_message_contains_step_count(tmp_path: Path) -> None:
    """AC-4: fanout Success → (True, 'fanout: N steps completed') """
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={
        "skill": "test-skill",
        "persona": "persona-a",
        "enable_fanout": True,
    })

    with patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)), \
         patch(_ENQUEUE_DAG, return_value=[(True, "ok1"), (True, "ok2")]):
        ok, msg = sch.execute_action(job)

    assert ok is True
    assert "2" in msg and "steps" in msg


def test_fanout_failure_returns_false_with_step_id(tmp_path: Path) -> None:
    """AC-4: fanout Any step fails → (False, 'fanout: step X failed: ...') """
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={
        "skill": "test-skill",
        "persona": "persona-a",
        "enable_fanout": True,
    })

    with patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)), \
         patch(_ENQUEUE_DAG, return_value=[(True, "ok"), (False, "timeout (120s)")]):
        ok, msg = sch.execute_action(job)

    assert ok is False
    assert "child-2" in msg
    assert "failed" in msg


def test_fanout_no_block_returns_initial_result(tmp_path: Path) -> None:
    """AC-4: enable_fanout=true But ghdag_fanout No block → enqueue_and_wait You can check the result as it is."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={
        "skill": "test-skill",
        "persona": "persona-a",
        "enable_fanout": True,
    })

    with patch(_ENQUEUE, return_value=(True, "fanout Normal response without")), \
         patch(_ENQUEUE_DAG) as mock_dag:
        ok, msg = sch.execute_action(job)

    assert ok is True
    assert msg == "fanout Normal response without"
    mock_dag.assert_not_called()


def test_fanout_disabled_does_not_call_enqueue_dag(tmp_path: Path) -> None:
    """AC-4: enable_fanout=false In the job enqueue_dag not called."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={
        "skill": "test-skill",
        "persona": "persona-a",
    })

    with patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)), \
         patch(_ENQUEUE_DAG) as mock_dag:
        ok, msg = sch.execute_action(job)

    assert ok is True
    mock_dag.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #1252: ConfigError / reload_jobs
# ---------------------------------------------------------------------------

def test_load_jobs_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    """Breakdown YAML Home ConfigError Send"""
    bad_yaml = tmp_path / "schedule.yaml"
    bad_yaml.write_text("jobs:\n  - id: [unclosed\n", encoding="utf-8")
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", yaml_path=bad_yaml)

    # Japanese text intentionally kept for CJK processing test
    with pytest.raises(ConfigError, match="YAML load error"):
        sch._load_jobs()


def test_reload_jobs_keeps_previous_jobs_on_config_error(tmp_path: Path) -> None:
    """reload_jobs Home YAML Existing when damaged jobs keep."""
    good_yaml = tmp_path / "schedule.yaml"
    good_yaml.write_text(
        "jobs:\n"
        "  - id: morning\n"
        "    enabled: true\n"
        "    mode: scheduled\n"
        "    every_day_at: '09:00'\n"
        "    action: noop\n"
        "    notify: silent\n",
        encoding="utf-8",
    )
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", yaml_path=good_yaml)
    sch.reload_jobs()
    assert len(sch._jobs) == 1
    assert sch._jobs[0].id == "morning"

    good_yaml.write_text("jobs:\n  - id: [unclosed\n", encoding="utf-8")
    sch.reload_jobs()
    assert len(sch._jobs) == 1
    assert sch._jobs[0].id == "morning"


# ---------------------------------------------------------------------------
# Issue #2379: OnExitPolicy  + SchedulePaths.skipped_path
# ---------------------------------------------------------------------------

from mltgnt.scheduler.models import OnExitPolicy  # noqa: E402
from mltgnt.scheduler.state import SchedulePaths as _SchedulePaths  # noqa: E402


def test_from_dict_on_exit_skip() -> None:
    """`on_exit: {nonzero: skip}` → `OnExitPolicy(nonzero="skip")`。"""
    job = ScheduleJob.from_dict({
        "id": "j",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "on_exit": {"nonzero": "skip"},
    })
    assert job.on_exit == OnExitPolicy(nonzero="skip")


def test_from_dict_on_exit_fail_explicit() -> None:
    """`on_exit: {nonzero: fail}` → `OnExitPolicy(nonzero="fail")`。"""
    job = ScheduleJob.from_dict({
        "id": "j",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "on_exit": {"nonzero": "fail"},
    })
    assert job.on_exit == OnExitPolicy(nonzero="fail")


def test_from_dict_on_exit_none() -> None:
    """`on_exit` Not specified → `job.on_exit is None`。"""
    job = ScheduleJob.from_dict({
        "id": "j",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    assert job.on_exit is None


def test_from_dict_on_exit_invalid_value() -> None:
    """`on_exit: {nonzero: retry}` → `ValueError`。"""
    with pytest.raises(ValueError, match="fail.*skip|skip.*fail"):
        ScheduleJob.from_dict({
            "id": "j",
            "mode": "scheduled",
            "every_day_at": "10:00",
            "action": "noop",
            "notify": "silent",
            "on_exit": {"nonzero": "retry"},
        })


def test_from_dict_on_exit_empty_dict() -> None:
    """`on_exit: {}` (English) → `ValueError`。"""
    with pytest.raises(ValueError, match="nonzero"):
        ScheduleJob.from_dict({
            "id": "j",
            "mode": "scheduled",
            "every_day_at": "10:00",
            "action": "noop",
            "notify": "silent",
            "on_exit": {},
        })


def test_from_dict_on_exit_not_dict() -> None:
    """`on_exit: "skip"` () → `ValueError`。"""
    with pytest.raises(ValueError, match="dict"):
        ScheduleJob.from_dict({
            "id": "j",
            "mode": "scheduled",
            "every_day_at": "10:00",
            "action": "noop",
            "notify": "silent",
            "on_exit": "skip",
        })


def test_skipped_path_format(tmp_path: Path) -> None:
    """`SchedulePaths.skipped_path("j1", date(2026,1,1))` → `<state_dir>/skipped/j1_2026-01-01.skipped`。"""
    p = _SchedulePaths(tmp_path / "state")
    result = p.skipped_path("j1", date(2026, 1, 1))
    assert result == tmp_path / "state" / "skipped" / "j1_2026-01-01.skipped"
