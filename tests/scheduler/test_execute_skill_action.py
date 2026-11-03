"""execute_action skill action and fanout tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path

from mltgnt.scheduler import PersonaScheduler, ScheduleJob
from mltgnt.skill.models import SkillMeta



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
        "---\n\n## Basic information\n\npersona body",
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

def _enqueue_side_effect(return_value=(True, "")):
    captured: list[dict] = []

    def _fake(**kwargs):
        captured.append(kwargs)
        return return_value

    return captured, _fake


def _dag_side_effect(return_value=None):
    if return_value is None:
        return_value = [(True, "ok")]
    captured: list[dict] = []
    positional: list = []

    def _fake(*args, **kwargs):
        if args:
            positional.append(args[0])
        captured.append(kwargs)
        return return_value

    return captured, positional, _fake



def test_skill_action_success(tmp_path: Path) -> None:
    """skill action: enqueue_and_wait Home (True, stdout) → (True, stdout) """
    sch, meta = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job()

    _capt, _fake = _enqueue_side_effect((True, "Response text"))
    with patch(_ENQUEUE, side_effect=_fake):
        ok, msg = sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert ok is True
    assert msg == "Response text"


def test_skill_action_persona_in_prompt(tmp_path: Path) -> None:
    """Persona body appears in the prompt before the skill body."""
    sch, meta = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job()

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    prompt = _enqueue_kw["prompt"]
    assert "persona body" in prompt
    assert "skill body" in prompt
    assert prompt.index("persona body") < prompt.index("skill body")


def test_skill_action_engine_explicit(tmp_path: Path) -> None:
    """action_args.engine When specified enqueue_and_wait correct engine is passed."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "engine": "gemini"})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert _enqueue_kw["engine"] == "gemini"


def test_skill_action_model_explicit(tmp_path: Path) -> None:
    """action_args.model When specified enqueue_and_wait correct model is passed."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="claude", model="claude-sonnet-4-6")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "model": "claude-opus-4-6"})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert _enqueue_kw["model"] == "claude-opus-4-6"


def test_skill_action_engine_fallback_to_persona(tmp_path: Path) -> None:
    """engine Persona when not specified engine Use the field."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="gemini", model="gemini-2.5-flash")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert _enqueue_kw["engine"] == "gemini"


def test_skill_action_model_fallback_to_persona(tmp_path: Path) -> None:
    """model Persona when not specified model Use the field."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a", engine="gemini", model="gemini-2.5-pro")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert _enqueue_kw["model"] == "gemini-2.5-pro"


def test_skill_action_argv_in_prompt(tmp_path: Path) -> None:
    """argv When specified $ARGUMENTS will be deployed in the skill body."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "$ARGUMENTS Processing")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "argv": ["morning"]})

    _capt, _fake = _enqueue_side_effect((True, "Result"))
    with patch(_ENQUEUE, side_effect=_fake):
        ok, msg = sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert ok is True
    assert msg == "Result"
    assert "morning Processing" in _enqueue_kw["prompt"]


def test_skill_action_no_argv(tmp_path: Path) -> None:
    """argv If not specified, it will be prompted 'Argument:' not included."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert "Argument:" not in _enqueue_kw["prompt"]


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

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    key = _enqueue_kw["idempotency_key"]
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

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    request_id = _enqueue_kw["request_id"]
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

    _capt, _fake = _enqueue_side_effect((True, _FANOUT_RESPONSE))
    _dag_capt, _dag_pos, _dag_fake = _dag_side_effect([(True, "ok1"), (True, "ok2")])
    with patch(_ENQUEUE, side_effect=_fake), patch(_ENQUEUE_DAG, side_effect=_dag_fake):
        sch.execute_action(job)

    wait_rid = _capt[-1]["request_id"]
    dag_rid = _dag_capt[-1]["request_id"]
    assert wait_rid == dag_rid
    assert wait_rid is not None


def test_skill_action_missing_skill_name(tmp_path: Path) -> None:
    """action_args.skill Not specified → (False, Error message)."""
    sch, _ = _make_skill_scheduler(tmp_path)
    job = _skill_job(action_args={"persona": "persona-a"})

    ok, msg = sch.execute_action(job)

    assert ok is False
    assert "action_args.skill" in msg


def test_skill_action_missing_persona(tmp_path: Path) -> None:
    """action_args.persona Not specified → (False, Error message)."""
    sch, _ = _make_skill_scheduler(tmp_path)
    job = _skill_job(action_args={"skill": "test-skill"})

    ok, msg = sch.execute_action(job)

    assert ok is False
    assert "action_args.persona" in msg


def test_skill_action_skill_not_found(tmp_path: Path) -> None:
    """A name absent from the skill registry → (False, 'Skill not found')."""
    sch, _ = _make_skill_scheduler(tmp_path)
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "nonexistent-skill", "persona": "persona-a"})

    ok, msg = sch.execute_action(job)

    assert ok is False
    assert "Skill not found" in msg
    assert "nonexistent-skill" in msg


def test_skill_action_persona_file_not_found(tmp_path: Path) -> None:
    """Persona file absent → (False, 'Persona file not found')."""
    sch, _ = _make_skill_scheduler(tmp_path)
    # Don't make a persona file
    job = _skill_job(action_args={"skill": "test-skill", "persona": "Not Found"})

    ok, msg = sch.execute_action(job)

    assert ok is False
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
# Issue #270: runner.run() Variable subst tion viaAC1-AC4)
# ---------------------------------------------------------------------------


def test_skill_action_substitutes_skill_dir(tmp_path: Path) -> None:
    """AC1: expand $SKILL_DIR to the skill file's parent directory."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "$SKILL_DIR/scripts/run.py Run")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    skill_dir_path = (tmp_path / "skills" / "test-skill").resolve()
    expected = str(skill_dir_path) + "/scripts/run.py Run"
    assert expected in _enqueue_kw["prompt"]


def test_skill_action_substitutes_arguments(tmp_path: Path) -> None:
    """AC2: expand $ARGUMENTS, $0, and $1."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "$ARGUMENTS → $0 $1")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "argv": ["hello", "world"]})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert "hello world → hello world" in _enqueue_kw["prompt"]


def test_skill_action_substitutes_persona_name(tmp_path: Path) -> None:
    """AC5: expand $PERSONA to persona.name."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "Assignee: $PERSONA")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert "Assignee: persona-a" in _enqueue_kw["prompt"]


def test_skill_action_arguments_empty_when_no_argv(tmp_path: Path) -> None:
    """AC2: expand $ARGUMENTS to empty when argv is omitted."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "Argument: [$ARGUMENTS]")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert "Argument: []" in _enqueue_kw["prompt"]


def test_skill_action_uses_format_prompt(tmp_path: Path) -> None:
    """AC3: build the prompt through persona.format_prompt()."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job()

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    prompt = _enqueue_kw["prompt"]
    # Product prompt template strings (src Englishization is out of scope).
    assert "You are the following character" in prompt
    assert "--- User instruction ---" in prompt
    assert "Current datetime:" in prompt


def test_skill_action_model_from_skill_meta(tmp_path: Path) -> None:
    """AC4: skill.meta.model takes priority over action_args.model."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "", model="sonnet")
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "model": "opus"})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert _enqueue_kw["model"] == "sonnet"


def test_skill_action_model_action_args_when_skill_meta_none(tmp_path: Path) -> None:
    """AC4: fall back to action_args.model when skill.meta.model is None."""
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta_with_body("test-skill", tmp_path, "", model=None)
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a", "model": "opus"})

    _capt, _fake = _enqueue_side_effect((True, ""))
    with patch(_ENQUEUE, side_effect=_fake):
        sch.execute_action(job)

    _enqueue_kw = _capt[-1]
    assert _enqueue_kw["model"] == "opus"
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

    _dag_capt, _dag_pos, _dag_fake = _dag_side_effect([(True, "ok1"), (True, "ok2")])
    with patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)), patch(
        _ENQUEUE_DAG, side_effect=_dag_fake
    ):
        ok, msg = sch.execute_action(job)

    assert ok is True
    steps = _dag_pos[0]
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
