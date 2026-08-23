"""tests/loops/test_component.py — LoopsComponent テスト。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mltgnt.config import LoopsConfig
from mltgnt.interfaces.loops import HumanThreadRef
from mltgnt.loops.component import LoopsComponent
from mltgnt.loops import store
from tests.loops.fakes import FakeExecutor, FakeHumanChannel


def _config(tmp_path: Path) -> LoopsConfig:
    obj_dir = tmp_path / "objectives"
    obj_dir.mkdir()
    return LoopsConfig(
        objectives_dir=obj_dir,
        state_dir=tmp_path / "state",
        status_dir=tmp_path / "status",
        jobs_dir=tmp_path / "jobs",
        exec_done_dir=tmp_path / "jobs" / "done",
        persona_dir=tmp_path / "personas",
        default_persona="mizuho",
        fallback_channel="C-fallback",
        poll_interval_sec=0.1,
        max_iterations=5,
    )


def _write_obj(cfg: LoopsConfig, name: str, content: str) -> Path:
    path = cfg.objectives_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def _write_request(cfg: LoopsConfig, name: str, **overrides) -> None:
    payload = {
        "objective_path": "foo.md",
        "channel_id": "C1",
        "thread_ts": "123.456",
        "persona": "mizuho",
        "requested_at": "2026-08-21T13:00:00+09:00",
    }
    payload.update(overrides)
    req_dir = cfg.state_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    (req_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_component_start_stop(tmp_path):
    cfg = _config(tmp_path)
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())
    comp.start()
    assert comp._thread is not None
    comp.stop()
    assert comp._thread is None


def test_excludes_subdirectory(tmp_path):
    cfg = _config(tmp_path)
    sub = cfg.objectives_dir / "status"
    sub.mkdir()
    (sub / "hidden.md").write_text("x", encoding="utf-8")
    (cfg.objectives_dir / "visible.md").write_text("---\nid: v\n---\nbody", encoding="utf-8")

    from mltgnt.loops.objective import list_objective_files

    files = list_objective_files(cfg.objectives_dir)
    assert len(files) == 1
    assert files[0].name == "visible.md"


def test_duplicate_ids_do_not_start_either_objective(tmp_path):
    cfg = _config(tmp_path)
    (cfg.objectives_dir / "one.md").write_text("---\nid: duplicate\n---\nbody one", encoding="utf-8")
    (cfg.objectives_dir / "two.md").write_text("---\nid: duplicate\n---\nbody two", encoding="utf-8")
    channel = FakeHumanChannel()
    comp = LoopsComponent(cfg, channel, FakeExecutor())

    comp._refresh_objectives(initial=True)
    _write_request(cfg, "req.json", objective_path="one.md")
    comp._process_requests()

    assert "duplicate" not in comp._objectives
    assert not (cfg.state_dir / "duplicate" / "state.json").exists()
    assert "duplicate id" in (cfg.status_dir / "duplicate.md").read_text(encoding="utf-8")
    assert any("duplicate id" in n["text"] for n in channel.notifies)
    assert not (cfg.state_dir / "requests" / "req.json").exists()


def test_malformed_yaml_writes_error_status(tmp_path):
    cfg = _config(tmp_path)
    (cfg.objectives_dir / "broken.md").write_text("---\nid: [unterminated\n---\nbody", encoding="utf-8")
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())

    comp._refresh_objectives(initial=True)

    assert "broken" in comp._errors
    assert "broken.md" in comp._errors_by_path
    assert (cfg.status_dir / "broken.md").exists()


def test_placing_md_does_not_auto_start(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(cfg, "foo.md", "# Foo\n\nDo the thing\n")
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())

    comp._refresh_objectives(initial=True)
    assert "foo" in comp._objectives
    assert not (cfg.state_dir / "foo" / "state.json").exists()

    # second refresh also must not start
    comp._refresh_objectives(initial=False)
    assert not (cfg.state_dir / "foo" / "state.json").exists()


def test_valid_request_starts_exactly_one_state(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(cfg, "foo.md", "# Foo\n\nbody\n")
    channel = FakeHumanChannel()
    comp = LoopsComponent(cfg, channel, FakeExecutor())
    comp._refresh_objectives(initial=True)
    _write_request(cfg, "a.json")
    comp._process_requests()

    state = store.load_state(cfg.state_dir, "foo")
    assert state is not None
    assert state.thread == HumanThreadRef("C1", "123.456")
    assert not (cfg.state_dir / "requests" / "a.json").exists()
    assert (cfg.state_dir / "requests" / "consumed" / "a.json").exists()


def test_request_persona_fallback_and_agent_priority(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(
        cfg,
        "with-agent.md",
        "---\nid: with-agent\ntitle: A\nstatus: active\nmax_iterations: 5\nagent: ando\n---\nbody\n",
    )
    _write_obj(
        cfg,
        "no-agent.md",
        "---\nid: no-agent\ntitle: B\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())
    comp._refresh_objectives(initial=True)

    _write_request(cfg, "1.json", objective_path="with-agent.md", persona="from-req")
    _write_request(cfg, "2.json", objective_path="no-agent.md", persona="from-req")
    comp._process_requests()

    assert store.load_state(cfg.state_dir, "with-agent").persona == "ando"
    assert store.load_state(cfg.state_dir, "no-agent").persona == "from-req"


def test_request_default_persona_when_both_empty(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(
        cfg,
        "foo.md",
        "---\nid: foo\ntitle: F\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())
    comp._refresh_objectives(initial=True)
    _write_request(cfg, "a.json", persona="")
    comp._process_requests()
    assert store.load_state(cfg.state_dir, "foo").persona == "mizuho"


def test_request_errors_for_bad_objectives(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(cfg, "empty.md", "---\nid: empty\ntitle: E\nstatus: active\nmax_iterations: 5\n---\n")
    _write_obj(cfg, "broken.md", "---\nid: [unterminated\n---\nbody")
    _write_obj(
        cfg,
        "cancelled.md",
        "---\nid: cancelled\ntitle: C\nstatus: cancelled\nmax_iterations: 5\n---\nbody\n",
    )
    channel = FakeHumanChannel()
    comp = LoopsComponent(cfg, channel, FakeExecutor())
    comp._refresh_objectives(initial=True)

    for name, obj in (
        ("missing.json", "nope.md"),
        ("empty.json", "empty.md"),
        ("broken.json", "broken.md"),
        ("cancelled.json", "cancelled.md"),
    ):
        _write_request(cfg, name, objective_path=obj)

    comp._process_requests()

    assert len(channel.notifies) == 4
    assert all(n["event_id"].startswith("loops:request:") for n in channel.notifies)
    assert not (cfg.state_dir / "empty" / "state.json").exists()
    assert not (cfg.state_dir / "cancelled" / "state.json").exists()
    for name in ("missing.json", "empty.json", "broken.json", "cancelled.json"):
        assert (cfg.state_dir / "requests" / "consumed" / name).exists()


def test_already_running_and_terminal_rerequest(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(
        cfg,
        "foo.md",
        "---\nid: foo\ntitle: F\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    channel = FakeHumanChannel()
    comp = LoopsComponent(cfg, channel, FakeExecutor())
    comp._refresh_objectives(initial=True)
    _write_request(cfg, "first.json")
    comp._process_requests()
    first = store.load_state(cfg.state_dir, "foo")
    assert first is not None

    _write_request(cfg, "dup.json")
    comp._process_requests()
    assert any("already running" in n["text"] for n in channel.notifies)
    assert (cfg.state_dir / "requests" / "consumed" / "dup.json").exists()

    first.status = "done"
    store.save_state(cfg.state_dir, first)
    old_created = first.created_at
    _write_request(cfg, "restart.json")
    comp._process_requests()

    archived = list((cfg.state_dir / "archive").iterdir())
    assert len(archived) == 1
    assert archived[0].name.startswith("foo.archived-")
    assert (archived[0] / "state.json").exists()
    restarted = store.load_state(cfg.state_dir, "foo")
    assert restarted is not None
    assert restarted.status == "clarifying"
    assert restarted.created_at != old_created
    assert "foo" in store.list_restorable_loops(cfg.state_dir)
    assert all(not name.startswith("foo.archived") for name in store.list_restorable_loops(cfg.state_dir))


def test_start_exception_leaves_request_for_retry(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(
        cfg,
        "foo.md",
        "---\nid: foo\ntitle: F\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())
    comp._refresh_objectives(initial=True)
    _write_request(cfg, "a.json")

    with patch.object(comp._engine, "start_loop", side_effect=RuntimeError("boom")):
        comp._process_requests()

    assert (cfg.state_dir / "requests" / "a.json").exists()
    assert store.load_state(cfg.state_dir, "foo") is None


def test_retry_after_save_before_consume_does_not_double_start(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(
        cfg,
        "foo.md",
        "---\nid: foo\ntitle: F\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    channel = FakeHumanChannel()
    comp = LoopsComponent(cfg, channel, FakeExecutor())
    comp._refresh_objectives(initial=True)
    _write_request(cfg, "a.json")

    from mltgnt.loops import requests as requests_mod

    real_consume = requests_mod.consume_request
    calls = {"n": 0}

    def consume_once(state_dir, filename, *, corrupt=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("stop after save")
        return real_consume(state_dir, filename, corrupt=corrupt)

    with patch("mltgnt.loops.component.consume_request", side_effect=consume_once):
        comp._process_requests()

    state = store.load_state(cfg.state_dir, "foo")
    assert state is not None
    assert (cfg.state_dir / "requests" / "a.json").exists()

    # second tick: already running path consumes without new state
    created_at = state.created_at
    with patch("mltgnt.loops.component.consume_request", side_effect=real_consume):
        comp._process_requests()
    state2 = store.load_state(cfg.state_dir, "foo")
    assert state2.created_at == created_at
    assert any("already running" in n["text"] for n in channel.notifies)


def test_frontmatter_refresh_does_not_create_state(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(cfg, "foo.md", "# Foo\n")
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())
    comp._refresh_objectives(initial=True)
    text = (cfg.objectives_dir / "foo.md").read_text(encoding="utf-8")
    assert "id: foo" in text
    assert "title: Foo" in text
    assert not (cfg.state_dir / "foo" / "state.json").exists()


def test_restore_nonterminal_and_cancel_on_delete(tmp_path):
    cfg = _config(tmp_path)
    _write_obj(
        cfg,
        "foo.md",
        "---\nid: foo\ntitle: F\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())
    comp._refresh_objectives(initial=True)
    _write_request(cfg, "a.json")
    comp._process_requests()
    state = store.load_state(cfg.state_dir, "foo")
    assert state.status == "clarifying"

    (cfg.objectives_dir / "foo.md").unlink()
    comp._snapshot = {"force-refresh": 0.0}
    comp._refresh_objectives(initial=False)
    state = store.load_state(cfg.state_dir, "foo")
    assert state.status == "cancelled"


def test_component_action_executor_optional_and_injectable(tmp_path):
    cfg = _config(tmp_path)
    comp = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor())
    assert comp.engine.action_executor is None

    class FakeAction:
        def execute(self, *, request, idempotency_key):
            from mltgnt.interfaces.loops import ActionResult

            return ActionResult(success=True, summary="ok")

    fake = FakeAction()
    comp2 = LoopsComponent(cfg, FakeHumanChannel(), FakeExecutor(), action_executor=fake)
    assert comp2.engine.action_executor is fake
