"""tests/loops/test_component.py — LoopsComponent テスト。"""
from __future__ import annotations

from pathlib import Path

from mltgnt.config import LoopsConfig
from mltgnt.loops.component import LoopsComponent
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
    )


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
