"""Pipeline and task_timeout_sec tests for run_skill_action."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler.actions.skill import run_skill_action
from mltgnt.scheduler.models import ScheduleJob

from .skill_helpers import (
    _ENQUEUE,
    _ENQUEUE_DAG,
    _FANOUT_RESPONSE,
    _make_persona,
    _make_skill_meta,
    _skill_job,
)

class TestEnablePipeline:
    """Issue #3031: enable_pipeline path."""

    def test_enable_pipeline_calls_compose_and_enqueue_dag(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from mltgnt.skill.models import ConsumesSpec, ProducesSpec, SkillMatchResult

        persona_dir = _make_persona(tmp_path)
        meta_a = _make_skill_meta("skill-a", tmp_path)
        meta_a.skill_io = "v1"
        meta_a.produces = ProducesSpec(content_type="text/markdown")
        meta_b = _make_skill_meta("skill-b", tmp_path)
        meta_b.skill_io = "v1"
        meta_b.produces = ProducesSpec(content_type="text/markdown")
        meta_b.consumes = [
            ConsumesSpec(producer="skill-a", content_type="text/markdown")
        ]
        job = _skill_job(
            action_args={
                "skill": "skill-a",
                "persona": "persona-a",
                "argv": ["/skill-a", "foo", "|", "/skill-b"],
                "enable_pipeline": True,
            }
        )
        match_results = [
            SkillMatchResult(
                decisive=meta_a,
                candidates=[meta_a],
                rationale="slash:skill-a",
                arguments="foo",
            ),
            SkillMatchResult(
                decisive=meta_b,
                candidates=[meta_b],
                rationale="slash:skill-b",
                arguments="",
            ),
        ]
        captured: dict = {}

        def capture_dag(steps, **kwargs):
            captured["steps"] = list(steps)
            return [(True, "out-a"), (True, "out-b")]

        with (
            patch(
                "mltgnt.skill.matcher.match_pipeline",
                new_callable=AsyncMock,
                return_value=match_results,
            ),
            patch(_ENQUEUE_DAG, side_effect=capture_dag),
            patch(_ENQUEUE) as mock_single,
        ):
            ok, msg = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"skill-a": meta_a, "skill-b": meta_b},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert msg == "out-b"
        assert len(captured["steps"]) == 2
        assert captured["steps"][0].id == "pipe_0_skill-a"
        assert captured["steps"][1].id == "pipe_1_skill-b"
        assert captured["steps"][1].depends == ["pipe_0_skill-a"]
        mock_single.assert_not_called()

    def test_enable_pipeline_prefers_over_fanout(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from mltgnt.skill.models import SkillMatchResult

        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(
            action_args={
                "skill": "test-skill",
                "persona": "persona-a",
                "argv": ["/test-skill", "x"],
                "enable_pipeline": True,
                "enable_fanout": True,
            }
        )
        match_results = [
            SkillMatchResult(
                decisive=meta,
                candidates=[meta],
                rationale="slash:test-skill",
                arguments="x",
            )
        ]

        with (
            patch(_ENQUEUE) as mock_single,
            patch(
                "mltgnt.skill.matcher.match_pipeline",
                new_callable=AsyncMock,
                return_value=match_results,
            ),
            patch(_ENQUEUE_DAG, return_value=[(True, "pipe-done")]) as mock_dag,
        ):
            ok, msg = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert msg == "pipe-done"
        mock_single.assert_not_called()
        mock_dag.assert_called_once()


    def _run_pipeline_engine(
        self, tmp_path: Path, *, persona_engine: str, job_engine: str | None
    ) -> dict:
        from unittest.mock import AsyncMock

        from mltgnt.skill.models import SkillMatchResult

        persona_dir = _make_persona(tmp_path, engine=persona_engine)
        meta = _make_skill_meta("test-skill", tmp_path)
        action_args = {
            "skill": "test-skill",
            "persona": "persona-a",
            "argv": ["/test-skill", "x"],
            "enable_pipeline": True,
        }
        if job_engine is not None:
            action_args["engine"] = job_engine
        job = _skill_job(action_args=action_args)
        match_results = [
            SkillMatchResult(
                decisive=meta,
                candidates=[meta],
                rationale="slash:test-skill",
                arguments="x",
            )
        ]
        with (
            patch(
                "mltgnt.skill.matcher.match_pipeline",
                new_callable=AsyncMock,
                return_value=match_results,
            ) as mock_match,
            patch(_ENQUEUE_DAG, return_value=[(True, "pipe-done")]),
        ):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )
        assert ok is True
        mock_match.assert_called_once()
        return mock_match.call_args.kwargs

    def test_enable_pipeline_passes_job_engine_to_match(self, tmp_path: Path) -> None:
        kwargs = self._run_pipeline_engine(
            tmp_path, persona_engine="claude", job_engine="cursor"
        )
        assert kwargs["engine"] == "cursor"

    def test_enable_pipeline_engine_defaults_to_claude(self, tmp_path: Path) -> None:
        kwargs = self._run_pipeline_engine(
            tmp_path, persona_engine='""', job_engine=None
        )
        assert kwargs["engine"] == "claude"
class TestTaskTimeoutSecPassthrough:
    """action_args.task_timeout_sec is forwarded as the ghdag task timeout."""

    @staticmethod
    def _capture_single(tmp_path: Path, action_args: dict) -> dict:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(
            action_args={"skill": "test-skill", "persona": "persona-a", **action_args}
        )
        captured_kwargs: dict = {}

        def capture_enqueue(**kwargs):
            captured_kwargs.update(kwargs)
            return True, "ok"

        with patch(_ENQUEUE, side_effect=capture_enqueue):
            run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )
        return captured_kwargs

    def test_passed_when_set(self, tmp_path: Path, capsys) -> None:
        kwargs = self._capture_single(tmp_path, {"task_timeout_sec": 300})
        assert kwargs["task_timeout_sec"] == 300.0
        assert isinstance(kwargs["task_timeout_sec"], float)
        assert "task_timeout_sec" not in capsys.readouterr().err

    def test_permission_none_when_unset(self, tmp_path: Path, capsys) -> None:
        kwargs = self._capture_single(tmp_path, {})
        assert "task_timeout_sec" in kwargs
        assert kwargs["task_timeout_sec"] is None
        assert "task_timeout_sec" not in capsys.readouterr().err

    def test_invalid_values_are_ignored_with_warning(
        self, tmp_path: Path, capsys
    ) -> None:
        for i, raw in enumerate([0, -5, "300", True]):
            sub = tmp_path / f"case{i}"
            sub.mkdir()
            kwargs = self._capture_single(sub, {"task_timeout_sec": raw})
            assert kwargs["task_timeout_sec"] is None, raw
            err_lines = [
                ln
                for ln in capsys.readouterr().err.splitlines()
                if "task_timeout_sec" in ln
            ]
            assert len(err_lines) == 1, (raw, err_lines)

    def test_fanout_enqueue_dag_receives_timeout(self, tmp_path: Path) -> None:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(action_args={
            "skill": "test-skill",
            "persona": "persona-a",
            "enable_fanout": True,
            "task_timeout_sec": 300,
        })
        single_kwargs: dict = {}
        dag_kwargs: dict = {}

        def capture_enqueue(**kwargs):
            single_kwargs.update(kwargs)
            return True, _FANOUT_RESPONSE

        def capture_dag(steps, **kwargs):
            dag_kwargs.update(kwargs)
            return [(True, "ok1"), (True, "ok2")]

        with patch(_ENQUEUE, side_effect=capture_enqueue), \
             patch(_ENQUEUE_DAG, side_effect=capture_dag):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert single_kwargs["task_timeout_sec"] == 300.0
        assert dag_kwargs["task_timeout_sec"] == 300.0

    def test_pipeline_enqueue_dag_receives_timeout(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from mltgnt.skill.models import SkillMatchResult

        persona_dir = _make_persona(tmp_path)
        meta_a = _make_skill_meta("skill-a", tmp_path)
        job = _skill_job(
            action_args={
                "skill": "skill-a",
                "persona": "persona-a",
                "argv": ["/skill-a", "foo"],
                "enable_pipeline": True,
                "task_timeout_sec": 300,
            }
        )
        match_results = [
            SkillMatchResult(
                decisive=meta_a,
                candidates=[meta_a],
                rationale="slash:skill-a",
                arguments="foo",
            ),
        ]
        dag_kwargs: dict = {}

        def capture_dag(steps, **kwargs):
            dag_kwargs.update(kwargs)
            return [(True, "out-a")]

        with (
            patch(
                "mltgnt.skill.matcher.match_pipeline",
                new_callable=AsyncMock,
                return_value=match_results,
            ),
            patch(_ENQUEUE_DAG, side_effect=capture_dag),
            patch(_ENQUEUE) as mock_single,
        ):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"skill-a": meta_a},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert dag_kwargs["task_timeout_sec"] == 300.0
        mock_single.assert_not_called()
