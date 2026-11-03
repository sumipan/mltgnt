"""typecheck_dag and compose_pipeline tests for ghdag_bridge."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.bridges.ghdag_bridge import (
    DagStep,
    SkillIOTypeError,
    compose_pipeline,
    enqueue_and_wait,
    enqueue_dag,
    typecheck_dag,
)
from mltgnt.skill.models import ConsumesSpec, ProducesSpec, SkillMatchResult, SkillMeta
from tests.ghdag_bridge_helpers import (
    _MD_READ,
    _WAIT,
    _make_jobs_dir,
    _make_jobs_dir_dag,
)

def _skill_meta(
    name: str,
    *,
    skill_io: str = "v1",
    produces: ProducesSpec | None = None,
    consumes: list[ConsumesSpec] | None = None,
) -> SkillMeta:
    return SkillMeta(
        name=name,
        description="",
        argument_hint="",
        model=None,
        path=Path("."),
        skill_io=skill_io,
        produces=produces,
        consumes=consumes or [],
    )


class TestTypecheckDag:
    def test_typecheck_dag_matching_types(self):
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/markdown"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        typecheck_dag(steps, skills)

    def test_typecheck_dag_content_type_mismatch(self):
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/plain"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        with pytest.raises(SkillIOTypeError) as exc_info:
            typecheck_dag(steps, skills)
        msg = str(exc_info.value)
        assert "content_type" in msg
        assert "text/markdown" in msg
        assert "text/plain" in msg
        assert "downstream" in msg or "consumer-b" in msg

    def test_typecheck_dag_producer_mismatch(self):
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/markdown"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="other-producer", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        with pytest.raises(SkillIOTypeError):
            typecheck_dag(steps, skills)

    def test_typecheck_dag_legacy_skip(self):
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/plain"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                skill_io="legacy",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        typecheck_dag(steps, skills)

    def test_typecheck_dag_no_skill_name_skip(self):
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/plain"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        typecheck_dag(steps, skills)

    def test_typecheck_dag_skill_not_in_dict_skip(self):
        skills = {
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="missing-upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        typecheck_dag(steps, skills)

    def test_typecheck_dag_empty_consumes_warn(self, capsys):
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/markdown"),
            ),
            "downstream": _skill_meta("consumer-b", consumes=[]),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        typecheck_dag(steps, skills)
        captured = capsys.readouterr()
        assert "WARN" in captured.err
        assert "no consumes" in captured.err

    def test_typecheck_dag_multiple_edges(self):
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/markdown"),
            ),
            "downstream_ok": _skill_meta(
                "consumer-ok",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
            "downstream_bad": _skill_meta(
                "consumer-bad",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/plain")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d1",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream_ok",
            ),
            DagStep(
                id="d2",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream_bad",
            ),
        ]
        with pytest.raises(SkillIOTypeError):
            typecheck_dag(steps, skills)


class TestEnqueueDagTypecheck:
    def test_enqueue_dag_typecheck_on_by_default(self, tmp_path):
        """An unchecked run mismatch raises SkillIOTypeError."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/plain"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(_WAIT, return_value=("success", "")),
        ):
            if "SKILL_IO_TYPECHECK" in os.environ:
                del os.environ["SKILL_IO_TYPECHECK"]
            with pytest.raises(SkillIOTypeError):
                enqueue_dag(
                    steps=steps,
                    timeout=5.0,
                    idempotency_key=f"dag:tc-default:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                    skills=skills,
                )
        assert (jobs_dir / "exec.jsonl").read_text().strip() == ""

    def test_enqueue_dag_typecheck_off_with_zero(self, tmp_path):
        """SKILL_IO_TYPECHECK=0 Home typecheck Skip to contentmismatch ."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/plain"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        with (
            patch.dict(os.environ, {"SKILL_IO_TYPECHECK": "0"}),
            patch(_WAIT, return_value=("success", "")),
        ):
            try:
                enqueue_dag(
                    steps=steps,
                    timeout=5.0,
                    idempotency_key=f"dag:tc-off:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                    skills=skills,
                )
            except (StopIteration, OSError):
                pass

    def test_enqueue_dag_typecheck_on_with_flag(self, tmp_path):
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        skills = {
            "upstream": _skill_meta(
                "producer-a",
                produces=ProducesSpec(content_type="text/plain"),
            ),
            "downstream": _skill_meta(
                "consumer-b",
                consumes=[
                    ConsumesSpec(producer="producer-a", content_type="text/markdown")
                ],
            ),
        }
        steps = [
            DagStep(id="u", prompt="P", engine="cursor", skill_name="upstream"),
            DagStep(
                id="d",
                prompt="P",
                engine="cursor",
                depends=["u"],
                skill_name="downstream",
            ),
        ]
        with patch.dict(os.environ, {"SKILL_IO_TYPECHECK": "1"}):
            with pytest.raises(SkillIOTypeError):
                enqueue_dag(
                    steps=steps,
                    timeout=5.0,
                    idempotency_key=f"dag:tc-on:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                    skills=skills,
                )
        assert (jobs_dir / "exec.jsonl").read_text().strip() == ""

    def test_enqueue_dag_typecheck_skills_none(self, tmp_path):
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        with patch.dict(os.environ, {"SKILL_IO_TYPECHECK": "1"}):
            try:
                enqueue_dag(
                    steps=[DagStep(id="s1", prompt="P", engine="cursor")],
                    timeout=5.0,
                    idempotency_key=f"dag:tc-none:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                    skills=None,
                )
            except (StopIteration, OSError):
                pass
def _match_result(
    meta: SkillMeta | None,
    *,
    arguments: str = "",
    rationale: str = "slash:x",
) -> SkillMatchResult:
    return SkillMatchResult(
        decisive=meta,
        candidates=[meta] if meta is not None else [],
        rationale=rationale if meta is not None else "none",
        arguments=arguments,
    )


class TestComposePipeline:
    """AC-1 / AC-3: match_results → DagStep Syn  of columns."""

    def test_linear_pipe_depends_and_step_ids(self):
        upstream = _skill_meta(
            "research",
            produces=ProducesSpec(content_type="text/markdown"),
        )
        downstream = _skill_meta(
            "summarize",
            consumes=[ConsumesSpec(producer="research", content_type="text/markdown")],
        )
        steps = compose_pipeline(
            [
                _match_result(upstream, arguments="topic"),
                _match_result(downstream, arguments=""),
            ],
            engine="cursor",
            model="gpt-5",
        )
        assert len(steps) == 2
        assert steps[0].id == "pipe_0_research"
        assert steps[0].skill_name == "research"
        assert steps[0].depends == []
        assert steps[0].prompt == "topic"
        assert steps[0].engine == "cursor"
        assert steps[0].model == "gpt-5"
        assert steps[1].id == "pipe_1_summarize"
        assert steps[1].skill_name == "summarize"
        assert steps[1].depends == ["pipe_0_research"]
        assert steps[1].engine == "cursor"

    @pytest.mark.parametrize("engine", ["claude", "cursor", "codex"])
    def test_engine_propagated_to_all_steps(self, engine: str):
        a = _skill_meta("a", produces=ProducesSpec())
        b = _skill_meta(
            "b",
            consumes=[ConsumesSpec(producer="a")],
        )
        steps = compose_pipeline(
            [_match_result(a), _match_result(b)],
            engine=engine,
        )
        assert all(s.engine == engine for s in steps)

    def test_decisive_none_raises_value_error(self):
        with pytest.raises(ValueError, match="decisive"):
            compose_pipeline(
                [_match_result(None)],
                engine="claude",
            )

    def test_producer_mismatch_raises_skill_io_type_error(self):
        upstream = _skill_meta(
            "research",
            produces=ProducesSpec(content_type="text/markdown"),
        )
        downstream = _skill_meta(
            "summarize",
            consumes=[
                ConsumesSpec(producer="other-skill", content_type="text/markdown")
            ],
        )
        with pytest.raises(SkillIOTypeError, match="producer"):
            compose_pipeline(
                [_match_result(upstream), _match_result(downstream)],
                engine="claude",
            )

    def test_legacy_downstream_skips_producer_check(self):
        upstream = _skill_meta(
            "research",
            produces=ProducesSpec(content_type="text/markdown"),
        )
        downstream = _skill_meta(
            "legacy-sum",
            skill_io="legacy",
            consumes=[
                ConsumesSpec(producer="other-skill", content_type="text/markdown")
            ],
        )
        steps = compose_pipeline(
            [_match_result(upstream), _match_result(downstream)],
            engine="claude",
        )
        assert steps[1].depends == ["pipe_0_research"]
