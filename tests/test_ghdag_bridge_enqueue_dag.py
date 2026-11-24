"""enqueue_dag tests for ghdag_bridge."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ghdag.core import capabilities

from mltgnt.bridges.ghdag_bridge import (
    DagStep,
    SkillIOTypeError,
    _extract_result_filename,
    check_pipeline_status,
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

Permission = capabilities.Permission


def _latest_result_filename(jobs_dir: Path) -> str:
    lines = [
        ln
        for ln in (jobs_dir / "exec.jsonl").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    return _extract_result_filename(lines[-1])


def _wait_writes_result(jobs_dir: Path, content: str):
    """Side effect for wait_for_result: materialize the result file on disk."""

    def _side(exec_done_dir, step_uuid, timeout=None, **kwargs):
        fn = _latest_result_filename(jobs_dir)
        if fn:
            (jobs_dir / fn).write_text(content, encoding="utf-8")
        return ("success", "")

    return _side


class TestCheckPipelineStatus:
    """Issue #7911: reply block first, then legacy PIPELINE_STATUS lines."""

    def test_reply_block_takes_priority_over_legacy_line(self, tmp_path) -> None:
        """AC-1: ghdag reply block wins when both formats appear."""
        result = tmp_path / "result.md"
        result.write_text(
            "PIPELINE_STATUS: LEGACY\n"
            "```ghdag\n"
            "status: ok\n"
            "code: REPLY_CODE\n"
            "```\n",
            encoding="utf-8",
        )
        assert check_pipeline_status(str(result)) == "REPLY_CODE"

    def test_legacy_line_uses_last_pipeline_status_match(self, tmp_path) -> None:
        """AC-2: legacy marker prefers the last ``PIPELINE_STATUS:`` line."""
        result = tmp_path / "result.md"
        result.write_text(
            "PIPELINE_STATUS: FIRST\n"
            "body\n"
            "PIPELINE_STATUS: LAST\n",
            encoding="utf-8",
        )
        assert check_pipeline_status(str(result)) == "LAST"

    def test_missing_file_returns_none(self, tmp_path) -> None:
        """AC-3: unreadable paths yield None."""
        assert check_pipeline_status(str(tmp_path / "missing.md")) is None

    def test_body_without_markers_returns_none(self, tmp_path) -> None:
        """AC-4: no reply block and no legacy line → None."""
        result = tmp_path / "result.md"
        result.write_text("plain output\n", encoding="utf-8")
        assert check_pipeline_status(str(result)) is None


class TestEnqueueDagPermissionPassThrough:
    """ghdag Permission values pass through enqueue_dag to each StepConfig."""

    def test_enqueue_dag_passes_permission_value_to_the_command(self, tmp_path):
        """Ref #7429: DAG-level permission embeds allowed_tools in every step command."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        perm = Permission(allowed_tools=("Read", "Bash(ls:*)"))
        steps = [
            DagStep(
                id="s1",
                prompt="P1",
                engine="claude",
                model="claude-sonnet-4-6",
            ),
            DagStep(
                id="s2",
                prompt="P2",
                engine="claude",
                model="claude-sonnet-4-6",
                depends=["s1"],
            ),
        ]

        with (
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_dag(
                steps=steps,
                timeout=5.0,
                idempotency_key=f"perm:dag:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
                permission=perm,
            )

        lines = [
            ln for ln in (jobs_dir / "exec.jsonl").read_text().splitlines() if ln.strip()
        ]
        assert len(lines) == 2
        for line in lines:
            command = json.loads(line)["command"]
            assert "Read" in command
            assert "Bash(ls:*)" in command


class TestEnqueueDag:
    """enqueue_dag() Acceptance test."""

    def test_compose_empty_steps_raises_value_error(self, tmp_path):
        """AC-3: empty list ValueError('empty' messages)."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with pytest.raises(ValueError, match="empty"):
            enqueue_dag(
                steps=[],
                timeout=5.0,
                idempotency_key="dag:ac3:k",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

    def test_compose_idempotency_prevents_duplicate_submission(self, tmp_path):
        """AC-4: Anti-injection due to idemic properties."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        key = f"dag:ac4:{uuid.uuid4()}"
        steps = [
            DagStep(id="s1", prompt="P1", engine="cursor"),
            DagStep(id="s2", prompt="P2", engine="cursor", depends=["s1"]),
        ]

        for _ in range(2):
            with patch(_WAIT, return_value=("success", "")):
                try:
                    enqueue_dag(
                        steps=steps,
                        timeout=5.0,
                        idempotency_key=key,
                        jobs_dir=jobs_dir,
                        exec_done_dir=done_dir,
                    )
                except (StopIteration, OSError):
                    pass

        exec_jsonl = jobs_dir / "exec.jsonl"
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2, f"line count grew after idempotency check: {len(lines)} lines"

        # 2calls [(True, ""), (True, "")]
        with patch(_WAIT, return_value=("success", "")):
            second_result = enqueue_dag(
                steps=steps,
                timeout=5.0,
                idempotency_key=key,
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )
        assert second_result == [(True, ""), (True, "")]


class TestEnqueueDagDataFlow:
    """AC-8–AC-12: verify result propagation between steps and sequential input model."""




    def test_compose_failed_step_blocks_dependent_submission(self, tmp_path):
        """AC-11: The following steps depend on the previous step failure submit not."""
        from ghdag.pipeline import TaskQueue

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        submitted_ids: list[str] = []
        original_submit = TaskQueue.enqueue

        def capture_submit(self_api, step_list, **kwargs):
            for s in step_list:
                submitted_ids.append(s.id)
            return original_submit(self_api, step_list, **kwargs)

        with (
            patch.object(TaskQueue, "enqueue", capture_submit),
            patch(_WAIT, return_value=("failed_exit", "exit code 1")),
        ):
            results = enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="P_a", engine="cursor"),
                    DagStep(id="step_b", prompt="P_b", engine="cursor", depends=["step_a"]),
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac11:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        # step_a Home submit  step_b Home submit Unrated
        assert "step_a" in submitted_ids
        assert "step_b" not in submitted_ids
        # both False
        assert results[0][0] is False
        assert results[1][0] is False

class TestEnqueueDagPipelineStatus:
    """AC-2: PIPELINE_STATUS Extraction / Injection /INVALID_STATE Home downstream ."""

    def test_pipeline_status_injected_into_downstream_context(self, tmp_path):
        from ghdag.pipeline import TaskQueue

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = TaskQueue.enqueue

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        mock_md_a = MagicMock()
        mock_md_a.content = "analysis result\nPIPELINE_STATUS: OK\n"
        mock_md_b = MagicMock()
        mock_md_b.content = "summary"

        with (
            patch.object(TaskQueue, "enqueue", capture_submit),
            patch(
                _WAIT,
                side_effect=_wait_writes_result(
                    jobs_dir,
                    "analysis result\nPIPELINE_STATUS: OK\n",
                ),
            ),
            patch(_MD_READ, side_effect=[mock_md_a, mock_md_b]),
        ):
            results = enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="P_a", engine="cursor"),
                    DagStep(
                        id="step_b",
                        prompt="P_b",
                        engine="cursor",
                        depends=["step_a"],
                    ),
                ],
                timeout=5.0,
                idempotency_key=f"dag:status:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert results[0][0] is True
        assert results[1][0] is True
        assert captured_contexts[1]["step_a_pipeline_status"] == "OK"
        assert "step_a_result" in captured_contexts[1]

    def test_invalid_state_blocks_downstream(self, tmp_path):
        from ghdag.pipeline import TaskQueue

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        submitted_ids: list[str] = []
        original_submit = TaskQueue.enqueue

        def capture_submit(self_api, step_list, **kwargs):
            for s in step_list:
                submitted_ids.append(s.id)
            return original_submit(self_api, step_list, **kwargs)

        mock_md = MagicMock()
        mock_md.content = "blocked\nPIPELINE_STATUS: INVALID_STATE\n"

        with (
            patch.object(TaskQueue, "enqueue", capture_submit),
            patch(
                _WAIT,
                side_effect=_wait_writes_result(
                    jobs_dir,
                    "blocked\nPIPELINE_STATUS: INVALID_STATE\n",
                ),
            ),
            patch(_MD_READ, return_value=mock_md),
        ):
            results = enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="P_a", engine="cursor"),
                    DagStep(
                        id="step_b",
                        prompt="P_b",
                        engine="cursor",
                        depends=["step_a"],
                    ),
                ],
                timeout=5.0,
                idempotency_key=f"dag:invalid:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert "step_a" in submitted_ids
        assert "step_b" not in submitted_ids
        assert results[0][0] is False
        assert "INVALID_STATE" in results[0][1]
        assert results[1] == (False, "dependency failed")

    def test_reply_block_invalid_state_blocks_downstream(self, tmp_path):
        """Issue #7554 AC3: reply block INVALID_STATE blocks downstream submit."""
        from ghdag.pipeline import TaskQueue

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        submitted_ids: list[str] = []
        original_submit = TaskQueue.enqueue

        def capture_submit(self_api, step_list, **kwargs):
            for s in step_list:
                submitted_ids.append(s.id)
            return original_submit(self_api, step_list, **kwargs)

        reply_block = (
            "```ghdag\n"
            "status: failed\n"
            "code: INVALID_STATE\n"
            "```\n"
        )
        mock_md = MagicMock()
        mock_md.content = reply_block

        with (
            patch.object(TaskQueue, "enqueue", capture_submit),
            patch(
                _WAIT,
                side_effect=_wait_writes_result(jobs_dir, reply_block),
            ),
            patch(_MD_READ, return_value=mock_md),
        ):
            results = enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="P_a", engine="cursor"),
                    DagStep(
                        id="step_b",
                        prompt="P_b",
                        engine="cursor",
                        depends=["step_a"],
                    ),
                ],
                timeout=5.0,
                idempotency_key=f"dag:reply-invalid:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert "step_a" in submitted_ids
        assert "step_b" not in submitted_ids
        assert results[0][0] is False
        assert "INVALID_STATE" in results[0][1]
        assert results[1] == (False, "dependency failed")
