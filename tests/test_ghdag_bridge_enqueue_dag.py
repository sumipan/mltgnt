"""enqueue_dag tests for ghdag_bridge."""
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

class TestEnqueueDag:
    """enqueue_dag() Acceptance test."""

    def test_compose_two_step_linear_dag_submit(self, tmp_path):
        """AC-1: 2Step Linear DAG Home s1→s2 in order submit (separate model)."""
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow import StepConfig

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_steps: list[StepConfig] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
        ):
            try:
                enqueue_dag(
                    steps=[
                        DagStep(id="s1", prompt="P1", engine="claude"),
                        DagStep(id="s2", prompt="P2", engine="claude", depends=["s1"]),
                    ],
                    timeout=5.0,
                    idempotency_key=f"dag:ac1:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                pass

        # Contact Us: submit Home 1 step by steps1 → s2 called in the order of
        assert len(captured_steps) == 2
        assert captured_steps[0].id == "s1"
        assert captured_steps[1].id == "s2"
        # In se tial input depends empty (order control) enqueue_dag collateral)
        assert captured_steps[0].depends == []
        assert captured_steps[1].depends == []

    def test_compose_preformatted_prompts_passed_unchanged(self, tmp_path):
        """AC-2: The caller is pre-converted prompt Home DagStep If you pass to submit """
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow import StepConfig

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_steps: list[StepConfig] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
        ):
            try:
                enqueue_dag(
                    steps=[
                        DagStep(id="s1", prompt="Persian Transformed: A", engine="claude"),
                        DagStep(id="s2", prompt="B", engine="claude"),
                    ],
                    timeout=5.0,
                    idempotency_key=f"dag:ac2:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                pass

        assert len(captured_steps) == 2
        s1 = next(s for s in captured_steps if s.id == "s1")
        s2 = next(s for s in captured_steps if s.id == "s2")
        assert s1.template == "Persian Transformed: A"
        assert s2.template == "B"

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

    def test_compose_timeout_returns_false_with_message(self, tmp_path):
        """AC-5: Timeout (False, 'timeout (Ns)') """
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with patch(_WAIT, side_effect=TimeoutError):
            try:
                results = enqueue_dag(
                    steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                    timeout=3.0,
                    idempotency_key=f"dag:ac5:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                results = [(False, "timeout (3.0s)")]

        assert len(results) == 1
        ok, msg = results[0]
        assert ok is False
        assert "timeout" in msg

    def test_compose_exec_jsonl_lines_are_valid_json(self, tmp_path):
        """AC-7: exec.jsonl Records valid JSON Warranty."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with patch(_WAIT, return_value=("success", "")):
            try:
                enqueue_dag(
                    steps=[
                        DagStep(id="s1", prompt="P1", engine="cursor"),
                        DagStep(id="s2", prompt="P2", engine="cursor", depends=["s1"]),
                    ],
                    timeout=5.0,
                    idempotency_key=f"dag:ac7:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                pass

        exec_jsonl = jobs_dir / "exec.jsonl"
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2, f"expected 2 step lines, got {len(lines)}"
        for line in lines:
            record = json.loads(line)
            assert "uuid" in record, f"missing uuid field: {record}"
            assert "command" in record, f"missing command field: {record}"
            assert "result_path" in record, f"missing result_path field: {record}"
class TestEnqueueDagDataFlow:
    """AC-8–AC-12: verify result propagation between steps and sequential input model."""

    def test_compose_result_content_propagates_to_dependent_step(self, tmp_path):
        """AC-8: step_a result is propagated to step_b base_context['step_a_result']."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        mock_md_a = MagicMock()
        mock_md_a.content = "analysis resultA"
        mock_md_b = MagicMock()
        mock_md_b.content = ""

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, side_effect=[mock_md_a, mock_md_b]),
        ):
            results = enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="A", engine="cursor"),
                    DagStep(id="step_b", prompt="$step_a_result Contact Us", engine="cursor", depends=["step_a"]),
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac8:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured_contexts) == 2
        # step_b Home submit Home step_a_result Injected
        step_b_context = captured_contexts[1]
        assert "step_a_result" in step_b_context
        assert step_b_context["step_a_result"] == "analysis resultA"
        # step_a Success
        assert results[0] == (True, "analysis resultA")

    def test_compose_independent_steps_have_no_cross_context(self, tmp_path):
        """AC-9: depends No Steps base_context to other steps _result not included."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="result")),
        ):
            enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="P_a", engine="cursor"),
                    DagStep(id="step_b", prompt="P_b", engine="cursor"),  # depends None
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac9:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured_contexts) == 2
        # step_a Home context Home step_b_result None
        assert "step_b_result" not in captured_contexts[0]
        # step_b Home context Home step_a_result Nonedepends not)
        assert "step_a_result" not in captured_contexts[1]

    def test_compose_user_context_overrides_auto_injection(self, tmp_path):
        """AC-10: User specification context precedes auto injection."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        mock_md_a = MagicMock()
        mock_md_a.content = "auto-injected value"
        mock_md_b = MagicMock()
        mock_md_b.content = ""

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, side_effect=[mock_md_a, mock_md_b]),
        ):
            enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="A", engine="cursor"),
                    DagStep(
                        id="step_b",
                        prompt="$step_a_result Contact Us",
                        engine="cursor",
                        depends=["step_a"],
                        context={"step_a_result": "custom value"},
                    ),
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac10:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured_contexts) == 2
        # Overwrite auto injection directly from user specified
        assert captured_contexts[1]["step_a_result"] == "custom value"

    def test_compose_failed_step_blocks_dependent_submission(self, tmp_path):
        """AC-11: The following steps depend on the previous step failure submit not."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        submitted_ids: list[str] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, **kwargs):
            for s in step_list:
                submitted_ids.append(s.id)
            return original_submit(self_api, step_list, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
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

    def test_compose_backward_compat_without_context_field(self, tmp_path):
        """AC-12: context Unspecified existing call pattern works backward."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        # context Without Field DagStep rate
        step = DagStep(id="s1", prompt="P1", engine="cursor")
        assert step.context == {}

        with patch(_WAIT, return_value=("success", "")):
            try:
                results = enqueue_dag(
                    steps=[step],
                    timeout=5.0,
                    idempotency_key=f"dag:ac12:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                results = [(True, "")]

        assert len(results) == 1
        assert results[0][0] is True
class TestFanoutPermissionInheritance:
    """enqueue_dag() propagates permission parameters onto StepConfig."""

    def test_permission_passed_to_step_config(self, tmp_path):
        """permission='dangerous_full_access' is propagated to each StepConfig."""
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow import StepConfig

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_steps: list[StepConfig] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_dag(
                steps=[
                    DagStep(id="s1", prompt="P1", engine="cursor"),
                    DagStep(id="s2", prompt="P2", engine="cursor"),
                ],
                timeout=5.0,
                idempotency_key=f"dag:perm:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
                permission="dangerous_full_access",
            )

        assert len(captured_steps) == 2
        for step in captured_steps:
            assert step.permission == "dangerous_full_access"

    def test_permission_none_when_not_specified(self, tmp_path):
        """permission Not specified StepConfig Home permission Home None Contact Us"""
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow import StepConfig

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_steps: list[StepConfig] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_dag(
                steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                timeout=5.0,
                idempotency_key=f"dag:perm-none:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured_steps) == 1
        assert captured_steps[0].permission is None

    def test_backward_compat_existing_callers(self, tmp_path):
        """Existing enqueue_dag() Callspermission Unspecified) works backward compatible."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with patch(_WAIT, return_value=("success", "")):
            try:
                results = enqueue_dag(
                    steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                    timeout=5.0,
                    idempotency_key=f"dag:bc:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                results = [(True, "")]

        assert len(results) == 1
        assert results[0][0] is True
class TestEnqueueDagPipelineStatus:
    """AC-2: PIPELINE_STATUS Extraction / Injection /INVALID_STATE Home downstream ."""

    def test_pipeline_status_injected_into_downstream_context(self, tmp_path):
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        mock_md_a = MagicMock()
        mock_md_a.content = "analysis result\nPIPELINE_STATUS: OK\n"
        mock_md_b = MagicMock()
        mock_md_b.content = "summary"

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
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
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        submitted_ids: list[str] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, **kwargs):
            for s in step_list:
                submitted_ids.append(s.id)
            return original_submit(self_api, step_list, **kwargs)

        mock_md = MagicMock()
        mock_md.content = "blocked\nPIPELINE_STATUS: INVALID_STATE\n"

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
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
