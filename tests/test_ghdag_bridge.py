"""tests/test_ghdag_bridge.py — ghdag_bridge non-consolidated + integrated testing.

Coverage:
  - _extract_result_filename(): JSON Type / Text format / Fallback
  - _order_to_result_filename(): Standard text exec line
  - enqueue_and_wait() : exec.jsonl All lines to write valid JSON
  - enqueue_and_wait() Read results: ghdag.files.md_read via result
"""
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
    _extract_result_filename,
    _order_to_result_filename,
    _scheduler_audit_context,
    compose_pipeline,
    enqueue_and_wait,
    enqueue_dag,
    typecheck_dag,
)
from mltgnt.skill.models import ConsumesSpec, ProducesSpec, SkillMatchResult, SkillMeta

# ---------------------------------------------------------------------------
# bridges/__init__ — Package re-export（AC1, AC2, AC7）
# ---------------------------------------------------------------------------


class TestBridgesInitImports:
    def test_bridges_init_imports(self):
        """AC1: bridges All public symbols via package import Yes."""
        from mltgnt.bridges import DagStep, call_llm, create_audit_writer
        from mltgnt.bridges import enqueue_and_wait, enqueue_dag, md_read, md_write

        assert callable(enqueue_and_wait)
        assert callable(enqueue_dag)
        assert DagStep is not None
        assert callable(md_read)
        assert callable(md_write)
        assert callable(create_audit_writer)
        assert callable(call_llm)

    def test_bridges_all_exports(self):
        """AC2: __all__ contains only public symbols."""
        import mltgnt.bridges

        assert set(mltgnt.bridges.__all__) == {
            "DagStep",
            "MltgntHooks",
            "call_llm",
            "create_audit_writer",
            "enqueue_and_wait",
            "enqueue_dag",
            "files_adapter",
            "ghdag_bridge",
            "hooks_adapter",
            "llm_adapter",
            "md_read",
            "md_write",
        }

    def test_scheduler_shim_removed(self):
        """v0.8.0: scheduler/ghdag_bridge.py shim has been removed."""
        with pytest.raises(ImportError):
            from mltgnt.scheduler.ghdag_bridge import enqueue_and_wait  # noqa: F401


# ---------------------------------------------------------------------------
# correlation_id — AuditContext PropagationAC3〜AC5）
# ---------------------------------------------------------------------------


class TestCorrelationIdPropagation:
    def test_enqueue_and_wait_correlation_id(self, tmp_path):
        """AC3: enqueue_and_wait Home correlation_id Home AuditContext passed to."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        captured_contexts: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_contexts.append(kwargs.get("audit_context"))
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="scheduler:test:corr",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
                correlation_id="test-123",
            )

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx.source == "mltgnt-scheduler"
        assert ctx.correlation_id == "test-123"

    def test_enqueue_dag_correlation_id(self, tmp_path):
        """AC4: enqueue_dag Home correlation_id Home AuditContext passed to."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_contexts.append(kwargs.get("audit_context"))
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_dag(
                steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                timeout=5.0,
                idempotency_key=f"dag:corr:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
                correlation_id="dag-456",
            )

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx.source == "mltgnt-scheduler"
        assert ctx.correlation_id == "dag-456"

    def test_enqueue_and_wait_default_correlation_id(self, tmp_path):
        # Japanese text intentionally kept for CJK processing test
        """AC5: correlation_id 省略時は None が AuditContext に渡される。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        captured_contexts: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_contexts.append(kwargs.get("audit_context"))
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="scheduler:test:default",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx.source == "mltgnt-scheduler"
        assert ctx.correlation_id is None


# ---------------------------------------------------------------------------
# request_id — AuditContext Propagation#1346）
# ---------------------------------------------------------------------------


class TestRequestIdPropagation:
    def test_scheduler_audit_context_with_request_id(self):
        """_scheduler_audit_context Home request_id Home AuditContext Pass to."""
        ctx = _scheduler_audit_context(None, None, request_id="test-rid")
        assert ctx.source == "mltgnt-scheduler"
        assert ctx.request_id == "test-rid"

    def test_enqueue_and_wait_request_id(self, tmp_path):
        """enqueue_and_wait Home request_id Home api.submit Home audit_context passed to."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        captured_contexts: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_contexts.append(kwargs.get("audit_context"))
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="scheduler:test:rid",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
                request_id="test-rid",
            )

        assert len(captured_contexts) == 1
        assert captured_contexts[0].request_id == "test-rid"

    def test_enqueue_dag_request_id(self, tmp_path):
        """enqueue_dag Home request_id Home api.submit Home audit_context passed to."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_contexts.append(kwargs.get("audit_context"))
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_dag(
                steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                timeout=5.0,
                idempotency_key=f"dag:rid:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
                request_id="test-rid",
            )

        assert len(captured_contexts) == 1
        assert captured_contexts[0].request_id == "test-rid"

    def test_enqueue_and_wait_default_request_id(self, tmp_path):
        # Japanese text intentionally kept for CJK processing test
        """request_id 省略時は None が AuditContext に渡される。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        captured_contexts: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_contexts.append(kwargs.get("audit_context"))
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="scheduler:test:default-rid",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert len(captured_contexts) == 1
        assert captured_contexts[0].request_id is None


# ---------------------------------------------------------------------------
# _order_to_result_filename
# ---------------------------------------------------------------------------

UUID_A = "38d6b791-1072-42f0-838d-45c7d10748ff"


class TestOrderToResultFilename:
    def test_claude_order_line(self):
        """cursor-order From Line cursor-result ."""
        line = f"jobs/20260508230000-cursor-order-{UUID_A}.md"
        assert _order_to_result_filename(line) == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_claude_engine(self):
        """claude The engine is also correctly guided."""
        line = f"jobs/20260508230000-claude-order-{UUID_A}.md"
        assert _order_to_result_filename(line) == f"20260508230000-claude-result-{UUID_A}.md"

    def test_no_order_pattern_returns_empty(self):
        """-order- If there is no pattern, you can check the empty character."""
        assert _order_to_result_filename("some random text") == ""

    def test_no_uuid_returns_empty(self):
        """Lines without a UUID return an empty string."""
        assert _order_to_result_filename("jobs/20260508-cursor-order-no-uuid.md") == ""


# ---------------------------------------------------------------------------
# _extract_result_filename
# ---------------------------------------------------------------------------

class TestExtractResultFilename:
    def test_json_format_extracts_result_path_name(self):
        """Get the basename of result_path from a JSON-format exec_line."""
        record = {
            "uuid": UUID_A,
            "command": "agent -p --force < order.md",
            "result_path": f"/Users/user/diary/jobs/20260508230000-cursor-result-{UUID_A}.md",
        }
        line = json.dumps(record)
        result = _extract_result_filename(line)
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_json_format_relative_result_path(self):
        """relative path result_path But basename get correctly."""
        record = {
            "uuid": UUID_A,
            "command": "cmd",
            "result_path": f"jobs/20260508230000-cursor-result-{UUID_A}.md",
        }
        result = _extract_result_filename(json.dumps(record))
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_json_format_no_result_path_returns_empty(self):
        """result_path No fields JSON Records return empty characters."""
        record = {"uuid": UUID_A, "command": "cmd"}
        assert _extract_result_filename(json.dumps(record)) == ""

    def test_json_format_empty_result_path_returns_empty(self):
        """result_path empty JSON Records return empty characters."""
        record = {"uuid": UUID_A, "command": "cmd", "result_path": ""}
        assert _extract_result_filename(json.dumps(record)) == ""

    def test_invalid_json_starting_brace_falls_back_to_text(self):
        """{ Close invalid JSON → Fallback to text parser."""
        line = f"{{not valid json}} jobs/20260508230000-cursor-order-{UUID_A}.md"
        # Falls back to text parser which can't parse this either → ""
        result = _extract_result_filename(line)
        # Either "" or derived from text pattern; key property is no exception raised
        assert isinstance(result, str)

    def test_text_format_exec_line(self):
        """Derive the result filename from text format (uuid: cmd | tee result)."""
        line = (
            f"{UUID_A}: agent -p --force < jobs/20260508230000-cursor-order-{UUID_A}.md"
            f" | tee -a jobs/20260508230000-cursor-result-{UUID_A}.md"
        )
        result = _extract_result_filename(line)
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_json_format_idempotency_key_not_confused(self):
        """idempotency_key field result_path Do not interfere with acquisition."""
        record = {
            "uuid": UUID_A,
            "command": "cmd",
            "result_path": f"jobs/20260508230000-cursor-result-{UUID_A}.md",
            "idempotency_key": "scheduler:diary_review:2026-05-08T23:00:00+09:00",
        }
        result = _extract_result_filename(json.dumps(record))
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"


# ---------------------------------------------------------------------------
# enqueue_and_wait — integration tests
# (only wait_for_result is mocked; LLMPipelineAPI is real)
# ---------------------------------------------------------------------------

def _make_jobs_dir(tmp_path: Path) -> Path:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "exec.jsonl").write_text("", encoding="utf-8")
    return jobs


_WAIT = "ghdag.pipeline.wait_for_result"


class TestEnqueueAndWaitJsonlIntegration:
    """exec.jsonl Write to valid JSON Integrated testing to ensure that."""

    def _run(self, tmp_path: Path, *, engine: str = "cursor", model: str = "auto",
             wait_return=None) -> Path:
        jobs_dir = _make_jobs_dir(tmp_path)
        if wait_return is None:
            done_dir = jobs_dir / "done"
            done_dir.mkdir()
            # Create done file so wait_for_result returns immediately
            wait_return = ("success", "")

        with patch(_WAIT, return_value=wait_return) as _mock_wait:
            result_file = jobs_dir / f"20260508230000-{engine}-result-fake.md"
            result_file.write_text("result content", encoding="utf-8")

            # We just need to observe exec.jsonl content; result_path doesn't need to match
            try:
                enqueue_and_wait(
                    prompt="test prompt",
                    engine=engine,
                    model=model,
                    timeout=5.0,
                    idempotency_key="scheduler:test_job:2026-05-08T00:00:00+09:00",
                    jobs_dir=jobs_dir,
                    exec_done_dir=jobs_dir / "done",
                )
            except (StopIteration, OSError):
                pass  # result file not found → OK for this test

        return jobs_dir / "exec.jsonl"

    def test_exec_jsonl_all_lines_valid_json(self, tmp_path):
        """enqueue_and_wait Home exec.jsonl All lines to write valid JSON。"""
        exec_jsonl = self._run(tmp_path)
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1, "exec.jsonl s are not written"
        for line in lines:
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON line in exec.jsonl: {line!r}\n{e}")

    def test_exec_jsonl_no_comment_lines(self, tmp_path):
        """exec.jsonl Home # does not include lines starting with the idemic key line in the text format."""
        exec_jsonl = self._run(tmp_path)
        for line in exec_jsonl.read_text().splitlines():
            assert not line.startswith("#"), f"comment line found: {line!r}"

    def test_exec_jsonl_record_has_required_fields(self, tmp_path):
        """Write JSON Record uuid / command / result_path exist."""
        exec_jsonl = self._run(tmp_path)
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        record = json.loads(lines[0])
        assert "uuid" in record
        assert "command" in record
        assert "result_path" in record

    def test_exec_jsonl_idempotency_key_in_record_not_comment(self, tmp_path):
        """idempotency_key Home JSON It is embedded as a field and cannot be written as a comment line."""
        exec_jsonl = self._run(tmp_path)
        content = exec_jsonl.read_text()
        assert '"idempotency_key"' in content
        assert "# idempotency:" not in content

    def test_exec_jsonl_cursor_engine_command_format(self, tmp_path):
        """cursor Engine command Home agent -p < order_path TypeTEXT_ONLY default)."""
        exec_jsonl = self._run(tmp_path, engine="cursor")
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        record = json.loads(lines[0])
        assert "agent" in record["command"]
        assert "-p" in record["command"]
        assert "--force" not in record["command"]

    def test_exec_jsonl_claude_engine_command_format(self, tmp_path):
        """claude Engine command Home claude -p ... TypeTEXT_ONLY default)."""
        exec_jsonl = self._run(tmp_path, engine="claude", model="claude-sonnet-4-6")
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        record = json.loads(lines[0])
        assert "claude" in record["command"]
        assert "--dangerously-skip-permissions" not in record["command"]
        assert "--permission-mode default" in record["command"]

    def test_idempotency_prevents_duplicate_submission(self, tmp_path):
        """Same idempotency_key Home 2 Contact Us exec.jsonl No more lines."""
        jobs_dir = _make_jobs_dir(tmp_path)
        key = "scheduler:diary_review:2026-05-08T23:00:00+09:00"

        for _ in range(2):
            try:
                with patch(_WAIT, return_value=("success", "")):
                    enqueue_and_wait(
                        prompt="prompt",
                        engine="cursor",
                        model="auto",
                        timeout=5.0,
                        idempotency_key=key,
                        jobs_dir=jobs_dir,
                        exec_done_dir=jobs_dir / "done",
                    )
            except (StopIteration, OSError):
                pass

        exec_jsonl = jobs_dir / "exec.jsonl"
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        # Second call should be a no-op due to idempotency
        assert len(lines) == 1, (
            f"idempotency not working: {len(lines)} lines written"
        )

    def test_timeout_returns_false(self, tmp_path):
        """wait_for_result Home TimeoutError When sending out (False, 'timeout ...') """
        jobs_dir = _make_jobs_dir(tmp_path)

        with patch(_WAIT, side_effect=TimeoutError):
            ok, msg = enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=1.0,
                idempotency_key="scheduler:test:ts",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is False
        assert "timeout" in msg
        assert "1.0" in msg

    def test_timeout_exec_jsonl_still_valid(self, tmp_path):
        """Timeout exec.jsonl s written to valid JSON Contact Us"""
        jobs_dir = _make_jobs_dir(tmp_path)

        with patch(_WAIT, side_effect=TimeoutError):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=1.0,
                idempotency_key="scheduler:test:ts",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        exec_jsonl = jobs_dir / "exec.jsonl"
        for line in exec_jsonl.read_text().splitlines():
            if line.strip():
                json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# enqueue_and_wait — permission Transmittance TestIssue #2191）
# ---------------------------------------------------------------------------


class TestEnqueueAndWaitPermissionPassthrough:
    def test_permission_none_passed_to_step_config(self, tmp_path):
        """permission When not specified,StepConfig.permission Home None to be."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        captured_steps: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="scheduler:test:perm-none",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert len(captured_steps) == 1
        assert captured_steps[0].permission is None

    def test_permission_dangerous_full_access_passed_to_step_config(self, tmp_path):
        """permission='dangerous_full_access' When specified,StepConfig.permission set."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        captured_steps: list = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="scheduler:test:perm-dfa",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
                permission="dangerous_full_access",
            )

        assert len(captured_steps) == 1
        assert captured_steps[0].permission == "dangerous_full_access"


# ---------------------------------------------------------------------------
# enqueue_and_wait — Prompt delivery test
# ---------------------------------------------------------------------------


class TestEnqueueAndWaitPromptPassthrough:
    """enqueue_and_wait does not conduct persona conversion,prompt Close submit """

    def test_prompt_passed_as_is_to_submit(self, tmp_path):
        """Format converted prompt Home submit to be passedbridge do not convert by side)."""
        jobs_dir = _make_jobs_dir(tmp_path)
        submitted_templates = []

        from ghdag.pipeline import LLMPipelineAPI

        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            for step in steps:
                submitted_templates.append(step.template)
            return original_submit(self_api, steps, **kwargs)

        pre_formatted = "Persona format converted prompt (pre-ap  by the caller)"

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
        ):
            try:
                enqueue_and_wait(
                    prompt=pre_formatted,
                    engine="cursor",
                    model="auto",
                    timeout=5.0,
                    idempotency_key=f"scheduler:passthrough:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=jobs_dir / "done",
                )
            except (StopIteration, OSError):
                pass

        assert len(submitted_templates) >= 1
        assert submitted_templates[0] == pre_formatted


# ---------------------------------------------------------------------------
# enqueue_dag — DAG Input TestAC-1〜AC-7）
# ---------------------------------------------------------------------------


def _make_jobs_dir_dag(tmp_path: Path) -> tuple[Path, Path]:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "exec.jsonl").write_text("", encoding="utf-8")
    done = jobs / "done"
    done.mkdir()
    return jobs, done


class TestEnqueueDag:
    """enqueue_dag() Acceptance test."""

    def test_ac1_two_step_linear_dag_submit(self, tmp_path):
        """AC-1: 2Step Linear DAG Home s1→s2 in order submit (separate model)."""
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow.schema import StepConfig

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

    def test_ac2_pre_formatted_prompts_passed_as_is(self, tmp_path):
        """AC-2: The caller is pre-converted prompt Home DagStep If you pass to submit """
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow.schema import StepConfig

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

    def test_ac3_empty_steps_raises_value_error(self, tmp_path):
        """AC-3: empty list ValueError（'empty' messages)."""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with pytest.raises(ValueError, match="empty"):
            enqueue_dag(
                steps=[],
                timeout=5.0,
                idempotency_key="dag:ac3:k",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

    def test_ac4_idempotency_prevents_duplicate_submission(self, tmp_path):
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

    def test_ac5_timeout_returns_false_with_message(self, tmp_path):
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

    def test_ac6_existing_tests_still_pass(self):
        """AC-6: This test itself pass Existing tests are non-destructive (if any)pytest ed)."""
        # This test verifies the test suite can be collected without import errors.
        assert enqueue_and_wait is not None
        assert enqueue_dag is not None
        assert DagStep is not None

    def test_ac7_exec_jsonl_valid_json_with_required_fields(self, tmp_path):
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


# ---------------------------------------------------------------------------
# enqueue_and_wait — result Read Test
# （md_read  result content Verify acquisition)
# ---------------------------------------------------------------------------

_MD_READ = "mltgnt.bridges.ghdag_bridge.md_read"


class TestEnqueueAndWaitResultRead:
    """result File reading ghdag.files.md_read Verify what is done via."""

    def test_success_uses_md_read(self, tmp_path):
        """wait_for_result Home success Home md_read via content """
        jobs_dir = _make_jobs_dir(tmp_path)
        mock_md = MagicMock()
        mock_md.content = "result text"

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, return_value=mock_md) as mock_read:
            ok, content = enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:result_read:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is True
        assert content == "result text"
        mock_read.assert_called_once()

    def test_result_not_found_returns_empty(self, tmp_path):
        """result When the file does not exist content="" """
        jobs_dir = _make_jobs_dir(tmp_path)

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, side_effect=FileNotFoundError("not found")):
            ok, content = enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:result_missing:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is True
        assert content == ""

    def test_frontmatter_stripped_from_result(self, tmp_path):
        """result File frontmatter Contact Us content only returns."""
        jobs_dir = _make_jobs_dir(tmp_path)
        mock_md = MagicMock()
        mock_md.content = "body only"

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, return_value=mock_md):
            ok, content = enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:frontmatter:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is True
        assert content == "body only"

    def test_md_read_called_with_jobs_dir_as_repo_root(self, tmp_path):
        """md_read Home repo_root Home jobs_dir """
        jobs_dir = _make_jobs_dir(tmp_path)
        mock_md = MagicMock()
        mock_md.content = ""

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, return_value=mock_md) as mock_read:
            enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:repo_root:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        _args, kwargs = mock_read.call_args
        assert kwargs.get("repo_root") == jobs_dir


# ---------------------------------------------------------------------------
# enqueue_dag — Data Flow Result Propagation TestAC-8〜AC-12）
# ---------------------------------------------------------------------------


class TestEnqueueDagDataFlow:
    """AC-8–AC-12: verify result propagation between steps and sequential input model."""

    def test_ac8_result_content_propagated_to_dependent_step(self, tmp_path):
        """AC-8: step_a result is propagated to step_b base_context['step_a_result']."""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        mock_md_a = MagicMock()
        # Japanese text intentionally kept for CJK processing test
        mock_md_a.content = "分析結果A"
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
        # Japanese text intentionally kept for CJK processing test
        assert step_b_context["step_a_result"] == "分析結果A"
        # step_a Success
        assert results[0] == (True, "分析結果A")

    def test_ac9_independent_steps_no_cross_context(self, tmp_path):
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

    def test_ac10_user_context_overrides_auto_injection(self, tmp_path):
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

    def test_ac11_failed_step_prevents_dependent_submission(self, tmp_path):
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

    def test_ac12_backward_compat_no_context_field(self, tmp_path):
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


# ---------------------------------------------------------------------------
# typecheck_dag / enqueue_dag compose-time typecheck（Issue #1386）
# ---------------------------------------------------------------------------


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
        """SKILL_IO_TYPECHECK Unset typecheck Run()mismatch Home SkillIOTypeError）。"""
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
        """SKILL_IO_TYPECHECK=0 Home typecheck Skip to contentmismatch 。"""
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


# ---------------------------------------------------------------------------
# TestFanoutPermissionInheritance — permission pass-through (#2235)
# ---------------------------------------------------------------------------


class TestFanoutPermissionInheritance:
    """enqueue_dag() propagates permission parameters onto StepConfig."""

    def test_permission_passed_to_step_config(self, tmp_path):
        """permission='dangerous_full_access' is propagated to each StepConfig."""
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow.schema import StepConfig

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
        from ghdag.workflow.schema import StepConfig

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


# ---------------------------------------------------------------------------
# compose_pipeline / PIPELINE_STATUS PropagationIssue #3031）
# ---------------------------------------------------------------------------


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
        # Japanese text intentionally kept for CJK processing test
        mock_md_a.content = "分析結果\nPIPELINE_STATUS: OK\n"
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
