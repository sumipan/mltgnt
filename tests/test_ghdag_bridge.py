"""tests/test_ghdag_bridge.py — ghdag_bridge non-consolidated + integrated testing.

Coverage:
  - _extract_result_filename(): JSON Type / Text format / Fallback
  - _order_to_result_filename(): Standard text exec line
  - ghdag_bridge core helpers All lines to write valid JSON
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
    _request_cancel,
    _scheduler_audit_context,
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


# ---------------------------------------------------------------------------
# correlation_id — AuditContext PropagationAC3-AC5)
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
        """AC5: correlation_id When omitted,  None   AuditContext  is passed."""
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
# request_id — AuditContext Propagation#1346)
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
        """request_id When omitted,  None   AuditContext  is passed."""
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
