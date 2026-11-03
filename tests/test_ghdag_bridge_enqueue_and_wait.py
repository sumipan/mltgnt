"""enqueue_and_wait integration tests for ghdag_bridge."""
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
        """enqueue_and_wait Home exec.jsonl All lines to write valid JSON."""
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
# enqueue_and_wait — permission Transmittance TestIssue #2191)
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
