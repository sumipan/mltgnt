"""enqueue_and_wait integration tests for ghdag_bridge."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ghdag.core import capabilities

Permission = capabilities.Permission

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




class TestEnqueueAndWaitPermissionPassThrough:
    """ghdag Permission values pass through enqueue_and_wait to StepConfig."""

    def test_permission_object_allowed_tools_reach_exec_command(self, tmp_path):
        """Ref #7429: Permission.allowed_tools appear in exec.jsonl command (claude)."""
        jobs_dir = _make_jobs_dir(tmp_path)
        (jobs_dir / "done").mkdir()
        perm = Permission(allowed_tools=("Read", "Bash(ls:*)"))

        with patch(_WAIT, return_value=("success", "")):
            try:
                enqueue_and_wait(
                    prompt="permission probe",
                    engine="claude",
                    model="claude-sonnet-4-6",
                    timeout=5.0,
                    idempotency_key=f"perm:wait:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=jobs_dir / "done",
                    permission=perm,
                )
            except (StopIteration, OSError):
                pass

        lines = [
            ln for ln in (jobs_dir / "exec.jsonl").read_text().splitlines() if ln.strip()
        ]
        assert len(lines) == 1
        command = json.loads(lines[0])["command"]
        assert "Read" in command
        assert "Bash(ls:*)" in command


