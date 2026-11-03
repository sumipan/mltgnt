"""Timeout annotation and cancel marker tests for ghdag_bridge."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.bridges.ghdag_bridge import (
    DagStep,
    _request_cancel,
    enqueue_and_wait,
    enqueue_dag,
)
from mltgnt.skill.models import ConsumesSpec, ProducesSpec, SkillMatchResult, SkillMeta
from tests.ghdag_bridge_helpers import (
    _MD_READ,
    _WAIT,
    _make_jobs_dir,
    _make_jobs_dir_dag,
)

class TestTaskTimeoutSecAnnotations:
    """task_timeout_sec is written to exec.jsonl annotations.timeout_sec."""

    @staticmethod
    def _records(jobs_dir: Path) -> list[dict]:
        text = (jobs_dir / "exec.jsonl").read_text(encoding="utf-8")
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    def _run_single(self, tmp_path: Path, **kwargs) -> list[dict]:
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        with (
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="test prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key=f"scheduler:tts:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
                **kwargs,
            )
        return self._records(jobs_dir)

    def _run_dag(self, tmp_path: Path, **kwargs) -> list[dict]:
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        with (
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_dag(
                steps=[
                    DagStep(id="s1", prompt="P1", engine="cursor"),
                    DagStep(id="s2", prompt="P2", engine="cursor", depends=["s1"]),
                ],
                timeout=5.0,
                idempotency_key=f"dag:tts:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
                **kwargs,
            )
        return self._records(jobs_dir)

    def test_enqueue_and_wait_writes_timeout_sec(self, tmp_path):
        records = self._run_single(tmp_path, task_timeout_sec=300)
        assert len(records) == 1
        annotations = records[0].get("annotations") or {}
        assert "timeout_sec" in annotations
        assert float(annotations["timeout_sec"]) == 300

    def test_enqueue_and_wait_omits_timeout_sec_by_default(self, tmp_path):
        records = self._run_single(tmp_path)
        assert len(records) == 1
        assert "timeout_sec" not in (records[0].get("annotations") or {})

    def test_enqueue_dag_writes_timeout_sec_on_all_steps(self, tmp_path):
        records = self._run_dag(tmp_path, task_timeout_sec=300)
        assert len(records) == 2
        for record in records:
            annotations = record.get("annotations") or {}
            assert float(annotations["timeout_sec"]) == 300

    def test_enqueue_dag_omits_timeout_sec_by_default(self, tmp_path):
        records = self._run_dag(tmp_path)
        assert len(records) == 2
        for record in records:
            assert "timeout_sec" not in (record.get("annotations") or {})


# ---------------------------------------------------------------------------
# Timeout requests cancel via jobs/cancel/<uuid> marker (#5034)
# ---------------------------------------------------------------------------


def _exec_uuids(jobs_dir: Path) -> list[str]:
    lines = (jobs_dir / "exec.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ln)["uuid"] for ln in lines if ln.strip()]


class TestTimeoutCancel:
    def test_enqueue_and_wait_timeout_writes_cancel_marker(self, tmp_path):
        jobs_dir = _make_jobs_dir(tmp_path)
        with patch(_WAIT, side_effect=TimeoutError):
            result = enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=3.0,
                idempotency_key=f"scheduler:cancel:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )
        assert result == (False, "timeout (3.0s)")
        uuids = _exec_uuids(jobs_dir)
        markers = list((jobs_dir / "cancel").iterdir())
        assert [m.name for m in markers] == uuids
        assert len(markers) == 1
        assert markers[0].read_text() == ""

    def test_enqueue_and_wait_cancel_failure_appended(self, tmp_path):
        jobs_dir = _make_jobs_dir(tmp_path)
        # A regular file where the cancel directory should be makes mkdir fail.
        (jobs_dir / "cancel").write_text("blocker", encoding="utf-8")
        with patch(_WAIT, side_effect=TimeoutError):
            ok, msg = enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=3.0,
                idempotency_key=f"scheduler:cancel:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )
        assert ok is False
        assert msg.startswith("timeout (3.0s); cancel failed: ")
        assert len(msg) > len("timeout (3.0s); cancel failed: ")

    def test_request_cancel_returns_error_string_on_oserror(self, tmp_path):
        blocker = tmp_path / "jobs" / "cancel"
        blocker.parent.mkdir()
        blocker.write_text("not a dir", encoding="utf-8")
        err = _request_cancel(tmp_path / "jobs" / "done", "abc")
        assert isinstance(err, str) and err

    def test_request_cancel_success_returns_none(self, tmp_path):
        assert _request_cancel(tmp_path / "jobs" / "done", "abc") is None
        assert (tmp_path / "jobs" / "cancel" / "abc").is_file()

    def test_enqueue_dag_timeout_writes_cancel_marker_for_step(self, tmp_path):
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        with patch(_WAIT, side_effect=TimeoutError):
            results = enqueue_dag(
                steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                timeout=3.0,
                idempotency_key=f"dag:cancel:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )
        assert results == [(False, "timeout (3.0s)")]
        uuids = _exec_uuids(jobs_dir)
        assert len(uuids) == 1
        assert (jobs_dir / "cancel" / uuids[0]).is_file()

    def test_enqueue_dag_remaining_exhausted_writes_cancel_marker(self, tmp_path):
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        with patch(_WAIT) as mock_wait:
            results = enqueue_dag(
                steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                timeout=0.0,
                idempotency_key=f"dag:cancel:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )
        mock_wait.assert_not_called()
        assert results == [(False, "timeout (0.0s)")]
        uuids = _exec_uuids(jobs_dir)
        assert (jobs_dir / "cancel" / uuids[0]).is_file()
