"""tests/test_ghdag_bridge_order_builder.py — order_builder 引数の注入テスト。

受け入れ条件:
  - enqueue_dag / enqueue_and_wait に order_builder 引数を渡すと LLMPipelineAPI に注入される
  - order_builder 省略時は InlineOrderBuilder() がフォールバックとして使われる（後方互換）
"""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mltgnt.bridges.ghdag_bridge import DagStep, enqueue_and_wait, enqueue_dag

_WAIT = "ghdag.pipeline.wait_for_result"
_MD_READ = "mltgnt.bridges.ghdag_bridge.md_read"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jobs_dir(tmp: Path) -> Path:
    d = tmp / "jobs"
    d.mkdir()
    (d / "done").mkdir()
    return d


def _make_jobs_dir_dag(tmp: Path) -> tuple[Path, Path]:
    jobs = tmp / "jobs"
    jobs.mkdir()
    done = jobs / "done"
    done.mkdir()
    return jobs, done


def _fake_exec_line(filename: str = "jobs/result-abc.md") -> str:
    import json

    return json.dumps({"result_filename": filename, "status": "success"})


# ---------------------------------------------------------------------------
# enqueue_and_wait — order_builder 注入
# ---------------------------------------------------------------------------


class TestEnqueueAndWaitOrderBuilder:
    def test_custom_order_builder_is_passed_to_pipeline_api(self, tmp_path):
        """order_builder を渡すと LLMPipelineAPI に注入される。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        custom_builder = MagicMock(name="custom_builder")
        custom_builder.build_order.return_value = "## Order\ntest prompt"
        captured: list = []
        original_init = LLMPipelineAPI.__init__

        def capture_init(self_api, *args, **kwargs):
            captured.append(kwargs.get("order_builder"))
            original_init(self_api, *args, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "__init__", capture_init),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="result")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key=f"test:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
                order_builder=custom_builder,
            )

        assert len(captured) == 1
        assert captured[0] is custom_builder

    def test_default_order_builder_is_inline(self, tmp_path):
        """order_builder 省略時は InlineOrderBuilder() がフォールバックに使われる。"""
        from ghdag.pipeline import InlineOrderBuilder, LLMPipelineAPI

        jobs_dir = _make_jobs_dir(tmp_path)
        captured: list = []
        original_init = LLMPipelineAPI.__init__

        def capture_init(self_api, *args, **kwargs):
            captured.append(kwargs.get("order_builder"))
            original_init(self_api, *args, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "__init__", capture_init),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key=f"test:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert len(captured) == 1
        assert isinstance(captured[0], InlineOrderBuilder)


# ---------------------------------------------------------------------------
# enqueue_dag — order_builder 注入
# ---------------------------------------------------------------------------


class TestEnqueueDagOrderBuilder:
    def test_custom_order_builder_is_passed_to_pipeline_api(self, tmp_path):
        """order_builder を渡すと LLMPipelineAPI に注入される。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        custom_builder = MagicMock(name="custom_builder")
        custom_builder.build_order.return_value = "## Order\ntest prompt"
        captured: list = []
        original_init = LLMPipelineAPI.__init__

        def capture_init(self_api, *args, **kwargs):
            captured.append(kwargs.get("order_builder"))
            original_init(self_api, *args, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "__init__", capture_init),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="result")),
        ):
            enqueue_dag(
                steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                timeout=5.0,
                idempotency_key=f"dag:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
                order_builder=custom_builder,
            )

        assert len(captured) == 1
        assert captured[0] is custom_builder

    def test_default_order_builder_is_inline(self, tmp_path):
        """order_builder 省略時は InlineOrderBuilder() がフォールバックに使われる。"""
        from ghdag.pipeline import InlineOrderBuilder, LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured: list = []
        original_init = LLMPipelineAPI.__init__

        def capture_init(self_api, *args, **kwargs):
            captured.append(kwargs.get("order_builder"))
            original_init(self_api, *args, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "__init__", capture_init),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="")),
        ):
            enqueue_dag(
                steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                timeout=5.0,
                idempotency_key=f"dag:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured) == 1
        assert isinstance(captured[0], InlineOrderBuilder)
