"""tests/test_ghdag_version.py — ghdag v0.28.3 互換性テスト。

Issue #1697: mltgnt が ghdag v0.28.3 の API に追従していることを検証する。
"""
from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path
import tomllib


def test_ghdag_version_is_at_least_0_28_3():
    """ghdag のインストール済みバージョンが 0.28.3 以上であることを確認する。"""
    version_str = importlib.metadata.version("ghdag")
    parts = [int(x) for x in version_str.split(".")[:3]]
    assert parts >= [0, 28, 3], (
        f"ghdag {version_str} は v0.28.3 より古い。pyproject.toml の依存ピンを更新してください。"
    )


def test_ghdag_llm_pipeline_api_submit_accepts_metadata():
    """LLMPipelineAPI.submit() が metadata 引数を受け付ける (v0.21.0 新機能)。"""
    from ghdag.pipeline import LLMPipelineAPI

    sig = inspect.signature(LLMPipelineAPI.submit)
    assert "metadata" in sig.parameters, (
        "LLMPipelineAPI.submit() に metadata パラメータがない。ghdag v0.21.0 以上が必要です。"
    )


def test_ghdag_dag_hooks_has_on_task_start():
    """DagHooks プロトコルに on_task_start が含まれる (v0.21.0 新機能)。"""
    from ghdag.dag.hooks import DagHooks

    assert hasattr(DagHooks, "on_task_start"), (
        "DagHooks に on_task_start が存在しない。ghdag v0.21.0 以上が必要です。"
    )


def test_ghdag_dag_hooks_has_check_promote_target():
    """DagHooks プロトコルに check_promote_target が含まれる (v0.21.0 新機能)。"""
    from ghdag.dag.hooks import DagHooks

    assert hasattr(DagHooks, "check_promote_target"), (
        "DagHooks に check_promote_target が存在しない。ghdag v0.21.0 以上が必要です。"
    )


def test_pyproject_pins_ghdag_v0_32_1():
    """Issue #2687: ghdag 依存が v0.32.1 に固定されていること。"""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    assert "ghdag @ git+https://github.com/sumipan/ghdag.git@v0.32.1" in project["dependencies"]


def test_issue_2687_required_imports_are_available():
    """Issue #2687: v0.32.1 追従で必要な import が維持されていること。"""
    from ghdag.dag._util import check_pipeline_status, default_check_rejected
    from ghdag.pipeline.status import interpret_done, read_done_content

    assert callable(interpret_done)
    assert callable(read_done_content)
    assert callable(check_pipeline_status)
    assert callable(default_check_rejected)
