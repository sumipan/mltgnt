"""tests/test_ghdag_version.py — ghdag 互換性テスト。

Issue #1697: mltgnt が ghdag v0.28.3 の API に追従していることを検証する。
"""
from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path
from typing import get_args
try:
    import tomllib
except ModuleNotFoundError:  # Python <3.11
    import tomli as tomllib


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


def test_pyproject_pins_ghdag_v0_34_4():
    """Issue #2743: ghdag 依存が v0.34.4 に固定されていること。"""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    dependency = "ghdag @ git+https://github.com/sumipan/ghdag.git@v0.34.4"
    assert project["dependencies"].count(dependency) == 1


def test_issue_2743_project_version_is_0_22_6():
    """Issue #2743: mltgnt のリリース版が 0.22.6 であること。"""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.22.6"


def test_issue_2702_required_imports_are_available():
    """Issue #2702: v0.33.0 追従で必要な import が維持されていること。"""
    from ghdag.dag._util import check_pipeline_status, default_check_rejected
    from ghdag.llm.engines import EngineError
    from ghdag.pipeline.status import interpret_done, read_done_content

    assert callable(interpret_done)
    assert callable(read_done_content)
    assert callable(check_pipeline_status)
    assert callable(default_check_rejected)
    assert EngineError is not None


def test_step_config_has_resume_from_field():
    """Issue #2702: StepConfig に resume_from 属性が存在すること。"""
    from ghdag.workflow.schema import StepConfig

    field_names: set[str] = set(getattr(StepConfig, "__annotations__", {}).keys())
    field_names.update(getattr(StepConfig, "model_fields", {}).keys())
    field_names.update(getattr(StepConfig, "__fields__", {}).keys())
    assert "resume_from" in field_names


def test_step_status_literal_includes_engine_error():
    """Issue #2721: StepStatus Literal に engine_error が含まれること。"""
    from mltgnt.interfaces.loops import StepStatus

    assert "engine_error" in get_args(StepStatus)


def test_interpret_done_engine_error_maps_to_engine_error():
    """Issue #2721: interpret_done が ENGINE_ERROR を engine_error に解釈すること。"""
    from ghdag.pipeline.status import interpret_done

    assert interpret_done("ENGINE_ERROR") == "engine_error"
