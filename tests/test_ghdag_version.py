"""tests/test_ghdag_version.py — ghdag compatibility tests.

Issue #1697: verify mltgnt tracks the ghdag v0.28.3 API.
"""
from __future__ import annotations

import ast
import importlib.metadata
import inspect
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python <3.11
    import tomli as tomllib


def test_ghdag_version_is_at_least_0_28_3():
    """Installed ghdag version must be at least 0.28.3."""
    version_str = importlib.metadata.version("ghdag")
    parts = [int(x) for x in version_str.split(".")[:3]]
    assert parts >= [0, 28, 3], (
        f"ghdag {version_str} is older than v0.28.3. Update the dependency pin in pyproject.toml."
    )


def test_ghdag_llm_pipeline_api_submit_accepts_metadata():
    """LLMPipelineAPI.submit() accepts a metadata argument (v0.21.0 feature)."""
    from ghdag.pipeline import LLMPipelineAPI

    sig = inspect.signature(LLMPipelineAPI.submit)
    assert "metadata" in sig.parameters, (
        "LLMPipelineAPI.submit() has no metadata parameter. ghdag v0.21.0+ is required."
    )


def test_ghdag_dag_hooks_has_on_task_start():
    """DagHooks protocol includes on_task_start (v0.21.0 feature)."""
    from ghdag.dag.hooks import DagHooks

    assert hasattr(DagHooks, "on_task_start"), (
        "DagHooks has no on_task_start. ghdag v0.21.0+ is required."
    )


def test_ghdag_dag_hooks_has_check_promote_target():
    """DagHooks protocol includes check_promote_target (v0.21.0 feature)."""
    from ghdag.dag.hooks import DagHooks

    assert hasattr(DagHooks, "check_promote_target"), (
        "DagHooks has no check_promote_target. ghdag v0.21.0+ is required."
    )


def test_pyproject_ghdag_pin_is_at_least_0_55_0():
    """Issue #3143: ghdag dependency pin must be at least v0.55.0.

    Check a lower bound, not an exact match. Pin updates are done deterministically
    by release-watcher / issuesmith bumps; an exact match would fail this test on
    every bump (six consecutive 'expected-value-only' fixes landed 2026-09-06–10).
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    pins = [
        dep for dep in project["dependencies"]
        if dep.startswith("ghdag @ git+https://github.com/sumipan/ghdag.git@v")
    ]
    assert len(pins) == 1, f"expected exactly one ghdag git pin: {pins}"
    tag = pins[0].rsplit("@v", 1)[1]
    parts = [int(x) for x in tag.split(".")[:3]]
    assert parts >= [0, 55, 0], f"ghdag pin v{tag} is older than v0.55.0"


def test_issue_2991_mltgnt_does_not_import_renamed_adapters():
    """Issue #2991: ghdag v0.43.0 adapter renames must not affect mltgnt.

    mltgnt must not import cursor/codex adapters directly; use public APIs only.
    """
    src_root = Path(__file__).resolve().parents[1] / "src"
    forbidden_modules = {
        "ghdag.llm.adapters.cursor",
        "ghdag.llm.adapters.codex",
    }
    forbidden_names = {
        "CursorAdapter",
        "CodexAdapter",
    }
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        offenders.append(f"{path.relative_to(src_root)}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in forbidden_modules:
                    offenders.append(f"{path.relative_to(src_root)}:{node.module}")
                elif node.module == "ghdag.llm.adapters":
                    for alias in node.names:
                        if alias.name in forbidden_names:
                            offenders.append(
                                f"{path.relative_to(src_root)}:{alias.name}"
                            )
    assert offenders == [], f"renamed adapter imports found: {offenders}"


def test_issue_2702_required_imports_are_available():
    """Issue #2702: imports required for v0.33.0 tracking remain available."""
    from ghdag.dag._util import check_pipeline_status, default_check_rejected
    from ghdag.llm.engines import EngineError
    from ghdag.pipeline.audit import write_task_exit_audit
    from ghdag.pipeline.status import interpret_done, read_done_content

    assert callable(write_task_exit_audit)
    assert callable(interpret_done)
    assert callable(read_done_content)
    assert callable(check_pipeline_status)
    assert callable(default_check_rejected)
    assert EngineError is not None


def test_step_config_has_resume_from_field():
    """Issue #2702: StepConfig has a resume_from attribute."""
    from ghdag.workflow.schema import StepConfig

    field_names: set[str] = set(getattr(StepConfig, "__annotations__", {}).keys())
    field_names.update(getattr(StepConfig, "model_fields", {}).keys())
    field_names.update(getattr(StepConfig, "__fields__", {}).keys())
    assert "resume_from" in field_names


def test_ghdag_interpret_done_recognizes_engine_error():
    """Issue #2721 / #3321: engine_error remains a done status after loops StepStatus removal."""
    from ghdag.pipeline.status import interpret_done

    assert interpret_done("ENGINE_ERROR\n") == "engine_error"


def test_interpret_done_engine_error_maps_to_engine_error():
    """Issue #2721: interpret_done maps ENGINE_ERROR to engine_error."""
    from ghdag.pipeline.status import interpret_done

    assert interpret_done("ENGINE_ERROR") == "engine_error"
