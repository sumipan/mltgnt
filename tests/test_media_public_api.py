"""Public surface for mltgnt.media (#5037)."""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

MEDIA_PUBLIC_NAMES: tuple[str, ...] = (
    "AWAITING_STATE",
    "CATCHUP_STATES",
    "DeliveryReconciler",
    "ExecDoneHandler",
    "HookRegistry",
    "MediaBridge",
    "MediaConfig",
    "MediaEvent",
    "PendingStore",
    "PENDING_KEY",
    "PlanGate",
    "PlanState",
    "ProgressState",
    "ProgressWatcher",
    "catchup_pending_on_startup",
    "dedupe_trailing_repeated_block",
    "enqueue_or_report",
    "expire_pending",
    "extract_final_assistant_text",
    "finalize_progress",
    "is_approval",
    "is_cancel_request",
    "iter_pending_with_done",
    "sanitize_result_body",
    "strip_leading_paragraphs",
    "strip_status_lines",
    "summarize_tool_use_block",
)

_CORE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("MediaBridge", "mltgnt.media._core.bridge", "MediaBridge"),
    ("HookRegistry", "mltgnt.media._core.hooks", "HookRegistry"),
    ("PendingStore", "mltgnt.media._core.pending", "PendingStore"),
    ("ProgressState", "mltgnt.media._core.progress", "ProgressState"),
    ("finalize_progress", "mltgnt.media._core.progress", "finalize_progress"),
    ("summarize_tool_use_block", "mltgnt.media._core.progress", "summarize_tool_use_block"),
    ("PlanGate", "mltgnt.media._core.plan_gate", "PlanGate"),
    ("PlanState", "mltgnt.media._core.plan_gate", "PlanState"),
    ("AWAITING_STATE", "mltgnt.media._core.plan_gate", "AWAITING_STATE"),
    ("PENDING_KEY", "mltgnt.media._core.plan_gate", "PENDING_KEY"),
    ("expire_pending", "mltgnt.media._core.plan_gate", "expire_pending"),
    ("is_approval", "mltgnt.media._core.plan_gate", "is_approval"),
    ("MediaConfig", "mltgnt.media._core.config", "MediaConfig"),
    ("MediaEvent", "mltgnt.media._core.types", "MediaEvent"),
    ("CATCHUP_STATES", "mltgnt.media._core.watchers", "CATCHUP_STATES"),
    ("DeliveryReconciler", "mltgnt.media._core.watchers", "DeliveryReconciler"),
    ("ExecDoneHandler", "mltgnt.media._core.watchers", "ExecDoneHandler"),
    ("ProgressWatcher", "mltgnt.media._core.watchers", "ProgressWatcher"),
    ("catchup_pending_on_startup", "mltgnt.media._core.watchers", "catchup_pending_on_startup"),
    ("iter_pending_with_done", "mltgnt.media._core.watchers", "iter_pending_with_done"),
    ("is_cancel_request", "mltgnt.media._core.cancel", "is_cancel_request"),
    ("enqueue_or_report", "mltgnt.media._core.enqueue_guard", "enqueue_or_report"),
    (
        "dedupe_trailing_repeated_block",
        "mltgnt.media._core.sanitizer",
        "dedupe_trailing_repeated_block",
    ),
    (
        "extract_final_assistant_text",
        "mltgnt.media._core.sanitizer",
        "extract_final_assistant_text",
    ),
    ("sanitize_result_body", "mltgnt.media._core.sanitizer", "sanitize_result_body"),
    ("strip_leading_paragraphs", "mltgnt.media._core.sanitizer", "strip_leading_paragraphs"),
    ("strip_status_lines", "mltgnt.media._core.sanitizer", "strip_status_lines"),
)


def test_media_public_imports_and_dunder_all() -> None:
    from mltgnt.media import (  # noqa: F401
        AWAITING_STATE,
        CATCHUP_STATES,
        DeliveryReconciler,
        ExecDoneHandler,
        HookRegistry,
        MediaBridge,
        MediaConfig,
        MediaEvent,
        PENDING_KEY,
        PendingStore,
        PlanGate,
        PlanState,
        ProgressState,
        ProgressWatcher,
        catchup_pending_on_startup,
        dedupe_trailing_repeated_block,
        enqueue_or_report,
        expire_pending,
        extract_final_assistant_text,
        finalize_progress,
        is_approval,
        is_cancel_request,
        iter_pending_with_done,
        sanitize_result_body,
        strip_leading_paragraphs,
        strip_status_lines,
        summarize_tool_use_block,
    )

    import mltgnt.media as media

    assert sorted(media.__all__) == sorted(MEDIA_PUBLIC_NAMES)
    assert len(media.__all__) == 27


@pytest.mark.parametrize("public_name,mod_path,attr", _CORE_SOURCES)
def test_media_public_symbols_match_core(public_name: str, mod_path: str, attr: str) -> None:
    import mltgnt.media as media

    core_mod = importlib.import_module(mod_path)
    assert getattr(media, public_name) is getattr(core_mod, attr)


def test_import_media_does_not_load_slack_or_webchat_submodules() -> None:
    code = (
        "import mltgnt.media; import sys; "
        "assert 'mltgnt.media.slack' not in sys.modules; "
        "assert 'mltgnt.media.webchat' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={**dict(**__import__("os").environ), "PYTHONPATH": "src"},
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
