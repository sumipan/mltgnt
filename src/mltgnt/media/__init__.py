"""Media layer (Slack / WebChat implementations live in submodules)."""

from mltgnt.media._core.bridge import MediaBridge
from mltgnt.media._core.cancel import is_cancel_request
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.enqueue_guard import enqueue_or_report
from mltgnt.media._core.hooks import HookRegistry
from mltgnt.media._core.pending import PendingStore
from mltgnt.media._core.plan_gate import (
    AWAITING_STATE,
    PENDING_KEY,
    PlanGate,
    PlanState,
    expire_pending,
    is_approval,
)
from mltgnt.media._core.progress import (
    ProgressState,
    finalize_progress,
    summarize_tool_use_block,
)
from mltgnt.media._core.sanitizer import (
    dedupe_trailing_repeated_block,
    extract_final_assistant_text,
    sanitize_result_body,
    strip_leading_paragraphs,
    strip_status_lines,
)
from mltgnt.media._core.types import MediaEvent
from mltgnt.media._core.watchers import (
    CATCHUP_STATES,
    DeliveryReconciler,
    ExecDoneHandler,
    ProgressWatcher,
    catchup_pending_on_startup,
    iter_pending_with_done,
)

__all__: list[str] = [
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
]
