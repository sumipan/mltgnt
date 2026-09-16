"""委譲判定（decision）の骨格（#3318）。

エージェント実行・セッション台帳・ツール名判定はホストが注入する。
プロンプト本文・ツール定義・plan 承認ポリシーは本モジュールに置かない。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone as _timezone
from typing import Callable

from mltgnt.agent.dispatch_preflight import PreflightContext
from mltgnt.conversation.session_store import SessionRecord
from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE

REASK_MESSAGE = "この場では完了できないため、対象と作業内容を明記して依頼し直してください"

MODE_REPLY = "reply"
MODE_DELEGATE = "delegate"


@dataclass
class DirectAgentResult:
    """直接応答エージェント（secretary_agent 相当）の実行結果。

    既存の ``mltgnt.agent.AgentResult``（AgentRunner 用）とは別型。
    """

    tool: str
    message: str
    raw_response: str
    tool_trace: list[dict] | None = None
    skill: str = "__unresolved__"
    session_id: str | None = None
    resume_failed: bool = False


@dataclass(frozen=True)
class DispatchDecision:
    mode: str  # "reply" | "delegate"
    reply_text: str | None
    agent_result: DirectAgentResult | None
    preempted: bool
    deferred_escalated: bool
    agent_spans: list[dict] = field(default_factory=list)
    session_id: str | None = None
    resumed_from_session: dict[str, str] | None = None


def _normalize_primary_engine_model(engine: str, model: str) -> tuple[str, str]:
    primary_engine = (engine or "").strip() or SYSTEM_DEFAULT_ENGINE
    primary_model = (model or "").strip()
    return primary_engine, primary_model


def _is_session_record(obj: object) -> bool:
    sid = getattr(obj, "session_id", None) if obj is not None else None
    return isinstance(sid, str) and bool(sid)


def _resumed_from_payload(
    conversation_key: str,
    record: SessionRecord,
) -> dict[str, str]:
    return {
        "key": conversation_key,
        "engine": record.engine,
        "session_id_prefix": record.session_id[:8],
    }


def _merge_memory_with_memo(
    memory_excerpt: str | None, secretary_memo: str | None
) -> str | None:
    merged = memory_excerpt
    if secretary_memo:
        memo_block = f"--- 今日の秘書メモ（状況認識） ---\n{secretary_memo}\n\n"
        merged = f"{memo_block}{merged}" if merged else memo_block
    return merged


def make_dispatch_decision(
    *,
    preflight: PreflightContext,
    instruction: str,
    space_id: str,
    thread_key: str,
    engine: str,
    model: str,
    run_agent_fn: Callable[..., DirectAgentResult | None],
    should_preempt_delegate_fn: Callable[[str], bool],
    match_deferred_promise_fn: Callable[[str], str | None],
    is_reply_tool_fn: Callable[[str], bool],
    latest_session_fn: Callable[[str, str], SessionRecord | None],
    invalidate_session_fn: Callable[[str, str], bool],
    resume_supported_fn: Callable[[str | None], bool] | None = None,
    ledger_key_fn: Callable[[str, str], str] | None = None,
    normalize_primary_engine_model_fn: Callable[
        [str, str], tuple[str, str]
    ] = _normalize_primary_engine_model,
    sleep_fn: Callable[[float], None] = time.sleep,
    utc_now_fn: Callable[[], str] | None = None,
    logger=None,
) -> DispatchDecision:
    """reply / delegate を決定する。ホスト依存はすべて注入必須。"""
    utc_now = utc_now_fn or (lambda: datetime.now(_timezone.utc).isoformat())
    resume_check = resume_supported_fn or (lambda _engine: False)
    key_fn = ledger_key_fn or (lambda space, thread: f"{space}-{thread}")

    if should_preempt_delegate_fn(instruction):
        preempt_agent: DirectAgentResult | None = None
        if preflight.selected_skill and preflight.selected_skill_is_deterministic:
            preempt_agent = DirectAgentResult(
                tool=MODE_DELEGATE,
                skill=preflight.selected_skill,
                message="deterministic_preempt_delegate",
                raw_response="",
            )
        return DispatchDecision(
            mode=MODE_DELEGATE,
            reply_text=None,
            agent_result=preempt_agent,
            preempted=True,
            deferred_escalated=False,
        )

    if preflight.selected_skill and preflight.selected_skill_is_deterministic:
        return DispatchDecision(
            mode=MODE_DELEGATE,
            reply_text=None,
            agent_result=DirectAgentResult(
                tool=MODE_DELEGATE,
                skill=preflight.selected_skill,
                message="shortcircuit",
                raw_response="",
            ),
            preempted=False,
            deferred_escalated=False,
        )

    agent_result: DirectAgentResult | None = None
    agent_spans: list[dict] = []
    primary_engine, primary_model = normalize_primary_engine_model_fn(engine, model)

    ledger = latest_session_fn(space_id, thread_key)
    resume_session_id: str | None = None
    resumed_from_session: dict[str, str] | None = None
    if (
        _is_session_record(ledger)
        and resume_check(primary_engine)
        and ledger is not None
        and ledger.engine == primary_engine
    ):
        resume_session_id = ledger.session_id
        resumed_from_session = _resumed_from_payload(
            key_fn(space_id, thread_key), ledger
        )

    def _call_agent(use_resume: str | None) -> DirectAgentResult | None:
        t_agent_mono = time.monotonic()
        t_agent_ts = utc_now()
        skill_hint = (
            preflight.selected_skill
            if preflight.selected_skill and not preflight.selected_skill_is_deterministic
            else None
        )
        result = run_agent_fn(
            instruction=instruction,
            profile_content=preflight.triage_profile_content,
            memory_tail=_merge_memory_with_memo(
                preflight.memory_excerpt, preflight.secretary_memo
            ),
            user_profile=preflight.user_preferences,
            engine=primary_engine,
            model=primary_model,
            logger=logger,
            space_id=space_id,
            thread_context=preflight.thread_context,
            persona_skills=preflight.persona_skills,
            thread_key=thread_key,
            resume_session_id=use_resume,
            skill_hint=skill_hint,
        )
        span: dict = {
            "start_ts": t_agent_ts,
            "duration_ms": int((time.monotonic() - t_agent_mono) * 1000),
            "attempt": len(agent_spans) + 1,
            "status": "ok" if result is not None and not result.resume_failed else "error",
            "engine": primary_engine,
            "model": primary_model,
            "resume": bool(use_resume),
        }
        if result is not None and result.tool_trace:
            span["tool_trace"] = result.tool_trace
        agent_spans.append(span)
        return result

    agent_result = _call_agent(resume_session_id)
    used_resume = bool(
        resume_session_id
        and agent_result is not None
        and not agent_result.resume_failed
    )
    if resume_session_id and (
        agent_result is None
        or (agent_result is not None and agent_result.resume_failed)
    ):
        invalidate_session_fn(space_id, thread_key)
        resume_session_id = None
        resumed_from_session = None
        agent_result = _call_agent(None)
        used_resume = False
    elif agent_result is None:
        for _attempt_idx in range(1):
            sleep_fn(1)
            agent_result = _call_agent(None)
            if agent_result is not None:
                break

    mode = MODE_DELEGATE
    reply: str | None = None
    if (
        agent_result is not None
        and is_reply_tool_fn(agent_result.tool)
        and agent_result.message
        and agent_result.message.strip()
    ):
        mode = MODE_REPLY
        reply = agent_result.message

    deferred_escalated = False
    if mode == MODE_REPLY and reply is not None:
        if match_deferred_promise_fn(reply) is not None:
            if not (instruction or "").strip():
                return DispatchDecision(
                    mode=MODE_REPLY,
                    reply_text=REASK_MESSAGE,
                    agent_result=agent_result,
                    preempted=False,
                    deferred_escalated=False,
                    agent_spans=agent_spans,
                    session_id=agent_result.session_id if agent_result else None,
                    resumed_from_session=resumed_from_session if used_resume else None,
                )
            mode = MODE_DELEGATE
            reply = None
            deferred_escalated = True

    return DispatchDecision(
        mode=mode,
        reply_text=reply,
        agent_result=agent_result,
        preempted=False,
        deferred_escalated=deferred_escalated,
        agent_spans=agent_spans,
        session_id=agent_result.session_id if agent_result else None,
        resumed_from_session=resumed_from_session if used_resume else None,
    )
