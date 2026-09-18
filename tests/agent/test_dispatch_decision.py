"""mltgnt.agent.dispatch_decision / deterministic_gate (#3318)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mltgnt.agent.dispatch_decision import (
    MODE_DELEGATE,
    MODE_REPLY,
    DirectAgentResult,
    make_dispatch_decision,
)
from mltgnt.agent.dispatch_preflight import (
    MemoryWorkerResult,
    PreflightContext,
    SkillWorkerResult,
    run_preflight,
)
from mltgnt.conversation.session_store import SessionRecord

# Persona names that must not appear as literals in agent source modules.
# Stored as bytes to keep this source file CJK-free while preserving the guard.
_SECRETARY_PROMPT = b"\xe3\x81\x82\xe3\x81\xaa\xe3\x81\x9f\xe3\x81\xaf\xe7\xa7\x98\xe6\x9b\xb8\xe3\x82\xa8\xe3\x83\xbc\xe3\x82\xb8\xe3\x82\xa7\xe3\x83\xb3\xe3\x83\x88".decode()


def _preflight(
    selected_skill: str | None = None,
    *,
    rationale: str | None = None,
) -> PreflightContext:
    return PreflightContext(
        triage_profile_content="profile",
        memory_excerpt="memory",
        memory_worker=MemoryWorkerResult("memory", "2026-01-01T00:00:00+00:00", 1, "ok"),
        skill_worker=SkillWorkerResult(
            ["system-diary"] if selected_skill else None,
            selected_skill,
            "2026-01-01T00:00:00+00:00",
            1,
            "ok",
            rationale,
        ),
        secretary_memo="memo",
        user_preferences=None,
        thread_context=None,
        preflight_note=None,
    )


def test_run_preflight_requires_injected_workers() -> None:
    from types import SimpleNamespace

    ctx = run_preflight(
        effective_persona="test-secretary",
        instruction="Please revise the script",
        thread_key="1.0",
        profile_content="base-profile",
        routing_result=SimpleNamespace(prebound_skills=[]),
        thread_messages=[{"user": "U1", "text": "x"}],
        load_persona_fn=lambda *_a, **_k: ("light-profile", None, None),
        memory_worker_fn=lambda *_a, **_k: MemoryWorkerResult(
            "memory", "2026-01-01T00:00:00+00:00", 10, "ok"
        ),
        skill_worker_fn=lambda *_a, **_k: SkillWorkerResult(
            ["system-diary"],
            "system-diary",
            "2026-01-01T00:00:00+00:00",
            20,
            "ok",
        ),
        read_secretary_memo_fn=lambda: "memo",
        read_memory_preferences_fn=lambda _p: "prefs",
        format_thread_for_prompt_fn=lambda *_a, **_k: "thread-ctx",
        build_preflight_note_fn=lambda _i: "note",
    )
    assert ctx.triage_profile_content == "light-profile"
    assert ctx.memory_excerpt == "memory"
    assert ctx.selected_skill == "system-diary"
    assert ctx.secretary_memo == "memo"
    assert ctx.preflight_note == "note"


def test_make_dispatch_decision_reply() -> None:
    decision = make_dispatch_decision(
        preflight=_preflight(),
        instruction="how about the weather",
        space_id="C1",
        thread_key="1.0",
        engine="claude",
        model="m",
        run_agent_fn=lambda **_kw: DirectAgentResult(
            tool=MODE_REPLY, message="sounds good", raw_response=""
        ),
        should_preempt_delegate_fn=lambda _i: False,
        match_deferred_promise_fn=lambda _t: None,
        is_reply_tool_fn=lambda tool: tool == MODE_REPLY,
        latest_session_fn=lambda *_a: None,
        invalidate_session_fn=lambda *_a: False,
        ledger_key_fn=lambda space, thread: f"conv-{space}-{thread}",
    )
    assert decision.mode == MODE_REPLY
    assert decision.reply_text == "sounds good"


def test_make_dispatch_decision_preempt() -> None:
    decision = make_dispatch_decision(
        preflight=_preflight(),
        instruction="please revise script.md",
        space_id="C1",
        thread_key="1.0",
        engine="claude",
        model="m",
        run_agent_fn=lambda **_kw: DirectAgentResult(
            tool=MODE_REPLY, message="x", raw_response=""
        ),
        should_preempt_delegate_fn=lambda _i: True,
        match_deferred_promise_fn=lambda _t: None,
        is_reply_tool_fn=lambda tool: tool == MODE_REPLY,
        latest_session_fn=lambda *_a: None,
        invalidate_session_fn=lambda *_a: False,
    )
    assert decision.mode == MODE_DELEGATE
    assert decision.preempted is True


def test_make_dispatch_decision_resume_three_engines() -> None:
    for engine in ("claude", "cursor", "codex"):
        calls: list[str | None] = []
        session_id = f"thread-{engine}-001"
        ledger = SessionRecord(
            engine=engine,
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
        )

        def _make_agent(eng: str):
            def _agent(**kwargs: object) -> DirectAgentResult:
                calls.append(kwargs.get("resume_session_id"))  # type: ignore[arg-type]
                return DirectAgentResult(
                    tool=MODE_REPLY,
                    message=f"{eng} resumed",
                    raw_response="",
                    session_id=f"thread-{eng}-002",
                )

            return _agent

        decision = make_dispatch_decision(
            preflight=_preflight(),
            instruction="continue",
            space_id="C1",
            thread_key="1.0",
            engine=engine,
            model="m",
            run_agent_fn=_make_agent(engine),
            should_preempt_delegate_fn=lambda _i: False,
            match_deferred_promise_fn=lambda _t: None,
            is_reply_tool_fn=lambda tool: tool == MODE_REPLY,
            latest_session_fn=lambda *_a, _ledger=ledger: _ledger,
            invalidate_session_fn=lambda *_a: False,
            resume_supported_fn=lambda eng: eng in {"claude", "cursor", "codex"},
            ledger_key_fn=lambda space, thread: f"conv-{space}-{thread}",
        )
        assert calls == [session_id]
        assert decision.resumed_from_session == {
            "key": "conv-C1-1.0",
            "engine": engine,
            "session_id_prefix": session_id[:8],
        }


def test_agent_modules_have_no_hardcoded_prompts() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "mltgnt" / "agent"
    for name in ("dispatch_preflight.py", "dispatch_decision.py", "deterministic_gate.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert _SECRETARY_PROMPT not in src
        assert "slack_sdk" not in src
