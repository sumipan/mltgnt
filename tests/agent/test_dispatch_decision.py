"""mltgnt.agent.dispatch_decision / deterministic_gate（#3318）。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mltgnt.agent.deterministic_gate import (
    extract_artifact_references,
    has_work_request,
    is_create_request,
    match_deferred_promise,
    should_force_delegate,
    should_preempt_delegate,
)
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


def test_deterministic_gate_work_plus_artifact() -> None:
    text = "かいちさんとの対談台本.mdをブラッシュアップしてもらえるかな"
    assert should_force_delegate(text) is True
    assert should_preempt_delegate(text) is True
    assert has_work_request(text) is True
    assert extract_artifact_references(text) == ("かいちさんとの対談台本.md",)


def test_deterministic_gate_plain_chat() -> None:
    assert should_force_delegate("今日は暑いね") is False
    assert match_deferred_promise("後でやっておくね") is not None
    assert is_create_request("新しくファイルつくって") is True


def test_run_preflight_requires_injected_workers() -> None:
    from types import SimpleNamespace

    ctx = run_preflight(
        effective_persona="テスト秘書",
        instruction="台本を修正して",
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
        instruction="今日は暑いね",
        space_id="C1",
        thread_key="1.0",
        engine="claude",
        model="m",
        run_agent_fn=lambda **_kw: DirectAgentResult(
            tool=MODE_REPLY, message="ほんと暑いね", raw_response=""
        ),
        should_preempt_delegate_fn=lambda _i: False,
        match_deferred_promise_fn=lambda _t: None,
        is_reply_tool_fn=lambda tool: tool == MODE_REPLY,
        latest_session_fn=lambda *_a: None,
        invalidate_session_fn=lambda *_a: False,
        ledger_key_fn=lambda space, thread: f"conv-{space}-{thread}",
    )
    assert decision.mode == MODE_REPLY
    assert decision.reply_text == "ほんと暑いね"


def test_make_dispatch_decision_preempt() -> None:
    decision = make_dispatch_decision(
        preflight=_preflight(),
        instruction="台本.md を修正して",
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
            instruction="続き",
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
        assert "あなたは秘書エージェント" not in src
        assert "slack_sdk" not in src
