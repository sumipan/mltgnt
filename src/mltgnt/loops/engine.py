"""mltgnt.loops.engine — 1 tick 1 遷移の状態機械。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from mltgnt.config import LoopsConfig
from mltgnt.execution.base_runner import BaseRunner
from mltgnt.interfaces.loops import (
    ConditionEvaluator,
    HumanChannel,
    HumanThreadRef,
    LoopStatus,
    SubtaskExecutor,
    WatchVerdict,
)
from mltgnt.loops.conditions import PathConditionEvaluator
from mltgnt.loops.models import (
    LoopState,
    PendingQuestion,
    Subtask,
    TERMINAL_STATUSES,
)
from mltgnt.loops.objective import Objective
from mltgnt.loops import prompts
from mltgnt.loops import store
from mltgnt.loops.status import render_progress_summary, write_status
from mltgnt.persona.loader import load as load_persona

logger = logging.getLogger("mltgnt.loops.engine")

_TZ = ZoneInfo("Asia/Tokyo")
_MAX_CONSECUTIVE_ERRORS = 3
_LOCAL_CONDITION_TYPES = frozenset({"path_exists", "path_changed"})
_PLAN_APPROVAL_ANSWERS = frozenset({"ok", "承認", "進めて", "go"})
_STATUS_EXACT = frozenset({"動いてる", "止まってる", "どう", "状況", "進捗", "status"})


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(_TZ)).isoformat()


def _parse_iso(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_TZ)
    return dt


def _event_id(loop_id: str, kind: str, stable_id: str) -> str:
    return f"loops:{loop_id}:{kind}:{stable_id}"


def is_status_inquiry(text: str) -> bool:
    """決定論的な状態問合せ判定。他語を伴う指示は status にしない。"""
    trimmed = text.strip()
    if not trimmed:
        return False
    normalized = trimmed.rstrip("?？").strip()
    if normalized in _STATUS_EXACT:
        return True
    if "状況" in trimmed or "進捗" in trimmed:
        return True
    return False


def _subtask_from_decompose(s: prompts.DecomposeSubtask) -> Subtask:
    return Subtask(
        id=s.id,
        title=s.title,
        kind=s.kind,
        prompt=s.prompt,
        condition=dict(s.condition) if s.condition is not None else None,
        depends=list(s.depends),
        timeout_sec=s.timeout_sec,
        poll_interval_sec=s.poll_interval_sec,
    )


@dataclass
class LoopsEngine(BaseRunner):
    config: LoopsConfig
    human_channel: HumanChannel
    executor: SubtaskExecutor
    objective_exists: Callable[[str], bool] | None = None
    objective_cancelled: Callable[[str], bool] | None = None
    objective_hash_changed: Callable[[str, str], bool] | None = None
    condition_evaluator: ConditionEvaluator | None = None

    def tick(self, now: datetime | None = None) -> None:
        if now is None:
            now = datetime.now(_TZ)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=_TZ)
        for loop_id in self._active_loop_ids():
            try:
                self._tick_loop(loop_id, now=now)
            except Exception:
                logger.exception("loop %s tick failed", loop_id)

    def _active_loop_ids(self) -> list[str]:
        return store.list_restorable_loops(self.config.state_dir)

    def start_loop(
        self, objective: Objective, *, thread: HumanThreadRef | None = None
    ) -> LoopState:
        state = LoopState(
            loop_id=objective.loop_id,
            objective_path=str(objective.path),
            objective_hash=objective.content_hash,
            title=objective.title,
            body=objective.body,
            status="clarifying",
            iteration=1,
            max_iterations=objective.max_iterations,
            persona=objective.agent,
            thread=thread,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            plan_approval=objective.plan_approval,
        )
        store.initialize_deliverable(self.config.state_dir, objective.loop_id, objective.body)
        store.save_state(self.config.state_dir, state)
        store.append_event(
            self.config.state_dir,
            objective.loop_id,
            "loop_started",
            {"objective_path": str(objective.path)},
            iteration=1,
        )
        store.append_event(
            self.config.state_dir,
            objective.loop_id,
            "deliverable_updated",
            store.deliverable_snapshot(self.config.state_dir, objective.loop_id),
            iteration=1,
        )
        self._write_status(state)
        return state

    def _transition(self, state: LoopState, to_status: LoopStatus, reason: str) -> None:
        if state.status == to_status:
            return
        from_status = state.status
        state.status = to_status
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            "state_change",
            {"from": from_status, "to": to_status, "reason": reason},
            iteration=state.iteration,
        )

    def _truncate(self, text: str) -> str:
        limit = self.config.result_summary_chars
        return text if len(text) <= limit else text[:limit]

    def _post_progress(self, state: LoopState, text: str, stable_id: str) -> bool:
        if not self.config.progress_notify:
            return False
        thread = state.thread
        if thread is None:
            return False
        eid = _event_id(state.loop_id, "progress", stable_id)
        if state.delivered_events.get(eid):
            return True
        try:
            state.delivered_events[eid] = False
            store.save_state(self.config.state_dir, state)
            ok = self.human_channel.post_progress(
                loop_id=state.loop_id,
                persona=state.persona,
                thread=thread,
                text=text,
                event_id=eid,
            )
            if ok:
                state.delivered_events[eid] = True
                return True
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "channel_error",
                {"action": "post_progress", "error": "false"},
                iteration=state.iteration,
            )
            return False
        except Exception as exc:
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "channel_error",
                {"action": "post_progress", "error": str(exc)},
                iteration=state.iteration,
            )
            return False

    def _post_deliverable(
        self, state: LoopState, *, summary: str, stable_id: str
    ) -> None:
        thread = state.thread
        if thread is None:
            return
        eid = _event_id(state.loop_id, "deliverable", stable_id)
        if state.delivered_events.get(eid):
            return
        path = str(store.deliverable_path(self.config.state_dir, state.loop_id))
        try:
            state.delivered_events[eid] = False
            store.save_state(self.config.state_dir, state)
            ok = self.human_channel.post_deliverable(
                loop_id=state.loop_id,
                persona=state.persona,
                thread=thread,
                deliverable_path=path,
                summary=summary,
                event_id=eid,
            )
            if ok:
                state.delivered_events[eid] = True
            else:
                store.append_event(
                    self.config.state_dir,
                    state.loop_id,
                    "channel_error",
                    {"action": "post_deliverable", "error": "false"},
                    iteration=state.iteration,
                )
        except Exception as exc:
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "channel_error",
                {"action": "post_deliverable", "error": str(exc)},
                iteration=state.iteration,
            )

    def _record_subtask_done(self, state: LoopState, st: Subtask) -> None:
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            "subtask_done",
            {"id": st.id, "status": st.status, "result_summary": st.result_summary},
            iteration=state.iteration,
        )
        self._post_progress(
            state,
            f"{st.title}\n{st.result_summary}",
            f"i{state.iteration}:subtask:{st.id}:done",
        )

    def _auto_prompt(self, state: LoopState, work_prompt: str) -> str:
        path = store.deliverable_path(self.config.state_dir, state.loop_id)
        excerpt = store.read_deliverable_excerpt(
            self.config.state_dir,
            state.loop_id,
            self.config.deliverable_excerpt_chars,
        )
        return prompts.build_auto_subtask_prompt(
            work_prompt,
            deliverable_path=str(path),
            deliverable_excerpt=excerpt,
        )

    def _load(self, loop_id: str) -> LoopState | None:
        try:
            return store.load_state(self.config.state_dir, loop_id)
        except ValueError as exc:
            logger.error("corrupt state for %s: %s", loop_id, exc)
            store.mark_state_corrupt(self.config.state_dir, loop_id, str(exc))
            return None

    def _tick_loop(self, loop_id: str, *, now: datetime) -> None:
        state = self._load(loop_id)
        if state is None or state.is_terminal():
            return

        comments_ingested = self._process_comments(state, now=now)

        if self._check_cancel(state):
            return

        if self._check_content_change(state):
            self._write_status(state)

        if state.is_terminal():
            return

        transitioned = False
        if state.status == "clarifying":
            transitioned = self._tick_clarifying(state)
        elif state.status == "awaiting_answer":
            transitioned = self._tick_awaiting_answer(state)
        elif state.status == "decomposing":
            transitioned = self._tick_decomposing(state)
        elif state.status == "replanning":
            transitioned = self._tick_replanning(state)
        elif state.status == "awaiting_plan_approval":
            transitioned = self._tick_awaiting_plan_approval(state)
        elif state.status == "executing":
            transitioned = self._tick_executing(state, now=now)
        elif state.status == "awaiting_human":
            transitioned = self._tick_awaiting_human(state, now=now)
        elif state.status == "evaluating":
            transitioned = self._tick_evaluating(state)

        if transitioned or comments_ingested:
            state.updated_at = _now_iso(now)
            store.save_state(self.config.state_dir, state)
            self._write_status(state)

    def _finalize(self, state: LoopState, status: LoopStatus) -> None:
        """done / cancelled / failed の終端処理を一元化する。

        status 更新 → 終端イベント記録 → close_thread の順で必ず実行する。
        """
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"non-terminal status for finalize: {status!r}")
        self._transition(state, status, f"finalize:{status}")
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            status,
            {},
            iteration=state.iteration,
        )
        self._close_thread(state)

    def _resolve_comment_engine_model(self, state: LoopState) -> tuple[str, str]:
        engine, model = self._resolve_llm_engine_model(state)
        override = self.config.comment_model.strip()
        if override:
            model = override
        return engine, model

    def _count_recent_comment_classified(self, state: LoopState, now: datetime) -> int:
        cutoff = now - timedelta(minutes=60)
        count = 0
        for event in store.read_events(self.config.state_dir, state.loop_id):
            if event.get("event") != "comment_classified":
                continue
            ts = _parse_iso(str(event.get("ts", "")))
            if ts is None:
                logger.warning(
                    "ignoring comment_classified with bad timestamp for %s",
                    state.loop_id,
                )
                continue
            if cutoff <= ts <= now:
                count += 1
        return count

    def _record_comment_classified(
        self,
        state: LoopState,
        *,
        message_id: str,
        intent: str,
        source: str,
        reason: str,
    ) -> None:
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            "comment_classified",
            {
                "message_id": message_id,
                "intent": intent,
                "source": source,
                "reason": reason,
            },
            iteration=state.iteration,
        )

    def _reply_comment(
        self,
        state: LoopState,
        *,
        message_id: str,
        intent: str,
        text: str,
    ) -> bool:
        ok = self._post_progress(state, text, f"comment:{message_id}:reply")
        if ok:
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "comment_replied",
                {
                    "message_id": message_id,
                    "intent": intent,
                    "chars": len(text),
                },
                iteration=state.iteration,
            )
        return ok

    def _handle_status_comment(
        self,
        state: LoopState,
        *,
        message_id: str,
        now: datetime,
        source: str,
        reason: str,
    ) -> None:
        self._record_comment_classified(
            state,
            message_id=message_id,
            intent="status",
            source=source,
            reason=reason,
        )
        summary = render_progress_summary(state, now)
        self._reply_comment(
            state, message_id=message_id, intent="status", text=summary
        )

    def _handle_instruction_comment(self, state: LoopState, text: str) -> None:
        if state.subtasks:
            state.replan_feedback = text
            state.replan_trigger_subtask_id = None
            self._transition(state, "replanning", "comment_instruction")
        else:
            state.clarification_context.append(f"補足: {text}")

    def _handle_question_comment(
        self, state: LoopState, *, message_id: str, text: str
    ) -> None:
        if not self._ensure_persona(state):
            state.clarification_context.append(f"補足: {text}")
            return
        deliverable_excerpt = store.read_deliverable_excerpt(
            self.config.state_dir,
            state.loop_id,
            self.config.deliverable_excerpt_chars,
        )
        plan_summary = self._format_plan_text(state) if state.subtasks else "（計画なし）"
        recent_results = "\n".join(
            f"- {st.id} ({st.status}): {st.result_summary or st.result[:200]}"
            for st in state.subtasks
            if st.status in ("success", "failed")
        ) or "（なし）"
        instruction = prompts.build_comment_reply_instruction(
            objective=state.body,
            deliverable_excerpt=deliverable_excerpt,
            plan_summary=plan_summary,
            recent_results=recent_results,
            comment_text=text,
            max_chars=self.config.comment_reply_max_chars,
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            engine, model = self._resolve_comment_engine_model(state)
            resp, trace = prompts.run_reply_comment(
                prompt, engine=engine, model=model
            )
            self._log_llm(state, trace)
            reply = resp.reply
            if len(reply) > self.config.comment_reply_max_chars:
                reply = reply[: self.config.comment_reply_max_chars]
            self._reply_comment(
                state, message_id=message_id, intent="question", text=reply
            )
        except Exception as exc:
            if isinstance(exc, prompts.LlmCallError):
                self._log_llm(state, exc.trace)
            else:
                store.append_event(
                    self.config.state_dir,
                    state.loop_id,
                    "llm_call",
                    {
                        "input": "",
                        "output": {"raw": "", "parsed": None},
                        "reasoning": "",
                        "config": {},
                        "metadata": {},
                        "uncertain_flag": False,
                        "error": str(exc),
                    },
                    iteration=state.iteration,
                )
            state.clarification_context.append(f"補足: {text}")

    def _process_comments(self, state: LoopState, *, now: datetime) -> bool:
        """active loop の kind=comment を分類・応答・再計画に振り分ける。"""
        messages = [
            m
            for m in store.list_inbox_messages(self.config.state_dir, state.loop_id)
            if m.kind == "comment"
        ]
        if not messages:
            return False

        consumed_ids = store.list_consumed_message_ids(
            self.config.state_dir, state.loop_id
        )
        changed = False
        handled = 0
        for msg in messages:
            if handled >= self.config.max_comments_per_tick:
                break
            if msg.message_id in consumed_ids:
                store.consume_inbox_message(
                    self.config.state_dir, state.loop_id, msg.filename
                )
                continue

            text = msg.text.strip()
            store.consume_inbox_message(
                self.config.state_dir, state.loop_id, msg.filename
            )
            consumed_ids.add(msg.message_id)
            handled += 1
            changed = True

            if not text:
                store.append_event(
                    self.config.state_dir,
                    state.loop_id,
                    "comment_warning",
                    {"message_id": msg.message_id, "reason": "empty_text"},
                    iteration=state.iteration,
                )
                continue

            if is_status_inquiry(text):
                self._handle_status_comment(
                    state,
                    message_id=msg.message_id,
                    now=now,
                    source="deterministic",
                    reason="status_keyword",
                )
                continue

            budget = self.config.comment_reply_budget_per_hour
            recent = self._count_recent_comment_classified(state, now)
            if budget <= 0 or recent >= budget:
                self._handle_status_comment(
                    state,
                    message_id=msg.message_id,
                    now=now,
                    source="budget_fallback",
                    reason="budget_exhausted" if budget > 0 else "budget_disabled",
                )
                continue

            try:
                if not self._ensure_persona(state):
                    state.clarification_context.append(f"補足: {text}")
                    continue
                instruction = prompts.build_comment_classify_instruction(text)
                prompt = self._format_with_persona(state, instruction)
                engine, model = self._resolve_comment_engine_model(state)
                resp, trace = prompts.run_classify_comment(
                    prompt, engine=engine, model=model
                )
                self._log_llm(state, trace)
                self._record_comment_classified(
                    state,
                    message_id=msg.message_id,
                    intent=resp.intent,
                    source="llm",
                    reason=resp.reason,
                )
                if resp.intent == "status":
                    summary = render_progress_summary(state, now)
                    self._reply_comment(
                        state,
                        message_id=msg.message_id,
                        intent="status",
                        text=summary,
                    )
                elif resp.intent == "instruction":
                    self._handle_instruction_comment(state, text)
                elif resp.intent == "question":
                    self._handle_question_comment(
                        state, message_id=msg.message_id, text=text
                    )
                else:
                    state.clarification_context.append(f"補足: {text}")
            except Exception as exc:
                if isinstance(exc, prompts.LlmCallError):
                    self._log_llm(state, exc.trace)
                else:
                    store.append_event(
                        self.config.state_dir,
                        state.loop_id,
                        "llm_call",
                        {
                            "input": "",
                            "output": {"raw": "", "parsed": None},
                            "reasoning": "",
                            "config": {},
                            "metadata": {},
                            "uncertain_flag": False,
                            "error": str(exc),
                        },
                        iteration=state.iteration,
                    )
                state.clarification_context.append(f"補足: {text}")

        return changed

    def _check_cancel(self, state: LoopState) -> bool:
        cancelled = False
        if self.objective_cancelled and self.objective_cancelled(state.loop_id):
            cancelled = True
        if self.objective_exists and not self.objective_exists(state.loop_id):
            cancelled = True
        for msg in store.list_inbox_messages(self.config.state_dir, state.loop_id):
            if msg.kind == "cancel" and msg.message_id not in store.list_consumed_message_ids(
                self.config.state_dir, state.loop_id
            ):
                store.consume_inbox_message(self.config.state_dir, state.loop_id, msg.filename)
                cancelled = True
        if cancelled and not state.is_terminal():
            self._finalize(state, "cancelled")
            state.updated_at = _now_iso()
            store.save_state(self.config.state_dir, state)
            self._write_status(state)
            return True
        return False

    def _check_content_change(self, state: LoopState) -> bool:
        if state.is_terminal():
            return False
        if self.objective_hash_changed and self.objective_hash_changed(
            state.loop_id, state.objective_hash
        ):
            warn = (
                "Objective content changed during execution. "
                "Create a new Objective with a new id to restart."
            )
            if state.content_change_warning != warn:
                state.content_change_warning = warn
                return True
        return False

    def _record_error(self, state: LoopState, event: str, data: dict) -> None:
        state.consecutive_errors += 1
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            event,
            data,
            iteration=state.iteration,
        )
        if state.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
            self._finalize(state, "failed")
            self._notify_fallback(state, f"Loop failed after {_MAX_CONSECUTIVE_ERRORS} errors")

    def _clear_errors(self, state: LoopState) -> None:
        state.consecutive_errors = 0

    def _format_with_persona(self, state: LoopState, instruction: str) -> str:
        persona_path = self.config.persona_dir / f"{state.persona}.md"
        persona = load_persona(persona_path)
        return persona.format_prompt(instruction, weight="heavy")

    def _resolve_persona_engine_model(
        self,
        state: LoopState,
        *,
        fallback_engine: str,
        fallback_model: str,
    ) -> tuple[str, str]:
        """ペルソナ fm.engine/model を優先し、空または読込失敗時は fallback を返す。"""
        persona_path = self.config.persona_dir / f"{state.persona}.md"
        try:
            persona = load_persona(persona_path)
            engine = persona.fm.engine or fallback_engine
            model = persona.fm.model or fallback_model
            return engine, model
        except Exception:
            return fallback_engine, fallback_model

    def _resolve_llm_engine_model(self, state: LoopState) -> tuple[str, str]:
        return self._resolve_persona_engine_model(
            state,
            fallback_engine=self.config.llm_engine,
            fallback_model=self.config.llm_model,
        )

    def _resolve_subtask_engine_model(self, state: LoopState) -> tuple[str, str]:
        return self._resolve_persona_engine_model(
            state,
            fallback_engine=self.config.subtask_engine,
            fallback_model=self.config.subtask_model,
        )

    def _ensure_persona(self, state: LoopState) -> bool:
        persona_path = self.config.persona_dir / f"{state.persona}.md"
        try:
            load_persona(persona_path)
            return True
        except Exception as exc:
            eid = _event_id(state.loop_id, "fallback", "persona")
            if not state.delivered_events.get(eid):
                state.delivered_events[eid] = False
                store.save_state(self.config.state_dir, state)
                try:
                    self.human_channel.notify_fallback(
                        loop_id=state.loop_id,
                        text=f"Persona {state.persona!r} not found: {exc}",
                        event_id=eid,
                    )
                    state.delivered_events[eid] = True
                except Exception as notify_exc:
                    store.append_event(
                        self.config.state_dir,
                        state.loop_id,
                        "channel_error",
                        {"action": "notify_fallback", "error": str(notify_exc)},
                        iteration=state.iteration,
                    )
            self._finalize(state, "failed")
            return False

    def _open_thread_if_needed(self, state: LoopState) -> HumanThreadRef | None:
        if state.thread is not None:
            return state.thread
        eid = _event_id(state.loop_id, "open_thread", "initial")
        if state.delivered_events.get(eid) and state.thread:
            return state.thread
        try:
            state.delivered_events[eid] = False
            store.save_state(self.config.state_dir, state)
            ref = self.human_channel.open_thread(
                loop_id=state.loop_id,
                persona=state.persona,
                title=state.title,
                body=state.body,
                event_id=eid,
            )
            if ref is not None:
                state.thread = ref
                state.delivered_events[eid] = True
                self._clear_errors(state)
            else:
                self._record_error(state, "channel_error", {"action": "open_thread", "error": "no thread returned"})
            return ref
        except Exception as exc:
            self._record_error(state, "channel_error", {"action": "open_thread", "error": str(exc)})
            return None

    def _ask(self, state: LoopState, question_id: str, text: str, *, kind: str) -> bool:
        thread = self._open_thread_if_needed(state)
        if thread is None:
            return False
        eid = _event_id(state.loop_id, "ask", question_id)
        if state.delivered_events.get(eid):
            return True
        try:
            state.delivered_events[eid] = False
            state.pending_question = PendingQuestion(question_id=question_id, text=text, kind=kind)
            store.save_state(self.config.state_dir, state)
            ok = self.human_channel.ask(
                loop_id=state.loop_id,
                persona=state.persona,
                thread=thread,
                question_id=question_id,
                text=text,
                event_id=eid,
            )
            if ok:
                state.delivered_events[eid] = True
                self._clear_errors(state)
                store.append_event(
                    self.config.state_dir,
                    state.loop_id,
                    "question_asked",
                    {
                        "question_id": question_id,
                        "kind": kind,
                        "text_excerpt": self._truncate(text),
                    },
                    iteration=state.iteration,
                )
            return ok
        except Exception as exc:
            self._record_error(state, "channel_error", {"action": "ask", "error": str(exc)})
            return False

    def _notify(self, state: LoopState, text: str, stable_id: str) -> None:
        thread = state.thread
        if thread is None:
            return
        eid = _event_id(state.loop_id, "notify", stable_id)
        if state.delivered_events.get(eid):
            return
        try:
            state.delivered_events[eid] = False
            store.save_state(self.config.state_dir, state)
            ok = self.human_channel.notify(
                loop_id=state.loop_id,
                persona=state.persona,
                thread=thread,
                text=text,
                event_id=eid,
            )
            if ok:
                state.delivered_events[eid] = True
                self._clear_errors(state)
        except Exception as exc:
            self._record_error(state, "channel_error", {"action": "notify", "error": str(exc)})

    def _notify_fallback(self, state: LoopState, text: str) -> None:
        eid = _event_id(state.loop_id, "fallback", text[:32])
        if state.delivered_events.get(eid):
            return
        try:
            self.human_channel.notify_fallback(
                loop_id=state.loop_id,
                text=text,
                event_id=eid,
            )
            state.delivered_events[eid] = True
        except Exception as exc:
            logger.warning("notify_fallback failed: %s", exc)

    def _close_thread(self, state: LoopState) -> None:
        if state.thread is None:
            return
        eid = _event_id(state.loop_id, "close_thread", "final")
        if state.delivered_events.get(eid):
            return
        try:
            state.delivered_events[eid] = False
            store.save_state(self.config.state_dir, state)
            ok = self.human_channel.close_thread(
                loop_id=state.loop_id,
                persona=state.persona,
                thread=state.thread,
                event_id=eid,
            )
            if ok:
                state.delivered_events[eid] = True
        except Exception as exc:
            logger.warning("close_thread failed: %s", exc)

    def _log_llm(self, state: LoopState, trace: prompts.LlmTrace) -> None:
        raw = trace.raw_output
        raw_s = raw if isinstance(raw, str) else str(raw)
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            "llm_call",
            {
                "input": trace.input,
                "output": {"raw": raw_s, "parsed": trace.parsed},
                "reasoning": trace.reasoning,
                "config": trace.config,
                "metadata": trace.metadata,
                "uncertain_flag": trace.uncertain_flag,
                "error": trace.error,
            },
            iteration=state.iteration,
        )

    @staticmethod
    def _summarize_llm_error(exc: Exception) -> str:
        """エラー記録用にプリミティブな文字列要約へ正規化する。"""
        if not isinstance(exc, prompts.LlmCallError):
            return str(exc)
        parts = [str(exc)]
        cfg = exc.trace.config or {}
        engine = cfg.get("engine")
        model = cfg.get("model")
        if engine:
            parts.append(f"engine={engine}")
        if model:
            parts.append(f"model={model}")
        raw = exc.trace.raw_output
        returncode = getattr(raw, "returncode", None)
        stderr = getattr(raw, "stderr", None)
        if returncode is not None:
            parts.append(f"returncode={returncode}")
        if stderr:
            stderr_s = stderr if isinstance(stderr, str) else str(stderr)
            parts.append(f"stderr={stderr_s[:200]}")
        return "; ".join(parts)

    def _record_llm_error(self, state: LoopState, phase: str, exc: Exception) -> None:
        """LLM 失敗を記録する。記録経路自体は例外を外へ出さない。"""
        summary = self._summarize_llm_error(exc)
        try:
            if isinstance(exc, prompts.LlmCallError):
                self._log_llm(state, exc.trace)
        except Exception as log_exc:
            logger.warning("failed to log llm trace during error record: %s", log_exc)
        errors_before_record = state.consecutive_errors
        try:
            self._record_error(state, "llm_error", {"phase": phase, "error": summary})
        except Exception as record_exc:
            logger.warning("failed to record llm_error event: %s", record_exc)
            if state.consecutive_errors == errors_before_record:
                state.consecutive_errors += 1
            if state.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                self._finalize(state, "failed")

    def _tick_clarifying(self, state: LoopState) -> bool:
        if not self._ensure_persona(state):
            return True
        if state.clarify_round >= self.config.max_clarify_rounds:
            # 上限到達は正常系: 仮定して decompose へ進む（エラー連発で failed にしない）
            self._notify(
                state,
                "明確化はここまでにして、不明点は仮定して進めるね",
                "clarify_limit",
            )
            try:
                store.append_event(
                    self.config.state_dir,
                    state.loop_id,
                    "clarify_limit",
                    {"rounds": state.clarify_round},
                    iteration=state.iteration,
                )
            except Exception as exc:
                logger.warning("failed to record clarify_limit event: %s", exc)
            self._transition(state, "decomposing", "clarify_limit")
            return True

        instruction = prompts.build_clarify_instruction(
            state.body,
            round_num=state.clarify_round + 1,
            max_rounds=self.config.max_clarify_rounds,
            clarification_context=state.clarification_context,
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            engine, model = self._resolve_llm_engine_model(state)
            resp, trace = prompts.run_clarify(
                prompt,
                engine=engine,
                model=model,
            )
            self._log_llm(state, trace)
            self._clear_errors(state)
        except Exception as exc:
            self._record_llm_error(state, "clarify", exc)
            return True

        if resp.clear:
            self._transition(state, "decomposing", "clarify_clear")
            state.clarify_round = 0
            state.pending_question = None
        else:
            state.clarify_round += 1
            qid = f"clarify-{state.clarify_round}"
            if not self._ask(state, qid, resp.question or "", kind="clarify"):
                return False
            self._transition(state, "awaiting_answer", "clarify_question")
        return True

    def _tick_awaiting_answer(self, state: LoopState) -> bool:
        consumed = store.list_consumed_message_ids(self.config.state_dir, state.loop_id)
        pending = state.pending_question
        if pending is None:
            self._transition(state, "clarifying", "awaiting_without_pending")
            return True
        for msg in store.list_inbox_messages(self.config.state_dir, state.loop_id):
            if msg.message_id in consumed:
                continue
            if msg.kind != "answer" or msg.question_id != pending.question_id:
                continue
            store.consume_inbox_message(self.config.state_dir, state.loop_id, msg.filename)
            state.clarification_context.append(
                f"Q: {pending.text}\nA: {msg.text}"
            )
            state.pending_question = None
            self._transition(state, "clarifying", "answer_received")
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "answer_received",
                {"question_id": msg.question_id, "text": msg.text},
                iteration=state.iteration,
            )
            return True
        return False

    def _tick_decomposing(self, state: LoopState) -> bool:
        if not self._ensure_persona(state):
            return True
        instruction = prompts.build_decompose_instruction(
            state.body,
            iteration=state.iteration,
            max_subtasks=self.config.max_subtasks_per_iteration,
            next_focus=state.next_focus,
            clarification_context=state.clarification_context,
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            engine, model = self._resolve_llm_engine_model(state)
            resp, trace = prompts.run_decompose(
                prompt,
                engine=engine,
                model=model,
                max_subtasks=self.config.max_subtasks_per_iteration,
            )
            self._log_llm(state, trace)
            self._clear_errors(state)
        except Exception as exc:
            self._record_llm_error(state, "decompose", exc)
            return True

        state.subtasks = [_subtask_from_decompose(s) for s in resp.subtasks]
        state.current_subtask_id = state.subtasks[0].id if state.subtasks else None
        state.replan_count = 0
        state.replan_feedback = ""
        state.replan_trigger_subtask_id = None
        state.plan_revision = 0
        state.next_focus = ""
        return self._after_plan_ready(state, event="plan_proposed", reason="decompose_done")

    def _plan_question_id(self, state: LoopState) -> str:
        return f"plan-approval-{state.iteration}-{state.plan_revision}"

    def _format_plan_text(self, state: LoopState) -> str:
        lines = [f"- [{s.kind}] {s.title} (`{s.id}`)" for s in state.subtasks]
        return "Plan:\n" + "\n".join(lines)

    def _after_plan_ready(
        self, state: LoopState, *, event: str, reason: str
    ) -> bool:
        plan_text = self._format_plan_text(state)
        self._post_progress(
            state,
            plan_text,
            f"i{state.iteration}:plan:r{state.plan_revision}",
        )
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            event,
            {
                "iteration": state.iteration,
                "plan_revision": state.plan_revision,
                "subtasks": [
                    {
                        "id": s.id,
                        "kind": s.kind,
                        "title": s.title,
                        "depends": list(s.depends),
                        "status": s.status,
                    }
                    for s in state.subtasks
                ],
            },
            iteration=state.iteration,
        )
        if state.plan_approval:
            qid = self._plan_question_id(state)
            ask_text = (
                f"{plan_text}\n\n"
                "承認する場合は「ok」「承認」「進めて」「go」のいずれか全文で返信してください。"
                "それ以外は修正指示として再計画します。"
            )
            if not self._ask(state, qid, ask_text, kind="plan_approval"):
                return False
            self._transition(state, "awaiting_plan_approval", reason)
        else:
            self._transition(state, "executing", reason)
        return True

    def _is_plan_approval(self, text: str) -> bool:
        return text.strip().casefold() in {a.casefold() for a in _PLAN_APPROVAL_ANSWERS}

    def _tick_awaiting_plan_approval(self, state: LoopState) -> bool:
        pending = state.pending_question
        if pending is None or pending.kind != "plan_approval":
            self._transition(state, "executing", "awaiting_plan_without_pending")
            return True
        consumed = store.list_consumed_message_ids(self.config.state_dir, state.loop_id)
        for msg in store.list_inbox_messages(self.config.state_dir, state.loop_id):
            if msg.message_id in consumed:
                continue
            if msg.kind != "answer" or msg.question_id != pending.question_id:
                continue
            store.consume_inbox_message(self.config.state_dir, state.loop_id, msg.filename)
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "answer_received",
                {"question_id": msg.question_id, "text": msg.text, "kind": "plan_approval"},
                iteration=state.iteration,
            )
            if self._is_plan_approval(msg.text):
                state.pending_question = None
                store.append_event(
                    self.config.state_dir,
                    state.loop_id,
                    "plan_approved",
                    {"plan_revision": state.plan_revision},
                    iteration=state.iteration,
                )
                self._transition(state, "executing", "plan_approved")
                return True

            # revision request
            state.pending_question = None
            state.replan_feedback = msg.text
            state.replan_trigger_subtask_id = None
            if state.plan_revision >= self.config.max_plan_revisions:
                self._notify(
                    state,
                    "計画修正の上限に達したため、最新案で実行を開始します。",
                    f"plan_revision_limit:{state.iteration}",
                )
                self._transition(state, "executing", "plan_revision_limit")
                return True
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "plan_revised",
                {
                    "plan_revision": state.plan_revision,
                    "feedback": msg.text,
                },
                iteration=state.iteration,
            )
            self._transition(state, "replanning", "plan_revision_requested")
            return True
        return False

    def _tick_replanning(self, state: LoopState) -> bool:
        if not self._ensure_persona(state):
            return True

        human_revision = bool(state.replan_feedback) and state.replan_trigger_subtask_id is None
        if not human_revision and state.replan_count >= self.config.max_replans_per_iteration:
            self._transition(state, "evaluating", "replan_limit")
            return True

        existing_ids = {s.id for s in state.subtasks}
        required_keep = {s.id for s in state.subtasks if s.status in ("running", "success")}
        plan_summary = "\n".join(
            f"- {s.id} kind={s.kind} depends={list(s.depends)} "
            f"status={s.status} summary={s.result_summary}"
            for s in state.subtasks
        )
        failure_detail = state.replan_feedback
        if state.replan_trigger_subtask_id:
            trigger = next(
                (s for s in state.subtasks if s.id == state.replan_trigger_subtask_id),
                None,
            )
            if trigger is not None:
                failure_detail = trigger.result_summary or trigger.result or failure_detail

        deliverable_excerpt = store.read_deliverable_excerpt(
            self.config.state_dir,
            state.loop_id,
            self.config.deliverable_excerpt_chars,
        )
        instruction = prompts.build_replan_instruction(
            state.body,
            plan_summary=plan_summary,
            failure_detail=failure_detail or "replan requested",
            deliverable_excerpt=deliverable_excerpt,
            human_feedback=state.replan_feedback if human_revision else "",
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            engine, model = self._resolve_llm_engine_model(state)
            resp, trace = prompts.run_replan(
                prompt,
                engine=engine,
                model=model,
                existing_ids=existing_ids,
                required_keep=required_keep,
                max_subtasks=self.config.max_subtasks_per_iteration,
            )
            self._log_llm(state, trace)
            self._clear_errors(state)
        except Exception as exc:
            self._record_llm_error(state, "replan", exc)
            return True

        old_plan = [s.to_dict() for s in state.subtasks]
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            "replan_triggered",
            {
                "reason": resp.reason,
                "trigger_subtask_id": state.replan_trigger_subtask_id,
                "human_revision": human_revision,
                "old_plan": old_plan,
                "keep": list(resp.keep),
                "add": [dict(s.__dict__) for s in resp.add],
            },
            iteration=state.iteration,
        )

        kept = [s for s in state.subtasks if s.id in set(resp.keep)]
        added = [_subtask_from_decompose(s) for s in resp.add]
        state.subtasks = kept + added
        state.current_subtask_id = next(
            (s.id for s in state.subtasks if s.status == "pending"),
            state.subtasks[0].id if state.subtasks else None,
        )
        if human_revision:
            state.plan_revision += 1
        else:
            state.replan_count += 1
        state.replan_feedback = ""
        state.replan_trigger_subtask_id = None

        event = "plan_revised" if human_revision else "plan_proposed"
        return self._after_plan_ready(state, event=event, reason="replan_done")

    def _by_id(self, state: LoopState) -> dict[str, Subtask]:
        return {s.id: s for s in state.subtasks}

    def _deps_satisfied(self, st: Subtask, by_id: dict[str, Subtask]) -> bool:
        return all(by_id[d].status == "success" for d in st.depends if d in by_id)

    def _deps_failed(self, st: Subtask, by_id: dict[str, Subtask]) -> bool:
        return any(by_id[d].status == "failed" for d in st.depends if d in by_id)

    def _all_subtasks_done(self, state: LoopState) -> bool:
        return bool(state.subtasks) and all(
            st.status in ("success", "failed") for st in state.subtasks
        )

    def _evaluate_condition(
        self, condition: dict[str, object], *, previous_token: str | None
    ) -> WatchVerdict:
        ctype = condition.get("type")
        if ctype in _LOCAL_CONDITION_TYPES:
            if self.config.watch_root is None:
                return WatchVerdict(
                    status="failed",
                    detail="watch_root is not configured for local path conditions",
                )
            return PathConditionEvaluator(self.config.watch_root).evaluate(
                condition, previous_token=previous_token
            )
        if self.condition_evaluator is None:
            return WatchVerdict(
                status="failed",
                detail=f"no host ConditionEvaluator for type {ctype!r}",
            )
        try:
            verdict = self.condition_evaluator.evaluate(
                condition, previous_token=previous_token
            )
        except Exception as exc:
            return WatchVerdict(
                status="failed",
                detail=f"evaluator raised {type(exc).__name__}: {exc}",
            )
        if verdict is None:
            return WatchVerdict(status="failed", detail="evaluator returned None")
        if verdict.status not in ("pending", "satisfied", "failed"):
            return WatchVerdict(
                status="failed",
                detail=f"invalid WatchStatus: {verdict.status!r}",
            )
        return verdict

    def _record_watch_polled(
        self,
        state: LoopState,
        st: Subtask,
        verdict: WatchVerdict,
        *,
        prev_status: str,
        prev_detail: str,
        prev_token: str | None,
    ) -> None:
        if (
            verdict.status == prev_status
            and verdict.detail == prev_detail
            and verdict.observed_token == prev_token
        ):
            return
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            "watch_polled",
            {
                "id": st.id,
                "status": verdict.status,
                "detail": verdict.detail,
                "observed_token": verdict.observed_token,
            },
            iteration=state.iteration,
        )

    def _poll_watches(self, state: LoopState, *, now: datetime) -> bool:
        """watch を評価する。replanning へ遷移したら True。"""
        by_id = self._by_id(state)
        for st in state.subtasks:
            if st.kind != "watch" or st.status not in ("pending", "running"):
                continue
            if self._deps_failed(st, by_id):
                st.status = "failed"
                st.result = "dependency failed"
                st.result_summary = self._truncate(st.result)
                self._record_subtask_done(state, st)
                continue
            if not self._deps_satisfied(st, by_id):
                continue

            if st.started_at is None:
                st.started_at = _now_iso(now)
                st.status = "running"

            timeout_sec = st.timeout_sec or prompts.DEFAULT_WATCH_TIMEOUT_SEC
            started = _parse_iso(st.started_at)
            if started is not None:
                elapsed = (now - started).total_seconds()
                if elapsed >= timeout_sec:
                    st.status = "failed"
                    st.result = f"watch timeout after {timeout_sec}s"
                    st.result_summary = self._truncate(st.result)
                    self._record_subtask_done(state, st)
                    state.replan_trigger_subtask_id = st.id
                    state.replan_feedback = st.result_summary
                    self._transition(state, "replanning", f"watch_timeout:{st.id}")
                    return True

            poll_interval = st.poll_interval_sec or prompts.DEFAULT_WATCH_POLL_INTERVAL_SEC
            if st.last_polled_at is not None:
                last = _parse_iso(st.last_polled_at)
                if last is not None and (now - last).total_seconds() < poll_interval:
                    continue

            if st.condition is None:
                verdict = WatchVerdict(status="failed", detail="missing condition")
            else:
                verdict = self._evaluate_condition(
                    st.condition, previous_token=st.watch_token
                )
            prev_detail = st.result_summary
            prev_token = st.watch_token
            prev_status = "pending" if st.status == "running" else st.status
            st.last_polled_at = _now_iso(now)
            if verdict.observed_token is not None:
                st.watch_token = verdict.observed_token
            st.result = verdict.detail
            st.result_summary = self._truncate(verdict.detail)
            self._record_watch_polled(
                state,
                st,
                verdict,
                prev_status=prev_status,
                prev_detail=prev_detail,
                prev_token=prev_token,
            )

            if verdict.status == "satisfied":
                st.status = "success"
                self._record_subtask_done(state, st)
            elif verdict.status == "failed":
                st.status = "failed"
                self._record_subtask_done(state, st)
                state.replan_trigger_subtask_id = st.id
                state.replan_feedback = st.result_summary
                self._transition(state, "replanning", f"watch_failed:{st.id}")
                return True
        return False

    def _fail_blocked_by_deps(self, state: LoopState) -> None:
        by_id = self._by_id(state)
        for st in state.subtasks:
            if st.status != "pending":
                continue
            if self._deps_failed(st, by_id):
                st.status = "failed"
                st.result = "dependency failed"
                st.result_summary = self._truncate(st.result)
                self._record_subtask_done(state, st)

    def _poll_running_autos(self, state: LoopState, *, now: datetime) -> bool:
        """in-flight auto を1件だけ poll する。状態が変わったら True。"""
        for st in state.subtasks:
            if st.kind != "auto" or st.status != "running" or st.submission is None:
                continue
            sub = st.submission
            try:
                poll = self.executor.poll(uuid=sub.uuid, result_filename=sub.result_filename)
                self._clear_errors(state)
            except Exception as exc:
                self._record_error(state, "executor_error", {"action": "poll", "error": str(exc)})
                return True

            if poll.status == "pending":
                if self._subtask_timed_out(sub.submitted_at, now=now):
                    st.status = "failed"
                    st.result = f"timeout after {self.config.subtask_timeout_sec}s"
                    st.result_summary = self._truncate(st.result)
                    st.result_filename = sub.result_filename
                    self._record_subtask_done(state, st)
                    return True
                return False

            if poll.status == "success":
                st.status = "success"
                st.result = poll.content
                st.result_summary = self._truncate(poll.content)
                st.result_filename = sub.result_filename
                store.append_event(
                    self.config.state_dir,
                    state.loop_id,
                    "deliverable_updated",
                    store.deliverable_snapshot(self.config.state_dir, state.loop_id),
                    iteration=state.iteration,
                )
            else:
                st.status = "failed"
                st.result = poll.content
                st.result_summary = self._truncate(poll.content)
                st.result_filename = sub.result_filename
            self._record_subtask_done(state, st)
            return True
        return False

    def _tick_executing(self, state: LoopState, *, now: datetime) -> bool:
        if self._poll_watches(state, now=now):
            return True
        if state.status == "replanning":
            return True

        self._fail_blocked_by_deps(state)
        self._poll_running_autos(state, now=now)
        self._fail_blocked_by_deps(state)
        by_id = self._by_id(state)

        auto_running = any(
            s.kind == "auto" and s.status == "running" for s in state.subtasks
        )
        if not auto_running:
            for st in state.subtasks:
                if st.kind != "auto" or st.status != "pending":
                    continue
                if not self._deps_satisfied(st, by_id):
                    continue
                key = f"loops:{state.loop_id}:i{state.iteration}:{st.id}"
                try:
                    engine, model = self._resolve_subtask_engine_model(state)
                    submission = self.executor.submit(
                        prompt=self._auto_prompt(state, st.prompt),
                        idempotency_key=key,
                        engine=engine,
                        model=model,
                    )
                    st.submission = submission
                    st.status = "running"
                    st.started_at = _now_iso(now)
                    state.current_subtask_id = st.id
                    store.append_event(
                        self.config.state_dir,
                        state.loop_id,
                        "subtask_submitted",
                        {
                            "id": st.id,
                            "uuid": submission.uuid,
                            "result_filename": submission.result_filename,
                        },
                        iteration=state.iteration,
                    )
                    self._clear_errors(state)
                except Exception as exc:
                    self._record_error(
                        state, "executor_error", {"action": "submit", "error": str(exc)}
                    )
                break

        # human: one at a time
        human_running = any(
            s.kind == "human" and s.status == "running" for s in state.subtasks
        )
        if not human_running:
            for st in state.subtasks:
                if st.kind != "human" or st.status != "pending":
                    continue
                if not self._deps_satisfied(st, by_id):
                    continue
                qid = f"human-{state.iteration}-{st.id}"
                excerpt = store.read_deliverable_excerpt(
                    self.config.state_dir,
                    state.loop_id,
                    self.config.deliverable_excerpt_chars,
                )
                self._post_deliverable(
                    state,
                    summary=self._truncate(excerpt or st.prompt),
                    stable_id=f"{st.id}:start",
                )
                if not self._ask(state, qid, st.prompt, kind="human_subtask"):
                    return False
                st.status = "running"
                st.started_at = _now_iso(now)
                state.current_subtask_id = st.id
                self._transition(state, "awaiting_human", f"human_subtask:{st.id}")
                return True

        if self._all_subtasks_done(state):
            self._transition(state, "evaluating", "all_subtasks_done")
        return True

    def _subtask_timed_out(self, submitted_at: str, now: datetime | None = None) -> bool:
        submitted = _parse_iso(submitted_at)
        if submitted is None:
            return False
        current = now or datetime.now(_TZ)
        elapsed = (current - submitted).total_seconds()
        return elapsed > self.config.subtask_timeout_sec

    def _tick_awaiting_human(self, state: LoopState, *, now: datetime) -> bool:
        if self._poll_watches(state, now=now):
            return True
        if state.status == "replanning":
            return True
        self._poll_running_autos(state, now=now)

        st = None
        if state.current_subtask_id is not None:
            st = next((s for s in state.subtasks if s.id == state.current_subtask_id), None)
        if st is None or st.kind != "human":
            st = next(
                (s for s in state.subtasks if s.kind == "human" and s.status == "running"),
                None,
            )
        if st is None:
            self._transition(state, "executing", "awaiting_human_without_subtask")
            return True
        consumed = store.list_consumed_message_ids(self.config.state_dir, state.loop_id)
        pending = state.pending_question
        expected_qid = f"human-{state.iteration}-{st.id}"
        for msg in store.list_inbox_messages(self.config.state_dir, state.loop_id):
            if msg.message_id in consumed:
                continue
            if msg.kind != "answer":
                continue
            if pending and msg.question_id != pending.question_id:
                continue
            if msg.question_id != expected_qid:
                continue
            store.consume_inbox_message(self.config.state_dir, state.loop_id, msg.filename)
            st.status = "success"
            st.result = msg.text
            st.result_summary = self._truncate(msg.text)
            st.result_filename = ""
            state.pending_question = None
            self._post_deliverable(
                state,
                summary=st.result_summary,
                stable_id=f"{st.id}:done",
            )
            self._record_subtask_done(state, st)
            self._transition(state, "executing", f"human_answer:{st.id}")
            if self._all_subtasks_done(state):
                self._transition(state, "evaluating", "all_subtasks_done")
            return True
        return True

    def _tick_evaluating(self, state: LoopState) -> bool:
        if not self._ensure_persona(state):
            return True
        results_summary = "\n".join(
            f"- {st.id} ({st.status}): {st.result_summary or st.result[:200]}"
            for st in state.subtasks
        )
        deliverable_excerpt = store.read_deliverable_excerpt(
            self.config.state_dir,
            state.loop_id,
            self.config.deliverable_excerpt_chars,
        )
        instruction = prompts.build_evaluate_instruction(
            state.body,
            results_summary=results_summary,
            iteration=state.iteration,
            max_iterations=state.max_iterations,
            deliverable_excerpt=deliverable_excerpt,
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            engine, model = self._resolve_llm_engine_model(state)
            resp, trace = prompts.run_evaluate(
                prompt,
                engine=engine,
                model=model,
            )
            self._log_llm(state, trace)
            self._clear_errors(state)
        except Exception as exc:
            self._record_llm_error(state, "evaluate", exc)
            return True

        if resp.achieved:
            self._notify(state, f"Loop completed: {resp.summary}", "done")
            self._finalize(state, "done")
        elif state.iteration < state.max_iterations:
            state.iteration += 1
            state.subtasks = []
            state.current_subtask_id = None
            state.next_focus = resp.next_focus
            state.replan_count = 0
            state.plan_revision = 0
            state.replan_feedback = ""
            state.replan_trigger_subtask_id = None
            self._transition(state, "decomposing", "evaluate_continue")
        else:
            self._notify(state, f"Loop failed: {resp.summary}", "failed")
            self._finalize(state, "failed")
        return True

    def _write_status(self, state: LoopState) -> None:
        write_status(
            self.config.status_dir,
            state,
            on_written=self.config.on_status_written,
        )
