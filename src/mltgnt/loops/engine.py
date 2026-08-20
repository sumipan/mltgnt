"""mltgnt.loops.engine — 1 tick 1 遷移の状態機械。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from mltgnt.config import LoopsConfig
from mltgnt.execution.base_runner import BaseRunner
from mltgnt.interfaces.loops import HumanChannel, HumanThreadRef, SubtaskExecutor
from mltgnt.loops.models import (
    LoopState,
    PendingQuestion,
    Subtask,
    TERMINAL_STATUSES,
)
from mltgnt.loops.objective import Objective
from mltgnt.loops import prompts
from mltgnt.loops import store
from mltgnt.loops.status import write_status
from mltgnt.persona.loader import load as load_persona

logger = logging.getLogger("mltgnt.loops.engine")

_TZ = ZoneInfo("Asia/Tokyo")
_MAX_CONSECUTIVE_ERRORS = 3


def _now_iso() -> str:
    return datetime.now(_TZ).isoformat()


def _event_id(loop_id: str, kind: str, stable_id: str) -> str:
    return f"loops:{loop_id}:{kind}:{stable_id}"


@dataclass
class LoopsEngine(BaseRunner):
    config: LoopsConfig
    human_channel: HumanChannel
    executor: SubtaskExecutor
    objective_exists: Callable[[str], bool] | None = None
    objective_cancelled: Callable[[str], bool] | None = None
    objective_hash_changed: Callable[[str, str], bool] | None = None

    def tick(self, now: datetime | None = None) -> None:
        for loop_id in self._active_loop_ids():
            try:
                self._tick_loop(loop_id)
            except Exception:
                logger.exception("loop %s tick failed", loop_id)

    def _active_loop_ids(self) -> list[str]:
        return store.list_restorable_loops(self.config.state_dir)

    def start_loop(self, objective: Objective) -> LoopState:
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
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        store.save_state(self.config.state_dir, state)
        store.append_event(
            self.config.state_dir,
            objective.loop_id,
            "loop_started",
            {"objective_path": str(objective.path)},
            iteration=1,
        )
        self._write_status(state)
        return state

    def _load(self, loop_id: str) -> LoopState | None:
        try:
            return store.load_state(self.config.state_dir, loop_id)
        except ValueError as exc:
            logger.error("corrupt state for %s: %s", loop_id, exc)
            store.mark_state_corrupt(self.config.state_dir, loop_id, str(exc))
            return None

    def _tick_loop(self, loop_id: str) -> None:
        state = self._load(loop_id)
        if state is None or state.is_terminal():
            return

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
        elif state.status == "executing":
            transitioned = self._tick_executing(state)
        elif state.status == "awaiting_human":
            transitioned = self._tick_awaiting_human(state)
        elif state.status == "evaluating":
            transitioned = self._tick_evaluating(state)

        if transitioned:
            state.updated_at = _now_iso()
            store.save_state(self.config.state_dir, state)
            self._write_status(state)

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
            state.status = "cancelled"
            state.updated_at = _now_iso()
            store.save_state(self.config.state_dir, state)
            store.append_event(
                self.config.state_dir,
                state.loop_id,
                "cancelled",
                {},
                iteration=state.iteration,
            )
            self._close_thread(state)
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
            state.status = "failed"
            self._notify_fallback(state, f"Loop failed after {_MAX_CONSECUTIVE_ERRORS} errors")

    def _clear_errors(self, state: LoopState) -> None:
        state.consecutive_errors = 0

    def _format_with_persona(self, state: LoopState, instruction: str) -> str:
        persona_path = self.config.persona_dir / f"{state.persona}.md"
        persona = load_persona(persona_path)
        return persona.format_prompt(instruction, weight="heavy")

    def _ensure_persona(self, state: LoopState) -> bool:
        persona_path = self.config.persona_dir / f"{state.persona}.md"
        try:
            load_persona(persona_path)
            return True
        except Exception as exc:
            eid = _event_id(state.loop_id, "fallback", "persona")
            if not state.delivered_events.get(eid):
                self.human_channel.notify_fallback(
                    loop_id=state.loop_id,
                    text=f"Persona {state.persona!r} not found: {exc}",
                    event_id=eid,
                )
                state.delivered_events[eid] = True
            state.status = "failed"
            return False

    def _open_thread_if_needed(self, state: LoopState) -> HumanThreadRef | None:
        if state.thread is not None:
            return state.thread
        eid = _event_id(state.loop_id, "open_thread", "initial")
        if state.delivered_events.get(eid) and state.thread:
            return state.thread
        try:
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
                state.pending_question = PendingQuestion(question_id=question_id, text=text, kind=kind)
                self._clear_errors(state)
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
        store.append_event(
            self.config.state_dir,
            state.loop_id,
            "llm_call",
            {
                "input": trace.input,
                "output": {"raw": trace.raw_output, "parsed": trace.parsed},
                "reasoning": trace.reasoning,
                "config": trace.config,
                "metadata": trace.metadata,
                "uncertain_flag": trace.uncertain_flag,
                "error": trace.error,
            },
            iteration=state.iteration,
        )

    def _tick_clarifying(self, state: LoopState) -> bool:
        if not self._ensure_persona(state):
            return True
        if state.clarify_round >= self.config.max_clarify_rounds:
            self._record_error(state, "clarify_limit", {"rounds": state.clarify_round})
            return True

        instruction = prompts.build_clarify_instruction(
            state.body,
            round_num=state.clarify_round + 1,
            max_rounds=self.config.max_clarify_rounds,
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            resp, trace = prompts.run_clarify(
                prompt,
                engine=self.config.llm_engine,
                model=self.config.llm_model,
            )
            self._log_llm(state, trace)
            self._clear_errors(state)
        except Exception as exc:
            self._record_error(state, "llm_error", {"phase": "clarify", "error": str(exc)})
            return True

        if resp.clear:
            state.status = "decomposing"
            state.clarify_round = 0
            state.pending_question = None
        else:
            state.clarify_round += 1
            qid = f"clarify-{state.clarify_round}"
            if not self._ask(state, qid, resp.question or "", kind="clarify"):
                return False
            state.status = "awaiting_answer"
        return True

    def _tick_awaiting_answer(self, state: LoopState) -> bool:
        consumed = store.list_consumed_message_ids(self.config.state_dir, state.loop_id)
        pending = state.pending_question
        if pending is None:
            state.status = "clarifying"
            return True
        for msg in store.list_inbox_messages(self.config.state_dir, state.loop_id):
            if msg.message_id in consumed:
                continue
            if msg.kind != "answer" or msg.question_id != pending.question_id:
                continue
            store.consume_inbox_message(self.config.state_dir, state.loop_id, msg.filename)
            state.pending_question = None
            state.status = "clarifying"
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
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            resp, trace = prompts.run_decompose(
                prompt,
                engine=self.config.llm_engine,
                model=self.config.llm_model,
                max_subtasks=self.config.max_subtasks_per_iteration,
            )
            self._log_llm(state, trace)
            self._clear_errors(state)
        except Exception as exc:
            self._record_error(state, "llm_error", {"phase": "decompose", "error": str(exc)})
            return True

        state.subtasks = [
            Subtask(id=s.id, title=s.title, kind=s.kind, prompt=s.prompt)
            for s in resp.subtasks
        ]
        state.current_subtask_id = state.subtasks[0].id if state.subtasks else None
        state.status = "executing"
        state.next_focus = ""
        return True

    def _current_subtask(self, state: LoopState) -> Subtask | None:
        if state.current_subtask_id is None:
            return None
        for st in state.subtasks:
            if st.id == state.current_subtask_id:
                return st
        return None

    def _advance_subtask(self, state: LoopState) -> None:
        if state.current_subtask_id is None:
            return
        found = False
        for i, st in enumerate(state.subtasks):
            if st.id == state.current_subtask_id:
                if i + 1 < len(state.subtasks):
                    state.current_subtask_id = state.subtasks[i + 1].id
                else:
                    state.current_subtask_id = None
                found = True
                break
        if not found:
            state.current_subtask_id = None

    def _all_subtasks_done(self, state: LoopState) -> bool:
        return all(st.status in ("success", "failed") for st in state.subtasks)

    def _tick_executing(self, state: LoopState) -> bool:
        st = self._current_subtask(state)
        if st is None:
            if self._all_subtasks_done(state):
                state.status = "evaluating"
            return True

        if st.kind == "human":
            if st.status == "pending":
                qid = f"human-{state.iteration}-{st.id}"
                if not self._ask(state, qid, st.prompt, kind="human_subtask"):
                    return False
                st.status = "running"
                state.status = "awaiting_human"
            return True

        if st.status == "pending":
            key = f"loops:{state.loop_id}:i{state.iteration}:{st.id}"
            try:
                submission = self.executor.submit(prompt=st.prompt, idempotency_key=key)
                st.submission = submission
                st.status = "running"
                self._clear_errors(state)
            except Exception as exc:
                self._record_error(state, "executor_error", {"action": "submit", "error": str(exc)})
            return True

        if st.status == "running" and st.submission is not None:
            sub = st.submission
            poll = self.executor.poll(uuid=sub.uuid, result_filename=sub.result_filename)

            if poll.status == "pending":
                if self._subtask_timed_out(sub.submitted_at):
                    st.status = "failed"
                    st.result = f"timeout after {self.config.subtask_timeout_sec}s"
                    self._advance_subtask(state)
                return True

            if poll.status == "success":
                st.status = "success"
                st.result = poll.content
            else:
                st.status = "failed"
                st.result = poll.content

            self._advance_subtask(state)
            if self._all_subtasks_done(state):
                state.status = "evaluating"
            return True

        return False

    def _subtask_timed_out(self, submitted_at: str) -> bool:
        try:
            submitted = datetime.fromisoformat(submitted_at)
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=_TZ)
            elapsed = (datetime.now(_TZ) - submitted).total_seconds()
            return elapsed > self.config.subtask_timeout_sec
        except ValueError:
            return False

    def _tick_awaiting_human(self, state: LoopState) -> bool:
        st = self._current_subtask(state)
        if st is None:
            state.status = "executing"
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
            state.pending_question = None
            self._advance_subtask(state)
            state.status = "executing"
            if self._all_subtasks_done(state):
                state.status = "evaluating"
            return True
        return False

    def _tick_evaluating(self, state: LoopState) -> bool:
        if not self._ensure_persona(state):
            return True
        results_summary = "\n".join(
            f"- {st.id} ({st.status}): {st.result[:200]}" for st in state.subtasks
        )
        instruction = prompts.build_evaluate_instruction(
            state.body,
            results_summary=results_summary,
            iteration=state.iteration,
            max_iterations=state.max_iterations,
        )
        try:
            prompt = self._format_with_persona(state, instruction)
            resp, trace = prompts.run_evaluate(
                prompt,
                engine=self.config.llm_engine,
                model=self.config.llm_model,
            )
            self._log_llm(state, trace)
            self._clear_errors(state)
        except Exception as exc:
            self._record_error(state, "llm_error", {"phase": "evaluate", "error": str(exc)})
            return True

        if resp.achieved:
            state.status = "done"
            self._notify(state, f"Loop completed: {resp.summary}", "done")
            self._close_thread(state)
        elif state.iteration < state.max_iterations:
            state.iteration += 1
            state.subtasks = []
            state.current_subtask_id = None
            state.next_focus = resp.next_focus
            state.status = "decomposing"
        else:
            state.status = "failed"
            self._notify(state, f"Loop failed: {resp.summary}", "failed")
            self._close_thread(state)
        return True

    def _write_status(self, state: LoopState) -> None:
        write_status(
            self.config.status_dir,
            state,
            on_written=self.config.on_status_written,
        )
