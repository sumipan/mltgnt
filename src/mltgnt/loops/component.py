"""mltgnt.loops.component — Objective snapshot watcher と DaemonComponent。"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from mltgnt.config import LoopsConfig
from mltgnt.interfaces.loops import HumanChannel, HumanThreadRef, SubtaskExecutor
from mltgnt.loops.engine import LoopsEngine
from mltgnt.loops.objective import (
    Objective,
    ObjectiveError,
    ensure_frontmatter,
    list_objective_files,
    parse_objective,
)
from mltgnt.loops import store
from mltgnt.loops.requests import RequestError, StartRequest, consume_request, list_requests
from mltgnt.loops.status import write_status

logger = logging.getLogger("mltgnt.loops.component")


def _objective_snapshot(paths: list[Path]) -> dict[str, float]:
    snap: dict[str, float] = {}
    for p in paths:
        try:
            snap[str(p)] = p.stat().st_mtime
        except OSError:
            pass
    return snap


class LoopsComponent:
    """objectives_dir をポーリングし、request 経由で LoopsEngine を駆動する。"""

    def __init__(
        self,
        config: LoopsConfig,
        human_channel: HumanChannel,
        executor: SubtaskExecutor,
    ) -> None:
        self._config = config
        self._engine = LoopsEngine(
            config=config,
            human_channel=human_channel,
            executor=executor,
            objective_exists=self._objective_exists,
            objective_cancelled=self._objective_cancelled,
            objective_hash_changed=self._objective_hash_changed,
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot: dict[str, float] = {}
        self._objectives: dict[str, Objective] = {}
        self._errors: dict[str, ObjectiveError] = {}
        self._objectives_by_path: dict[str, Objective] = {}
        self._errors_by_path: dict[str, ObjectiveError] = {}

    @property
    def name(self) -> str:
        return "loops"

    @property
    def engine(self) -> LoopsEngine:
        return self._engine

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._refresh_objectives(initial=True)
        self._restore_states()
        self._thread = threading.Thread(target=self._watch_loop, name="loops-watcher", daemon=True)
        self._thread.start()
        logger.info("LoopsComponent: started (interval=%.1fs)", self._config.poll_interval_sec)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._config.poll_interval_sec + 2)
            self._thread = None
        logger.info("LoopsComponent: stopped")

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(self._config.poll_interval_sec):
            try:
                self._refresh_objectives(initial=False)
                self._process_requests()
                self._engine.tick()
            except Exception:
                logger.exception("LoopsComponent watch loop error")

    def _refresh_objectives(self, *, initial: bool) -> None:
        paths = list_objective_files(self._config.objectives_dir)
        current = _objective_snapshot(paths)
        if not initial and current == self._snapshot:
            return

        changed: set[str] = set(current)
        if not initial:
            changed = {p for p, m in current.items() if self._snapshot.get(p) != m}
            changed |= {p for p in self._snapshot if p not in current}

        for path in paths:
            if initial or str(path) in changed:
                ensure_frontmatter(
                    path, default_max_iterations=self._config.max_iterations
                )

        self._snapshot = _objective_snapshot(paths)

        new_objectives: dict[str, Objective] = {}
        new_errors: dict[str, ObjectiveError] = {}
        new_by_path: dict[str, Objective] = {}
        new_errors_by_path: dict[str, ObjectiveError] = {}

        parsed: list[Objective | ObjectiveError] = [
            parse_objective(
                path,
                default_persona=self._config.default_persona,
                default_max_iterations=self._config.max_iterations,
                plan_approval_default=self._config.plan_approval_default,
            )
            for path in paths
        ]
        id_counts: dict[str, int] = {}
        for result in parsed:
            if isinstance(result, Objective):
                id_counts[result.loop_id] = id_counts.get(result.loop_id, 0) + 1

        for result in parsed:
            if isinstance(result, ObjectiveError):
                new_errors[result.loop_id] = result
                new_errors_by_path[result.path.name] = result
                self._write_error_status(result)
                continue
            if id_counts[result.loop_id] > 1:
                err = ObjectiveError(
                    loop_id=result.loop_id,
                    message=f"duplicate id: {result.loop_id!r}",
                    path=result.path,
                )
                new_errors[result.loop_id] = err
                new_errors_by_path[result.path.name] = err
                self._write_error_status(err)
                continue
            new_objectives[result.loop_id] = result
            new_by_path[result.path.name] = result

        removed = set(self._objectives.keys()) - set(new_objectives.keys())
        self._objectives = new_objectives
        self._errors = new_errors
        self._objectives_by_path = new_by_path
        self._errors_by_path = new_errors_by_path

        for loop_id in removed:
            state = store.load_state(self._config.state_dir, loop_id)
            if state and not state.is_terminal():
                self._engine._check_cancel(state)

    def _process_requests(self) -> None:
        requests, errors = list_requests(
            self._config.state_dir, self._config.objectives_dir
        )
        for err in errors:
            self._notify_request_error(err)
        for req in requests:
            self._handle_start_request(req)

    def _notify_request_error(self, err: RequestError) -> None:
        event_id = f"loops:request:{err.filename}:error"
        text = f"Invalid start request ({err.filename}): {err.message}"
        if err.channel_id and err.thread_ts:
            try:
                self._engine.human_channel.notify(
                    loop_id=err.filename,
                    persona=err.persona or self._config.default_persona,
                    thread=HumanThreadRef(err.channel_id, err.thread_ts),
                    text=text,
                    event_id=event_id,
                )
            except Exception:
                logger.exception("failed to notify request error for %s", err.filename)
            return
        try:
            self._engine.human_channel.notify_fallback(
                loop_id=err.filename,
                text=text,
                event_id=event_id,
            )
        except Exception:
            logger.exception("failed to fallback-notify request error for %s", err.filename)

    def _notify_request(
        self, req: StartRequest, *, text: str, kind: str, loop_id: str
    ) -> None:
        event_id = f"loops:request:{req.filename}:{kind}"
        self._engine.human_channel.notify(
            loop_id=loop_id,
            persona=req.persona or self._config.default_persona,
            thread=HumanThreadRef(req.channel_id, req.thread_ts),
            text=text,
            event_id=event_id,
        )

    def _handle_start_request(self, req: StartRequest) -> None:
        try:
            err = self._errors_by_path.get(req.objective_path)
            obj = self._objectives_by_path.get(req.objective_path)
            if err is not None:
                self._notify_request(
                    req,
                    text=f"Cannot start objective {req.objective_path}: {err.message}",
                    kind="error",
                    loop_id=err.loop_id,
                )
                consume_request(self._config.state_dir, req.filename)
                return
            if obj is None:
                self._notify_request(
                    req,
                    text=f"Cannot start objective {req.objective_path}: not found",
                    kind="error",
                    loop_id=Path(req.objective_path).stem,
                )
                consume_request(self._config.state_dir, req.filename)
                return
            if obj.status == "cancelled":
                self._notify_request(
                    req,
                    text=f"Cannot start objective {req.objective_path}: status is cancelled",
                    kind="error",
                    loop_id=obj.loop_id,
                )
                consume_request(self._config.state_dir, req.filename)
                return

            existing = store.load_state(self._config.state_dir, obj.loop_id)
            if existing is not None and not existing.is_terminal():
                self._notify_request(
                    req,
                    text=f"Loop {obj.loop_id} is already running",
                    kind="already_running",
                    loop_id=obj.loop_id,
                )
                consume_request(self._config.state_dir, req.filename)
                return
            if existing is not None and existing.is_terminal():
                store.archive_terminal_state(self._config.state_dir, obj.loop_id)

            parsed = parse_objective(
                obj.path,
                default_persona=req.persona or self._config.default_persona,
                default_max_iterations=self._config.max_iterations,
                plan_approval_default=self._config.plan_approval_default,
            )
            if isinstance(parsed, ObjectiveError):
                self._notify_request(
                    req,
                    text=f"Cannot start objective {req.objective_path}: {parsed.message}",
                    kind="error",
                    loop_id=parsed.loop_id,
                )
                consume_request(self._config.state_dir, req.filename)
                return

            self._engine.start_loop(
                parsed,
                thread=HumanThreadRef(req.channel_id, req.thread_ts),
            )
            consume_request(self._config.state_dir, req.filename)
        except Exception:
            logger.exception("failed to process start request %s", req.filename)

    def _restore_states(self) -> None:
        for loop_id in store.list_restorable_loops(self._config.state_dir):
            try:
                state = store.load_state(self._config.state_dir, loop_id)
            except ValueError as exc:
                store.mark_state_corrupt(self._config.state_dir, loop_id, str(exc))
                try:
                    self._engine.human_channel.notify_fallback(
                        loop_id=loop_id,
                        text=f"Corrupt loop state isolated: {exc}",
                        event_id=f"loops:{loop_id}:fallback:corrupt-state",
                    )
                except Exception:
                    logger.exception("failed to notify corrupt state for %s", loop_id)
                continue
            if state and not state.is_terminal():
                logger.info("restored non-terminal loop: %s", loop_id)

    def _objective_exists(self, loop_id: str) -> bool:
        return loop_id in self._objectives

    def _objective_cancelled(self, loop_id: str) -> bool:
        obj = self._objectives.get(loop_id)
        return obj is not None and obj.status == "cancelled"

    def _objective_hash_changed(self, loop_id: str, stored_hash: str) -> bool:
        obj = self._objectives.get(loop_id)
        if obj is None:
            return False
        return obj.content_hash != stored_hash

    def _write_error_status(self, err: ObjectiveError) -> None:
        from mltgnt.loops.models import LoopState

        state = LoopState(
            loop_id=err.loop_id,
            objective_path=str(err.path),
            objective_hash="",
            title=err.loop_id,
            body=err.message,
            status="failed",
            iteration=0,
            max_iterations=0,
            persona=self._config.default_persona,
        )
        write_status(
            self._config.status_dir,
            state,
            on_written=self._config.on_status_written,
        )
