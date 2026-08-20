"""mltgnt.loops.component — Objective snapshot watcher と DaemonComponent。"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from mltgnt.config import LoopsConfig
from mltgnt.interfaces.loops import HumanChannel, SubtaskExecutor
from mltgnt.loops.engine import LoopsEngine
from mltgnt.loops.objective import Objective, ObjectiveError, list_objective_files, parse_objective
from mltgnt.loops import store
from mltgnt.loops.models import TERMINAL_STATUSES
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
    """objectives_dir をポーリングし LoopsEngine で各 loop を 1 tick 1 遷移する。"""

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
                self._engine.tick()
            except Exception:
                logger.exception("LoopsComponent watch loop error")

    def _refresh_objectives(self, *, initial: bool) -> None:
        paths = list_objective_files(self._config.objectives_dir)
        current = _objective_snapshot(paths)
        if not initial and current == self._snapshot:
            return
        self._snapshot = current

        new_objectives: dict[str, Objective] = {}
        new_errors: dict[str, ObjectiveError] = {}

        parsed: list[Objective | ObjectiveError] = [
            parse_objective(
                path,
                default_persona=self._config.default_persona,
                default_max_iterations=self._config.max_iterations,
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
                self._write_error_status(result)
                continue
            if id_counts[result.loop_id] > 1:
                err = ObjectiveError(
                    loop_id=result.loop_id,
                    message=f"duplicate id: {result.loop_id!r}",
                    path=result.path,
                )
                new_errors[result.loop_id] = err
                self._write_error_status(err)
                continue
            new_objectives[result.loop_id] = result

        for loop_id, obj in new_objectives.items():
            if loop_id not in self._objectives:
                self._maybe_start(obj)

        removed = set(self._objectives.keys()) - set(new_objectives.keys())
        for loop_id in removed:
            state = store.load_state(self._config.state_dir, loop_id)
            if state and not state.is_terminal():
                self._engine._check_cancel(state)

        self._objectives = new_objectives
        self._errors = new_errors

    def _maybe_start(self, objective: Objective) -> None:
        existing = store.load_state(self._config.state_dir, objective.loop_id)
        if existing is not None:
            if existing.is_terminal():
                return
            return
        if objective.status == "cancelled":
            return
        try:
            self._engine.start_loop(objective)
        except Exception:
            logger.exception("failed to start loop %s", objective.loop_id)

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
