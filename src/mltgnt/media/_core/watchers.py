"""Job completion detection, startup catch-up, delivery reconciliation and progress polling.

Directories come from ``MediaConfig`` (``pending_dir`` / ``events_dir``); where a
job's done marker lives is the host's business, so it is passed in as ``is_done``.

Pending record keys used here: ``state``, ``space``, ``thread`` and
``progress_message_id`` (written back after the first progress post).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from mltgnt.interfaces.media import MediaClient
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.pending import PendingStore
from mltgnt.media._core.plan_gate import AWAITING_STATE, expire_pending
from mltgnt.media._core.progress import ProgressState

__all__ = [
    "CATCHUP_STATES",
    "DeliveryReconciler",
    "ExecDoneHandler",
    "ProgressWatcher",
    "catchup_pending_on_startup",
    "iter_pending_with_done",
]

_log = logging.getLogger(__name__)

_UUID_RE = re.compile(r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}")
CATCHUP_STATES = frozenset({"running", AWAITING_STATE})

Deliver = Callable[[str], None]
IsDone = Callable[[str], bool]


def _iter_pending(store: PendingStore, config: MediaConfig, prefix: str) -> list[tuple[str, dict[str, Any]]]:
    if not config.pending_dir.is_dir():
        return []
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(config.pending_dir.glob(f"{prefix}*.json")):
        uid = path.stem.removeprefix(prefix)
        if not _UUID_RE.fullmatch(uid):
            continue
        pending = store.load(uid)
        if pending is not None:
            found.append((uid, pending))
    return found


def iter_pending_with_done(
    config: MediaConfig,
    *,
    is_done: IsDone,
    states: frozenset[str] = CATCHUP_STATES,
    prefix: str = "pending-",
) -> list[tuple[str, dict[str, Any]]]:
    """``(uid, pending)`` whose state is in ``states`` and whose job is done."""
    store = PendingStore.from_config(config, prefix=prefix)
    return [
        (uid, pending)
        for uid, pending in _iter_pending(store, config, prefix)
        if pending.get("state") in states and is_done(uid)
    ]


def catchup_pending_on_startup(
    config: MediaConfig,
    *,
    deliver: Deliver,
    is_done: IsDone,
    prefix: str = "pending-",
) -> list[str]:
    """Deliver once every job that finished while the host was down. Return the uids tried."""
    attempted: list[str] = []
    for uid, _pending in iter_pending_with_done(config, is_done=is_done, prefix=prefix):
        try:
            deliver(uid)
        except Exception:
            _log.exception("[catchup] uid=%s status=error", uid)
        attempted.append(uid)
    return attempted


class _PollingThread:
    """``poll_once`` every ``interval_sec`` on a daemon thread until ``stop``."""

    _thread_name = "MediaPoller"

    def __init__(self, interval_sec: float) -> None:
        self._interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=self._thread_name)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_sec):
            try:
                self.poll_once()
            except Exception:
                _log.exception("[%s] poll failed", self._thread_name)

    def poll_once(self) -> None:
        raise NotImplementedError


class DeliveryReconciler(_PollingThread):
    """Retry delivery of done jobs and report ones that stay undelivered.

    ``on_stale(uid, elapsed_sec)`` fires once per uid whose ``running`` record keeps
    existing for ``stale_threshold_sec`` after its job is done.
    """

    _thread_name = "DeliveryReconciler"

    def __init__(
        self,
        config: MediaConfig,
        *,
        deliver: Deliver,
        is_done: IsDone,
        on_stale: Callable[[str, float], None] | None = None,
        interval_sec: float = 30.0,
        stale_threshold_sec: float = 120.0,
        prefix: str = "pending-",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(interval_sec)
        self._config = config
        self._deliver = deliver
        self._is_done = is_done
        self._on_stale = on_stale
        self._stale_threshold_sec = stale_threshold_sec
        self._store = PendingStore.from_config(config, prefix=prefix)
        self._prefix = prefix
        self._clock = clock
        self._stale_since: dict[str, float] = {}
        self._stale_warned: set[str] = set()

    def _forget(self, uid: str) -> None:
        self._stale_since.pop(uid, None)
        self._stale_warned.discard(uid)

    def poll_once(self) -> None:
        now = self._clock()
        seen: set[str] = set()
        for uid, pending in iter_pending_with_done(self._config, is_done=self._is_done, prefix=self._prefix):
            seen.add(uid)
            try:
                self._deliver(uid)
            except Exception:
                _log.exception("[DeliveryReconciler] deliver failed uid=%s", uid)
            if pending.get("state") != "running" or self._store.load(uid) is None:
                self._forget(uid)
                continue
            since = self._stale_since.setdefault(uid, now)
            elapsed = now - since
            if elapsed >= self._stale_threshold_sec and uid not in self._stale_warned:
                self._stale_warned.add(uid)
                if self._on_stale is not None:
                    self._on_stale(uid, elapsed)
        for uid in list(self._stale_since):
            if uid not in seen:
                self._forget(uid)


class ExecDoneHandler:
    """watchdog-compatible handler: a created file named ``<uuid>`` triggers ``deliver(uuid)``."""

    def __init__(self, deliver: Deliver) -> None:
        self._deliver = deliver

    def dispatch(self, event: Any) -> None:
        if getattr(event, "event_type", None) == "created":
            self.on_created(event)

    def on_created(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        uid = str(getattr(event, "src_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if _UUID_RE.fullmatch(uid):
            self._deliver(uid)


class ProgressWatcher(_PollingThread):
    """Tail ``<events_dir>/<uid>.jsonl`` for running jobs and rewrite their progress message.

    Records in ``AWAITING_STATE`` are checked for plan expiry instead;
    ``on_plan_expired(uid, claimed)`` receives the consumed record.
    """

    _thread_name = "ProgressWatcher"

    def __init__(
        self,
        config: MediaConfig,
        client: MediaClient,
        *,
        is_done: IsDone,
        on_plan_expired: Callable[[str, dict[str, Any]], None] | None = None,
        interval_sec: float = 1.0,
        max_lines: int = 1,
        prefix: str = "pending-",
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(interval_sec)
        self._config = config
        self._client = client
        self._is_done = is_done
        self._on_plan_expired = on_plan_expired
        self._max_lines = max_lines
        self._store = PendingStore.from_config(config, prefix=prefix)
        self._prefix = prefix
        self._clock = clock
        self._wall_clock = wall_clock
        self._trackers: dict[str, ProgressState] = {}

    def poll_once(self) -> None:
        seen: set[str] = set()
        for uid, pending in _iter_pending(self._store, self._config, self._prefix):
            seen.add(uid)
            if pending.get("state") == AWAITING_STATE:
                self._check_plan_expiry(uid, pending)
                continue
            if self._is_done(uid):
                self._trackers.pop(uid, None)
                continue
            events_path = self._config.events_dir / f"{uid}.jsonl"
            if not events_path.is_file():
                continue
            tracker = self._trackers.get(uid)
            if tracker is None:
                tracker = ProgressState.from_config(self._config, max_lines=self._max_lines, clock=self._clock)
                self._trackers[uid] = tracker
            tracker.read_new_lines(events_path)
            self._publish(uid, pending, tracker)
        for uid in list(self._trackers):
            if uid not in seen:
                self._trackers.pop(uid, None)

    def _check_plan_expiry(self, uid: str, pending: dict[str, Any]) -> None:
        claimed = expire_pending(self._store, uid, pending, now=self._wall_clock())
        if claimed is not None and self._on_plan_expired is not None:
            self._on_plan_expired(uid, claimed)

    def _publish(self, uid: str, pending: dict[str, Any], tracker: ProgressState) -> None:
        message_id = pending.get("progress_message_id")
        if message_id:
            tracker.maybe_update(self._client, str(message_id))
            return
        space = pending.get("space")
        if not space or not tracker.is_due():
            return
        text = tracker.render_text()
        thread = pending.get("thread")
        posted = self._client.post(text, str(space), str(thread) if thread else None)
        if not posted:
            _log.warning("[ProgressWatcher] progress post failed uid=%s", uid)
            return
        tracker.mark_sent(text)
        pending["progress_message_id"] = posted
        self._store.save(uid, pending)
