"""mltgnt.media._core.watchers (#4030)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mltgnt.interfaces.media import Status
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.pending import PendingStore
from mltgnt.media._core.plan_gate import AWAITING_STATE, PENDING_KEY, PlanGate
from mltgnt.media._core.watchers import (
    DeliveryReconciler,
    ExecDoneHandler,
    ProgressWatcher,
    catchup_pending_on_startup,
    iter_pending_with_done,
)

U1 = "11111111-1111-1111-1111-111111111111"
U2 = "22222222-2222-2222-2222-222222222222"
U3 = "33333333-3333-3333-3333-333333333333"


class FakeClient:
    def __init__(self, post_result: str | None = "p1") -> None:
        self.posts: list[tuple[str, str, str | None]] = []
        self.updates: list[tuple[str, str]] = []
        self.post_result = post_result

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        self.posts.append((text, space, thread))
        return self.post_result

    def update(self, message_id: str, text: str) -> bool:
        self.updates.append((message_id, text))
        return True

    def set_status(self, message_id: str, status: Status) -> bool:
        return True


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeFsEvent:
    src_path: str
    event_type: str = "created"
    is_directory: bool = False


def _config(tmp_path: Path, interval: float = 0.0) -> MediaConfig:
    return MediaConfig(
        state_dir=tmp_path / "state",
        pending_dir=tmp_path / "pending",
        events_dir=tmp_path / "events",
        progress_min_interval_sec=interval,
    )


def _progress_line(text: str) -> str:
    return json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": f"[progress] {text}"}]}})


def test_iter_pending_with_done_filters(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert iter_pending_with_done(config, is_done=lambda uid: True) == []
    store = PendingStore.from_config(config)
    store.save(U1, {"state": "running"})
    store.save(U2, {"state": "delivered"})
    store.save(U3, {"state": AWAITING_STATE})
    store.save("not-a-uuid", {"state": "running"})
    (config.pending_dir / f"pending-{'4' * 8}-4444-4444-4444-{'4' * 12}.json").write_text("{bad", encoding="utf-8")
    done = {U1, U2}
    found = iter_pending_with_done(config, is_done=done.__contains__)
    assert [uid for uid, _ in found] == [U1]


def test_catchup_delivers_done_jobs_and_survives_errors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = PendingStore.from_config(config)
    store.save(U1, {"state": "running"})
    store.save(U2, {"state": AWAITING_STATE})
    store.save(U3, {"state": "running"})
    delivered: list[str] = []

    def deliver(uid: str) -> None:
        delivered.append(uid)
        if uid == U1:
            raise RuntimeError("post failed")

    attempted = catchup_pending_on_startup(config, deliver=deliver, is_done=lambda uid: uid != U3)
    assert attempted == [U1, U2]
    assert delivered == [U1, U2]


def test_exec_done_handler_delivers_uuid_files() -> None:
    delivered: list[str] = []
    handler = ExecDoneHandler(delivered.append)
    handler.dispatch(FakeFsEvent(f"/x/done/{U1}"))
    handler.dispatch(FakeFsEvent(f"/x/done/{U2}", event_type="modified"))
    handler.dispatch(FakeFsEvent(f"/x/done/{U3}", is_directory=True))
    handler.dispatch(FakeFsEvent("/x/done/readme.txt"))
    assert delivered == [U1]


def test_reconciler_redelivers_and_warns_stale_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = PendingStore.from_config(config)
    store.save(U1, {"state": "running"})
    store.save(U2, {"state": "running"})
    store.save(U3, {"state": AWAITING_STATE})
    clock = FakeClock()
    stale: list[tuple[str, float]] = []
    delivered: list[str] = []

    def deliver(uid: str) -> None:
        delivered.append(uid)
        if uid == U2:
            store.delete(uid)
        if uid == U3:
            raise RuntimeError("x")

    reconciler = DeliveryReconciler(
        config,
        deliver=deliver,
        is_done=lambda uid: True,
        on_stale=lambda uid, sec: stale.append((uid, sec)),
        stale_threshold_sec=120.0,
        clock=clock,
    )
    reconciler.poll_once()
    assert delivered == [U1, U2, U3]
    clock.now = 130.0
    reconciler.poll_once()
    clock.now = 300.0
    reconciler.poll_once()
    assert stale == [(U1, 130.0)]
    store.delete(U1)
    reconciler.poll_once()
    store.save(U1, {"state": "running"})
    reconciler.poll_once()
    clock.now = 500.0
    reconciler.poll_once()
    assert stale == [(U1, 130.0), (U1, 200.0)]


def test_reconciler_thread_start_stop(tmp_path: Path) -> None:
    reconciler = DeliveryReconciler(_config(tmp_path), deliver=lambda uid: None, is_done=lambda uid: True)
    reconciler.start()
    reconciler.stop(timeout=5)


def test_progress_watcher_posts_then_updates(tmp_path: Path) -> None:
    config = _config(tmp_path, interval=3.0)
    store = PendingStore.from_config(config)
    store.save(U1, {"state": "running", "space": "C", "thread": "T"})
    config.events_dir.mkdir(parents=True)
    events = config.events_dir / f"{U1}.jsonl"
    events.write_text("\n".join(_progress_line(f"s{i}") for i in range(5)) + "\n", encoding="utf-8")
    clock = FakeClock()
    client = FakeClient()
    watcher = ProgressWatcher(config, client, is_done=lambda uid: False, clock=clock)
    watcher.poll_once()
    assert client.posts == []
    clock.now = 3.0
    watcher.poll_once()
    assert client.posts == [("s4", "C", "T")]
    assert store.load(U1)["progress_message_id"] == "p1"
    with events.open("a", encoding="utf-8") as f:
        f.write(_progress_line("s5") + "\n")
    clock.now = 6.0
    watcher.poll_once()
    assert client.updates == [("p1", "s5")]


def test_progress_watcher_skips_done_missing_and_failed_post(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = PendingStore.from_config(config)
    store.save(U1, {"state": "running", "space": "C"})
    store.save(U2, {"state": "running", "space": "C"})
    store.save(U3, {"state": "running"})
    config.events_dir.mkdir(parents=True)
    for uid in (U1, U3):
        (config.events_dir / f"{uid}.jsonl").write_text(_progress_line("x") + "\n", encoding="utf-8")
    client = FakeClient(post_result=None)
    done: set[str] = set()
    watcher = ProgressWatcher(config, client, is_done=done.__contains__)
    watcher.poll_once()
    assert client.posts == [("x", "C", None)]
    assert "progress_message_id" not in store.load(U1)
    done.add(U1)
    watcher.poll_once()
    assert len(client.posts) == 1
    store.delete(U3)
    watcher.poll_once()


def test_progress_watcher_expires_awaiting_plan(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = PendingStore.from_config(config)
    store.save(U1, {"state": AWAITING_STATE, PENDING_KEY: PlanGate(expires_at=50.0).to_dict()})
    store.save(U2, {"state": AWAITING_STATE, PENDING_KEY: PlanGate(expires_at=500.0).to_dict()})
    expired: list[tuple[str, dict]] = []
    wall = FakeClock(100.0)
    watcher = ProgressWatcher(
        config,
        FakeClient(),
        is_done=lambda uid: False,
        on_plan_expired=lambda uid, claimed: expired.append((uid, claimed)),
        wall_clock=wall,
    )
    watcher.poll_once()
    assert [uid for uid, _ in expired] == [U1]
    assert expired[0][1][PENDING_KEY]["state"] == "expired"
    assert store.load(U1) is None
    assert store.load(U2) is not None
