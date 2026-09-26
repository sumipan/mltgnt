"""mltgnt.media._core.component (#4030)."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from mltgnt.media._core.component import MediaBridgeComponent


def _exit_three() -> None:
    sys.exit(3)


def test_three_quick_exits_restart_in_backoff_order() -> None:
    started: list[float] = []
    codes: list[int] = []
    codes_iter = iter([2, 5, 7])

    def start() -> None:
        started.append(time.monotonic())
        raise SystemExit(next(codes_iter))

    component = MediaBridgeComponent(start, mode="thread", backoff_sec=(0.05, 0.15), on_exit=codes.append)
    component.run()
    assert codes == [2, 5, 7]
    assert len(started) == 3
    gaps = [b - a for a, b in zip(started, started[1:])]
    assert gaps[0] >= 0.05
    assert gaps[1] >= 0.15


def test_thread_mode_exit_codes() -> None:
    outcomes = iter([None, "boom", RuntimeError("x")])
    codes: list[int] = []

    def start() -> None:
        outcome = next(outcomes)
        if outcome is None:
            return
        if isinstance(outcome, str):
            raise SystemExit(outcome)
        raise outcome

    MediaBridgeComponent(start, mode="thread", backoff_sec=(0, 0), on_exit=codes.append).run()
    assert codes == [0, 1, 1]


def test_process_mode_reports_child_exit_code() -> None:
    codes: list[int] = []
    MediaBridgeComponent(_exit_three, mode="process", backoff_sec=(), on_exit=codes.append).run()
    assert codes == [3]


def test_stop_during_backoff_prevents_restart() -> None:
    calls: list[int] = []
    exited = threading.Event()

    def on_exit(code: int) -> None:
        calls.append(code)
        exited.set()

    component = MediaBridgeComponent(lambda: None, mode="thread", backoff_sec=(30.0,), on_exit=on_exit)
    component.start()
    assert exited.wait(5)
    component.stop(timeout=5)
    component.join(timeout=5)
    assert calls == [0]


def test_on_exit_failure_does_not_stop_supervision() -> None:
    runs: list[int] = []

    def on_exit(code: int) -> None:
        raise ValueError("audit failed")

    MediaBridgeComponent(lambda: runs.append(1), mode="thread", backoff_sec=(0,), on_exit=on_exit).run()
    assert runs == [1, 1]


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        MediaBridgeComponent(lambda: None, mode="fork", backoff_sec=())  # type: ignore[arg-type]
    assert MediaBridgeComponent(lambda: None, mode="thread", backoff_sec=()).name == "media_bridge"
