import os
import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from mltgnt.daemon import DaemonComponent, DaemonRunner, PidLock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockComponent:
    def __init__(self, name: str, start_side_effect=None, stop_side_effect=None):
        self._name = name
        self.mock = MagicMock()
        if start_side_effect:
            self.mock.start.side_effect = start_side_effect
        if stop_side_effect:
            self.mock.stop.side_effect = stop_side_effect

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self.mock.start()

    def stop(self) -> None:
        self.mock.stop()


# ---------------------------------------------------------------------------
# PidLock tests
# ---------------------------------------------------------------------------

def test_pidlock_acquire_and_release(tmp_path):
    pid_file = tmp_path / "test.pid"
    lock = PidLock(pid_file)

    assert lock.acquire() is True
    assert pid_file.exists()
    assert int(pid_file.read_text()) == os.getpid()

    lock.release()
    assert not pid_file.exists()


def test_pidlock_stale_pid(tmp_path):
    pid_file = tmp_path / "test.pid"
    # Write a PID that almost certainly doesn't exist
    pid_file.write_text("99999999")

    lock = PidLock(pid_file)
    assert lock.acquire() is True
    assert int(pid_file.read_text()) == os.getpid()
    lock.release()


def test_pidlock_conflict(tmp_path):
    pid_file = tmp_path / "test.pid"
    # Write current process's PID -- it's alive
    pid_file.write_text(str(os.getpid()))

    lock = PidLock(pid_file)
    assert lock.acquire() is False


# ---------------------------------------------------------------------------
# DaemonRunner tests
# ---------------------------------------------------------------------------

def _run_and_stop(runner: DaemonRunner, delay: float = 0.05):
    """Run runner in a background thread and stop it after delay."""
    def _target():
        runner.run()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    time.sleep(delay)
    runner.stop()
    t.join(timeout=5)
    assert not t.is_alive(), "runner did not stop in time"


def test_runner_start_stop_order(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    comp_a = MockComponent("a")
    comp_b = MockComponent("b")

    runner = DaemonRunner(pid_file=pid_file, components=[comp_a, comp_b])
    call_order = []
    comp_a.mock.start.side_effect = lambda: call_order.append("a.start")
    comp_b.mock.start.side_effect = lambda: call_order.append("b.start")
    comp_a.mock.stop.side_effect = lambda: call_order.append("a.stop")
    comp_b.mock.stop.side_effect = lambda: call_order.append("b.stop")

    _run_and_stop(runner)

    assert call_order == ["a.start", "b.start", "b.stop", "a.stop"]


def test_runner_sigterm(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    comp = MockComponent("c")

    runner = DaemonRunner(pid_file=pid_file, components=[comp])

    def _send_signal():
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    t_signal = threading.Thread(target=_send_signal, daemon=True)
    t_run = threading.Thread(target=runner.run, daemon=True)

    t_run.start()
    t_signal.start()
    t_run.join(timeout=5)

    assert not t_run.is_alive()
    comp.mock.stop.assert_called_once()
    assert not pid_file.exists()


def test_runner_component_start_failure(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    comp_a = MockComponent("a")
    comp_b = MockComponent("b", start_side_effect=RuntimeError("boom"))

    runner = DaemonRunner(pid_file=pid_file, components=[comp_a, comp_b])

    with pytest.raises(RuntimeError, match="boom"):
        runner.run()

    comp_a.mock.start.assert_called_once()
    comp_a.mock.stop.assert_called_once()
    comp_b.mock.stop.assert_not_called()
    assert not pid_file.exists()


def test_runner_stop_exception_does_not_prevent_others(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    comp_a = MockComponent("a", stop_side_effect=RuntimeError("stop error"))
    comp_b = MockComponent("b")

    runner = DaemonRunner(pid_file=pid_file, components=[comp_a, comp_b])
    _run_and_stop(runner)

    comp_b.mock.stop.assert_called_once()
    assert not pid_file.exists()


def test_daemon_component_protocol():
    comp = MockComponent("x")
    assert isinstance(comp, DaemonComponent)
