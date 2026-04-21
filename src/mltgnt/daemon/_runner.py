import logging
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import DaemonComponent

logger = logging.getLogger("mltgnt.daemon")


class DaemonRunner:
    """コンポーネントの起動・停止とシグナルハンドリングを管理する。"""

    def __init__(
        self,
        *,
        pid_file: Path,
        components: list["DaemonComponent"],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        from ._pidlock import PidLock

        self._pid_lock = PidLock(pid_file)
        self._components = list(components)
        self._logger = logger or logging.getLogger("mltgnt.daemon")
        self._stop_event = threading.Event()

        # Register signal handlers on the main thread at construction time so
        # that tests can instantiate DaemonRunner from the main thread and then
        # call run() from a worker thread (signal.signal() requires main thread).
        if threading.current_thread() is threading.main_thread():
            def _signal_handler(signum, frame):
                self._logger.info("Signal %s received, shutting down.", signum)
                self._stop_event.set()

            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)

    def run(self) -> None:
        """
        メインエントリポイント。
        1. PIDロック取得（失敗時は SystemExit）
        2. SIGTERM/SIGINT ハンドラ登録（メインスレッドの場合）
        3. 全コンポーネントの start() 呼び出し（登録順）
        4. シグナル受信まで待機
        5. 全コンポーネントの stop() 呼び出し（逆順）
        6. PIDロック解放
        """
        if not self._pid_lock.acquire():
            self._logger.error("Another instance is already running.")
            raise SystemExit(1)

        if threading.current_thread() is threading.main_thread():
            def _signal_handler(signum, frame):
                self._logger.info("Signal %s received, shutting down.", signum)
                self._stop_event.set()

            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)

        started: list["DaemonComponent"] = []
        try:
            for component in self._components:
                self._logger.info("Starting component: %s", component.name)
                component.start()
                started.append(component)
        except Exception:
            self._logger.exception("Component start failed, rolling back.")
            for comp in reversed(started):
                try:
                    comp.stop()
                except Exception:
                    self._logger.exception("Error stopping component %s during rollback.", comp.name)
            self._pid_lock.release()
            raise

        try:
            self._stop_event.wait()
        finally:
            self._logger.info("Stopping components.")
            for comp in reversed(self._components):
                try:
                    comp.stop()
                except Exception:
                    self._logger.exception("Error stopping component %s.", comp.name)
            self._pid_lock.release()

    def stop(self) -> None:
        """外部から停止を要求する（テスト用）。"""
        self._stop_event.set()
