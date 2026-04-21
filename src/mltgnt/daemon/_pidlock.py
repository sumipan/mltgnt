import os
from pathlib import Path


class PidLock:
    """PIDファイルによる単一インスタンス制御。"""

    def __init__(self, pid_file: Path) -> None:
        self._pid_file = pid_file

    def acquire(self) -> bool:
        """ロック取得。成功時 True、既存プロセス稼働中なら False。"""
        if self._pid_file.exists():
            try:
                pid = int(self._pid_file.read_text().strip())
                os.kill(pid, 0)
                # Process is alive
                return False
            except (ValueError, ProcessLookupError, PermissionError):
                # stale PID or unreadable -- fall through to overwrite
                pass
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(os.getpid()))
        return True

    def release(self) -> None:
        """PIDファイルを削除する。"""
        try:
            self._pid_file.unlink()
        except FileNotFoundError:
            pass
