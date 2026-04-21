from typing import Protocol, runtime_checkable

from ._pidlock import PidLock
from ._runner import DaemonRunner


@runtime_checkable
class DaemonComponent(Protocol):
    """デーモンに登録できるコンポーネントの型契約。"""

    @property
    def name(self) -> str: ...

    def start(self) -> None:
        """コンポーネントを起動する。ブロックしない。"""
        ...

    def stop(self) -> None:
        """コンポーネントを停止する。リソース解放を含む。"""
        ...


__all__ = ["DaemonComponent", "DaemonRunner", "PidLock"]
