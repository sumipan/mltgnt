"""
mltgnt.daemon._skill_watcher — skills/ ディレクトリの変更を監視してホットリロードする DaemonComponent。

設計方針:
- 外部ライブラリ不要: os.stat() によるポーリング（デフォルト 5 秒）
- SkillRegistry.reload() を変更検知時に呼ぶだけ
- DaemonComponent プロトコル準拠（start/stop/name）
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.skill._registry import SkillRegistry

logger = logging.getLogger("mltgnt.daemon.skill_watcher")


def _collect_snapshot(paths: list[Path], entry_file: str) -> dict[str, float]:
    """監視対象パス配下の SKILL.md ファイルの {path_str: mtime} を返す。"""
    snapshot: dict[str, float] = {}
    for base in paths:
        if not base.exists():
            continue
        for skill_file in base.rglob(entry_file):
            try:
                snapshot[str(skill_file)] = skill_file.stat().st_mtime
            except OSError:
                pass
    return snapshot


class SkillWatcherComponent:
    """skills/ ディレクトリを監視して変更時に SkillRegistry をリロードする。

    使い方（secretary/components.py など）:
        registry = SkillRegistry(paths=[REPO_ROOT / "skills"])
        registry.reload()  # 起動時に初回ロード
        watcher = SkillWatcherComponent(registry=registry)
        runner = DaemonRunner(components=[..., watcher])
    """

    def __init__(
        self,
        registry: "SkillRegistry",
        interval: float = 5.0,
    ) -> None:
        self._registry = registry
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return "skill_watcher"

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="skill-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info("SkillWatcherComponent: 開始（interval=%.1fs）", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            self._thread = None
        logger.info("SkillWatcherComponent: 停止")

    def _watch_loop(self) -> None:
        registry = self._registry
        paths = registry._paths
        entry_file = registry._entry_file
        prev_snapshot = _collect_snapshot(paths, entry_file)

        while not self._stop_event.wait(self._interval):
            current_snapshot = _collect_snapshot(paths, entry_file)
            if current_snapshot != prev_snapshot:
                added = set(current_snapshot) - set(prev_snapshot)
                removed = set(prev_snapshot) - set(current_snapshot)
                modified = {
                    k for k in current_snapshot
                    if k in prev_snapshot and current_snapshot[k] != prev_snapshot[k]
                }
                logger.info(
                    "SkillWatcherComponent: 変更検知 added=%s removed=%s modified=%s — リロード",
                    added, removed, modified,
                )
                try:
                    registry.reload()
                except Exception:
                    logger.exception("SkillWatcherComponent: リロード失敗")
                prev_snapshot = current_snapshot
