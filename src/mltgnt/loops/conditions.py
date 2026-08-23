"""mltgnt.loops.conditions — ローカル path 条件の決定論評価。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from mltgnt.interfaces.loops import WatchVerdict

MISSING_PATH_TOKEN = "<missing>"
_LOCAL_TYPES = frozenset({"path_exists", "path_changed"})


class PathConditionEvaluator:
    """root 配下の path_exists / path_changed のみを評価する。"""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def evaluate(
        self,
        condition: Mapping[str, object],
        *,
        previous_token: str | None,
    ) -> WatchVerdict:
        ctype = condition.get("type")
        if ctype not in _LOCAL_TYPES:
            return WatchVerdict(
                status="failed",
                detail=f"unsupported local condition type: {ctype!r}",
            )
        raw_path = condition.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return WatchVerdict(status="failed", detail="path is required")

        resolved = self._resolve_safe(raw_path)
        if isinstance(resolved, WatchVerdict):
            return resolved

        if ctype == "path_exists":
            if resolved.exists():
                return WatchVerdict(status="satisfied", detail="path exists")
            return WatchVerdict(status="pending", detail="path does not exist")

        # path_changed
        if resolved.exists() and resolved.is_dir():
            return WatchVerdict(
                status="failed",
                detail="path_changed does not accept directories",
            )
        token = self._file_token(resolved)
        if previous_token is None:
            return WatchVerdict(
                status="pending",
                detail="initial path token recorded",
                observed_token=token,
            )
        if token == previous_token:
            return WatchVerdict(
                status="pending",
                detail="path unchanged",
                observed_token=token,
            )
        return WatchVerdict(
            status="satisfied",
            detail="path changed",
            observed_token=token,
        )

    def _resolve_safe(self, raw_path: str) -> Path | WatchVerdict:
        if Path(raw_path).is_absolute():
            return WatchVerdict(status="failed", detail="absolute path is not allowed")
        candidate = (self._root / raw_path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            return WatchVerdict(
                status="failed",
                detail="path escapes watch_root",
            )
        return candidate

    @staticmethod
    def _file_token(path: Path) -> str:
        if not path.exists():
            return MISSING_PATH_TOKEN
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
