"""mltgnt.loops.requests — 起動依頼 JSON の検証・列挙・隔離・consume。"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("mltgnt.loops.requests")

_REQUIRED_KEYS = frozenset(
    {"objective_path", "channel_id", "thread_ts", "persona", "requested_at"}
)


@dataclass(frozen=True)
class StartRequest:
    objective_path: str
    channel_id: str
    thread_ts: str
    persona: str
    requested_at: str
    filename: str


@dataclass(frozen=True)
class RequestError:
    filename: str
    message: str
    channel_id: str | None
    thread_ts: str | None
    persona: str | None


def _requests_dir(state_dir: Path) -> Path:
    return state_dir / "requests"


def _is_safe_basename(filename: str) -> bool:
    if not filename or filename in (".", ".."):
        return False
    if "/" in filename or "\\" in filename:
        return False
    return Path(filename).name == filename


def _validate_objective_path(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("/") or "\\" in raw or ".." in raw.split("/") or "/" in raw:
        return None
    if Path(raw).name != raw:
        return None
    if not raw.endswith(".md"):
        return None
    return raw


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if isinstance(value, str):
        return value
    return None


def _isolate_corrupt(state_dir: Path, path: Path) -> None:
    dest_dir = _requests_dir(state_dir) / "corrupt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    try:
        os.replace(path, dest)
    except OSError:
        logger.exception("failed to isolate corrupt request %s", path)


def _parse_request(path: Path) -> StartRequest | RequestError:
    filename = path.name
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return RequestError(
            filename=filename,
            message=f"invalid JSON: {exc}",
            channel_id=None,
            thread_ts=None,
            persona=None,
        )

    if not isinstance(data, dict):
        return RequestError(
            filename=filename,
            message="request must be a JSON object",
            channel_id=None,
            thread_ts=None,
            persona=None,
        )

    channel_id = _optional_str(data, "channel_id")
    thread_ts = _optional_str(data, "thread_ts")
    persona = _optional_str(data, "persona")

    keys = set(data.keys())
    if keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - keys)
        extra = sorted(keys - _REQUIRED_KEYS)
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if extra:
            parts.append(f"unexpected keys: {extra}")
        return RequestError(
            filename=filename,
            message="; ".join(parts) or "invalid keys",
            channel_id=channel_id,
            thread_ts=thread_ts,
            persona=persona,
        )

    for key in ("objective_path", "channel_id", "thread_ts", "persona", "requested_at"):
        if not isinstance(data[key], str):
            return RequestError(
                filename=filename,
                message=f"{key} must be a string",
                channel_id=channel_id,
                thread_ts=thread_ts,
                persona=persona,
            )

    objective_path = _validate_objective_path(data["objective_path"])
    if objective_path is None:
        return RequestError(
            filename=filename,
            message=f"invalid objective_path: {data['objective_path']!r}",
            channel_id=channel_id,
            thread_ts=thread_ts,
            persona=persona,
        )

    if not data["channel_id"]:
        return RequestError(
            filename=filename,
            message="channel_id must be non-empty",
            channel_id=channel_id,
            thread_ts=thread_ts,
            persona=persona,
        )
    if not data["thread_ts"]:
        return RequestError(
            filename=filename,
            message="thread_ts must be non-empty",
            channel_id=channel_id,
            thread_ts=thread_ts,
            persona=persona,
        )

    try:
        parsed_at = datetime.fromisoformat(data["requested_at"])
    except ValueError:
        return RequestError(
            filename=filename,
            message=f"invalid requested_at: {data['requested_at']!r}",
            channel_id=channel_id,
            thread_ts=thread_ts,
            persona=persona,
        )
    if parsed_at.tzinfo is None:
        return RequestError(
            filename=filename,
            message="requested_at must include timezone",
            channel_id=channel_id,
            thread_ts=thread_ts,
            persona=persona,
        )

    return StartRequest(
        objective_path=objective_path,
        channel_id=data["channel_id"],
        thread_ts=data["thread_ts"],
        persona=data["persona"],
        requested_at=data["requested_at"],
        filename=filename,
    )


def list_requests(
    state_dir: Path, objectives_dir: Path
) -> tuple[list[StartRequest], list[RequestError]]:
    """state_dir/requests/*.json をファイル名昇順で返す。不正分は corrupt/ へ隔離する。"""
    del objectives_dir  # path 検証は basename 契約のみ（存在確認は component 側）
    inbox = _requests_dir(state_dir)
    if not inbox.is_dir():
        return [], []

    ok: list[StartRequest] = []
    errors: list[RequestError] = []
    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        result = _parse_request(path)
        if isinstance(result, RequestError):
            _isolate_corrupt(state_dir, path)
            errors.append(result)
        else:
            ok.append(result)
    return ok, errors


def consume_request(state_dir: Path, filename: str, *, corrupt: bool = False) -> bool:
    """inbox の request を consumed/ または corrupt/ へ移す。再実行は False。"""
    if not _is_safe_basename(filename):
        return False

    src = _requests_dir(state_dir) / filename
    dest_dir = _requests_dir(state_dir) / ("corrupt" if corrupt else "consumed")
    dest = dest_dir / filename

    if not src.is_file():
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dest)
    except OSError:
        logger.exception("failed to consume request %s", filename)
        return False
    return True
