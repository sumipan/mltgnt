"""mltgnt.loops.objective — Objective Markdown の解析・検証。"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yaml import YAMLError

from mltgnt.bridges.files_adapter import md_read

logger = logging.getLogger("mltgnt.loops.objective")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class Objective:
    loop_id: str
    title: str
    body: str
    agent: str
    max_iterations: int
    status: str  # active | cancelled
    path: Path
    content_hash: str


@dataclass(frozen=True)
class ObjectiveError:
    loop_id: str
    message: str
    path: Path


def _content_hash(body: str, meta: dict[str, Any]) -> str:
    payload = body + "\n" + repr(sorted(meta.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _first_body_line(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _parse_max_iterations(raw: Any, default: int) -> int | str:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return "bool not allowed"
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return f"invalid integer: {raw!r}"
    if not (1 <= val <= 10):
        return f"out of range: {val}"
    return val


def parse_objective(
    path: Path,
    *,
    default_persona: str,
    default_max_iterations: int,
    known_ids: set[str] | None = None,
) -> Objective | ObjectiveError:
    """Objective Markdown を解析する。失敗時は ObjectiveError を返す。"""
    stem = path.stem
    try:
        md = md_read(path.name, repo_root=path.parent)
    except (OSError, YAMLError) as exc:
        return ObjectiveError(loop_id=stem, message=f"read failed: {exc}", path=path)

    meta = md.frontmatter or {}
    unknown_keys = set(meta.keys()) - {"id", "title", "agent", "max_iterations", "status"}
    for key in sorted(unknown_keys):
        logger.warning("unknown YAML key in objective %s: %s", path, key)

    loop_id = meta.get("id", stem)
    if not isinstance(loop_id, str):
        return ObjectiveError(loop_id=stem, message="id must be a string", path=path)
    if not _ID_RE.match(loop_id):
        return ObjectiveError(
            loop_id=stem,
            message=f"invalid id format: {loop_id!r}",
            path=path,
        )
    if known_ids and loop_id in known_ids:
        return ObjectiveError(
            loop_id=loop_id,
            message=f"duplicate id: {loop_id!r}",
            path=path,
        )

    body = (md.content or "").strip()
    if not body:
        return ObjectiveError(loop_id=loop_id, message="empty body", path=path)

    title_raw = meta.get("title")
    if title_raw is not None and not isinstance(title_raw, str):
        return ObjectiveError(loop_id=loop_id, message="title must be a string", path=path)
    title = title_raw if title_raw else (_first_body_line(body) or loop_id)

    agent_raw = meta.get("agent")
    if agent_raw is not None and not isinstance(agent_raw, str):
        return ObjectiveError(loop_id=loop_id, message="agent must be a string", path=path)
    agent = agent_raw if agent_raw else default_persona

    max_raw = _parse_max_iterations(meta.get("max_iterations"), default_max_iterations)
    if isinstance(max_raw, str):
        return ObjectiveError(loop_id=loop_id, message=max_raw, path=path)

    status_raw = meta.get("status", "active")
    if status_raw is not None and not isinstance(status_raw, str):
        return ObjectiveError(loop_id=loop_id, message="status must be a string", path=path)
    status = status_raw if status_raw else "active"
    if status not in ("active", "cancelled"):
        return ObjectiveError(loop_id=loop_id, message=f"invalid status: {status!r}", path=path)

    content_hash = _content_hash(body, meta)

    return Objective(
        loop_id=loop_id,
        title=title,
        body=body,
        agent=agent,
        max_iterations=max_raw,
        status=status,
        path=path,
        content_hash=content_hash,
    )


def list_objective_files(objectives_dir: Path) -> list[Path]:
    """objectives_dir 直下の .md のみを返す（非再帰）。"""
    if not objectives_dir.is_dir():
        return []
    return sorted(
        p for p in objectives_dir.iterdir()
        if p.is_file() and p.suffix == ".md"
    )
