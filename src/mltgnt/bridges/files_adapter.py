"""mltgnt.bridges.files_adapter

L2 bridge: thin wrappers around ghdag.files md_read / md_write and the
ghdag.vcs sink commit. Isolates L3 (domain) from direct L0 (ghdag) dependency.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ghdag.vcs import CommitResult


def md_read(path: str, *, repo_root: Path | None = None):
    """Thin wrapper around ghdag.files.md_read."""
    from ghdag.files import md_read as _md_read
    return _md_read(path, repo_root=repo_root)


def md_write(path: str, content: str, *, repo_root: Path | None = None):
    """Thin wrapper around ghdag.files.md_write."""
    from ghdag.files import md_write as _md_write
    return _md_write(path, content, repo_root=repo_root)


def commit(
    paths: Sequence[str | Path],
    message: str,
    *,
    sink: str = "memory",
    trailers: Mapping[str, str] | None = None,
) -> "CommitResult":
    """Commit ``paths`` through the ghdag.vcs sink named ``sink``.

    Absolute paths are made relative to the sink's ``repo_root`` (``ValueError``
    when outside it). Sink errors (``OwnershipError`` / ``ConflictError``) propagate.
    """
    from ghdag.vcs import NullSink, get_sink

    s = get_sink(sink)
    if isinstance(s, NullSink):
        return s.commit([str(p) for p in paths], message, trailers=trailers)
    root = Path(s.repo_root).resolve()
    rel: list[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_absolute():
            rel.append(pp.resolve().relative_to(root).as_posix())
        else:
            rel.append(pp.as_posix())
    return s.commit(rel, message, trailers=trailers)
