"""mltgnt.bridges.files_adapter

L2 bridge: thin wrappers around ghdag.files md_read / md_write.
Isolates L3 (domain) from direct L0 (ghdag) dependency.
"""
from __future__ import annotations

from pathlib import Path


def md_read(path: str, *, repo_root: Path | None = None):
    """Thin wrapper around ghdag.files.md_read."""
    from ghdag.files import md_read as _md_read
    return _md_read(path, repo_root=repo_root)


def md_write(path: str, content: str, *, repo_root: Path | None = None):
    """Thin wrapper around ghdag.files.md_write."""
    from ghdag.files import md_write as _md_write
    return _md_write(path, content, repo_root=repo_root)
