"""mltgnt.bridges.files_adapter

L2 ブリッジ層: ghdag.files の md_read / md_write の薄いラッパ。
L3（domain）→ L0（ghdag）の直接依存を隔離する。
"""
from __future__ import annotations

from pathlib import Path


def md_read(path: str, *, repo_root: Path | None = None):
    """ghdag.files.md_read の薄いラッパ。"""
    from ghdag.files import md_read as _md_read
    return _md_read(path, repo_root=repo_root)


def md_write(path: str, content: str, *, repo_root: Path | None = None):
    """ghdag.files.md_write の薄いラッパ。"""
    from ghdag.files import md_write as _md_write
    return _md_write(path, content, repo_root=repo_root)
