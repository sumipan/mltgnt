"""
mltgnt.exceptions — 共通例外型階層。

設計: diary Issue #1252
"""
from __future__ import annotations

__all__ = [
    "MltgntError",
    "ConfigError",
    "DependencyError",
]


class MltgntError(Exception):
    """mltgnt 共通基底例外。外部コードは except MltgntError で一括捕捉可能。"""


class ConfigError(MltgntError):
    """設定ファイル（YAML 等）の読み込み・パースエラー。"""


class DependencyError(MltgntError):
    """外部依存（callable, subprocess, API）の呼び出し失敗。"""
