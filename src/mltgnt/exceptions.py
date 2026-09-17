"""
mltgnt.exceptions — shared exception type hierarchy.

Design: diary Issue #1252
"""
from __future__ import annotations

__all__ = [
    "MltgntError",
    "ConfigError",
    "DependencyError",
]


class MltgntError(Exception):
    """mltgnt common base exception. Callers can catch MltgntError for all."""


class ConfigError(MltgntError):
    """Config file (YAML etc.) load / parse error."""


class DependencyError(MltgntError):
    """External dependency (callable, subprocess, API) call failure."""
