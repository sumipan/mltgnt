"""Shared pytest fixtures and compatibility shims."""

from __future__ import annotations

import re
import string
import sys


def _polyfill_template_get_identifiers() -> None:
    """Python 3.10 lacks string.Template.get_identifiers (added in 3.11).

    ghdag pipeline order expansion calls this API; compat CI matrix includes 3.10.
    """
    if hasattr(string.Template, "get_identifiers"):
        return

    def get_identifiers(self: string.Template) -> list[str]:
        ids: list[str] = []
        for match in re.finditer(r"\$(?:(\w+)|{([^}]+)})", self.template):
            name = match.group(1) or match.group(2)
            if name:
                ids.append(name)
        return ids

    string.Template.get_identifiers = get_identifiers  # type: ignore[attr-defined]


_polyfill_template_get_identifiers()
