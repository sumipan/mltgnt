"""Shared pytest fixtures and compatibility shims."""

from __future__ import annotations

import re
import string
import sys

import pytest

from mltgnt.config.language import LanguagePack


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


@pytest.fixture(scope="session")
def ascii_pack() -> LanguagePack:
    """ASCII LanguagePack for verifying detection logic without CJK text."""
    return LanguagePack(
        work_request_markers=("please", "update", "revise", "polish", "fix", "create", "make"),
        create_request_markers=("create", "make", "new-file"),
        deferred_patterns=(
            re.compile(r"will do later"),
            re.compile(r"hold on"),
            re.compile(r"working on it"),
            re.compile(r"later"),
        ),
        compress_prompt_template=(
            "Generate a v2.1 light block from the heavy block below.\n\n"
            "Required sections: **tone**, **values**, **positive-reaction**, **friction**\n\n"
            "Heavy block:\n{heavy_text}"
        ),
        v21_required_sections=("**tone**", "**values**", "**positive-reaction**", "**friction**"),
        v21_example_section="**speech-example**",
        meta_header_needles=(
            "as-persona (stdout-equivalent)",
            "as-persona",
        ),
        dedupe_opener_re=re.compile(r"^this-week \([^)]{1,80}\) plan[,]", re.MULTILINE),
        persona_cut_re=re.compile(r"\n\npersona-\w+ tone-body starts"),
        exclude_stems=frozenset({"sample-persona"}),
    )
