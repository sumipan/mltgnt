"""mltgnt.config.language — language-pack dataclass for locale-specific vocabulary."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["LanguagePack", "EN", "get_language_pack", "set_language_pack"]


@dataclass(frozen=True)
class LanguagePack:
    """Locale-specific vocabulary injected into processing functions.

    Locale-specific strings are collected here so callers can substitute a
    different locale by passing a custom pack or calling set_language_pack.
    None-default arguments on each consuming function fall back to
    get_language_pack() (EN by default).
    """

    work_request_markers: tuple[str, ...]
    create_request_markers: tuple[str, ...]
    deferred_patterns: tuple[re.Pattern[str], ...]
    compress_prompt_template: str
    v21_required_sections: tuple[str, ...]
    v21_example_section: str
    meta_header_needles: tuple[str, ...]
    dedupe_opener_re: re.Pattern[str]
    persona_cut_re: re.Pattern[str]
    # Persona stems to exclude from listing by default (e.g. sample/template files)
    exclude_stems: frozenset[str]
    persona_end_re: re.Pattern[str] = re.compile(r"\n-{3,}\s*\n\s*\(end\)")
    cancel_words: frozenset[str] = frozenset({"cancel", "stop"})
    composite_header: str = (
        "The following messages arrived while you were working. Take them into account."
    )
    composite_cancel_suffix: str = (
        "Note: a stop request is included. Stop the current work and report that you stopped."
    )
    # Media-layer vocabulary (mltgnt.media)
    approval_words: frozenset[str] = frozenset({"OK", "ok", "yes", "approve"})
    # Status value (mltgnt.interfaces.media.Status) -> display label
    status_labels: dict[str, str] = field(
        hash=False,
        default_factory=lambda: {
            "received": "Received",
            "working": "Working",
            "done": "Done",
            "failed": "Failed",
            "cancelled": "Cancelled",
        }
    )
    enqueue_failed_text: str = "Failed to enqueue the request. Please try again later."
    progress_line_pattern: re.Pattern[str] = re.compile(r"^\s*\[progress\]:?\s*(?P<text>.+)$", re.MULTILINE)
    # Memory-tool trigger words (mltgnt.memory.tools); empty by default, injected by callers
    remember_trigger_words: frozenset[str] = frozenset()
    forget_trigger_words: frozenset[str] = frozenset()
    # Localized legacy persona headings -> canonical English names; empty by default
    persona_section_aliases: dict[str, str] = field(hash=False, default_factory=dict)
    # Compaction Phase 1 meta lines to drop, in addition to the English ones
    phase1_meta_prefixes: tuple[str, ...] = ()
    phase1_meta_markers: tuple[str, ...] = ()


_EN_WORK_REQUEST_MARKERS: tuple[str, ...] = (
    "please",
    "could you",
    "can you",
    "fix",
    "update",
    "create",
    "make",
    "write",
    "brush up",
)

_EN_CREATE_REQUEST_MARKERS: tuple[str, ...] = (
    "create",
    "make a new",
    "new file",
    "write a new",
)

_EN_DEFERRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"will do (it )?later", re.IGNORECASE),
    re.compile(r"\bhold on\b", re.IGNORECASE),
    re.compile(r"\blater\b", re.IGNORECASE),
    re.compile(r"\bI'll (get|work) on it\b", re.IGNORECASE),
    re.compile(r"\basked [^\s]{1,24} to\b", re.IGNORECASE),
)

_EN_COMPRESS_PROMPT_TEMPLATE = (
    "Generate a v2.1 light block from the persona's heavy block below.\n\n"
    "## Output format\n\n"
    "1. Lead sentence (summarize the essence of the persona in 1-2 sentences)\n"
    "2. Required subsections (bold headings):\n"
    "   - **Tone** - characteristics of how they speak\n"
    "   - **Values** - what they care about\n"
    "   - **Positive reactions** - when they are pleased\n"
    "   - **Friction** - when they are bothered\n"
    "3. Recommended (optional):\n"
    "   - **Examples** - 1-3 examples as block quotes (> )\n\n"
    "## Constraints\n"
    "- At most 1500 characters (strict)\n"
    "- Use the bold heading names above exactly as written\n"
    "- If examples are included, always format them as block quotes\n\n"
    "## Heavy block:\n"
    "{heavy_text}"
)

_EN_V21_REQUIRED_SECTIONS: tuple[str, ...] = (
    "**Tone**",
    "**Values**",
    "**Positive reactions**",
    "**Friction**",
)

_EN_META_HEADER_NEEDLES: tuple[str, ...] = (
    "response (stdout)",
    "'s response",
)

EN = LanguagePack(
    work_request_markers=_EN_WORK_REQUEST_MARKERS,
    create_request_markers=_EN_CREATE_REQUEST_MARKERS,
    deferred_patterns=_EN_DEFERRED_PATTERNS,
    compress_prompt_template=_EN_COMPRESS_PROMPT_TEMPLATE,
    v21_required_sections=_EN_V21_REQUIRED_SECTIONS,
    v21_example_section="**Examples**",
    meta_header_needles=_EN_META_HEADER_NEEDLES,
    dedupe_opener_re=re.compile(r"^This week's plan \([^)]{1,80}\)[,:]", re.MULTILINE),
    persona_cut_re=re.compile(r"\n\n\S+ tone body:"),
    exclude_stems=frozenset({"sample"}),
    persona_end_re=re.compile(r"\n-{3,}\s*\n\s*\(end\)"),
    cancel_words=frozenset({"cancel", "stop"}),
    composite_header=(
        "The following messages arrived while you were working. Take them into account."
    ),
    composite_cancel_suffix=(
        "Note: a stop request is included. Stop the current work and report that you stopped."
    ),
    approval_words=frozenset({"OK", "ok", "yes", "approve"}),
    status_labels={
        "received": "Received",
        "working": "Working",
        "done": "Done",
        "failed": "Failed",
        "cancelled": "Cancelled",
    },
    enqueue_failed_text="Failed to enqueue the request. Please try again later.",
    progress_line_pattern=re.compile(r"^\s*\[progress\]:?\s*(?P<text>.+)$", re.MULTILINE),
    remember_trigger_words=frozenset(),
    forget_trigger_words=frozenset(),
    persona_section_aliases={},
    phase1_meta_prefixes=(),
    phase1_meta_markers=(),
)

_current: LanguagePack = EN


def get_language_pack() -> LanguagePack:
    """Return the current language pack (EN until set_language_pack is called)."""
    return _current


def set_language_pack(pack: LanguagePack) -> None:
    """Replace the current language pack. Raises TypeError for non-LanguagePack values."""
    global _current
    if not isinstance(pack, LanguagePack):
        raise TypeError(f"pack must be a LanguagePack, got {type(pack).__name__}")
    _current = pack
