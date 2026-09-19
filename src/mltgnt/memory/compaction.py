"""
mltgnt.memory.compaction — memory compaction (per-section cap).

Design: Issue #123, #137, #823, #1135
Issue #1135: upstream diary's advanced compression logic.
- per-section cap (preferences / long_term / mid_term at 25% each)
- Phase 1: recent → preferences extract and merge
- Rollup loop: recent → mid_term chunk split + one-line summary
- mid_term → long_term cascade promote
- Incremental save (per section / chunk)
- ratio guard (exclude [slack-observe] entries)
- date coverage post-check
- entry reclassification (_redistribute_entries)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

from mltgnt.memory._format import MemoryEntry, parse_jsonl, serialize_entry

_log = logging.getLogger(__name__)

# For extracting timestamps from entry headings
_ENTRY_HEADER_TS_RE = re.compile(
    r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) — (?:user|assistant)",
    re.MULTILINE,
)

LlmCall = Callable[[str], str]

# per-section cap ratios
PREFS_CAP_RATIO = 0.25
LONG_TERM_CAP_RATIO = 0.25
MID_TERM_CAP_RATIO = 0.25
# Max compression ratio for preferences/long_term merge (reject below this)
PREFS_MAX_RATIO = 0.90
LONG_TERM_MAX_RATIO = 0.90
# Constants for mid_term → long_term cascade promote
LONG_TERM_PROMOTE_FLUSH_BYTES = 10 * 1024  # Fire LLM compress when buffer exceeds this
LONG_TERM_PROMOTE_MAX_ITER = 10  # Max LLM calls; remainder deferred to next run

ROLLUP_CHUNK = 50 * 1024  # 50KB: max bytes taken per rollup
ROLLUP_FINE_CHUNK = 5 * 1024  # 5KB: fine-grain mode cut limit
ROLLUP_MIN_KEEP_BYTES = 10 * 1024  # 10KB: min recent keep (stop loop at or below)
ROLLUP_MAX_ITER = 20  # Cap against infinite loops
ROLLUP_SUMMARY_TARGET_BYTES = 5 * 1024  # 5KB: target max bytes for one-line summary after LLM
ROLLUP_SUMMARY_MAX_RETRIES = 2  # Max retries (excluding first attempt)

__all__ = [
    "LlmCallError",
    "CompactionResult",
    "PromoteCandidate",
    "extract_promote_candidates",
    "needs_compaction",
    "compact",
    "_effective_bytes_for_ratio",
    "_build_section_prompt",
    "_strip_heading",
    "_compact_section",
    "_promote_with_compression",
    "_promote_mid_to_long",
    "_extract_and_merge_preferences",
    "_sanitize_phase1_output",
    "_strip_observe_entries",
    "_rollup_recent_chunk",
    "_extract_chunk_date_range",
    "_check_date_coverage",
    "_compress_rollup_chunk",
    "_redistribute_entries",
    "_entry_to_block",
    "_entries_to_body",
]


class LlmCallError(RuntimeError):
    """Exception wrapping an error raised during llm_call."""


@dataclass(frozen=True)
class PromoteCandidate:
    topic: str
    summary: str
    source_entries: int
    recurrence: int


@dataclass(frozen=True)
class CompactionResult:
    before_bytes: int
    after_bytes: int
    summary: str
    warnings: list[str] = field(default_factory=list)
    promote_candidates: list[PromoteCandidate] = field(default_factory=list)


def extract_promote_candidates(
    entries: list[MemoryEntry],
    *,
    min_recurrence: int = 3,
) -> list[PromoteCandidate]:
    """Extract promote candidates from compaction entries.

    Candidates are entries whose source_tag appears at least min_recurrence times.
    Whether to promote is left to the caller.

    Args:
        entries: MemoryEntry list to compact
        min_recurrence: Minimum occurrences of the same topic

    Returns:
        List of PromoteCandidate.
    """
    from collections import defaultdict

    groups: dict[str, list[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.source_tag].append(entry)

    candidates: list[PromoteCandidate] = []
    for source_tag, group in groups.items():
        if len(group) < min_recurrence:
            continue
        summary = "\n\n".join(e.content for e in group if e.content.strip())
        candidates.append(
            PromoteCandidate(
                topic=source_tag,
                summary=summary,
                source_entries=len(group),
                recurrence=len(group),
            )
        )
    return candidates


def needs_compaction(config: "MemoryConfig", persona_stem: str) -> bool:
    """Whether the memory file exceeds the compaction threshold."""
    from mltgnt.memory import memory_file_path

    path = memory_file_path(config, persona_stem)
    if not path.exists():
        return False
    return path.stat().st_size >= config.compact_threshold_bytes


def _effective_bytes_for_ratio(text: str) -> int:
    """Effective byte count for ratio checks (exclude [slack-observe] blocks).

    ``[slack-observe]``-tagged entries are low-density and shrink drastically
    after LLM compression, so ratio checks false-positive easily.
    Returning size without them aborts only when
    non-observe content compresses below 5%.

    Supports both JSONL (one JSON per line) and Markdown (``\n---\n`` delimited).
    """
    import json as _json

    lines = text.splitlines()
    # JSONL detection: first non-empty line starts with {
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty and non_empty[0].strip().startswith("{"):
        # JSONL: count only lines whose source_tag/content lack [slack-observe]
        kept_lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                obj = _json.loads(line)
                tag = obj.get("source_tag", "") or ""
                content = obj.get("content", "") or ""
                if "[slack-observe]" not in tag and "[slack-observe]" not in content:
                    kept_lines.append(line)
            except _json.JSONDecodeError:
                kept_lines.append(line)
        cleaned = "\n".join(kept_lines)
        result = len(cleaned.encode("utf-8"))
    else:
        # Markdown: delimited by \n---\n
        blocks = re.split(r"\n---\n", text)
        kept = [b for b in blocks if "[slack-observe]" not in b]
        cleaned = "\n---\n".join(kept)
        result = len(cleaned.encode("utf-8"))
    return result if result > 0 else len(text.encode("utf-8"))


def _build_section_prompt(section_text: str, target_bytes: int) -> str:
    """Build a compaction prompt for a single section."""
    return (
        "Please summarize and compress the following text."
        "Keep each entry date heading line (form '## YYYY-MM-DD HH:MM — user/assistant') "
        "unchanged; do not delete or alter them."
        "Compress only the body text outside headings."
        f"Target size: within {target_bytes} bytes."
        "Output only the summarized body."
        "Do not include meta such as byte/token counts, size info, or compression ratio."
        "Do not self-refer to the prompt (e.g. 'compressed as instructed')."
        "\n\n"
        f"{section_text}"
    )


def _strip_heading(section_text: str) -> str:
    """Strip a leading ``## ...`` heading line from section text; return body only."""
    return re.sub(r"^##\s+[^\n]*\n*", "", section_text, count=1).strip()


def _compact_section(
    section_name: str,
    section_text: str,
    target_bytes: int,
    llm_call: LlmCall,
    *,
    skip_min_ratio: bool = False,
) -> tuple[str, str | None]:
    """Compact one section.

    Returns:
        (compacted_body, warning_or_none)
        On failure, return the original body and put the reason in warning.
    """
    body = _strip_heading(section_text)
    if not body:
        return body, None

    MIN_RATIO = 0.05
    original_size = len(body.encode("utf-8"))

    try:
        prompt = _build_section_prompt(body, target_bytes)
        result = llm_call(prompt)
    except Exception as e:
        warning = f"{section_name}: LLM call failed ({e}), using original text"
        _log.warning(warning)
        return body, warning

    result_size = len(result.encode("utf-8"))
    if result_size < original_size * MIN_RATIO:
        if skip_min_ratio:
            warning = (
                f"{section_name}: result very small "
                f"({result_size}B < {original_size}B * {MIN_RATIO}), "
                f"accepted (--no-min-ratio-guard)"
            )
            _log.warning(warning)
            return result.strip(), warning
        warning = (
            f"{section_name}: result too small ({result_size}B < {original_size}B * {MIN_RATIO}), using original text"
        )
        _log.warning(warning)
        return body, warning

    return result.strip(), None


def _promote_with_compression(
    section_name: str,
    existing_body: str,
    incoming_body: str,
    cap_bytes: int,
    llm_call: LlmCall,
    *,
    max_ratio: float = 0.90,
    skip_min_ratio: bool = False,
) -> tuple[str, str | None]:
    """Promote-merge compress: LLM-compress existing_body + incoming_body together.

    For preferences / long_term. Called only when over cap.
    max_ratio guard: if result is below ``existing_body * max_ratio``, reject as
    over-compression and return the concatenated text unchanged.

    Args:
        section_name: Section name (for logs)
        existing_body: Existing section body (baseline size for max_ratio)
        incoming_body: Text added by promote (may be empty)
        cap_bytes: Target byte cap
        llm_call: LLM call callable
        max_ratio: Min compression ratio (reject if result < existing * max_ratio)

    Returns:
        (result_body, warning_or_none)
    """
    _SEP = "\n\n---\n\n"
    if existing_body and incoming_body:
        combined = existing_body + _SEP + incoming_body
    else:
        combined = existing_body or incoming_body

    if not combined:
        return "", None

    # max_ratio guard baseline is existing_body size (when present)
    check_size = len(existing_body.encode("utf-8")) if existing_body else len(combined.encode("utf-8"))

    try:
        prompt = _build_section_prompt(combined, cap_bytes)
        result = llm_call(prompt)
    except Exception as e:
        warning = f"{section_name}: LLM call failed ({e}), using original text"
        _log.warning(warning)
        return combined, warning

    result_size = len(result.encode("utf-8"))

    # max_ratio guard: reject over-compression below existing * max_ratio
    if result_size < check_size * max_ratio:
        if skip_min_ratio:
            warning = (
                f"{section_name}: result over-compressed "
                f"({result_size}B < {check_size}B * {max_ratio}), "
                f"accepted (--no-min-ratio-guard)"
            )
            _log.warning(warning)
            return result.strip(), warning
        warning = (
            f"{section_name}: result over-compressed "
            f"({result_size}B < {check_size}B * {max_ratio}), "
            f"using original text"
        )
        _log.warning(warning)
        return combined, warning

    return result.strip(), None


def _promote_mid_to_long(
    compacted: dict[str, str],
    mid_term_cap: int,
    long_term_cap: int,
    llm_call: LlmCall,
    *,
    flush_threshold: int = LONG_TERM_PROMOTE_FLUSH_BYTES,
    max_iter: int = LONG_TERM_PROMOTE_MAX_ITER,
    skip_min_ratio: bool = False,
) -> list[str]:
    """Cascade-promote older mid_term entries into long_term.

    Mutates compacted in place; returns warnings only.

    Algorithm:
    1. Split mid_term into blocks ("\n\n---\n\n" delimited)
    2. Accumulate oldest-first into buffer; flush when buffer > flush_threshold:
       - long_term + buffer → _promote_with_compression(long_term_cap, max_ratio=0.90)
       - LLM input cap = long_term_cap + flush_threshold (structurally fixed)
    3. Stop when mid_term_size <= mid_term_cap
    4. Stop after max_iter LLM calls; defer remainder to next run
    5. After the loop, final-flush leftover buffer (also counts toward iter_count)
    """
    SEP = "\n\n---\n\n"
    blocks = compacted["mid_term"].split(SEP)
    blocks = [b for b in blocks if b.strip()]
    buffer: list[str] = []
    warnings: list[str] = []
    iter_count = 0

    while blocks and iter_count < max_iter:
        remaining = SEP.join(blocks)
        if len(remaining.encode("utf-8")) <= mid_term_cap:
            break

        block = blocks.pop(0)
        buffer.append(block)
        buffer_size = sum(len(b.encode("utf-8")) for b in buffer)

        if buffer_size >= flush_threshold:
            incoming = SEP.join(buffer)
            new_long_term, warning = _promote_with_compression(
                "long_term",
                compacted["long_term"],
                incoming,
                long_term_cap,
                llm_call,
                max_ratio=LONG_TERM_MAX_RATIO,
                skip_min_ratio=skip_min_ratio,
            )
            if warning:
                warnings.append(warning)
            compacted["long_term"] = new_long_term
            buffer = []
            iter_count += 1

    # Final flush if buffer remains and iter_count < max_iter
    if buffer and iter_count < max_iter:
        incoming = SEP.join(buffer)
        new_long_term, warning = _promote_with_compression(
            "long_term",
            compacted["long_term"],
            incoming,
            long_term_cap,
            llm_call,
            max_ratio=LONG_TERM_MAX_RATIO,
            skip_min_ratio=skip_min_ratio,
        )
        if warning:
            warnings.append(warning)
        compacted["long_term"] = new_long_term
        buffer = []
        iter_count += 1

    # Rebuild mid_term from remaining blocks + undigested buffer
    remaining_blocks = buffer + blocks
    compacted["mid_term"] = SEP.join(remaining_blocks) if remaining_blocks else ""

    if iter_count >= max_iter and (blocks or buffer):
        _log.warning(
            "mid→long promotion hit max_iter=%d, %d blocks remain in mid_term",
            max_iter,
            len(blocks) + len(buffer),
        )

    return warnings


_PHASE1_PROMPT_TEMPLATE = """\
Please process the following two texts.

[Task]
1. From "Recent records", extract user preferences, tendencies, habits, and patterns.
   - Exclude transient states (e.g. "tired today"); keep only recurring tendencies.
2. Merge the extracted content with "Existing preferences/tendencies".
   - Unify duplicate items.
   - On conflict, prefer the newer side (recent records).
3. Output only a bullet list of preferences/tendencies.
   Target size: within {target_bytes} bytes.

Output rules (strict):
- Bullet body only. No preamble, postscript, headings, or size info
- No self-referential lines such as "Understood" / "Analyzing"
- No interactive replies (questions, confirmations, suggestions)
- If input is empty or "(none)", return an empty string (no explanation)

[Existing preferences/tendencies]
{existing_prefs}

[Recent records]
{recent_text}"""


def _sanitize_phase1_output(text: str) -> str:
    """Strip meta-speech, headings, and meta lines from Phase 1 LLM raw output.

    Strip targets:
    - Lines starting with acknowledgment/analysis meta tokens
    - Heading lines starting with ``## ``
    - Lines containing size/stats/analysis bold markers
    """
    if not text:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(
            (
                "\N{CJK UNIFIED IDEOGRAPH-627F}\N{CJK UNIFIED IDEOGRAPH-77E5}",
                "\N{CJK UNIFIED IDEOGRAPH-5206}\N{CJK UNIFIED IDEOGRAPH-6790}",
                "\N{CJK UNIFIED IDEOGRAPH-4E86}\N{CJK UNIFIED IDEOGRAPH-89E3}",
                "\N{CJK UNIFIED IDEOGRAPH-4EE5}\N{CJK UNIFIED IDEOGRAPH-4E0B}",
            )
        ):
            continue
        if stripped.startswith(("Understood", "Analyzing", "Analysis", "Below")):
            continue
        if stripped.startswith("## "):
            continue
        if any(
            marker in stripped
            for marker in (
                "**\N{KATAKANA LETTER SA}\N{KATAKANA LETTER I}\N{KATAKANA LETTER ZU}**",
                "**\N{CJK UNIFIED IDEOGRAPH-7D71}\N{CJK UNIFIED IDEOGRAPH-8A08}**",
                "**\N{CJK UNIFIED IDEOGRAPH-5206}\N{CJK UNIFIED IDEOGRAPH-6790}**",
                "**Size**",
                "**Statistics**",
                "**Analysis**",
            )
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def _extract_and_merge_preferences(
    existing_prefs: str,
    recent_text: str,
    target_bytes: int,
    llm_call: LlmCall,
    *,
    skip_min_ratio: bool = False,
) -> tuple[str, str | None]:
    """Extract preferences/tendencies from recent text and merge with existing preferences.

    Args:
        existing_prefs: Current preferences section body (heading stripped)
        recent_text: Recent section body (heading stripped)
        target_bytes: Preferences cap in bytes
        llm_call: LLM call callable

    Returns:
        (merged_prefs, warning)
        - merged_prefs: Merged preferences body
        - warning: Warning message on anomaly (None when OK)
    """
    if not recent_text:
        return existing_prefs, None

    prompt = _PHASE1_PROMPT_TEMPLATE.format(
        target_bytes=target_bytes,
        existing_prefs=existing_prefs
        if existing_prefs
        else "(No existing preferences/tendencies. Initialize from recent extraction only.)",
        recent_text=recent_text,
    )

    try:
        result = llm_call(prompt)
    except Exception as e:
        warning = f"Phase 1 extract-merge failed: {e}"
        _log.warning(warning)
        return existing_prefs, warning

    if not result or not result.strip():
        warning = "Phase 1 extract-merge: empty output from LLM"
        _log.warning(warning)
        return existing_prefs, warning

    result = _sanitize_phase1_output(result).strip()

    # Over-compression guard: when existing preferences are present,
    # reject LLM output below existing * PREFS_MAX_RATIO
    existing_bytes = len(existing_prefs.encode("utf-8"))
    if existing_bytes > 0:
        result_bytes = len(result.encode("utf-8"))
        if result_bytes < existing_bytes * PREFS_MAX_RATIO:
            if skip_min_ratio:
                warning = (
                    f"Phase 1 extract-merge: over-compressed "
                    f"({result_bytes}B < {existing_bytes}B * {PREFS_MAX_RATIO}), "
                    f"accepted (--no-min-ratio-guard)"
                )
                _log.warning(warning)
                return result.strip(), warning
            warning = (
                f"Phase 1 extract-merge: over-compressed "
                f"({result_bytes}B < {existing_bytes}B * {PREFS_MAX_RATIO}), "
                f"keeping original preferences"
            )
            _log.warning(warning)
            return existing_prefs, warning

    return result, None


def _strip_observe_entries(body: str) -> str:
    """Remove entry blocks containing ``[slack-observe]`` from recent section body.

    Split on ``\n---\n`` and rejoin only blocks without ``[slack-observe]``.
    """
    if not body:
        return body
    blocks = re.split(r"\n---\n", body)
    kept = [b for b in blocks if "[slack-observe]" not in b]
    return "\n---\n".join(kept)


def _rollup_recent_chunk(recent_body: str, rollup_chunk: int) -> tuple[str, str]:
    """Take up to rollup_chunk bytes of oldest entries from recent body.

    Args:
        recent_body: Recent section body
        rollup_chunk: Max bytes to take

    Returns:
        (remaining_body, promoted_body)
        - remaining_body: Text left in recent
        - promoted_body: Text promoted to mid_term (raw entries)

    Edge cases:
    - 0 entries: return (recent_body, "")
    - 1 entry over rollup_chunk: promote that one whole
    """
    if not recent_body:
        return recent_body, ""

    blocks = re.split(r"\n---\n", recent_body)
    blocks = [b for b in blocks if b]  # drop empty blocks
    if not blocks:
        return recent_body, ""

    accumulated: list[str] = []
    acc_bytes = 0

    for i, block in enumerate(blocks):
        block_bytes = len(block.encode("utf-8"))
        if acc_bytes + block_bytes > rollup_chunk and accumulated:
            # Already over rollup_chunk → finalize
            break
        accumulated.append(block)
        acc_bytes += block_bytes
        if acc_bytes > rollup_chunk:
            # Single entry alone exceeds: promote whole
            break

    if not accumulated:
        # Could not take even one entry (should be rare; safety valve)
        return recent_body, ""

    # Index-based split (handles duplicate entries)
    promoted_blocks = blocks[: len(accumulated)]
    remaining_blocks = blocks[len(accumulated) :]

    promoted_body = "\n---\n".join(promoted_blocks)
    remaining_body = "\n---\n".join(remaining_blocks)
    return remaining_body, promoted_body


def _extract_chunk_date_range(promoted: str) -> tuple[str, str] | None:
    """Extract start/end entry dates from chunk text.

    Args:
        promoted: Promoted text returned by _rollup_recent_chunk

    Returns:
        On success: (start_date, end_date) — each "YYYY-MM-DD"
        On failure (no dates found): None
    """
    import warnings

    matches = _ENTRY_HEADER_TS_RE.findall(promoted)
    if not matches:
        warnings.warn(
            f"_extract_chunk_date_range: no date headers found in chunk (first 200 bytes: {promoted[:200]!r})",
            stacklevel=2,
        )
        return None
    start_date = matches[0].split(" ")[0]
    end_date = matches[-1].split(" ")[0]
    return (start_date, end_date)


def _check_date_coverage(
    observed_ranges: list[tuple[str, str]],
    final_text: str,
) -> list[tuple[str, str]]:
    """Return observed_ranges whose neither date appears in final_text.

    Args:
        observed_ranges: (start_date, end_date) list from rollup.
                         Each date is a "YYYY-MM-DD" string.
        final_text: Full assemble_memory() output.

    Returns:
        List of missing ranges (empty if every range appears on at least one end).
    """
    missed = []
    for start, end in observed_ranges:
        if start not in final_text and end not in final_text:
            missed.append((start, end))
    return missed


_ROLLUP_SUMMARY_PROMPT = """\
Please summarize the following conversation log in one line.
- Output exactly one line (no newlines)
- Stay within {target} bytes
- Do not include dates (caller will attach them)
- Briefly list key topics, decisions, and artifacts

---
{chunk}"""


def _compress_rollup_chunk(
    promoted: str,
    llm_call: "LlmCall",
    *,
    target: int = ROLLUP_SUMMARY_TARGET_BYTES,
    max_retries: int = ROLLUP_SUMMARY_MAX_RETRIES,
) -> str:
    """Compress chunk text to a one-line summary via LLM.

    Args:
        promoted: Chunk text to compress
        llm_call: LLM call function
        target: Target bytes after compression (default: 5KB)
        max_retries: Max retries (default: 2)

    Returns:
        Summary text (no newlines, no date prefix).
        On all retries failing, return promoted unchanged.
    """
    import warnings

    prompt = _ROLLUP_SUMMARY_PROMPT.format(target=target, chunk=promoted)
    for attempt in range(max_retries + 1):
        try:
            output = llm_call(prompt)
        except Exception as e:
            _log.warning("_compress_rollup_chunk: LLM call failed (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries:
                continue
            break
        # Mechanically strip all newlines
        output = output.replace("\n", " ").strip()
        # Length check: retry if over target * 3
        if len(output.encode("utf-8")) > target * 3:
            _log.warning(
                "_compress_rollup_chunk: output too long (%dB > %dB), retrying (attempt %d/%d)",
                len(output.encode("utf-8")),
                target * 3,
                attempt + 1,
                max_retries + 1,
            )
            if attempt < max_retries:
                continue
            break
        return output

    warnings.warn(
        f"_compress_rollup_chunk: all {max_retries + 1} attempts failed, falling back to raw promoted text",
        stacklevel=2,
    )
    return promoted


def _redistribute_entries(
    entries: list["MemoryEntry"],
    now: datetime,
    config: "MemoryConfig",
    *,
    raw_days_override: int | None = None,
) -> list["MemoryEntry"]:
    """Reclassify entry layers by age (pure function).

    Keep protected / preferences entries as-is.
    Compute age with config.timezone (no diary-specific dependency).
    """
    tz = ZoneInfo(config.timezone)
    effective_raw_days = raw_days_override if raw_days_override is not None else config.raw_days
    mid_threshold_days = config.mid_weeks * 7

    result = []
    for entry in entries:
        if (entry.layer is not None and entry.layer in config.protected_layers) or entry.source_tag == "preferences":
            result.append(entry)
            continue
        ts = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
            try:
                ts = datetime.strptime(entry.timestamp, fmt)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=tz)
                break
            except ValueError:
                continue
        if ts is None:
            result.append(entry)
            continue
        age_days = (now.astimezone(tz) - ts.astimezone(tz)).total_seconds() / 86400
        if age_days <= effective_raw_days:
            new_layer = "recent"
        elif age_days <= mid_threshold_days:
            new_layer = "mid_term"
        else:
            new_layer = "long_term"
        if entry.layer == new_layer:
            result.append(entry)
        else:
            result.append(
                MemoryEntry(
                    timestamp=entry.timestamp,
                    role=entry.role,
                    content=entry.content,
                    source_tag=entry.source_tag,
                    layer=new_layer,
                    dedupe_key=entry.dedupe_key,
                )
            )
    return result


def _entry_to_block(e: "MemoryEntry") -> str:
    """Convert a MemoryEntry to text-block form (for rollup).

    Normalize timestamps to YYYY-MM-DD HH:MM (_ENTRY_HEADER_TS_RE compatible).
    Also convert ISO 8601 (e.g. 2026-04-20T10:00:00+09:00).
    """
    ts = e.timestamp
    # Replace ISO 8601 'T' with space; strip trailing timezone/seconds
    if "T" in ts:
        ts = ts.replace("T", " ")
    # Strip timezone (+09:00 etc.)
    if "+" in ts:
        ts = ts[: ts.index("+")]
    elif ts.endswith("Z"):
        ts = ts[:-1]
    # Strip seconds (:SS): HH:MM:SS → HH:MM
    parts = ts.split(":")
    if len(parts) >= 3:
        ts = ":".join(parts[:2])
    return f"## {ts} — {e.role}\n\n{e.content}"


def _entries_to_body(entries: list["MemoryEntry"]) -> str:
    """Convert a MemoryEntry list to text body (for rollup)."""
    if not entries:
        return ""
    return "\n---\n".join(_entry_to_block(e) for e in entries)


def compact(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    llm_call: LlmCall,
    dry_run: bool = False,
    max_retries: int = 3,
    skip_min_ratio: bool = False,
) -> CompactionResult:
    """Compact a memory file (per-section cap).

    llm_call takes a prompt string and returns compacted text.
    Hosts can wrap llm_call to inject timestamps or trace info.

    Example:
        def wrapped_llm_call(prompt: str) -> str:
            enriched = f"[date-context]\\n{prompt}"
            return base_llm_call(enriched)

        compact(config, persona_stem, llm_call=wrapped_llm_call)

    When dry_run=True, do not write the file.

    **per-section cap**:
    - preferences: merge-compress only when over cap (dedupe only, max_ratio=0.90)
    - long_term: merge-compress only when over cap (max_ratio=0.90)
    - mid_term: pass-through buffer receiving promotes from recent; when over cap, cascade to long_term via _promote_mid_to_long; LLM-compress if still over
    - recent: chunked LLM compress only when over cap; if still over, shorten raw_days for early promote

    **Incremental save**: write after each long_term/mid_term step and each recent chunk.
    Large files shrink gradually; partial progress survives interruption.

    New params max_retries / skip_min_ratio all have defaults, so
    existing compact(config, stem, llm_call=fn) calls need no change (backward compatible).
    """
    from mltgnt.memory import memory_file_path, persona_memory_lock

    path = memory_file_path(config, persona_stem)
    if not path.exists():
        raise FileNotFoundError(f"Memory file not found: {path}")

    tz = ZoneInfo(config.timezone)

    with persona_memory_lock(config, persona_stem) as ok:
        if not ok:
            raise TimeoutError(f"Failed to acquire memory lock for {persona_stem}")

        original_text = path.read_text(encoding="utf-8")
        before_bytes = len(original_text.encode("utf-8"))

        warnings_list: list[str] = []
        observed_date_ranges: list[tuple[str, str]] = []
        new_text = original_text

        for attempt in range(max_retries):
            entries = parse_jsonl(path)

            protected_entries = [e for e in entries if e.layer is not None and e.layer in config.protected_layers]
            prefs_entries = [e for e in entries if e.source_tag == "preferences"]
            _classified_ids = set(id(e) for e in protected_entries + prefs_entries)
            other_entries = [e for e in entries if id(e) not in _classified_ids]

            long_entries = [e for e in other_entries if e.layer == "long_term"]
            mid_entries = [e for e in other_entries if e.layer == "mid_term"]
            recent_entries = [e for e in other_entries if e.layer not in ("long_term", "mid_term")]

            prefs_body = "\n".join(e.content for e in prefs_entries)
            long_term_body = _entries_to_body(long_entries)
            mid_term_body = _entries_to_body(mid_entries)
            recent_body = _entries_to_body(recent_entries)

            # --- per-section cap calculation ---
            compact_target = config.compact_target_bytes
            prefs_cap = int(compact_target * PREFS_CAP_RATIO)
            long_term_cap = int(compact_target * LONG_TERM_CAP_RATIO)
            mid_term_cap = int(compact_target * MID_TERM_CAP_RATIO)

            prefs_size = len(prefs_body.encode("utf-8"))
            long_term_size = len(long_term_body.encode("utf-8"))
            recent_size = len(recent_body.encode("utf-8"))

            # recent target: compact_target minus other 3 sections actual size (capped)
            # and mid_term cap
            prefs_budget = min(prefs_size, prefs_cap)
            long_term_budget = min(long_term_size, long_term_cap)
            recent_target = max(
                compact_target - prefs_budget - long_term_budget - mid_term_cap,
                ROLLUP_MIN_KEEP_BYTES,
            )

            # compacted dict: track processed sections; keep original for unprocessed
            compacted: dict[str, str] = {
                "long_term": long_term_body,
                "mid_term": mid_term_body,
                "recent": recent_body,
            }

            # Always strip [slack-observe] entries first regardless of capacity
            stripped = _strip_observe_entries(recent_body)
            if stripped != recent_body:
                recent_body = stripped
                recent_size = len(recent_body.encode("utf-8"))
                compacted["recent"] = recent_body
                _log.info(
                    "stripped observe entries for %s, recent=%.1fKB",
                    persona_stem,
                    recent_size / 1024,
                )

            # --- rollup when recent is over capacity ---
            # (1) rollup loop → (2) Phase 1 once
            if recent_size > recent_target:
                # Rollup loop (still over after observe removal)
                promoted_text_parts: list[str] = []
                if recent_size > recent_target:
                    for _iter in range(ROLLUP_MAX_ITER):
                        current_recent = compacted["recent"]
                        if not current_recent:
                            break
                        current_recent_size = len(current_recent.encode("utf-8"))
                        if current_recent_size <= recent_target:
                            break

                        # Min-keep guard
                        if current_recent_size <= ROLLUP_MIN_KEEP_BYTES:
                            break

                        # Dynamic chunk_size
                        if current_recent_size <= ROLLUP_CHUNK:
                            chunk_size = min(ROLLUP_FINE_CHUNK, current_recent_size - ROLLUP_MIN_KEEP_BYTES)
                            if chunk_size <= 0:
                                break
                        else:
                            chunk_size = ROLLUP_CHUNK

                        remaining, promoted = _rollup_recent_chunk(current_recent, chunk_size)
                        if not promoted:
                            break

                        # Date extract
                        dates = _extract_chunk_date_range(promoted)
                        if dates is None:
                            _log.warning(
                                "compact: skipping undateable chunk for %s (first 200 bytes: %r)",
                                persona_stem,
                                promoted[:200],
                            )
                            compacted["recent"] = remaining
                            remaining_kb = len(remaining.encode("utf-8")) / 1024
                            _log.info(
                                "rollup iter=%d for %s: skipped (no dates), remaining=%.1fKB",
                                _iter + 1,
                                persona_stem,
                                remaining_kb,
                            )
                            continue

                        observed_date_ranges.append(dates)

                        # LLM compress
                        summary = _compress_rollup_chunk(
                            promoted,
                            llm_call,
                            target=ROLLUP_SUMMARY_TARGET_BYTES,
                            max_retries=ROLLUP_SUMMARY_MAX_RETRIES,
                        )

                        # Format
                        if summary is promoted:
                            # LLM compress fallback: do not feed into Phase 1 (avoid oversized input)
                            final = promoted
                            warnings_list.append(
                                f"rollup chunk LLM compression failed for "
                                f"{dates[0]}-{dates[1]}, "
                                f"excluded from Phase 1 input"
                            )
                        else:
                            final = f"{dates[0]} - {dates[1]} {summary}"
                            promoted_text_parts.append(final)  # only formatted one-line summaries

                        # Append to mid_term
                        existing_mid = compacted["mid_term"]
                        if existing_mid:
                            compacted["mid_term"] = existing_mid + "\n---\n" + final
                        else:
                            compacted["mid_term"] = final
                        compacted["recent"] = remaining

                        promoted_kb = len(promoted.encode("utf-8")) / 1024
                        remaining_kb = len(remaining.encode("utf-8")) / 1024
                        chunk_kb = chunk_size / 1024
                        _log.info(
                            "rollup iter=%d for %s: chunk=%.1fKB, promoted=%.1fKB, remaining=%.1fKB",
                            _iter + 1,
                            persona_stem,
                            chunk_kb,
                            promoted_kb,
                            remaining_kb,
                        )

                # (3) After rollup, run Phase 1 once on all accumulated promote text
                if promoted_text_parts:
                    all_promoted = "\n---\n".join(promoted_text_parts)
                    _log.info(
                        "Phase 1 input size: %d bytes (%d chunks)",
                        len(all_promoted.encode()),
                        len(promoted_text_parts),
                    )
                    prefs_body, p1_warning = _extract_and_merge_preferences(
                        prefs_body,
                        all_promoted,
                        prefs_cap,
                        llm_call,
                        skip_min_ratio=skip_min_ratio,
                    )
                    if p1_warning:
                        warnings_list.append(f"[attempt {attempt + 1}] {p1_warning}")
                    prefs_size = len(prefs_body.encode("utf-8"))
                    _log.info(
                        "Phase 1 done for %s, prefs=%.1fKB",
                        persona_stem,
                        prefs_size / 1024,
                    )

            # --- [B] mid_term → long_term cascade promote ---
            mid_term_size_now = len(compacted["mid_term"].encode("utf-8"))
            if mid_term_size_now > mid_term_cap:
                promote_warnings = _promote_mid_to_long(
                    compacted,
                    mid_term_cap,
                    long_term_cap,
                    llm_call,
                    skip_min_ratio=skip_min_ratio,
                )
                for w in promote_warnings:
                    warnings_list.append(f"[attempt {attempt + 1}] {w}")

            # --- [C] preferences merge-compress when over cap (dedupe, max_ratio=0.90) ---
            if prefs_size > prefs_cap:
                body, warning = _promote_with_compression(
                    "preferences",
                    prefs_body,
                    "",
                    prefs_cap,
                    llm_call,
                    max_ratio=PREFS_MAX_RATIO,
                    skip_min_ratio=skip_min_ratio,
                )
                if warning:
                    warnings_list.append(f"[attempt {attempt + 1}] {warning}")
                prefs_body = body

            # --- [D] long_term (insurance: long_term over cap after B max_iter stop) ---
            long_term_size_now = len(compacted["long_term"].encode("utf-8"))
            if long_term_size_now > long_term_cap:
                body, warning = _promote_with_compression(
                    "long_term",
                    compacted["long_term"],
                    "",
                    long_term_cap,
                    llm_call,
                    max_ratio=LONG_TERM_MAX_RATIO,
                    skip_min_ratio=skip_min_ratio,
                )
                if warning:
                    warnings_list.append(f"[attempt {attempt + 1}] {warning}")
                compacted["long_term"] = body

            # --- [E] mid_term fallback (still over cap after B max_iter stop) ---
            mid_term_size_now = len(compacted["mid_term"].encode("utf-8"))
            if mid_term_size_now > mid_term_cap:
                body, warning = _compact_section(
                    "mid_term",
                    "## mid_term\n" + compacted["mid_term"],
                    mid_term_cap,
                    llm_call,
                    skip_min_ratio=skip_min_ratio,
                )
                if warning:
                    warnings_list.append(f"[attempt {attempt + 1}] {warning}")
                compacted["mid_term"] = body

            # Assemble entries in JSONL form
            now_ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
            _recent_by_key = {(e.timestamp, e.role): e for e in recent_entries}

            def _block_to_entry(block: str, default_layer: str) -> "MemoryEntry":
                m = re.match(r"^## (\S+ \S+) — (user|assistant)\n\n(.*)", block, re.DOTALL)
                if m:
                    ts_str, role, content = m.group(1), m.group(2), m.group(3).strip()
                    key = (ts_str, role)
                    if default_layer == "recent" and key in _recent_by_key:
                        return _recent_by_key[key]
                    return MemoryEntry(
                        timestamp=ts_str, role=role, content=content, source_tag="compaction", layer=default_layer
                    )
                return MemoryEntry(
                    timestamp=now_ts,
                    role="assistant",
                    content=block.strip(),
                    source_tag="compaction",
                    layer=default_layer,
                )

            def _text_to_entries(body: str, layer: str) -> list["MemoryEntry"]:
                if not body.strip():
                    return []
                blocks = [b.strip() for b in re.split(r"\n---\n", body) if b.strip()]
                return [_block_to_entry(b, layer) for b in blocks]

            final_prefs = (
                [MemoryEntry(timestamp=now_ts, role="assistant", content=prefs_body.strip(), source_tag="preferences")]
                if prefs_body.strip()
                else []
            )
            final_long = _text_to_entries(compacted["long_term"], "long_term")
            final_mid = _text_to_entries(compacted["mid_term"], "mid_term")
            final_recent = _text_to_entries(compacted["recent"], "recent")

            all_final_entries = protected_entries + final_prefs + final_long + final_mid + final_recent
            new_text = "".join(serialize_entry(e) + "\n" for e in all_final_entries if e.content.strip())
            after_bytes = len(new_text.encode("utf-8"))

            if after_bytes <= config.compact_target_bytes * 1.3:
                break

            _log.warning(
                "compact: attempt %d/%d result still large for %s (%dB > %dB), retrying",
                attempt + 1,
                max_retries,
                persona_stem,
                after_bytes,
                int(config.compact_target_bytes * 1.3),
            )
            # JSONL: parse_jsonl reads from file, so write before retry
            if not dry_run:
                path.write_text(new_text, encoding="utf-8")

        after_bytes = len(new_text.encode("utf-8"))

        # Min-size check: under 5% of original (excl. [slack-observe]) is abnormal
        MIN_RATIO = 0.05
        effective_before = _effective_bytes_for_ratio(original_text)
        ratio = after_bytes / effective_before if effective_before > 0 else 1.0
        if after_bytes < effective_before * MIN_RATIO:
            if skip_min_ratio:
                if ratio < 0.01:
                    _log.warning(
                        "compact: extreme ratio for %s (%.4f) but accepted (--no-min-ratio-guard)",
                        persona_stem,
                        ratio,
                    )
                # Lift guard: continue normal path (to write)
            else:
                if not dry_run:
                    path.write_text(original_text, encoding="utf-8")
                    _log.warning(
                        "compact: restored original text for %s due to near-empty result "
                        "(%dB -> %dB, effective_before=%dB, ratio %.3f < %.2f)",
                        persona_stem,
                        before_bytes,
                        after_bytes,
                        effective_before,
                        ratio,
                        MIN_RATIO,
                    )
                raise ValueError(
                    f"Compaction produced near-empty result for {persona_stem}: "
                    f"{before_bytes}B -> {after_bytes}B "
                    f"(effective_before={effective_before}B, "
                    f"ratio {ratio:.3f} < {MIN_RATIO}) "
                    f"— aborting to prevent data loss"
                )

        # Date coverage post-check
        if observed_date_ranges:
            missed = _check_date_coverage(observed_date_ranges, new_text)
            if missed:
                missed_str = ", ".join(f"{s}..{e}" for s, e in missed)
                _log.warning(
                    "compact: post-check missed dates for %s: [%s]",
                    persona_stem,
                    missed_str,
                )
                warnings_list.append(f"post-check missed dates: [{missed_str}]")

        if not dry_run:
            path.write_text(new_text, encoding="utf-8")

        return CompactionResult(
            before_bytes=before_bytes,
            after_bytes=after_bytes,
            summary=f"compacted {persona_stem}: {before_bytes}B -> {after_bytes}B",
            warnings=warnings_list,
            promote_candidates=[],
        )
