"""mltgnt.persona.compress

Generate a light block from a heavy block via LLM compression and persist it in the persona file.

Public API:
    compress_heavy_to_light(heavy_text, *, engine, model, timeout) -> str
    compute_block_hash(text)                                         -> str
    regenerate_light_block(persona_path, *, engine, model, timeout) -> RegenerationResult
    RegenerationResult                                               (dataclass)
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from mltgnt.config.language import JA, LanguagePack

logger = logging.getLogger(__name__)

LIGHT_BLOCK_MAX_CHARS = 1500


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegenerationResult:
    """Result of regenerating a light block."""

    persona_name: str
    old_hash: str
    new_hash: str
    light_text: str
    changed: bool

    @property
    def is_first_generation(self) -> bool:
        return self.old_hash == ""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_block_hash(text: str) -> str:
    """Return the sha256 hash of text.

    Normalize: strip surrounding whitespace and unify newlines to LF before hashing.

    Args:
        text: Text to hash

    Returns:
        sha256 hex digest string
    """
    normalized = text.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compress_heavy_to_light(
    heavy_text: str,
    *,
    engine: str = "claude",
    model: str | None = None,
    timeout: int = 120,
    pack: LanguagePack | None = None,
) -> str:
    """LLM-compress heavy-block text into a light-block summary.

    Args:
        heavy_text: Full heavy-block text (including H3 and below)
        engine: LLM engine name (default: "claude")
        model: Model name. Use engine default when None
        timeout: LLM call timeout seconds
        pack: Language pack; defaults to JA when None

    Returns:
        v2.1 text within 1500 chars (lead + required subsections)

    Raises:
        RuntimeError: When the LLM call fails or heavy_text is empty
    """
    from mltgnt.bridges.llm_adapter import call_llm as ghdag_llm_call

    if not heavy_text.strip():
        raise RuntimeError("heavy_text is empty. Provide text to compress.")

    prompt = (pack or JA).compress_prompt_template.format(heavy_text=heavy_text)

    kwargs: dict = {"engine": engine, "timeout": timeout}
    if model is not None:
        kwargs["model"] = model

    try:
        result = ghdag_llm_call(prompt, **kwargs)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    if not result.success:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"LLM returned ok=False: {stderr}")

    return (result.body or "").strip()


def regenerate_light_block(
    persona_path: Path,
    *,
    engine: str = "claude",
    model: str | None = None,
    timeout: int = 120,
    pack: LanguagePack | None = None,
) -> RegenerationResult:
    """Regenerate the light block from the persona heavy block and write it back.

    Flow:
    1. Read the persona file and split H2 blocks
    2. Record sha256 of the existing light block
    3. LLM-compress the heavy block into a new light block
    4. Validate the result as v2.1
    5. Overwrite the file with the new light block
    6. Compare sha256 and log a warning if changed

    Args:
        persona_path: Path to the persona file
        engine: LLM engine name
        model: Model name
        timeout: LLM call timeout seconds

    Returns:
        RegenerationResult

    Raises:
        # Japanese text intentionally kept for CJK processing test
        ValueError: Not v2 (missing ## 重量) or result not v2.1
        RuntimeError: LLM compression failed
    """
    from mltgnt.bridges.files_adapter import md_read, md_write

    md = md_read(persona_path.name, repo_root=persona_path.parent)
    fm_dict = md.frontmatter
    body = md.content

    blocks = _split_h2_blocks(body)

    # Japanese text intentionally kept for CJK processing test
    if "重量" not in blocks:
        raise ValueError(
            f"v2 形式ではありません: {persona_path.name} に '## 重量' ブロックが存在しません"
        )

    # Japanese text intentionally kept for CJK processing test
    heavy_text = blocks["重量"]
    existing_light = blocks.get("軽量", "")

    # First-time generation if light block is empty
    old_hash = "" if not existing_light.strip() else compute_block_hash(existing_light)

    # LLM compress
    new_light = compress_heavy_to_light(heavy_text, engine=engine, model=model, timeout=timeout, pack=pack)

    # v2.1 validation
    _validate_v21_light_block(new_light, pack=pack)

    new_hash = compute_block_hash(new_light)

    changed = old_hash != new_hash

    if changed and old_hash != "":
        logger.warning(
            "[compress] drift detected for %r: old_hash=%s new_hash=%s",
            persona_path.stem,
            old_hash,
            new_hash,
        )

    # Write back to file
    new_content = _rebuild_file(fm_dict, blocks, new_light)
    md_write(persona_path.name, new_content, repo_root=persona_path.parent)

    return RegenerationResult(
        persona_name=persona_path.stem,
        old_hash=old_hash,
        new_hash=new_hash,
        light_text=new_light,
        changed=changed,
    )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _validate_v21_light_block(text: str, pack: LanguagePack | None = None) -> None:
    """Validate that a generated light block conforms to v2.1.

    v2.1 requirements:
    - Lead text (one or more non-empty lines) before the first **
    - Required subsections present (from pack.v21_required_sections)
    - If pack.v21_example_section is present, a following > line is required

    Args:
        text: Text to validate
        pack: Language pack; defaults to JA when None

    Raises:
        ValueError: When format requirements are not met
    """
    _pack = pack or JA
    lines = text.strip().splitlines()

    # Lead check: non-empty line required before first ** heading
    first_bold_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("**"):
            first_bold_index = i
            break

    if first_bold_index is None:
        # If no ** headings at all, required-section check will catch it
        # Treat lead check as pass for now
        pass
    else:
        # Confirm a non-empty line exists before the first ** line
        lead_lines = [ln for ln in lines[:first_bold_index] if ln.strip()]
        if not lead_lines:
            raise ValueError(
                "v2.1 format error: missing lead text. Write a persona intro before the first bold heading (**)."
            )

    # Required subsection check
    for section in _pack.v21_required_sections:
        found = any(section in line for line in lines)
        if not found:
            section_name = section.strip("*")
            raise ValueError(
                f"v2.1 format error: required subsection {section} is missing."
                f" (Write as {section_name} — ...)"
            )

    # Example-speech check: if v21_example_section present, a later > line is required
    example_section = _pack.v21_example_section
    for i, line in enumerate(lines):
        if example_section in line:
            has_quote = False
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip().startswith("**") and example_section not in next_line:
                    break
                if next_line.strip().startswith("> ") or next_line.strip() == ">":
                    has_quote = True
                    break
            if not has_quote:
                raise ValueError(
                    f"v2.1 format error: {example_section} must be followed by a quote block (> line)."
                )
            break


def _split_h2_blocks(body: str) -> dict[str, str]:
    """Split body by H2 headings into {heading: text}.

    # Japanese text intentionally kept for CJK processing test
    v2 expects three blocks: "軽量", "重量", "参照".
    """
    blocks: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        m = re.match(r"^##\s+(.+)", line)
        if m:
            if current_key is not None:
                blocks[current_key] = "\n".join(current_lines).strip()
            current_key = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        blocks[current_key] = "\n".join(current_lines).strip()

    return blocks


def _rebuild_file(
    fm_dict: dict,
    blocks: dict[str, str],
    new_light: str,
) -> str:
    """Rebuild the whole file content replacing the light block with new_light.

    Reserialize frontmatter from fm_dict via yaml.dump.
    # Japanese text intentionally kept for CJK processing test
    Keep H2 order: light → heavy → reference (軽量→重量→参照).
    """
    if fm_dict:
        fm_text = yaml.dump(fm_dict, sort_keys=False, allow_unicode=True)
        frontmatter_section = f"---\n{fm_text}---\n"
    else:
        frontmatter_section = ""

    section_order = list(blocks.keys())
    new_sections: list[str] = []
    for key in section_order:
        # Japanese text intentionally kept for CJK processing test
        if key == "軽量":
            text = new_light
        else:
            text = blocks[key]
        if text:
            new_sections.append(f"## {key}\n\n{text}\n")
        else:
            new_sections.append(f"## {key}\n")

    new_body = "\n".join(new_sections)
    if frontmatter_section:
        return f"{frontmatter_section}\n{new_body}"
    return new_body
