"""
mltgnt.skill.models — SkillMeta / SkillFile dataclass definitions.

Design: Issue #124 §6.1, Issue #1382 U1 (SKILL I/O redesign Phase 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.interfaces.types import ChatInput


@dataclass
class ArtifactSpec:
    """Element of produces.artifacts. Out-of-body file reference."""

    path: str
    role: str = "primary"  # "primary" | "log" | "attachment"


@dataclass
class ProducesSpec:
    """Skill output contract (when skill_io: v1)."""

    content_type: str = "text/markdown"  # "text/markdown" | "text/plain"
    artifacts: list[ArtifactSpec] = field(default_factory=list)
    status_markers: list[str] = field(default_factory=list)


@dataclass
class ConsumesSpec:
    """Input contract when participating in a pipe."""

    producer: str  # upstream skill name (SkillMeta.name)
    content_type: str = "text/markdown"


class ExitStatus:
    SUCCESS = 0
    ALREADY_APPLIED = 1
    INVALID_STATE = 2
    CONTRACT_VIOLATION = 3
    USAGE_ERROR = 64


@dataclass
class SideEffectsSpec:
    """Side-effect declaration for skill runs (SKILL.md frontmatter side_effects)."""

    writes: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)
    mutates: list[str] = field(default_factory=list)
    conditional: list[str] = field(default_factory=list)


@dataclass
class SkillRunResult:
    """Return value of runner.run() (pre-execution + post-execution combined)."""

    chat_input: "ChatInput"
    expected_markers: list[str]
    skill_io: str
    content: str = ""
    exit_code: int = 0
    diagnostics: list[str] = field(default_factory=list)
    artifacts: list[ArtifactSpec] = field(default_factory=list)
    status_markers: list[str] = field(default_factory=list)
    produces: ProducesSpec | None = None


@dataclass
class SkillMatchResult:
    """Return value of matcher.match(). Includes match path and candidates."""

    decisive: SkillMeta | None
    candidates: list[SkillMeta]
    rationale: str  # "slash:<name>" | "literal:<name>" | "trigger:<keyword>" | "llm:<name>" | "none"
    arguments: str


class SkillLoadError(Exception):
    """Raised on skill load / Tool validation failure."""


@dataclass
class SkillMeta:
    """Metadata loaded at discover time (Progressive Disclosure)."""

    name: str
    description: str
    argument_hint: str
    model: str | None
    path: Path
    triggers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    skill_io: str = "legacy"  # "legacy" | "v1"
    input_schema: dict = field(default_factory=dict)  # JSON Schema object
    produces: ProducesSpec | None = None
    consumes: list[ConsumesSpec] = field(default_factory=list)
    side_effects: SideEffectsSpec | None = None
    knowledge_paths: list[Path] = field(default_factory=list)


@dataclass
class SkillFile:
    """Full-text data loaded at run time."""

    meta: SkillMeta
    body: str
