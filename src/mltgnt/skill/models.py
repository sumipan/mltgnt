"""
mltgnt.skill.models — SkillMeta / SkillFile dataclass 定義。

設計: Issue #124 §6.1, Issue #1382 U1 (SKILL I/O 再設計 Phase 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArtifactSpec:
    """produces.artifacts の要素。本文外ファイル参照。"""

    path: str
    role: str = "primary"  # "primary" | "log" | "attachment"


@dataclass
class ProducesSpec:
    """スキルの出力契約（skill_io: v1 時）。"""

    content_type: str = "text/markdown"  # "text/markdown" | "text/plain"
    artifacts: list[ArtifactSpec] = field(default_factory=list)
    status_markers: list[str] = field(default_factory=list)


@dataclass
class ConsumesSpec:
    """パイプ参加時の入力契約。"""

    producer: str  # 上流スキル名（SkillMeta.name）
    content_type: str = "text/markdown"


@dataclass
class SkillRunResult:
    """runner の構造化戻り値（raw text からの昇格）。"""

    content: str
    exit_code: int = 0
    artifacts: list[ArtifactSpec] = field(default_factory=list)
    status_markers: list[str] = field(default_factory=list)


@dataclass
class SkillMeta:
    """discover 時にロードされるメタ情報（Progressive Disclosure）。"""

    name: str
    description: str
    argument_hint: str
    model: str | None
    path: Path
    triggers: list[str] = field(default_factory=list)
    skill_io: str = "legacy"  # "legacy" | "v1"
    input_schema: dict = field(default_factory=dict)  # JSON Schema object
    produces: ProducesSpec | None = None
    consumes: list[ConsumesSpec] = field(default_factory=list)


@dataclass
class SkillFile:
    """実行時にロードされる全文データ。"""

    meta: SkillMeta
    body: str
