from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mltgnt.improvement.analyzer import FailurePattern


@dataclass
class ImprovementProposal:
    proposal_id: str
    target_type: str
    target_name: str
    action: str
    description: str
    diff_preview: str
    confidence: float
    source_patterns: list[str]


def _persona_exists(persona_dir: Path, persona_name: str) -> bool:
    return (
        (persona_dir / f"{persona_name}.md").exists()
        or (persona_dir / persona_name / "PERSONA.md").exists()
        or (persona_dir / persona_name).exists()
    )


def _skill_exists(skills_dir: Path, skill_name: str) -> bool:
    return (
        (skills_dir / skill_name / "SKILL.md").exists()
        or (skills_dir / f"{skill_name}.md").exists()
        or (skills_dir / skill_name).exists()
    )


def generate_proposals(
    patterns: list[FailurePattern],
    persona_dir: Path,
    skills_dir: Path,
    *,
    min_count: int = 3,
    min_confidence: float = 0.5,
) -> list[ImprovementProposal]:
    """失敗パターンから改善提案を生成する。min_count 未満のパターンは無視する。"""
    proposals: list[ImprovementProposal] = []

    for pattern in patterns:
        if pattern.count < min_count:
            continue

        if pattern.category == "triage_error":
            target_name = pattern.affected_persona
            if not target_name or not _persona_exists(persona_dir, target_name):
                continue
            proposal = ImprovementProposal(
                proposal_id=f"proposal:{pattern.pattern_id}",
                target_type="persona",
                target_name=target_name,
                action="adjust_section",
                description="軽量セクションにトリアージヒントを追加する提案",
                diff_preview=f"+ ## 軽量\n+ - 追加トリアージヒント ({pattern.pattern_id})",
                confidence=0.8,
                source_patterns=[pattern.pattern_id],
            )
        elif pattern.category == "skill_mismatch":
            target_name = pattern.affected_skill
            if not target_name or not _skill_exists(skills_dir, target_name):
                continue
            proposal = ImprovementProposal(
                proposal_id=f"proposal:{pattern.pattern_id}",
                target_type="trigger",
                target_name=target_name,
                action="add_trigger",
                description="不足キーワードを trigger に追加する提案",
                diff_preview=f"+ triggers:\n+   - {target_name} 関連キーワード",
                confidence=0.85,
                source_patterns=[pattern.pattern_id],
            )
        elif pattern.category == "timeout":
            target_name = pattern.affected_persona
            if not target_name or not _persona_exists(persona_dir, target_name):
                continue
            proposal = ImprovementProposal(
                proposal_id=f"proposal:{pattern.pattern_id}",
                target_type="persona",
                target_name=target_name,
                action="add_rule",
                description="タイムアウト回避の応答長制限ルールを追加する提案",
                diff_preview="+ - 応答は 8 行以内を目安にする",
                confidence=0.7,
                source_patterns=[pattern.pattern_id],
            )
        elif pattern.category == "persona_quality":
            target_name = pattern.affected_persona
            if not target_name or not _persona_exists(persona_dir, target_name):
                continue
            proposal = ImprovementProposal(
                proposal_id=f"proposal:{pattern.pattern_id}",
                target_type="persona",
                target_name=target_name,
                action="adjust_section",
                description="persona 品質改善のため該当セクションを見直す提案",
                diff_preview=f"~ persona section tune: {target_name}",
                confidence=0.55,
                source_patterns=[pattern.pattern_id],
            )
        else:
            continue

        if proposal.confidence >= min_confidence and proposal.diff_preview.strip():
            proposals.append(proposal)

    proposals.sort(key=lambda item: (-item.confidence, item.proposal_id))
    return proposals
