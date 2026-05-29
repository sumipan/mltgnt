from __future__ import annotations

from mltgnt.improvement.loop import CycleResult


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_cycle_report(result: CycleResult) -> str:
    sections: list[str] = [
        "# サマリ",
        "",
        f"- 対象期間: {result.period_start} 〜 {result.period_end}",
        f"- 検出パターン数: {len(result.patterns)}",
        f"- 提案数: {len(result.proposals)}",
        "",
    ]

    if not result.patterns and not result.proposals:
        sections.append("対象期間に失敗パターンは検出されませんでした")
        return "\n".join(sections)

    sections.extend(["# 失敗パターン一覧", ""])
    if result.patterns:
        pattern_rows = [
            [
                pattern.category,
                str(pattern.count),
                ", ".join(pattern.example_correlation_ids),
            ]
            for pattern in result.patterns
        ]
        sections.append(
            _markdown_table(
                ["category", "count", "example_correlation_ids"],
                pattern_rows,
            )
        )
    else:
        sections.append("（なし）")

    sections.extend(["", "# 改善提案一覧", ""])
    if result.proposals:
        proposal_rows = [
            [
                proposal.target_type,
                proposal.target_name,
                proposal.action,
                proposal.description,
                str(proposal.confidence),
            ]
            for proposal in result.proposals
        ]
        sections.append(
            _markdown_table(
                [
                    "target_type",
                    "target_name",
                    "action",
                    "description",
                    "confidence",
                ],
                proposal_rows,
            )
        )
    else:
        sections.append("（なし）")

    return "\n".join(sections)
