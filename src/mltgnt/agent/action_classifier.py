"""mltgnt.agent.action_classifier — ツール副作用レベルの自動分類。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ActionClass(Enum):
    """ツール実行の副作用レベル。"""

    SAFE = "safe"
    NEEDS_REVIEW = "needs-review"
    DANGEROUS = "dangerous"


@dataclass
class _Rule:
    tool_name: str
    approved_count: int = 0
    rejected_count: int = 0


class ActionClassifier:
    """実行履歴とユーザーフィードバックから分類ルールを学習する。"""

    def __init__(self, rules_path: Path, *, threshold: int = 5) -> None:
        self._rules_path = rules_path
        self._threshold = threshold
        self._rules: dict[str, _Rule] = {}
        self._load()

    def classify(self, tool_name: str, tool_args: dict) -> ActionClass:
        rule = self._rules.get(tool_name)
        if rule is None:
            return ActionClass.NEEDS_REVIEW

        if rule.rejected_count > 0:
            return ActionClass.NEEDS_REVIEW

        if rule.approved_count >= self._threshold:
            return ActionClass.SAFE

        return ActionClass.NEEDS_REVIEW

    def record_feedback(self, tool_name: str, user_approved: bool) -> None:
        rule = self._rules.get(tool_name)
        if rule is None:
            rule = _Rule(tool_name=tool_name)
            self._rules[tool_name] = rule

        if user_approved:
            rule.approved_count += 1
            rule.rejected_count = 0
        else:
            rule.rejected_count += 1
            rule.approved_count = 0

        self._persist()

    def _load(self) -> None:
        if not self._rules_path.exists():
            return

        text = self._rules_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tool_name = data["tool_name"]
            self._rules[tool_name] = _Rule(
                tool_name=tool_name,
                approved_count=int(data.get("approved_count", 0)),
                rejected_count=int(data.get("rejected_count", 0)),
            )

    def _persist(self) -> None:
        self._rules_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "tool_name": rule.tool_name,
                    "approved_count": rule.approved_count,
                    "rejected_count": rule.rejected_count,
                },
                ensure_ascii=False,
            )
            for rule in self._rules.values()
        ]
        content = "\n".join(lines)
        if content:
            content += "\n"
        self._rules_path.write_text(content, encoding="utf-8")
