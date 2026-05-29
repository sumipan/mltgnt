"""
mltgnt.skill.lint — SKILL.md フロントマターの構造検証（V1–V9）。

設計: Issue #1383 U3
"""
from __future__ import annotations

from pathlib import Path


def lint_skill_meta(fm: dict, path: Path) -> list[str]:
    """フロントマター dict を V1–V9 で検証し、エラーメッセージのリストを返す。

    空リスト = 検証通過。
    """
    errors: list[str] = []

    # V1: description 非空
    if not fm.get("description"):
        errors.append("V1: description is required")

    # V2: triggers が list 型
    triggers = fm.get("triggers")
    if triggers is not None and not isinstance(triggers, list):
        errors.append("V2: triggers must be a list")

    # V3: name == ディレクトリ名
    name = fm.get("name") or path.parent.name
    if name != path.parent.name:
        errors.append(f"V3: name '{name}' does not match directory '{path.parent.name}'")

    # V4: skill_io ∈ {legacy, v1}
    skill_io = fm.get("skill_io", "legacy")
    if skill_io not in ("legacy", "v1"):
        errors.append(f"V4: skill_io must be 'legacy' or 'v1', got {skill_io!r}")

    # V5: skill_io: v1 → produces 必須
    if skill_io == "v1" and not fm.get("produces"):
        errors.append("V5: skill_io=v1 requires produces field")

    # V6–V7: produces 構造
    produces = fm.get("produces")
    if produces is not None:
        if isinstance(produces, dict):
            content_type = produces.get("content_type", "text/markdown")
            if not isinstance(content_type, str):
                errors.append(
                    f"V6: produces.content_type must be str, got {type(content_type).__name__}"
                )
            artifacts = produces.get("artifacts") or []
            if isinstance(artifacts, list):
                for i, artifact in enumerate(artifacts):
                    if not isinstance(artifact, dict):
                        errors.append(f"V7: produces.artifacts[{i}].path is required")
                    elif "path" not in artifact or not isinstance(artifact["path"], str):
                        errors.append(f"V7: produces.artifacts[{i}].path is required")
        # produces が dict 以外の場合は V6/V7 は lint 時点では触れず V5/V4 等に委譲

    # V8: consumes[*].producer 非空 str
    consumes = fm.get("consumes") or []
    if isinstance(consumes, list):
        for i, item in enumerate(consumes):
            if not isinstance(item, dict):
                errors.append(f"V8: consumes[{i}].producer must be non-empty str")
            else:
                producer = item.get("producer")
                if not isinstance(producer, str) or not producer:
                    errors.append(f"V8: consumes[{i}].producer must be non-empty str")

    # V9: input_schema が dict（v1 のみ。legacy は list 形式を許容）
    if skill_io == "v1":
        input_schema = fm.get("input_schema")
        if input_schema is not None and not isinstance(input_schema, dict):
            errors.append(f"V9: input_schema must be dict, got {type(input_schema).__name__}")

    return errors
