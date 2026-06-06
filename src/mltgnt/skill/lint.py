"""
mltgnt.skill.lint — SKILL.md フロントマターの構造検証（V1–V12, V13）。

設計: Issue #1383 U3, Issue #1832 (V10–V12 warning), Issue #1828 V13
"""
from __future__ import annotations

from pathlib import Path


_ALLOWED_MUTATES = frozenset({"config", "env", "git", "github", "process"})


def lint_skill_meta(fm: dict, path: Path) -> list[str]:
    """フロントマター dict を V1–V12, V13 で検証し、エラーメッセージのリストを返す。

    空リスト = 検証通過。warning レベル（V10–V12, V13 等）はメッセージ末尾 ``(warning)`` で区別する。
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
    if produces is not None and isinstance(produces, dict):
        content_type = produces.get("content_type", "text/markdown")
        if not isinstance(content_type, str):
            errors.append(
                f"V6: produces.content_type must be str, got {type(content_type).__name__}"
            )
        artifacts = produces.get("artifacts") or []
        if isinstance(artifacts, list):
            for i, artifact in enumerate(artifacts):
                if not isinstance(artifact, dict) or "path" not in artifact or not isinstance(artifact["path"], str):
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

    # V10–V12: side_effects（warning レベル。キー存在時のみ検証）
    if "side_effects" in fm:
        side_effects = fm["side_effects"]
        if not isinstance(side_effects, dict):
            errors.append("side_effects must be a mapping (warning)")
        else:
            if "writes" in side_effects:
                writes = side_effects["writes"]
                if not isinstance(writes, list) or not all(isinstance(w, str) for w in writes):
                    errors.append(
                        "V10: side_effects.writes must be a list of strings (warning)"
                    )
            if "network" in side_effects:
                network = side_effects["network"]
                if not isinstance(network, list) or not all(isinstance(n, str) for n in network):
                    errors.append(
                        "V11: side_effects.network must be a list of strings (warning)"
                    )
            if "mutates" in side_effects:
                mutates = side_effects["mutates"]
                if isinstance(mutates, list):
                    for value in mutates:
                        if value not in _ALLOWED_MUTATES:
                            errors.append(
                                "V12: side_effects.mutates values must be from "
                                "{config, env, git, github, process} (warning)"
                            )
                            break

    # V13: scripts/ あり + README.md なし → warning
    scripts_dir = path.parent / "scripts"
    if scripts_dir.is_dir() and not (path.parent / "README.md").is_file():
        errors.append("V13: skills with scripts/ should have README.md (warning)")

    return errors
