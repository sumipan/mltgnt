"""
mltgnt.skill.loader — SKILL.md の glob 探索とフロントマターパース。

設計: Issue #124 §6.2
"""
from __future__ import annotations

import logging
from pathlib import Path

from mltgnt.bridges.files_adapter import md_read
from mltgnt.skill.lint import lint_skill_meta
from mltgnt.skill.models import (
    ArtifactSpec,
    ConsumesSpec,
    ProducesSpec,
    SkillFile,
    SkillMeta,
)

_log = logging.getLogger(__name__)


def _build_meta(fm: dict, path: Path) -> SkillMeta:
    """フロントマター dict から SkillMeta を構築する。"""
    name: str = fm.get("name") or path.parent.name
    description: str | None = fm.get("description")
    if not description:
        raise ValueError("description フィールドが必須です")
    argument_hint: str = fm.get("argument_hint") or ""
    model: str | None = fm.get("model") or None
    triggers_raw = fm.get("triggers")
    if triggers_raw is None:
        triggers: list[str] = []
    elif not isinstance(triggers_raw, list):
        raise ValueError(f"triggers フィールドはリストである必要があります: {triggers_raw!r}")
    else:
        triggers = [str(t) for t in triggers_raw]

    skill_io: str = fm.get("skill_io", "legacy")
    input_schema: dict = fm.get("input_schema") or {}

    produces_raw = fm.get("produces")
    produces: ProducesSpec | None = None
    if produces_raw is not None and isinstance(produces_raw, dict):
        artifacts_raw = produces_raw.get("artifacts") or []
        artifacts = [
            ArtifactSpec(
                path=a["path"],
                role=a.get("role", "primary"),
            )
            for a in artifacts_raw
            if isinstance(a, dict) and "path" in a
        ]
        produces = ProducesSpec(
            content_type=produces_raw.get("content_type", "text/markdown"),
            artifacts=artifacts,
            status_markers=produces_raw.get("status_markers") or [],
        )

    consumes_raw = fm.get("consumes") or []
    consumes: list[ConsumesSpec] = []
    if isinstance(consumes_raw, list):
        for c in consumes_raw:
            if isinstance(c, dict) and "producer" in c:
                consumes.append(
                    ConsumesSpec(
                        producer=c["producer"],
                        content_type=c.get("content_type", "text/markdown"),
                    )
                )

    return SkillMeta(
        name=name,
        description=str(description).strip(),
        argument_hint=argument_hint,
        model=model,
        path=path.resolve(),
        triggers=triggers,
        skill_io=skill_io,
        input_schema=input_schema,
        produces=produces,
        consumes=consumes,
    )


def discover(
    paths: list[Path],
    entry_file: str = "SKILL.md",
) -> dict[str, SkillMeta]:
    """
    指定パスから SKILL.md を再帰的に探索し、フロントマターのみパースする。

    戻り値: {skill_name: SkillMeta}。name 重複時は先勝ち（stderr に警告）。
    個別パースエラーは stderr 出力してスキップ。
    """
    result: dict[str, SkillMeta] = {}

    for base in paths:
        base = Path(base)
        if not base.exists():
            _log.warning("パスが存在しません: %s", base)
            continue

        for skill_file in sorted(base.rglob(entry_file)):
            # _ 始まりディレクトリ（内部・フォールバック用）はスキップ
            if any(p.startswith("_") for p in skill_file.relative_to(base).parts[:-1]):
                continue
            try:
                md = md_read(str(skill_file.relative_to(base)), repo_root=base)
                meta = _build_meta(md.frontmatter, skill_file)
            except Exception as e:
                _log.warning("パースエラー（スキップ）: %s: %s", skill_file, e)
                continue

            errors = lint_skill_meta(md.frontmatter, skill_file)
            unresolved_errors = [
                e
                for e in errors
                if e.startswith(("V3", "V4", "V5", "V6", "V7", "V8", "V9"))
            ]
            if unresolved_errors:
                for err in unresolved_errors:
                    _log.warning("skill lint failed: %s: %s", skill_file, err)
                continue

            for err in errors:
                _log.warning("skill lint warning: %s: %s", skill_file, err)

            if meta.name in result:
                _log.warning(
                    "スキル名重複（先勝ち）: '%s' (%s vs %s)",
                    meta.name,
                    result[meta.name].path,
                    meta.path,
                )
                continue

            result[meta.name] = meta

    return result


def load(meta: SkillMeta) -> SkillFile:
    """
    SkillMeta.path から全文を読み込み、SkillFile を返す。

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: フロントマターのパースに失敗した場合
    """
    path = meta.path
    if not path.exists():
        raise FileNotFoundError(f"SKILL.md が見つかりません: {path}")

    md = md_read(path.name, repo_root=path.parent)
    loaded_meta = _build_meta(md.frontmatter, path)
    return SkillFile(meta=loaded_meta, body=md.content)
