"""
mltgnt.skill.loader — SKILL.md の glob 探索とフロントマターパース。

設計: Issue #124 §6.2
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from mltgnt.skill.models import SkillFile, SkillMeta

_FRONTMATTER_DELIM = "---"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """テキストを frontmatter dict と body に分割する。"""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        raise ValueError("frontmatter が存在しません（--- で始まる行が必要）")

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("frontmatter の終端 --- が見つかりません")

    fm_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1:])
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"frontmatter の YAML パースに失敗しました: {e}") from e

    return fm, body


def _build_meta(fm: dict, path: Path) -> SkillMeta:
    """フロントマター dict から SkillMeta を構築する。"""
    name: str = fm.get("name") or path.parent.name
    description: str | None = fm.get("description")
    if not description:
        raise ValueError("description フィールドが必須です")
    argument_hint: str = fm.get("argument_hint") or ""
    model: str | None = fm.get("model") or None
    return SkillMeta(
        name=name,
        description=str(description).strip(),
        argument_hint=argument_hint,
        model=model,
        path=path.resolve(),
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
            print(f"[skill.loader] WARNING: パスが存在しません: {base}", file=sys.stderr)
            continue

        for skill_file in sorted(base.rglob(entry_file)):
            try:
                text = skill_file.read_text(encoding="utf-8")
                fm, _ = _parse_frontmatter(text)
                meta = _build_meta(fm, skill_file)
            except Exception as e:
                print(
                    f"[skill.loader] パースエラー（スキップ）: {skill_file}: {e}",
                    file=sys.stderr,
                )
                continue

            if meta.name in result:
                print(
                    f"[skill.loader] WARNING: スキル名重複（先勝ち）: '{meta.name}' "
                    f"({result[meta.name].path} vs {meta.path})",
                    file=sys.stderr,
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

    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    # Re-validate meta (may differ from cached discover result)
    loaded_meta = _build_meta(fm, path)
    return SkillFile(meta=loaded_meta, body=body)
