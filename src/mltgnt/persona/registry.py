"""mltgnt.persona.registry

ペルソナディレクトリの一覧取得・名前解決（エイリアス含む）。

- 最終ファイル（`<persona_dir>/<名前>.md`）のみを返す
- サブディレクトリ（`<persona_dir>/<名前>/`）は除外
- `サンプル.md` など除外対象を EXCLUDE_STEMS で設定可能
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

EXCLUDE_STEMS: frozenset[str] = frozenset({"サンプル"})


def resolve(name: str, persona_dir: Path) -> Path:
    """ペルソナ名またはファイルパスを受け取り、絶対パスを返す。

    - 絶対パスまたは `.md` が付いた文字列はそのまま Path に変換
    - それ以外は `<persona_dir>/<name>.md` に補完する
    - エイリアス解決は含まない（エイリアス解決は resolve_with_alias を使う）
    """
    p = Path(name)
    if p.is_absolute():
        return p
    if name.endswith(".md"):
        return (persona_dir / p).resolve()
    return persona_dir / f"{name}.md"


def resolve_with_alias(name: str, persona_dir: Path) -> Path:
    """名前またはエイリアスでペルソナファイルのパスを解決する。

    1. `<persona_dir>/<name>.md` が存在すれば返す
    2. 存在しなければ全ペルソナの aliases を走査して一致するものを探す

    Raises:
        FileNotFoundError: 名前・エイリアスいずれにも一致しないとき
    """
    # まず直接名前検索
    direct = resolve(name, persona_dir)
    if direct.exists():
        return direct

    # エイリアス検索（全ファイルの frontmatter を読む）
    from mltgnt.persona.frontmatter import split_yaml_frontmatter

    for p in sorted(persona_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".md":
            continue
        if p.stem in EXCLUDE_STEMS:
            continue
        try:
            raw = p.read_text(encoding="utf-8")
            meta, _ = split_yaml_frontmatter(raw)
            if meta is None:
                continue
            persona_ns = meta.get("persona") or {}
            if not isinstance(persona_ns, dict):
                continue
            aliases_raw = persona_ns.get("aliases") or []
            aliases = list(aliases_raw) if isinstance(aliases_raw, list) else []
            if name in aliases:
                return p
        except (OSError, UnicodeDecodeError):
            continue

    raise FileNotFoundError(
        f"ペルソナ '{name}' が見つかりません（名前・エイリアスいずれも不一致）: {persona_dir}"
    )


def list_personas(persona_dir: Path) -> list[str]:
    """有効なペルソナ名の一覧（stem）を返す。

    - `<persona_dir>/*.md` のファイルのみ（サブディレクトリ内ファイルは除外）
    - EXCLUDE_STEMS に含まれる stem は除外
    - 名前順でソートして返す
    """
    if not persona_dir.is_dir():
        return []

    stems: list[str] = []
    for p in persona_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".md":
            continue
        if p.stem in EXCLUDE_STEMS:
            continue
        stems.append(p.stem)

    return sorted(stems)
