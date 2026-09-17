"""mltgnt.persona.registry

List personas in a directory and resolve names (including aliases).

- Return only leaf files (`<persona_dir>/<name>.md`)
- Exclude subdirectories (`<persona_dir>/<name>/`)
- sample persona stem (CJK via escapes) is excluded via EXCLUDE_STEMS
"""

from __future__ import annotations

from pathlib import Path

EXCLUDE_STEMS: frozenset[str] = frozenset({"\u30b5\u30f3\u30d7\u30eb"})


def resolve(name: str, persona_dir: Path) -> Path:
    """Accept a persona name or file path; return an absolute path.

    - Absolute paths or strings ending in `.md` become Path as-is
    - Otherwise complete to `<persona_dir>/<name>.md`
    - Does not resolve aliases (use resolve_with_alias for that)
    """
    p = Path(name)
    if p.is_absolute():
        return p
    if name.endswith(".md"):
        return (persona_dir / p).resolve()
    return persona_dir / f"{name}.md"


def resolve_with_alias(name: str, persona_dir: Path) -> Path:
    """Resolve a persona file path by name or alias.

    1. Return `<persona_dir>/<name>.md` if it exists
    2. Otherwise scan all persona aliases for a match

    Raises:
        FileNotFoundError: When neither name nor alias matches
    """
    # Direct name lookup first
    direct = resolve(name, persona_dir)
    if direct.exists():
        return direct

    # Alias scan (read frontmatter of all files)
    from mltgnt.bridges.files_adapter import md_read

    for p in sorted(persona_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".md":
            continue
        if p.stem in EXCLUDE_STEMS:
            continue
        try:
            md = md_read(p.name, repo_root=p.parent)
            persona_ns = md.frontmatter.get("persona") or {}
            if not isinstance(persona_ns, dict):
                continue
            aliases_raw = persona_ns.get("aliases") or []
            aliases = list(aliases_raw) if isinstance(aliases_raw, list) else []
            if name in aliases:
                return p
        except OSError:
            continue

    raise FileNotFoundError(
        f"Persona '{name}' not found (name/alias mismatch): {persona_dir}"
    )


def list_personas(persona_dir: Path) -> list[str]:
    """Return valid persona name stems.

    - Only `<persona_dir>/*.md` (exclude files in subdirs)
    - Exclude stems in EXCLUDE_STEMS
    - Return sorted by name
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
