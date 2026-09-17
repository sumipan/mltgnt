"""
mltgnt.skill.loader — glob SKILL.md and parse frontmatter.

Design: Issue #124 §6.2
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from mltgnt.bridges.files_adapter import md_read
from mltgnt.skill.lint import lint_skill_meta
from mltgnt.skill.models import (
    ArtifactSpec,
    ConsumesSpec,
    ProducesSpec,
    SideEffectsSpec,
    SkillFile,
    SkillLoadError,
    SkillMeta,
)

_log = logging.getLogger(__name__)

_VIOLATION_ID_RE = re.compile(r"^(V\d+)")


def _write_unresolved_diagnosis(
    diagnostics_dir: Path,
    skill_name: str,
    skill_path: Path,
    base: Path,
    unresolved_errors: list[str],
) -> None:
    """Write lint-failure diagnostics JSON to `diagnostics_dir/{skill_name}.json`."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for msg in unresolved_errors:
        m = _VIOLATION_ID_RE.match(msg)
        errors.append({"id": m.group(1) if m else "", "message": msg})
    payload = {
        "skill_name": skill_name,
        "path": str(skill_path.relative_to(base)),
        "errors": errors,
    }
    (diagnostics_dir / f"{skill_name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _clear_unresolved_diagnosis(diagnostics_dir: Path, skill_name: str) -> None:
    """Delete diagnostics JSON for a resolved skill."""
    diag = diagnostics_dir / f"{skill_name}.json"
    if diag.is_file():
        diag.unlink()


def build_meta(fm: dict, path: Path) -> SkillMeta:
    """Build SkillMeta from a frontmatter dict (public API)."""
    name: str = fm.get("name") or path.parent.name
    description: str | None = fm.get("description")
    if not description:
        raise ValueError("description field is required")
    argument_hint: str = fm.get("argument_hint") or ""
    model: str | None = fm.get("model") or None
    triggers_raw = fm.get("triggers")
    if triggers_raw is None:
        triggers: list[str] = []
    elif not isinstance(triggers_raw, list):
        raise ValueError(f"triggers field must be a list: {triggers_raw!r}")
    else:
        triggers = [str(t) for t in triggers_raw]

    tools_raw = fm.get("tools")
    if tools_raw is None:
        tools: list[str] = []
    elif not isinstance(tools_raw, list):
        raise ValueError(f"tools field must be a list: {tools_raw!r}")
    else:
        tools = [str(t) for t in tools_raw]

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

    se_raw = fm.get("side_effects")
    side_effects: SideEffectsSpec | None = None
    if isinstance(se_raw, dict):
        side_effects = SideEffectsSpec(
            writes=se_raw.get("writes") or [],
            network=se_raw.get("network") or [],
            mutates=se_raw.get("mutates") or [],
            conditional=se_raw.get("conditional") or [],
        )

    return SkillMeta(
        name=name,
        description=str(description).strip(),
        argument_hint=argument_hint,
        model=model,
        path=path.resolve(),
        triggers=triggers,
        tools=tools,
        skill_io=skill_io,
        input_schema=input_schema,
        produces=produces,
        consumes=consumes,
        side_effects=side_effects,
    )


_build_meta = build_meta  # backward-compat alias


def discover(
    paths: list[Path],
    entry_file: str = "SKILL.md",
    *,
    diagnostics_dir: Path | None = None,
) -> dict[str, SkillMeta]:
    """
    Recursively find SKILL.md under the given paths; parse frontmatter only.

    Returns: {skill_name: SkillMeta}. First wins on name collision (warn on stderr).
    Skip individual parse errors after logging to stderr.

    diagnostics_dir:
        None (default): do not write diagnostics JSON.
        When set, write lint failures to ``diagnostics_dir/{name}.json``
        and delete stale JSON on pass.
    """
    result: dict[str, SkillMeta] = {}

    for base in paths:
        base = Path(base)
        if not base.exists():
            _log.warning("Path does not exist: %s", base)
            continue

        for skill_file in sorted(base.rglob(entry_file)):
            # Skip directories starting with _ (internal / fallback)
            if any(p.startswith("_") for p in skill_file.relative_to(base).parts[:-1]):
                continue
            try:
                md = md_read(str(skill_file.relative_to(base)), repo_root=base)
                meta = build_meta(md.frontmatter, skill_file)
            except Exception as e:
                _log.warning("Parse error (skip): %s: %s", skill_file, e)
                continue

            errors = lint_skill_meta(md.frontmatter, skill_file)
            unresolved_errors = [
                e
                for e in errors
                if e.startswith(
                    ("V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V14")
                )
            ]
            if unresolved_errors:
                for err in unresolved_errors:
                    _log.warning("skill lint failed: %s: %s", skill_file, err)
                if diagnostics_dir is not None:
                    _write_unresolved_diagnosis(
                        diagnostics_dir,
                        meta.name,
                        skill_file,
                        base,
                        unresolved_errors,
                    )
                continue

            if diagnostics_dir is not None:
                _clear_unresolved_diagnosis(diagnostics_dir, meta.name)

            for err in errors:
                _log.warning("skill lint warning: %s: %s", skill_file, err)

            if meta.name in result:
                _log.warning(
                    "Duplicate skill name (first wins): '%s' (%s vs %s)",
                    meta.name,
                    result[meta.name].path,
                    meta.path,
                )
                continue

            skill_dir = skill_file.parent
            knowledge_paths: list[Path] = []
            single = skill_dir / "knowledge.md"
            if single.is_file():
                knowledge_paths.append(single)
            subdir = skill_dir / "knowledge"
            if subdir.is_dir():
                knowledge_paths.extend(sorted(subdir.glob("*.md")))
            meta.knowledge_paths = knowledge_paths

            result[meta.name] = meta

    return result


def _get_available_tools(tools_path: Path) -> set[str]:
    """Call `ghdag tools list --path <path> --json` via subprocess; return Tool name set.

    Raises:
        SkillLoadError: On subprocess failure or JSON parse error
    """
    try:
        result = subprocess.run(
            ["ghdag", "tools", "list", "--path", str(tools_path), "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as e:
        raise SkillLoadError(f"ghdag tools list timed out: {tools_path}") from e

    if result.returncode != 0:
        raise SkillLoadError(
            f"ghdag tools list failed (exit {result.returncode}): {result.stderr}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise SkillLoadError(f"Failed to parse ghdag tools list JSON: {e}") from e

    tools_list = data.get("tools", [])
    return {
        t["name"]
        for t in tools_list
        if isinstance(t, dict) and "name" in t
    }


def validate_tool_refs(
    skills: dict[str, SkillMeta],
    tools_path: Path,
) -> None:
    """Cross-check each skill tools field against available Tools.

    - Skip skills with empty tools
    - Collect all unknown-Tool errors and raise one SkillLoadError
    - Error form: "skill '<name>': unknown Tool ['x', 'y']"

    Raises:
        SkillLoadError: On unknown Tool refs or ghdag tools list failure
    """
    skills_with_tools = {name: meta for name, meta in skills.items() if meta.tools}
    if not skills_with_tools:
        return

    available = _get_available_tools(tools_path)
    errors: list[str] = []
    for name, meta in skills_with_tools.items():
        unknown = [t for t in meta.tools if t not in available]
        if unknown:
            errors.append(f"skill '{name}': unknown Tool {unknown!r}")
    if errors:
        raise SkillLoadError("\n".join(errors))


def load(meta: SkillMeta) -> SkillFile:
    """
    Read the full text from SkillMeta.path and return SkillFile.

    Raises:
        FileNotFoundError: File missing
        ValueError: Frontmatter parse failed
    """
    path = meta.path
    if not path.exists():
        raise FileNotFoundError(f"SKILL.md not found: {path}")

    md = md_read(path.name, repo_root=path.parent)
    loaded_meta = build_meta(md.frontmatter, path)
    return SkillFile(meta=loaded_meta, body=md.content)
