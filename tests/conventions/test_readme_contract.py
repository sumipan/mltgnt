"""README.md must stay aligned with the installed package (Issue #6025).

Verifies release metadata, section structure, Quick Start, CLI, public API,
architecture, configuration, errors, and v0.135.x hub / WebChat contracts.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from argparse import ArgumentParser
from collections.abc import Iterator
from dataclasses import MISSING, Field, fields
from pathlib import Path
from typing import Any, get_type_hints

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
SRC_ROOT = REPO_ROOT / "src" / "mltgnt"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

REQUIRED_SECTIONS = (
    "Title / Positioning",
    "Status",
    "Not",
    "Installation",
    "Quick Start",
    "CLI Reference",
    "Public API",
    "Architecture",
    "Configuration",
    "Error Reference",
    "Public API Stability",
    "Deprecated API",
    "License",
)

CLI_COMMANDS = (
    ("run",),
    ("memory", "dream", "show"),
    ("memory", "dream", "forget"),
    ("hub", "serve"),
)

REMOVED_README_TOKENS = (
    "resolve_responding_persona",
    "PERSONA_SECTION_ALIASES",
    "from mltgnt.config.language import JA",
    "_thread_queue_config",
    "_read_state",
    "_write_post_content",
    "_build_meta",
)

V0135_README_MARKERS = (
    "header_meters_url",
    "too_fast",
    "HubClient.last_error",
    "UrllibTransport.last_error",
    "http:<status>:<reason>",
)


def _readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _section_slice(text: str, heading: str, *, stop_before: tuple[str, ...] = ()) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return ""
    start = match.end()
    rest = text[start:]
    stops = [rest.find(f"\n## {name}\n") for name in stop_before if f"\n## {name}\n" in rest]
    end = min(stops) if stops else len(rest)
    return rest[:end]


def _heading_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            title = line[3:].strip()
            counts[title] = counts.get(title, 0) + 1
    return counts


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _ghdag_pin(version: str) -> str:
    for dep in _pyproject()["project"]["dependencies"]:
        if "ghdag" in dep:
            match = re.search(r"@v([0-9.]+)", dep)
            if match:
                return f"v{match.group(1)}"
    raise AssertionError("ghdag dependency pin not found in pyproject.toml")


def _install_urls(text: str) -> list[str]:
    return re.findall(r"git\+https://github\.com/sumipan/mltgnt\.git@v[0-9.]+", text)


def _collect_all_exports(src_root: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        names.add(elt.value)
    return names


def _tokens_from_table_cell(cell: str) -> Iterator[str]:
    for part in re.split(r"[,/]", cell):
        token = part.strip().strip("`").strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
            yield token


def _table_claimed_api_names(text: str) -> set[str]:
    """Symbols explicitly listed as API rows (not prose examples)."""
    block = _section_slice(text, "Public API", stop_before=("Protocols / Extension Points",))
    error_block = _section_slice(text, "Error Reference", stop_before=("Public API Stability",))
    kind_markers = {
        "function",
        "class",
        "constant",
        "protocol",
        "frozen dataclass",
        "dataclass",
        "exception",
        "typeddict",
        "str",
        "mapping",
    }
    names: set[str] = set()
    supplement = _section_slice(block, "Submodule export index (`__all__` supplement)", stop_before=())
    if supplement:
        names.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", supplement))
    for section in (block, error_block):
        for line in section.splitlines():
            if not line.startswith("|") or line.startswith("|--"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            if cells[0] in ("Name", "Exception", "Deprecated", "Purpose"):
                continue
            second = cells[1].strip("` ").lower()
            if cells[0] == "Names" and len(cells) > 1 and cells[1].strip() != "Purpose":
                for token in _tokens_from_table_cell(cells[1]):
                    names.add(token)
                continue
            if second.split()[0] in kind_markers or second in kind_markers:
                for token in _tokens_from_table_cell(cells[0]):
                    names.add(token)
    return names


def _parse_readme_symbol_names(text: str) -> set[str]:
    block = _section_slice(text, "Public API", stop_before=("Protocols / Extension Points",))
    error_block = _section_slice(text, "Error Reference", stop_before=("Public API Stability",))
    names: set[str] = set()
    for section in (block, error_block):
        names.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", section))
        for line in section.splitlines():
            if not line.startswith("|") or line.startswith("|--"):
                continue
            if line.startswith("| Name |") or line.startswith("| Exception |"):
                continue
            for cell in line.strip("|").split("|"):
                for token in _tokens_from_table_cell(cell):
                    names.add(token)
    return names


def _public_api_table_names(text: str) -> set[str]:
    exports = _collect_all_exports(SRC_ROOT)
    return _parse_readme_symbol_names(text) & exports


def _architecture_module_rows(text: str) -> dict[str, str]:
    block = _section_slice(text, "Architecture", stop_before=("Configuration",))
    modules: dict[str, str] = {}
    in_modules = False
    for line in block.splitlines():
        if line.startswith("### Modules"):
            in_modules = True
            continue
        if in_modules and line.startswith("### "):
            break
        if not in_modules or not line.startswith("|"):
            continue
        if line.startswith("|--") or "Module |" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        mod = cells[0].strip("` ")
        if mod.startswith("mltgnt."):
            modules[mod] = cells[1]
    return modules


def _discovered_python_modules(src_root: Path) -> set[str]:
    mods: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root.parent).with_suffix("")
        mod = ".".join(rel.parts)
        mods.add(mod)
    return mods


def _direct_env_vars(src_root: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("getenv", "environ")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            if (
                isinstance(func, ast.Name)
                and func.id == "getenv"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
    return keys


def _public_config_dataclasses() -> tuple[type[Any], ...]:
    from mltgnt.config import (
        ConversationConfig,
        MemoryConfig,
        PersonaConfig,
        SchedulerConfig,
        SkillConfig,
    )
    from mltgnt.media._core.config import MediaConfig
    from mltgnt.media.hub.config import HubMediaConfig
    from mltgnt.media.hub.server import HubServerConfig
    from mltgnt.media.slack.config import SlackMediaConfig
    from mltgnt.media.webchat.config import WebChatMediaConfig, WebPushConfig

    return (
        PersonaConfig,
        MemoryConfig,
        SchedulerConfig,
        SkillConfig,
        ConversationConfig,
        MediaConfig,
        SlackMediaConfig,
        WebChatMediaConfig,
        WebPushConfig,
        HubMediaConfig,
        HubServerConfig,
    )


def _field_type_tokens(field: Field[Any], cls: type[Any]) -> set[str]:
    hints = get_type_hints(cls)
    annotation = hints.get(field.name, field.type)
    text = str(annotation)
    tokens: set[str] = set()
    for name in (
        "Path",
        "str",
        "float",
        "bool",
        "int",
        "tuple",
        "frozenset",
        "dict",
        "list",
        "LanguagePack",
        "GuardConfig",
        "WebPushConfig",
        "Mapping",
    ):
        if name in text:
            tokens.add(name)
    if not tokens and text not in {"typing.Any", "Any"}:
        tokens.add(text.replace("typing.", ""))
    return tokens


def _field_default_token(field: Field[Any]) -> str | None:
    if field.default is not MISSING:
        value = field.default
    elif field.default_factory is not MISSING:
        value = field.default_factory()
    else:
        return None
    if isinstance(value, Path):
        return "Path"
    if isinstance(value, str):
        return f'"{value}"' if value else '""'
    if isinstance(value, float):
        return str(value).rstrip("0").rstrip(".") if value == int(value) else str(value)
    return repr(value)


def _field_is_required(field: Field[Any]) -> bool:
    return field.default is MISSING and field.default_factory is MISSING


def _config_field_names() -> set[str]:
    names: set[str] = set()
    for cls in _public_config_dataclasses():
        for field in fields(cls):
            names.add(field.name)
            if not _field_is_required(field):
                token = _field_default_token(field)
                assert token is not None, f"missing default metadata for {cls.__name__}.{field.name}"
            type_tokens = _field_type_tokens(field, cls)
            assert type_tokens, f"missing type metadata for {cls.__name__}.{field.name}"
    return names


def _assert_readme_config_field(block: str, field: Field[Any], cls: type[Any]) -> None:
    assert field.name in block, f"undocumented config field {field.name}"
    type_tokens = _field_type_tokens(field, cls)
    default_token = None if _field_is_required(field) else _field_default_token(field)
    type_ok = any(token in block for token in type_tokens)
    default_ok = bool(default_token and default_token in block)
    assert type_ok or default_ok, (
        f"README missing type or default for {cls.__name__}.{field.name}: "
        f"types={sorted(type_tokens)} default={default_token!r}"
    )


def _exception_class_names(src_root: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _ast_class_inherits_exception(node):
                continue
            names.add(node.name)
    return names


def _ast_class_inherits_exception(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in {
            "Exception",
            "ValueError",
            "TypeError",
            "RuntimeError",
            "MltgntError",
        }:
            return True
        if isinstance(base, ast.Attribute) and base.attr in {
            "Exception",
            "ValueError",
            "TypeError",
            "RuntimeError",
        }:
            return True
    return False


def _raised_builtin_exception_names(src_root: Path) -> set[str]:
    documented = {"ValueError", "TypeError", "RuntimeError", "ImportError"}
    found: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            exc = node.exc
            if exc is None:
                continue
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Name) and exc.id in documented:
                found.add(exc.id)
    return found


def _public_exception_names(src_root: Path) -> set[str]:
    return _exception_class_names(src_root) | _raised_builtin_exception_names(src_root)


def _error_table_names(text: str) -> set[str]:
    block = _section_slice(text, "Error Reference", stop_before=("Public API Stability",))
    names: set[str] = set()
    for line in block.splitlines():
        if not line.startswith("|") or line.startswith("|--"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0].strip("` ")
        if first in ("Exception", "Name"):
            continue
        if re.match(r"^[A-Z][A-Za-z0-9_]*$", first):
            names.add(first)
    return names


def _cli_parser_for_command(
    root: ArgumentParser,
    cmd: tuple[str, ...],
) -> ArgumentParser:
    subparsers = [
        action
        for action in root._actions
        if getattr(action, "choices", None) is not None and action.dest == "subcommand"
    ]
    if not subparsers:
        raise AssertionError("CLI root parser has no subcommands")
    parser = subparsers[0].choices[cmd[0]]
    for part in cmd[1:]:
        nested = [
            action
            for action in parser._actions
            if getattr(action, "choices", None) is not None and part in action.choices
        ]
        if not nested:
            raise AssertionError(f"CLI subcommand path not found: {cmd}")
        parser = nested[0].choices[part]
    return parser


def _cli_argument_spec(parser: ArgumentParser) -> dict[str, set[str]]:
    options: set[str] = set()
    args: set[str] = set()
    for action in parser._actions:
        if action.dest in {"help"}:
            continue
        if action.option_strings:
            long_opts = [opt for opt in action.option_strings if opt.startswith("--")]
            if long_opts:
                options.add(long_opts[0])
            continue
        args.add(action.dest)
    if options and not args:
        return {"options": options}
    return {"args": args | options}


def _function_exit_codes(path: Path, function_names: set[str]) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    codes: set[int] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in function_names:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Return)
                and isinstance(inner.value, ast.Constant)
                and isinstance(inner.value.value, int)
            ):
                codes.add(inner.value.value)
            if isinstance(inner, ast.Raise) and isinstance(inner.exc, ast.Call):
                func = inner.exc.func
                if (
                    isinstance(func, ast.Name)
                    and func.id == "SystemExit"
                    and inner.exc.args
                    and isinstance(inner.exc.args[0], ast.Constant)
                    and isinstance(inner.exc.args[0].value, int)
                ):
                    codes.add(inner.exc.args[0].value)
    return codes


def _cli_exit_codes_from_sources(cmd: tuple[str, ...]) -> set[int]:
    cli_root = REPO_ROOT / "src" / "mltgnt" / "cli"
    if cmd == ("run",):
        return _function_exit_codes(cli_root / "main.py", {"main"})
    if cmd == ("memory", "dream", "show"):
        return _function_exit_codes(cli_root / "memory.py", {"_show"})
    if cmd == ("memory", "dream", "forget"):
        return _function_exit_codes(cli_root / "memory.py", {"_forget"})
    if cmd == ("hub", "serve"):
        return _function_exit_codes(cli_root / "hub.py", {"_serve"})
    raise AssertionError(f"unsupported CLI command inventory: {cmd}")


def _cli_inventory_from_sources() -> dict[tuple[str, ...], dict[str, Any]]:
    from mltgnt.cli import hub as hub_module
    from mltgnt.cli import memory as memory_module

    parser = ArgumentParser(prog="mltgnt")
    sub = parser.add_subparsers(dest="subcommand")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--components", required=True)
    run_parser.add_argument("--pid-file", default="/tmp/mltgnt_daemon.pid")
    memory_module.register(sub)
    hub_module.register(sub)

    inventory: dict[tuple[str, ...], dict[str, Any]] = {}
    for cmd in CLI_COMMANDS:
        leaf = _cli_parser_for_command(parser, cmd)
        spec = _cli_argument_spec(leaf)
        spec["exit_codes"] = _cli_exit_codes_from_sources(cmd)
        inventory[cmd] = spec
    return inventory


def _quick_start_python_block(text: str) -> str:
    block = _section_slice(text, "Quick Start", stop_before=("CLI Reference",))
    lines = block.splitlines()
    collecting = False
    buf: list[str] = []
    for line in lines:
        if line.strip() == "```python":
            collecting = True
            buf = []
            continue
        if collecting and line.strip() == "```":
            return "\n".join(buf)
        if collecting:
            buf.append(line)
    raise AssertionError("Quick Start python fence not found")


def test_readme_release_metadata_matches_pyproject() -> None:
    """Release badge, install URLs, ghdag pin, Python floor, and license match pyproject."""
    text = _readme_text()
    project = _pyproject()["project"]
    version = project["version"]
    tag = f"v{version}"
    ghdag_tag = _ghdag_pin(version)

    assert f"`{version}`" in text or version in text
    assert tag in text
    assert ghdag_tag in text
    assert project["requires-python"] in text
    assert "MIT" in text

    urls = _install_urls(text)
    assert urls, "no mltgnt git install URLs in README"
    assert all(url.endswith(f"@{tag}") for url in urls)

    install_block = _section_slice(text, "Installation", stop_before=("Quick Start",))
    assert ghdag_tag in install_block


def test_readme_required_sections_are_present_once() -> None:
    """OSS quality headings appear exactly once."""
    counts = _heading_counts(_readme_text())
    missing = [name for name in REQUIRED_SECTIONS if counts.get(name, 0) != 1]
    assert missing == []


def test_readme_quick_start_runs_offline() -> None:
    """Quick Start example runs without network or external services."""
    code = _quick_start_python_block(_readme_text())
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_readme_cli_inventory_matches_argparse_sources() -> None:
    """Documented CLI commands match argparse sources and list exits."""
    text = _readme_text()
    cli_block = _section_slice(text, "CLI Reference", stop_before=("Public API",))
    inventory = _cli_inventory_from_sources()

    for cmd in CLI_COMMANDS:
        label = "mltgnt " + " ".join(cmd)
        assert label in cli_block or f"`{label}`" in cli_block

    for cmd, spec in inventory.items():
        section_title = "### `" + " ".join(("mltgnt", *cmd)) + "`"
        assert section_title in cli_block, f"missing CLI section for {cmd}"
        chunk = cli_block.split(section_title, 1)[1].split("### ", 1)[0]
        if "options" in spec:
            for opt in spec["options"]:
                assert opt in chunk
        if "args" in spec:
            for arg in spec["args"]:
                assert arg.strip("-") in chunk or arg in chunk
        for code in spec["exit_codes"]:
            assert f"`{code}`" in chunk


def test_readme_public_api_inventory_covers_exports() -> None:
    """Every __all__ export appears in Public API tables and no stray names remain."""
    exports = _collect_all_exports(SRC_ROOT)
    parsed = _parse_readme_symbol_names(_readme_text())
    documented = parsed & exports
    missing = sorted(exports - documented)
    extra = sorted(_table_claimed_api_names(_readme_text()) - exports)
    assert missing == [], f"README missing __all__ exports: {missing[:20]}"
    assert extra == [], f"README documents unknown API names: {extra[:20]}"


def test_readme_architecture_inventory_covers_modules() -> None:
    """Each Python module under src/mltgnt has an Architecture row with a one-line role."""
    text = _readme_text()
    rows = _architecture_module_rows(text)
    expected = _discovered_python_modules(SRC_ROOT)
    missing = sorted(expected - set(rows))
    empty = sorted(mod for mod, role in rows.items() if not role.strip())
    assert missing == [], f"missing architecture rows: {missing[:15]}"
    assert empty == [], f"empty architecture roles: {empty[:15]}"


def test_readme_configuration_inventory_covers_runtime_contracts() -> None:
    """Env vars and public config fields are documented; header meters contract is stated."""
    text = _readme_text()
    config_block = _section_slice(text, "Configuration", stop_before=("Error Reference",))

    for var in sorted(_direct_env_vars(SRC_ROOT)):
        if var.startswith("_"):
            continue
        assert var in config_block, f"undocumented env var {var}"

    _ = _config_field_names()
    for cls in _public_config_dataclasses():
        for field in fields(cls):
            _assert_readme_config_field(config_block, field, cls)

    assert "header_meters_url" in config_block
    assert "same-origin" in config_block.lower()
    assert "chat" in config_block.lower() and "SSE" in config_block


def test_readme_error_inventory_covers_public_exceptions() -> None:
    """Public exceptions in Error Reference match the documented set."""
    expected = _public_exception_names(SRC_ROOT)
    documented = _error_table_names(_readme_text())
    missing = sorted(expected - documented)
    extra = sorted(documented - expected)
    assert missing == [], missing
    assert extra == [], extra


def test_readme_deprecated_tokens_are_absent_and_v0135_features_present() -> None:
    """Removed aliases stay out of README; v0.135 hub and WebChat features are documented."""
    text = _readme_text()
    for token in REMOVED_README_TOKENS:
        assert token not in text
    for marker in V0135_README_MARKERS:
        assert marker in text
