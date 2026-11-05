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
import argparse
from argparse import SUPPRESS, ArgumentParser
from collections.abc import Iterator
from dataclasses import MISSING, Field, fields
from pathlib import Path
from typing import Any, get_type_hints
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

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
        string_constants: dict[str, str] = {}
        for node in tree.body:
            if (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        string_constants[target.id] = node.value.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_os_getenv = (
                isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
            )
            is_os_environ_get = (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
            )
            is_imported_getenv = isinstance(func, ast.Name) and func.id == "getenv"
            if not (is_os_getenv or is_os_environ_get or is_imported_getenv) or not node.args:
                continue
            key_arg = node.args[0]
            if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                keys.add(key_arg.value)
            elif isinstance(key_arg, ast.Name) and key_arg.id in string_constants:
                keys.add(string_constants[key_arg.id])
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


def _config_summary_table_row(block: str, cls_name: str) -> list[str] | None:
    for line in block.splitlines():
        if not line.startswith("|") or line.startswith("|--"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0].strip("` ") != cls_name:
            continue
        return cells
    return None


def _readme_documents_required_config_field(block: str, field: Field[Any], cls: type[Any]) -> bool:
    row = _config_summary_table_row(block, cls.__name__)
    if row is not None and len(row) >= 2 and field.name in row[1]:
        return True
    if re.search(
        rf"\|\s*`?{re.escape(field.name)}`?\s*\|\s*required\s*\|",
        block,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            rf"`{re.escape(field.name)}`[^\n]*\(\s*required",
            block,
            flags=re.IGNORECASE,
        )
    )


def _readme_documents_optional_config_default(
    block: str,
    field: Field[Any],
    cls: type[Any],
    default_token: str,
) -> bool:
    if default_token in block:
        return True
    row = _config_summary_table_row(block, cls.__name__)
    if row is not None and len(row) >= 3 and field.name in row[2]:
        return True
    if field.name in block and default_token.strip('"') in block:
        return True
    if field.name == "language" and "get_language_pack()" in block:
        return True
    if field.name == "guard" and "GuardConfig()" in block:
        return True
    type_tokens = _field_type_tokens(field, cls)
    if "LanguagePack" in type_tokens and field.name in block and "LanguagePack" in block:
        return True
    if re.search(
        rf"{re.escape(field.name)}[^\n]*\(\s*`?[A-Z][A-Z0-9_]+`?\)",
        block,
    ):
        return True
    return bool(
        re.search(rf"{re.escape(field.name)}[^\n]*\([^)]+\)", block)
    )


def _readme_documents_config_type(
    block: str,
    field: Field[Any],
    cls: type[Any],
    type_tokens: set[str],
    default_token: str | None,
) -> bool:
    if any(token in block for token in type_tokens):
        return True
    if default_token and default_token in block:
        if "float" in type_tokens:
            return True
        if "int" in type_tokens:
            return True
        if "bool" in type_tokens and default_token in {"True", "False"}:
            return True
        if "str" in type_tokens and default_token.startswith('"'):
            return True
    return False


def _assert_readme_config_field(block: str, field: Field[Any], cls: type[Any]) -> None:
    assert field.name in block, f"undocumented config field {field.name}"
    type_tokens = _field_type_tokens(field, cls)
    default_token = None if _field_is_required(field) else _field_default_token(field)
    assert _readme_documents_config_type(block, field, cls, type_tokens, default_token), (
        f"README missing type for {cls.__name__}.{field.name}: types={sorted(type_tokens)}"
    )
    if _field_is_required(field):
        assert _readme_documents_required_config_field(block, field, cls), (
            f"README missing required marker for {cls.__name__}.{field.name}"
        )
        return
    default_token = _field_default_token(field)
    assert default_token is not None, f"missing default metadata for {cls.__name__}.{field.name}"
    assert _readme_documents_optional_config_default(block, field, cls, default_token), (
        f"README missing default for {cls.__name__}.{field.name}: default={default_token!r}"
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


def _ast_direct_base_name(node: ast.ClassDef) -> str | None:
    if not node.bases:
        return None
    base = node.bases[0]
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _exception_class_metadata(src_root: Path) -> dict[str, tuple[str, str]]:
    meta: dict[str, tuple[str, str]] = {}
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root.parent)
        module = ".".join(rel.with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _ast_class_inherits_exception(node):
                continue
            base = _ast_direct_base_name(node)
            if base is None:
                continue
            meta[node.name] = (module, base)
    return meta


def _error_reference_custom_rows(text: str) -> dict[str, dict[str, str]]:
    block = _section_slice(text, "Error Reference", stop_before=("Public API Stability",))
    rows: dict[str, dict[str, str]] = {}
    for line in block.splitlines():
        if line.startswith("Built-in exceptions"):
            break
        if not line.startswith("|") or line.startswith("|--"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in ("Exception", "Name"):
            continue
        name = cells[0].strip("` ")
        if not re.match(r"^[A-Z][A-Za-z0-9_]*$", name):
            continue
        module_match = re.search(r"`(mltgnt\.[^`]+)`", cells[1])
        module = module_match.group(1) if module_match else cells[1].strip("` ").split()[0]
        base_match = re.search(r"`([A-Za-z][A-Za-z0-9_]*)`", cells[2])
        base = base_match.group(1) if base_match else cells[2].strip("` ")
        rows[name] = {"module": module, "base": base}
    return rows


def _assert_error_reference_matches_implementation(text: str) -> None:
    source_meta = _exception_class_metadata(SRC_ROOT)
    readme_rows = _error_reference_custom_rows(text)
    for name, (module, base) in sorted(source_meta.items()):
        if name not in readme_rows:
            continue
        doc = readme_rows[name]
        assert doc["module"] == module, f"{name} module: README={doc['module']!r} source={module!r}"
        assert doc["base"] == base, f"{name} base: README={doc['base']!r} source={base!r}"


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


def _assert_readme_error_module_base(text: str) -> None:
    _assert_error_reference_matches_implementation(text)


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


_CLI_SKIP_ACTION_DESTS = frozenset(
    {"help", "subcommand", "memory_cmd", "dream_cmd", "hub_cmd", "handler"},
)


def _build_cli_parser() -> ArgumentParser:
    from mltgnt.cli import main as main_module

    captured: dict[str, ArgumentParser] = {}

    def capture_parser(parser: ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
        captured["parser"] = parser
        return argparse.Namespace(subcommand=None)

    with (
        patch.object(ArgumentParser, "parse_args", capture_parser),
        patch.object(ArgumentParser, "print_help"),
    ):
        try:
            main_module.main([])
        except SystemExit as exc:
            assert exc.code == 0

    assert "parser" in captured, "mltgnt.cli.main did not construct an ArgumentParser"
    return captured["parser"]


def _cli_action_command_key(action: argparse.Action) -> str:
    if action.option_strings:
        long_opts = [opt for opt in action.option_strings if opt.startswith("--")]
        return long_opts[0] if long_opts else action.option_strings[0]
    return action.dest


def _cli_action_default_literal(action: argparse.Action) -> str | None:
    if action.default is SUPPRESS:
        return None
    if action.option_strings:
        if action.default is None:
            return None
        if isinstance(action.default, str):
            return action.default
        return repr(action.default)
    return None


def _cli_argument_entries(parser: ArgumentParser) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for action in parser._actions:
        if action.dest in _CLI_SKIP_ACTION_DESTS:
            continue
        entries.append(
            {
                "command": _cli_action_command_key(action),
                "required": bool(action.required),
                "default": _cli_action_default_literal(action),
            }
        )
    return entries


def _readme_cli_section_chunk(cli_block: str, cmd: tuple[str, ...]) -> str:
    section_title = "### `" + " ".join(("mltgnt", *cmd)) + "`"
    assert section_title in cli_block, f"missing CLI section for {cmd}"
    return cli_block.split(section_title, 1)[1].split("### ", 1)[0]


def _readme_cli_argument_table(chunk: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    has_default_column = False
    for line in chunk.splitlines():
        if not line.startswith("|") or line.startswith("|--"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        if cells[0] in ("Option", "Argument / option", "Argument"):
            has_default_column = "Default" in cells
            continue
        command = cells[0].strip("`").split()[0]
        required = cells[1].lower() == "yes" if len(cells) > 1 else False
        default: str | None = None
        if has_default_column and len(cells) > 2:
            raw = cells[2].strip("` ")
            default = None if raw in {"", "-"} else raw
        rows[command] = {"required": required, "default": default}
    return rows


def _assert_readme_cli_argument_matches_source(
    readme_row: dict[str, Any],
    source_row: dict[str, Any],
    *,
    cmd: tuple[str, ...],
) -> None:
    command = source_row["command"]
    assert readme_row["required"] == source_row["required"], (
        f"{cmd} {command}: required README={readme_row['required']!r} "
        f"parser={source_row['required']!r}"
    )
    if source_row["default"] is not None:
        assert readme_row["default"] == source_row["default"], (
            f"{cmd} {command}: default README={readme_row['default']!r} "
            f"parser={source_row['default']!r}"
        )


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


def _cli_inventory_from_sources(*, readme_cli_block: str | None = None) -> dict[tuple[str, ...], dict[str, Any]]:
    parser = _build_cli_parser()
    inventory: dict[tuple[str, ...], dict[str, Any]] = {}
    for cmd in CLI_COMMANDS:
        leaf = _cli_parser_for_command(parser, cmd)
        inventory[cmd] = {
            "arguments": _cli_argument_entries(leaf),
            "exit_codes": _cli_exit_codes_from_sources(cmd),
        }

    if readme_cli_block is not None:
        for cmd, spec in inventory.items():
            chunk = _readme_cli_section_chunk(readme_cli_block, cmd)
            readme_args = _readme_cli_argument_table(chunk)
            for source_row in spec["arguments"]:
                command = source_row["command"]
                assert command in readme_args, f"README missing CLI argument {command} for {cmd}"
                _assert_readme_cli_argument_matches_source(
                    readme_args[command],
                    source_row,
                    cmd=cmd,
                )
            for code in spec["exit_codes"]:
                assert f"`{code}`" in chunk, f"README missing exit code {code} for {cmd}"
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


def test_readme_quick_start_runs_offline(tmp_path: Path) -> None:
    """Quick Start example runs without network or external services."""
    code = _quick_start_python_block(_readme_text())
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
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
    _cli_inventory_from_sources(readme_cli_block=cli_block)

    for cmd in CLI_COMMANDS:
        label = "mltgnt " + " ".join(cmd)
        assert label in cli_block or f"`{label}`" in cli_block


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
    text = _readme_text()
    _assert_readme_error_module_base(text)
    expected = _public_exception_names(SRC_ROOT)
    documented = _error_table_names(text)
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
