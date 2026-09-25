"""Every documented ``mltgnt <subcommand>`` must be registered in the CLI.

Commands are collected from Markdown code spans / fences (README, CHANGELOG)
and from code spans inside Python string literals under ``src/``. Prose such as
"mltgnt import" outside code spans is ignored.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from mltgnt.cli.main import main

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "mltgnt"

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _code_fragments(text: str) -> list[str]:
    """Return lines inside fenced blocks and the contents of inline code spans."""
    fragments: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            fragments.append(line)
            continue
        fragments.extend(m.group(2) for m in _CODE_SPAN_RE.finditer(line))
    return fragments


def _command_tokens(fragment: str) -> tuple[str, ...]:
    words = fragment.strip().lstrip("$").split()
    if len(words) < 2 or words[0] != "mltgnt":
        return ()
    tokens: list[str] = []
    for word in words[1:]:
        if not _TOKEN_RE.match(word):
            break
        tokens.append(word)
    return tuple(tokens)


def extract_documented_commands(texts: list[str]) -> set[tuple[str, ...]]:
    """Extract subcommand sequences from code fragments starting with ``mltgnt ``."""
    commands: set[tuple[str, ...]] = set()
    for text in texts:
        for fragment in _code_fragments(text):
            tokens = _command_tokens(fragment)
            if tokens:
                commands.add(tokens)
    return commands


def find_unregistered(cmds: set[tuple[str, ...]] | list[tuple[str, ...]]) -> list[str]:
    """Return commands for which ``mltgnt <cmd> --help`` does not exit with 0."""
    unregistered: list[str] = []
    for cmd in sorted(cmds):
        try:
            main([*cmd, "--help"])
            code: object = 0
        except SystemExit as exc:
            code = exc.code
        if code not in (0, None):
            unregistered.append(" ".join(cmd))
    return unregistered


def _python_string_literals(src_root: Path) -> list[str]:
    literals: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.append(node.value)
    return literals


def _documented_texts() -> list[str]:
    texts = [
        (REPO_ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "CHANGELOG.md")
        if (REPO_ROOT / name).exists()
    ]
    return texts + _python_string_literals(SRC_ROOT)


def test_known_subcommands_help_exit_zero() -> None:
    cmds = [("run",), ("memory", "dream", "show"), ("memory", "dream", "forget")]
    assert find_unregistered(cmds) == []


def test_documented_commands_are_registered() -> None:
    cmds = extract_documented_commands(_documented_texts())
    assert ("run",) in cmds
    assert find_unregistered(cmds) == []


def test_unregistered_command_is_reported() -> None:
    text = "Usage:\n\n```bash\nmltgnt nosuchcmd --flag\n```\n"
    cmds = extract_documented_commands([text])
    assert cmds == {("nosuchcmd",)}
    assert find_unregistered(cmds) == ["nosuchcmd"]


def test_prose_is_not_extracted() -> None:
    text = "mltgnt import works in prose.\nSee `mltgnt memory dream show <persona>`."
    assert extract_documented_commands([text]) == {("memory", "dream", "show")}
