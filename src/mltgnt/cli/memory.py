"""mltgnt.cli.memory — memory dream show/forget サブコマンド。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mltgnt.memory.dream import DreamSummary, read_dream, write_dream


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    memory_parser = subparsers.add_parser("memory", help="Memory management commands")
    memory_sub = memory_parser.add_subparsers(dest="memory_cmd", required=True)

    dream_parser = memory_sub.add_parser("dream", help="Dream summary commands")
    dream_sub = dream_parser.add_subparsers(dest="dream_cmd", required=True)

    show_parser = dream_sub.add_parser("show", help="Show dream summary for a persona")
    show_parser.add_argument("persona", help="Persona name/stem")
    show_parser.add_argument(
        "--chat-dir",
        required=True,
        help="Parent path of persona directories",
    )
    show_parser.set_defaults(handler=_show)

    forget_parser = dream_sub.add_parser("forget", help="Remove a dream category")
    forget_parser.add_argument("persona", help="Persona name/stem")
    forget_parser.add_argument(
        "--category",
        required=True,
        help="Category name to remove",
    )
    forget_parser.add_argument(
        "--chat-dir",
        required=True,
        help="Parent path of persona directories",
    )
    forget_parser.set_defaults(handler=_forget)


def execute(args: argparse.Namespace) -> int:
    return int(args.handler(args))


def _persona_dir(chat_dir: str, persona: str) -> Path:
    return Path(chat_dir) / persona


def _show(args: argparse.Namespace) -> int:
    persona_dir = _persona_dir(args.chat_dir, args.persona)
    summary = read_dream(persona_dir)
    if summary is None:
        print(f"No dream summary found for {args.persona}")
        return 0

    for section in summary.sections:
        print(f"=== {section.category} (source_entries: {section.source_entries}) ===")
        print(section.content)
        print()
    return 0


def _forget(args: argparse.Namespace) -> int:
    persona_dir = _persona_dir(args.chat_dir, args.persona)
    summary = read_dream(persona_dir)
    if summary is None:
        print("No dream summary found", file=sys.stderr)
        return 1

    remaining = [s for s in summary.sections if s.category != args.category]
    if len(remaining) == len(summary.sections):
        print(f"Category not found: {args.category}", file=sys.stderr)
        return 1

    updated = DreamSummary(
        persona=summary.persona,
        sections=remaining,
        updated_at=summary.updated_at,
    )
    write_dream(persona_dir, updated)
    print(f"Removed category '{args.category}' from dream summary for {args.persona}")
    return 0
