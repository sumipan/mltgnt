"""mltgnt.cli.hub — hub serve subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mltgnt.exceptions import ConfigError, DependencyError
from mltgnt.media.hub.server import load_server_config, serve


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    hub_parser = subparsers.add_parser("hub", help="Hub relay server commands")
    hub_sub = hub_parser.add_subparsers(dest="hub_cmd", required=True)

    serve_parser = hub_sub.add_parser("serve", help="Run the hub FastAPI server from a YAML config")
    serve_parser.add_argument("config", help="Path to hub server YAML")
    serve_parser.set_defaults(handler=_serve)


def execute(args: argparse.Namespace) -> int:
    return int(args.handler(args))


def _serve(args: argparse.Namespace) -> int:
    path = Path(args.config)
    try:
        config = load_server_config(path)
    except ConfigError as exc:
        print(f"ConfigError: {exc}", file=sys.stderr)
        return 2
    try:
        serve(config)
    except ImportError as exc:
        print(f"DependencyError: {exc}", file=sys.stderr)
        return 3
    except DependencyError as exc:
        print(f"DependencyError: {exc}", file=sys.stderr)
        return 3
    return 0
