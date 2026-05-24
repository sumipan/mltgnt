import argparse
from typing import Optional

from mltgnt.cli import run as run_module


def main(argv: Optional[list[str]] = None) -> None:
    """CLI エントリポイント。argparse でサブコマンドを dispatch する。"""
    parser = argparse.ArgumentParser(
        prog="mltgnt",
        description="mltgnt daemon management",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    run_parser = subparsers.add_parser("run", help="Start the mltgnt daemon")
    run_parser.add_argument(
        "--components",
        required=True,
        metavar="MODULE:FUNCTION",
        help="Component factory as 'module.path:function_name'",
    )
    run_parser.add_argument(
        "--pid-file",
        default="/tmp/mltgnt_daemon.pid",
        help="Path to PID file (default: /tmp/mltgnt_daemon.pid)",
    )

    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        raise SystemExit(0)

    if args.subcommand == "run":
        run_module.execute(args)
