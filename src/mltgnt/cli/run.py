import importlib
import sys
import argparse
from pathlib import Path

from mltgnt.daemon import DaemonRunner


def execute(args: argparse.Namespace) -> None:
    components_spec: str = args.components
    pid_file = Path(args.pid_file)

    if ":" not in components_spec:
        print(
            f"Error: --components must be in 'module:function' format, got: {components_spec!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    module_path, func_name = components_spec.rsplit(":", 1)

    if not module_path or not func_name:
        print(
            f"Error: --components must be in 'module:function' format, got: {components_spec!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        print(f"ModuleNotFoundError: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        factory = getattr(module, func_name)
    except AttributeError as exc:
        print(f"AttributeError: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not callable(factory):
        print(
            f"Error: '{func_name}' in module '{module_path}' is not callable",
            file=sys.stderr,
        )
        raise SystemExit(1)

    components = factory()
    runner = DaemonRunner(pid_file=pid_file, components=components)
    runner.run()
