import argparse
import importlib
from pathlib import Path

from mltgnt.daemon import DaemonRunner
from mltgnt.exceptions import ConfigError


def execute(args: argparse.Namespace) -> None:
    components_spec: str = args.components
    pid_file = Path(args.pid_file)

    if ":" not in components_spec:
        raise ConfigError(
            f"--components must be in 'module:function' format, got: {components_spec!r}"
        )

    module_path, func_name = components_spec.rsplit(":", 1)

    if not module_path or not func_name:
        raise ConfigError(
            f"--components must be in 'module:function' format, got: {components_spec!r}"
        )

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ConfigError(f"module not found: {module_path}") from exc

    try:
        factory = getattr(module, func_name)
    except AttributeError as exc:
        raise ConfigError(
            f"function {func_name!r} not found in module {module_path!r}"
        ) from exc

    if not callable(factory):
        raise ConfigError(
            f"'{func_name}' in module '{module_path}' is not callable"
        )

    components = factory()
    runner = DaemonRunner(pid_file=pid_file, components=components)
    runner.run()
