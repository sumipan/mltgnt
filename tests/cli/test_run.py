import argparse

import pytest

from mltgnt.cli.run import execute
from mltgnt.exceptions import ConfigError


def _make_args(components: str, pid_file: str = "/tmp/mltgnt_daemon.pid") -> argparse.Namespace:
    return argparse.Namespace(components=components, pid_file=pid_file)


def test_missing_colon_raises_config_error():
    with pytest.raises(ConfigError, match="module:function"):
        execute(_make_args("no_colon_spec"))


def test_empty_module_raises_config_error():
    with pytest.raises(ConfigError, match="module:function"):
        execute(_make_args(":func"))


def test_empty_function_raises_config_error():
    with pytest.raises(ConfigError, match="module:function"):
        execute(_make_args("os:"))


def test_invalid_module_raises_config_error():
    with pytest.raises(ConfigError, match="nonexistent"):
        execute(_make_args("nonexistent.module.path:build_components"))


def test_invalid_function_raises_config_error():
    with pytest.raises(ConfigError, match="nonexistent_func"):
        execute(_make_args("mltgnt.daemon:nonexistent_func"))


def test_non_callable_raises_config_error():
    with pytest.raises(ConfigError, match="not callable"):
        execute(_make_args("mltgnt.daemon:__all__"))


