import argparse
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_valid_components_starts_daemon(tmp_path):
    mock_components = [MagicMock()]
    mock_factory = MagicMock(return_value=mock_components)
    mock_runner = MagicMock()
    mock_module = MagicMock()
    mock_module.build_components = mock_factory

    with patch("mltgnt.cli.run.importlib") as mock_importlib, \
         patch("mltgnt.cli.run.DaemonRunner", return_value=mock_runner) as mock_runner_cls:
        mock_importlib.import_module.return_value = mock_module

        pid_file = str(tmp_path / "test.pid")
        execute(_make_args("mymod:build_components", pid_file=pid_file))

        mock_importlib.import_module.assert_called_once_with("mymod")
        mock_factory.assert_called_once()
        mock_runner_cls.assert_called_once_with(
            pid_file=Path(pid_file),
            components=mock_components,
        )
        mock_runner.run.assert_called_once()


