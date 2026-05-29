import argparse
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mltgnt.cli.run import execute


def _make_args(components: str, pid_file: str = "/tmp/mltgnt_daemon.pid") -> argparse.Namespace:
    return argparse.Namespace(components=components, pid_file=pid_file)


def test_missing_components_exits_2(capsys):
    # argparse already enforces required; but execute itself should also guard
    # This test exercises the module:func split and import path
    with pytest.raises(SystemExit) as exc_info:
        execute(_make_args(""))
    assert exc_info.value.code == 1


def test_invalid_module_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        execute(_make_args("nonexistent.module.path:build_components"))
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "ModuleNotFoundError" in err or "nonexistent" in err.lower() or "module" in err.lower()


def test_invalid_function_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        execute(_make_args("mltgnt.daemon:nonexistent_func"))
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "nonexistent_func" in err or "AttributeError" in err


def test_non_callable_exits_1(capsys):
    # mltgnt.daemon.__all__ is a list, not callable
    with pytest.raises(SystemExit) as exc_info:
        execute(_make_args("mltgnt.daemon:__all__"))
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "callable" in err.lower() or "not callable" in err.lower()


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


def test_malformed_components_spec_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        execute(_make_args("no_colon_spec"))
    assert exc_info.value.code == 1
