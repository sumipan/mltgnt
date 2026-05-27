"""CLI 終了コード 0/1/2/3 の検証（Issue #1254）。"""
import os
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.cli.main import main
from mltgnt.exceptions import ConfigError, DependencyError


def test_invalid_components_format_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--components", "invalid"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "ConfigError" in err


def test_empty_module_name_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--components", ":func"])
    assert exc_info.value.code == 2


def test_empty_function_name_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--components", "os:"])
    assert exc_info.value.code == 2


def test_module_not_found_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--components", "nonexistent.module:fn"])
    assert exc_info.value.code == 2


def test_valid_os_getcwd_resolves(capsys, tmp_path):
    mock_components = [MagicMock()]
    mock_runner = MagicMock()

    with patch("mltgnt.cli.run.DaemonRunner", return_value=mock_runner) as mock_runner_cls:
        pid_file = str(tmp_path / "test.pid")
        main(["run", "--components", "os:getcwd", "--pid-file", pid_file])

        mock_runner_cls.assert_called_once()
        call_kwargs = mock_runner_cls.call_args.kwargs
        assert call_kwargs["components"] == os.getcwd()
        mock_runner.run.assert_called_once()


def test_pid_lock_failure_exits_3(capsys, tmp_path):
    mock_components = [MagicMock()]
    mock_factory = MagicMock(return_value=mock_components)

    with patch("mltgnt.cli.run.importlib") as mock_importlib, \
         patch("mltgnt.daemon._pidlock.PidLock") as mock_pidlock_cls:
        mock_module = MagicMock()
        mock_module.build = mock_factory
        mock_importlib.import_module.return_value = mock_module
        mock_pidlock_cls.return_value.acquire.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            main([
                "run",
                "--components", "mymod:build",
                "--pid-file", str(tmp_path / "daemon.pid"),
            ])
        assert exc_info.value.code == 3
        err = capsys.readouterr().err
        assert "DependencyError" in err


def test_mltgnt_error_exits_1(capsys):
    from mltgnt.exceptions import MltgntError

    with patch("mltgnt.cli.run.execute", side_effect=MltgntError("generic")):
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--components", "m:m"])
        assert exc_info.value.code == 1


def test_config_error_from_execute_exits_2(capsys):
    with patch("mltgnt.cli.run.execute", side_effect=ConfigError("bad config")):
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--components", "m:m"])
        assert exc_info.value.code == 2


def test_dependency_error_from_execute_exits_3(capsys):
    with patch("mltgnt.cli.run.execute", side_effect=DependencyError("dep failed")):
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--components", "m:m"])
        assert exc_info.value.code == 3
