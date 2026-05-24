import pytest
from unittest.mock import patch, MagicMock

from mltgnt.cli.main import main


def test_help_shows_run_subcommand(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "run" in out


def test_run_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--components" in out
    assert "--pid-file" in out


def test_no_subcommand_shows_help_and_exits_0(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "run" in out


def test_dispatch_calls_run_handler():
    with patch("mltgnt.cli.run.execute") as mock_execute:
        main(["run", "--components", "mymod:build"])
        mock_execute.assert_called_once()
        args = mock_execute.call_args[0][0]
        assert args.components == "mymod:build"


def test_dispatch_passes_pid_file():
    with patch("mltgnt.cli.run.execute") as mock_execute:
        main(["run", "--components", "mymod:build", "--pid-file", "/tmp/test.pid"])
        mock_execute.assert_called_once()
        args = mock_execute.call_args[0][0]
        assert args.pid_file == "/tmp/test.pid"
