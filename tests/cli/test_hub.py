"""mltgnt hub CLI registration and exit codes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.cli.main import main
def test_hub_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["hub", "--help"])
    assert exc.value.code == 0


def test_hub_serve_missing_token_env_exits_2(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
store_dir: /tmp/hub
rooms: ["lobby"]
token_env:
  host-a: TOKEN_A
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["hub", "serve", str(cfg)])
    assert exc.value.code == 2
    assert "ConfigError" in capsys.readouterr().err


def test_hub_serve_missing_fastapi_exits_3(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
store_dir: /tmp/hub
rooms: ["lobby"]
token_env:
  host-a: TOKEN_A
""",
        encoding="utf-8",
    )
    with patch.dict("os.environ", {"TOKEN_A": "secret"}, clear=False):
        with patch("mltgnt.cli.hub.serve", side_effect=ImportError("no fastapi")):
            with pytest.raises(SystemExit) as exc:
                main(["hub", "serve", str(cfg)])
    assert exc.value.code == 3
    assert "DependencyError" in capsys.readouterr().err


def test_hub_serve_config_error_from_load_exits_2(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "hub.yaml"
    cfg.write_text("rooms: []\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["hub", "serve", str(cfg)])
    assert exc.value.code == 2
