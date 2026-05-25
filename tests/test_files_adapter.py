"""tests/test_files_adapter.py

mltgnt.bridges.files_adapter の単体テスト。
md_read / md_write が ghdag.files へ正しく委譲し、repo_root が伝搬することを mock で検証する。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from mltgnt.bridges.files_adapter import md_read, md_write


class TestMdRead:
    def test_delegates_to_ghdag_files_md_read(self):
        with patch("ghdag.files.md_read", return_value="content") as mock:
            result = md_read("some/path.md")
        mock.assert_called_once_with("some/path.md", repo_root=None)
        assert result == "content"

    def test_repo_root_propagates(self):
        root = Path("/fake/root")
        with patch("ghdag.files.md_read", return_value="data") as mock:
            result = md_read("file.md", repo_root=root)
        mock.assert_called_once_with("file.md", repo_root=root)
        assert result == "data"

    def test_return_value_is_passed_through(self):
        with patch("ghdag.files.md_read", return_value={"key": "value"}):
            result = md_read("path.md")
        assert result == {"key": "value"}


class TestMdWrite:
    def test_delegates_to_ghdag_files_md_write(self):
        with patch("ghdag.files.md_write") as mock:
            md_write("out.md", "hello")
        mock.assert_called_once_with("out.md", "hello", repo_root=None)

    def test_repo_root_propagates(self):
        root = Path("/tmp/repo")
        with patch("ghdag.files.md_write") as mock:
            md_write("out.md", "body", repo_root=root)
        mock.assert_called_once_with("out.md", "body", repo_root=root)

    def test_return_value_is_passed_through(self):
        with patch("ghdag.files.md_write", return_value=42):
            result = md_write("out.md", "x")
        assert result == 42


class TestBridgesAll:
    def test_bridges_all_contains_expected_modules(self):
        from mltgnt import bridges
        assert hasattr(bridges, "__all__"), "bridges/__init__.py must define __all__"
        assert set(bridges.__all__) == {
            "MltgntHooks",
            "create_audit_writer",
            "files_adapter",
            "ghdag_bridge",
            "hooks_adapter",
            "llm_adapter",
        }
