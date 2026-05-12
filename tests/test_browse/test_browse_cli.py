"""tests/test_browse/test_browse_cli.py — CLI 統合テスト (AC-1, AC-3, AC-4, AC-5, AC-6)"""
from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestFetchCommand:
    """AC-1: fetch サブコマンド正常系"""

    def _run_main(self, args):
        from skills.browse.browse import main

        captured = StringIO()
        with patch("sys.stdout", captured):
            try:
                main(args)
                exit_code = 0
            except SystemExit as e:
                exit_code = e.code
        return exit_code, captured.getvalue()

    def test_fetch_returns_json_with_url_title_content(self):
        html = "<html><head><title>Test Title</title></head><body><p>Test content text</p></body></html>"
        with patch("skills.browse.browse.fetch", return_value=html), \
             patch("skills.browse.browse.extract", return_value="Test content text"):
            code, out = self._run_main(["fetch", "https://example.com"])
        assert code == 0
        data = json.loads(out)
        assert data["url"] == "https://example.com"
        assert "content" in data
        assert "truncated" in data

    def test_fetch_truncation_at_30000(self):
        long_content = "A" * 35000
        html = f"<html><body><p>{long_content}</p></body></html>"
        with patch("skills.browse.browse.fetch", return_value=html), \
             patch("skills.browse.browse.extract", return_value=long_content):
            code, out = self._run_main(["fetch", "https://example.com"])
        assert code == 0
        data = json.loads(out)
        assert data["truncated"] is True
        assert len(data["content"]) == 30000

    def test_fetch_no_truncation_under_30000(self):
        short_content = "B" * 100
        html = f"<html><body><p>{short_content}</p></body></html>"
        with patch("skills.browse.browse.fetch", return_value=html), \
             patch("skills.browse.browse.extract", return_value=short_content):
            code, out = self._run_main(["fetch", "https://example.com"])
        assert code == 0
        data = json.loads(out)
        assert data["truncated"] is False


class TestFetchErrorCases:
    """AC-3: fetch サブコマンド異常系"""

    def _run_main(self, args):
        from skills.browse.browse import main

        captured = StringIO()
        with patch("sys.stdout", captured):
            try:
                main(args)
                exit_code = 0
            except SystemExit as e:
                exit_code = e.code
        return exit_code, captured.getvalue()

    def test_private_url_returns_error_json_exit1(self):
        with patch("skills.browse.browse.fetch", side_effect=ValueError("blocked private address")):
            code, out = self._run_main(["fetch", "http://127.0.0.1"])
        assert code == 1
        data = json.loads(out)
        assert "error" in data

    def test_runtime_error_returns_error_json_exit1(self):
        with patch("skills.browse.browse.fetch", side_effect=RuntimeError("connection failed")):
            code, out = self._run_main(["fetch", "https://nonexistent.example.invalid"])
        assert code == 1
        data = json.loads(out)
        assert "error" in data

    def test_timeout_returns_error_json_exit1(self):
        with patch("skills.browse.browse.fetch", side_effect=TimeoutError("timed out")):
            code, out = self._run_main(["fetch", "https://slow.example.com"])
        assert code == 1
        data = json.loads(out)
        assert "error" in data


class TestSearchCommand:
    """AC-4: search サブコマンド"""

    def _run_main(self, args):
        from skills.browse.browse import main

        captured = StringIO()
        with patch("sys.stdout", captured):
            try:
                main(args)
                exit_code = 0
            except SystemExit as e:
                exit_code = e.code
        return exit_code, captured.getvalue()

    def test_search_returns_json_with_query_and_results(self):
        mock_results = [
            {"url": "https://example.com/1", "title": "Result 1", "content": "content 1"},
            {"url": "https://example.com/2", "title": "Result 2", "content": "content 2"},
        ]
        with patch("skills.browse.browse.search", return_value=mock_results):
            code, out = self._run_main(["search", "Python asyncio tutorial"])
        assert code == 0
        data = json.loads(out)
        assert data["query"] == "Python asyncio tutorial"
        assert isinstance(data["results"], list)

    def test_search_top_flag(self):
        mock_results = [
            {"url": f"https://example.com/{i}", "title": f"Title {i}", "content": "c"}
            for i in range(5)
        ]
        with patch("skills.browse.browse.search", return_value=mock_results) as mock_s:
            code, out = self._run_main(["search", "query", "--top", "5"])
        assert code == 0
        mock_s.assert_called_once_with("query", top=5)

    def test_search_top_exceeds_limit_returns_error(self):
        with patch("skills.browse.browse.search", side_effect=ValueError("top must be 1-10")):
            code, out = self._run_main(["search", "query", "--top", "11"])
        assert code == 1
        data = json.loads(out)
        assert "error" in data


class TestSkillMdDiscovery:
    """AC-6: SKILL.md が discover() で検出される"""

    def test_browse_skill_discovered(self, tmp_path):
        import shutil
        from pathlib import Path
        from mltgnt.skill.loader import discover

        # skills/browse/SKILL.md を tmp_path に複製して discover をテスト
        src = Path("skills/browse/SKILL.md")
        dest = tmp_path / "browse" / "SKILL.md"
        dest.parent.mkdir(parents=True)
        shutil.copy(src, dest)

        skills = discover([tmp_path])
        assert "browse" in skills
        meta = skills["browse"]
        assert meta.name == "browse"
        assert meta.description
        assert meta.argument_hint
        assert isinstance(meta.triggers, list)
        assert len(meta.triggers) > 0
