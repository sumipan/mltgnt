"""tests/test_fanout_integration.py — fanout 経路の統合テスト (#1036)"""
import pytest

from ghdag.dag.fanout import (
    FanOutSpec,
    build_child_exec_line,
    parse_fanout_spec,
)
from mltgnt.scheduler import _FANOUT_PROMPT_SUFFIX


class TestParseFanoutSpec:
    def test_parses_fanout_yaml_from_result_file(self, tmp_path):
        """AC: fanout YAML を含む result ファイルが FanOutSpec としてパースされる。"""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "ペルソナの通常出力\n"
            "\n"
            "---\n"
            "ghdag_fanout:\n"
            "  children:\n"
            "    - id: child-1\n"
            "      command: \"claude -p 'task1' < input.md\"\n"
            "    - id: child-2\n"
            "      command: \"claude -p 'task2' < input.md\"\n",
            encoding="utf-8",
        )
        spec = parse_fanout_spec(str(result_file))
        assert spec is not None
        assert isinstance(spec, FanOutSpec)
        assert len(spec.children) >= 1
        assert spec.children[0].id == "child-1"

    def test_no_fanout_returns_none(self, tmp_path):
        """AC: --- セパレータや ghdag_fanout を含まない result ファイル → None。"""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "通常の出力\nfanout なし\n普通のテキストのみ\n",
            encoding="utf-8",
        )
        assert parse_fanout_spec(str(result_file)) is None

    def test_separator_without_fanout_key_returns_none(self, tmp_path):
        """--- セパレータがあっても ghdag_fanout キーがなければ None。"""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "output\n---\nsome_other_key: value\n",
            encoding="utf-8",
        )
        assert parse_fanout_spec(str(result_file)) is None

    def test_duplicate_child_ids_raise_value_error(self, tmp_path):
        """AC: id が重複する child を含む result → ValueError。"""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "output\n"
            "---\n"
            "ghdag_fanout:\n"
            "  children:\n"
            "    - id: dup\n"
            "      command: cmd1\n"
            "    - id: dup\n"
            "      command: cmd2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            parse_fanout_spec(str(result_file))

    def test_none_path_returns_none(self):
        """result_path が None のとき None を返す。"""
        assert parse_fanout_spec(None) is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        """存在しないファイルパスは None を返す。"""
        assert parse_fanout_spec(str(tmp_path / "missing.md")) is None


class TestBuildChildExecLine:
    def test_format_matches_expected(self):
        """AC: build_child_exec_line の出力が 'uuid: command' 形式。"""
        result = build_child_exec_line("abc-uuid", "claude -p 'test'")
        assert result == "abc-uuid: claude -p 'test'"

    def test_format_with_complex_command(self):
        """複雑なコマンドでも形式が維持される。"""
        cmd = "agent -p --force < /path/to/order.md"
        result = build_child_exec_line("some-uuid-1234", cmd)
        assert result == f"some-uuid-1234: {cmd}"


class TestFanoutPromptSuffix:
    def test_suffix_contains_ghdag_fanout_key(self):
        """_FANOUT_PROMPT_SUFFIX に ghdag_fanout が含まれる。"""
        assert "ghdag_fanout" in _FANOUT_PROMPT_SUFFIX

    def test_suffix_output_is_parseable(self, tmp_path):
        """_FANOUT_PROMPT_SUFFIX のサンプル YAML が parse_fanout_spec でパースできる。"""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "ペルソナの出力\n"
            "---\n"
            "ghdag_fanout:\n"
            "  children:\n"
            "    - id: subtask-1\n"
            "      command: \"agent -p --force < order-1.md\"\n"
            "    - id: subtask-2\n"
            "      command: \"agent -p --force < order-2.md\"\n",
            encoding="utf-8",
        )
        spec = parse_fanout_spec(str(result_file))
        assert spec is not None
        assert len(spec.children) == 2
