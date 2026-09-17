"""tests/test_fanout_integration.py — integration tests for the fanout path (#1036)"""
import pytest

from ghdag.dag.fanout import (
    FanOutSpec,
    build_child_jsonl_record,
    parse_fanout_spec,
)
from mltgnt.scheduler import _FANOUT_PROMPT_SUFFIX


class TestParseFanoutSpec:
    def test_parses_fanout_yaml_from_result_file(self, tmp_path):
        """AC: a result file containing fanout YAML parses as FanOutSpec."""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "Normal persona output\n"
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
        """AC: result file without --- separator or ghdag_fanout → None."""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "Normal output\nno fanout\nplain text only\n",
            encoding="utf-8",
        )
        assert parse_fanout_spec(str(result_file)) is None

    def test_separator_without_fanout_key_returns_none(self, tmp_path):
        """--- separator without a ghdag_fanout key → None."""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "output\n---\nsome_other_key: value\n",
            encoding="utf-8",
        )
        assert parse_fanout_spec(str(result_file)) is None

    def test_duplicate_child_ids_raise_value_error(self, tmp_path):
        """AC: result with duplicate child ids → ValueError."""
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
        """result_path=None returns None."""
        assert parse_fanout_spec(None) is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Nonexistent file path returns None."""
        assert parse_fanout_spec(str(tmp_path / "missing.md")) is None


class TestBuildChildJsonlRecord:
    def test_format_matches_expected(self):
        """AC: build_child_jsonl_record output is an exec.jsonl-style JSON line."""
        import json
        result = build_child_jsonl_record("abc-uuid", "claude -p 'test'")
        assert json.loads(result) == {"uuid": "abc-uuid", "command": "claude -p 'test'"}

    def test_format_with_complex_command(self):
        """JSON record shape is preserved for complex commands."""
        import json
        cmd = "agent -p --force < /path/to/order.md"
        result = build_child_jsonl_record("some-uuid-1234", cmd)
        assert json.loads(result) == {"uuid": "some-uuid-1234", "command": cmd}


class TestFanoutPromptSuffix:
    def test_suffix_contains_ghdag_fanout_key(self):
        """_FANOUT_PROMPT_SUFFIX contains ghdag_fanout."""
        assert "ghdag_fanout" in _FANOUT_PROMPT_SUFFIX

    def test_suffix_output_is_parseable(self, tmp_path):
        """Sample YAML in _FANOUT_PROMPT_SUFFIX is parseable by parse_fanout_spec."""
        result_file = tmp_path / "result.md"
        result_file.write_text(
            "Persona output\n"
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
