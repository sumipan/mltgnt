"""tests/test_hooks_adapter.py — hooks_adapter 受け入れ条件テスト (#1036)"""
import json

import pytest

from mltgnt.agent import AgentRunner, create_audit_writer


def make_llm(responses: list):
    calls = iter(responses)

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(calls)

    return llm_call


def make_executor(results: dict):
    def executor(tool_name: str, tool_args: dict) -> str:
        return results[tool_name]

    return executor


class TestCreateAuditWriter:
    def test_writes_tool_exec_event(self, tmp_path):
        """AC-1: ツール実行で event_type=tool_exec のレコードが audit_path に追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        runner = AgentRunner(
            llm_call=make_llm([
                '{"tool": "search", "args": {"q": "test"}}',
                '{"tool": "done", "args": {}}',
            ]),
            tool_executor=make_executor({"search": "result"}),
            terminal_tools=frozenset({"done"}),
            audit_writer=create_audit_writer(audit_path),
        )
        runner.run("prompt")
        assert audit_path.exists()
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "tool_exec"
        assert record["correlation_id"] == "search"

    def test_no_audit_file_when_writer_is_none(self, tmp_path):
        """AC-2: audit_writer=None(デフォルト)の場合、audit ファイルは作成されない。"""
        audit_path = tmp_path / "audit.jsonl"
        runner = AgentRunner(
            llm_call=make_llm([
                '{"tool": "search", "args": {"q": "x"}}',
                '{"tool": "done", "args": {}}',
            ]),
            tool_executor=make_executor({"search": "result"}),
            terminal_tools=frozenset({"done"}),
        )
        runner.run("prompt")
        assert not audit_path.exists()

    def test_tool_args_not_in_record(self, tmp_path):
        """AC-3: tool_args の値がレコードに含まれない（機密データ混入防止）。"""
        audit_path = tmp_path / "audit.jsonl"
        runner = AgentRunner(
            llm_call=make_llm([
                '{"tool": "search", "args": {"api_key": "secret123", "query": "test"}}',
                '{"tool": "done", "args": {}}',
            ]),
            tool_executor=make_executor({"search": "result"}),
            terminal_tools=frozenset({"done"}),
            audit_writer=create_audit_writer(audit_path),
        )
        runner.run("prompt")
        audit_text = audit_path.read_text()
        assert "secret123" not in audit_text

    def test_multiple_tools_separate_records_unique_uuids(self, tmp_path):
        """AC-4: ツール 3 回実行 → 行数 3、UUID 全て異なる。"""
        audit_path = tmp_path / "audit.jsonl"
        runner = AgentRunner(
            llm_call=make_llm([
                '{"tool": "search", "args": {"q": "a"}}',
                '{"tool": "search", "args": {"q": "b"}}',
                '{"tool": "search", "args": {"q": "c"}}',
                '{"tool": "done", "args": {}}',
            ]),
            tool_executor=make_executor({"search": "ok"}),
            terminal_tools=frozenset({"done"}),
            max_iterations=5,
            audit_writer=create_audit_writer(audit_path),
        )
        runner.run("prompt")
        lines = audit_path.read_text().splitlines()
        assert len(lines) == 3
        uuids = [json.loads(ln)["uuid"] for ln in lines]
        assert len(set(uuids)) == 3

    def test_source_appears_in_engine_field(self, tmp_path):
        """source 引数が engine フィールドとして記録される。"""
        audit_path = tmp_path / "audit.jsonl"
        runner = AgentRunner(
            llm_call=make_llm([
                '{"tool": "search", "args": {}}',
                '{"tool": "done", "args": {}}',
            ]),
            tool_executor=make_executor({"search": "ok"}),
            terminal_tools=frozenset({"done"}),
            audit_writer=create_audit_writer(audit_path, source="my-agent"),
        )
        runner.run("prompt")
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["engine"] == "my-agent"

    def test_default_source_is_mltgnt_agent(self, tmp_path):
        """source を省略したとき engine は 'mltgnt-agent'。"""
        audit_path = tmp_path / "audit.jsonl"
        runner = AgentRunner(
            llm_call=make_llm([
                '{"tool": "search", "args": {}}',
                '{"tool": "done", "args": {}}',
            ]),
            tool_executor=make_executor({"search": "ok"}),
            terminal_tools=frozenset({"done"}),
            audit_writer=create_audit_writer(audit_path),
        )
        runner.run("prompt")
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["engine"] == "mltgnt-agent"
