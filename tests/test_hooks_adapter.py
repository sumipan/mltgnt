"""tests/test_hooks_adapter.py — hooks_adapter 受け入れ条件テスト (#1036, #1128)"""
import json

from ghdag.dag.hooks import Task, TaskMetrics

from mltgnt.agent import AgentRunner, create_audit_writer
from mltgnt.bridges.hooks_adapter import MltgntHooks


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

    def test_uses_explicit_correlation_id_when_given(self, tmp_path):
        """correlation_id 指定時は tool_name ではなく指定値を使う。"""
        audit_path = tmp_path / "audit.jsonl"
        runner = AgentRunner(
            llm_call=make_llm([
                '{"tool": "search", "args": {}}',
                '{"tool": "done", "args": {}}',
            ]),
            tool_executor=make_executor({"search": "ok"}),
            terminal_tools=frozenset({"done"}),
            audit_writer=create_audit_writer(audit_path, correlation_id="session-123"),
        )
        runner.run("prompt")
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["correlation_id"] == "session-123"

    def test_falls_back_to_tool_name_when_correlation_id_omitted(self, tmp_path):
        """correlation_id 省略時は従来どおり tool_name を使う。"""
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
        assert record["correlation_id"] == "search"


def _make_task(uuid: str = "t1") -> Task:
    return Task(uuid=uuid, command="echo hello")


def _make_metrics(uuid: str = "t1", status: str = "success") -> TaskMetrics:
    return TaskMetrics(
        uuid=uuid,
        engine="claude",
        model="claude-sonnet-4-6",
        wall_time_sec=1.0,
        token_count=100,
        status=status,
        started_at=1.0,
        finished_at=2.0,
        correlation_id="test-corr",
    )


class TestMltgntHooks:
    def test_protocol_compliance(self, tmp_path):
        """AC-1: DagHooks Protocol の全 10 メソッドが実装されている（structural typing チェック）。"""
        hooks = MltgntHooks(tmp_path / "audit.jsonl")
        required = [
            "on_task_start", "on_task_success", "on_task_failure",
            "on_task_rejected", "on_task_dep_failed", "on_task_empty_result",
            "on_shutdown", "check_rejected", "check_pipeline_status", "check_promote_target",
        ]
        for method in required:
            assert callable(getattr(hooks, method, None)), f"{method} が実装されていない"

    def test_on_task_start_writes_task_started(self, tmp_path):
        """AC-1: on_task_start → audit_path に event_type=task_started のレコードが 1 行追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_task_start("t1", _make_task())
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "task_started"

    def test_on_task_success_writes_success(self, tmp_path):
        """AC-1: on_task_success → event_type=task_success + status=success のレコードが追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_task_success("t1", _make_task(), _make_metrics())
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "task_success"
        assert record["status"] == "success"

    def test_on_task_failure_writes_failure(self, tmp_path):
        """AC-1: on_task_failure → event_type=task_failure + status=failure のレコードが追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_task_failure("t1", _make_task(), 1, "error msg", _make_metrics(status="failure"))
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "task_failure"
        assert record["status"] == "failure"

    def test_on_task_rejected_writes_rejected(self, tmp_path):
        """on_task_rejected → event_type=task_rejected が追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_task_rejected("t1", _make_task(), retry_depth=1, is_final=False, metrics=_make_metrics())
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "task_rejected"

    def test_on_task_dep_failed_writes_dep_failed(self, tmp_path):
        """on_task_dep_failed → event_type=task_dep_failed が追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_task_dep_failed("t1", _make_task(), "dep-uuid")
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "task_dep_failed"

    def test_on_task_empty_result_writes_empty_result(self, tmp_path):
        """on_task_empty_result → event_type=task_empty_result が追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_task_empty_result("t1", _make_task(), "stderr text", _make_metrics())
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "task_empty_result"

    def test_on_shutdown_writes_shutdown(self, tmp_path):
        """on_shutdown → event_type=shutdown が追記される。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_shutdown(15)
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["event_type"] == "shutdown"

    def test_check_rejected_true(self, tmp_path):
        """AC-2: REJECTED: マーカーを含む結果ファイルに対して True を返す。"""
        result_file = tmp_path / "result.md"
        result_file.write_text("REJECTED: 不適切な応答\n", encoding="utf-8")
        hooks = MltgntHooks(tmp_path / "audit.jsonl")
        assert hooks.check_rejected(str(result_file)) is True

    def test_check_rejected_false(self, tmp_path):
        """AC-2: 通常の結果ファイルに対して False を返す。"""
        result_file = tmp_path / "result.md"
        result_file.write_text("ACCEPTED\n通常の出力\n", encoding="utf-8")
        hooks = MltgntHooks(tmp_path / "audit.jsonl")
        assert hooks.check_rejected(str(result_file)) is False

    def test_check_pipeline_status(self, tmp_path):
        """AC-2: PIPELINE_STATUS: BRUSHUP_DONE を含むファイルに対して 'BRUSHUP_DONE' を返す。"""
        result_file = tmp_path / "result.md"
        result_file.write_text("出力\nPIPELINE_STATUS: BRUSHUP_DONE\n", encoding="utf-8")
        hooks = MltgntHooks(tmp_path / "audit.jsonl")
        assert hooks.check_pipeline_status(str(result_file)) == "BRUSHUP_DONE"

    def test_check_promote_target_returns_none(self, tmp_path):
        """AC-2: check_promote_target は任意の入力に対して None を返す。"""
        hooks = MltgntHooks(tmp_path / "audit.jsonl")
        assert hooks.check_promote_target("any/path.md") is None

    def test_custom_source_in_records(self, tmp_path):
        """source 引数がレコードの engine フィールドに反映される（on_task_start 経由）。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path, source="my-scheduler")
        hooks.on_task_start("t1", _make_task())
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["engine"] == "my-scheduler"

    def test_default_source_is_mltgnt_scheduler(self, tmp_path):
        """source を省略したとき engine は 'mltgnt-scheduler'（on_task_start で確認）。"""
        audit_path = tmp_path / "audit.jsonl"
        hooks = MltgntHooks(audit_path)
        hooks.on_task_start("t1", _make_task())
        record = json.loads(audit_path.read_text().splitlines()[0])
        assert record["engine"] == "mltgnt-scheduler"
