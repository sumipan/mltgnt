"""tests/test_ghdag_bridge.py — ghdag_bridge の単体＋統合テスト。

カバレッジ対象:
  - _extract_result_filename(): JSON 形式 / テキスト形式 / フォールバック
  - _order_to_result_filename(): 標準テキスト exec 行からの導出
  - enqueue_and_wait() 統合: exec.jsonl に書き込む行が全て valid JSON
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.scheduler.ghdag_bridge import (
    _extract_result_filename,
    _order_to_result_filename,
    enqueue_and_wait,
)

# ---------------------------------------------------------------------------
# _order_to_result_filename
# ---------------------------------------------------------------------------

UUID_A = "38d6b791-1072-42f0-838d-45c7d10748ff"


class TestOrderToResultFilename:
    def test_claude_order_line(self):
        """cursor-order 行から cursor-result 行を導出する。"""
        line = f"jobs/20260508230000-cursor-order-{UUID_A}.md"
        assert _order_to_result_filename(line) == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_claude_engine(self):
        """claude エンジンでも正しく導出する。"""
        line = f"jobs/20260508230000-claude-order-{UUID_A}.md"
        assert _order_to_result_filename(line) == f"20260508230000-claude-result-{UUID_A}.md"

    def test_no_order_pattern_returns_empty(self):
        """-order- パターンがなければ空文字を返す。"""
        assert _order_to_result_filename("some random text") == ""

    def test_no_uuid_returns_empty(self):
        """UUID が含まれない行では空文字を返す。"""
        assert _order_to_result_filename("jobs/20260508-cursor-order-no-uuid.md") == ""


# ---------------------------------------------------------------------------
# _extract_result_filename
# ---------------------------------------------------------------------------

class TestExtractResultFilename:
    def test_json_format_extracts_result_path_name(self):
        """JSON 形式の exec_line から result_path の basename を取得する。"""
        record = {
            "uuid": UUID_A,
            "command": "agent -p --force < order.md",
            "result_path": f"/Users/user/diary/jobs/20260508230000-cursor-result-{UUID_A}.md",
        }
        line = json.dumps(record)
        result = _extract_result_filename(line)
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_json_format_relative_result_path(self):
        """相対パスの result_path でも basename を正しく取得する。"""
        record = {
            "uuid": UUID_A,
            "command": "cmd",
            "result_path": f"jobs/20260508230000-cursor-result-{UUID_A}.md",
        }
        result = _extract_result_filename(json.dumps(record))
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_json_format_no_result_path_returns_empty(self):
        """result_path フィールドがない JSON レコードは空文字を返す。"""
        record = {"uuid": UUID_A, "command": "cmd"}
        assert _extract_result_filename(json.dumps(record)) == ""

    def test_json_format_empty_result_path_returns_empty(self):
        """result_path が空文字の JSON レコードは空文字を返す。"""
        record = {"uuid": UUID_A, "command": "cmd", "result_path": ""}
        assert _extract_result_filename(json.dumps(record)) == ""

    def test_invalid_json_starting_brace_falls_back_to_text(self):
        """{ で始まるが invalid JSON → テキスト形式パーサーへフォールバック。"""
        line = f"{{not valid json}} jobs/20260508230000-cursor-order-{UUID_A}.md"
        # Falls back to text parser which can't parse this either → ""
        result = _extract_result_filename(line)
        # Either "" or derived from text pattern; key property is no exception raised
        assert isinstance(result, str)

    def test_text_format_exec_line(self):
        """テキスト形式（uuid: cmd | tee result）から result ファイル名を導出する。"""
        line = (
            f"{UUID_A}: agent -p --force < jobs/20260508230000-cursor-order-{UUID_A}.md"
            f" | tee -a jobs/20260508230000-cursor-result-{UUID_A}.md"
        )
        result = _extract_result_filename(line)
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"

    def test_json_format_idempotency_key_not_confused(self):
        """idempotency_key フィールドがあっても result_path 取得に干渉しない。"""
        record = {
            "uuid": UUID_A,
            "command": "cmd",
            "result_path": f"jobs/20260508230000-cursor-result-{UUID_A}.md",
            "idempotency_key": "scheduler:diary_review:2026-05-08T23:00:00+09:00",
        }
        result = _extract_result_filename(json.dumps(record))
        assert result == f"20260508230000-cursor-result-{UUID_A}.md"


# ---------------------------------------------------------------------------
# enqueue_and_wait — 統合テスト
# （wait_for_result だけモック; LLMPipelineAPI は実物）
# ---------------------------------------------------------------------------

def _make_jobs_dir(tmp_path: Path) -> Path:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "exec.jsonl").write_text("", encoding="utf-8")
    return jobs


_WAIT = "ghdag.pipeline.wait_for_result"


class TestEnqueueAndWaitJsonlIntegration:
    """exec.jsonl に書き込む内容が valid JSON であることを保証する統合テスト。"""

    def _run(self, tmp_path: Path, *, engine: str = "cursor", model: str = "auto",
             wait_return=None) -> Path:
        jobs_dir = _make_jobs_dir(tmp_path)
        if wait_return is None:
            step_uuid = str(uuid.uuid4())
            done_dir = jobs_dir / "done"
            done_dir.mkdir()
            # Create done file so wait_for_result returns immediately
            wait_return = ("success", "")

        with patch(_WAIT, return_value=wait_return) as _mock_wait:
            result_file = jobs_dir / f"20260508230000-{engine}-result-fake.md"
            result_file.write_text("result content", encoding="utf-8")

            # We just need to observe exec.jsonl content; result_path doesn't need to match
            try:
                enqueue_and_wait(
                    prompt="テストプロンプト",
                    engine=engine,
                    model=model,
                    timeout=5.0,
                    idempotency_key="scheduler:test_job:2026-05-08T00:00:00+09:00",
                    jobs_dir=jobs_dir,
                    exec_done_dir=jobs_dir / "done",
                )
            except (StopIteration, OSError):
                pass  # result file not found → OK for this test

        return jobs_dir / "exec.jsonl"

    def test_exec_jsonl_all_lines_valid_json(self, tmp_path):
        """enqueue_and_wait が exec.jsonl に書き込む全行が valid JSON。"""
        exec_jsonl = self._run(tmp_path)
        lines = [l for l in exec_jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1, "exec.jsonl に行が書き込まれていない"
        for line in lines:
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON line in exec.jsonl: {line!r}\n{e}")

    def test_exec_jsonl_no_comment_lines(self, tmp_path):
        """exec.jsonl に # で始まる行（テキスト形式の冪等キー行）が含まれない。"""
        exec_jsonl = self._run(tmp_path)
        for line in exec_jsonl.read_text().splitlines():
            assert not line.startswith("#"), f"comment line found: {line!r}"

    def test_exec_jsonl_record_has_required_fields(self, tmp_path):
        """書き込まれた JSON レコードに uuid / command / result_path が存在する。"""
        exec_jsonl = self._run(tmp_path)
        lines = [l for l in exec_jsonl.read_text().splitlines() if l.strip()]
        record = json.loads(lines[0])
        assert "uuid" in record
        assert "command" in record
        assert "result_path" in record

    def test_exec_jsonl_idempotency_key_in_record_not_comment(self, tmp_path):
        """idempotency_key が JSON フィールドとして埋め込まれ、コメント行として書かれない。"""
        exec_jsonl = self._run(tmp_path)
        content = exec_jsonl.read_text()
        assert '"idempotency_key"' in content
        assert "# idempotency:" not in content

    def test_exec_jsonl_cursor_engine_command_format(self, tmp_path):
        """cursor エンジン時の command が agent -p --force < order_path 形式。"""
        exec_jsonl = self._run(tmp_path, engine="cursor")
        lines = [l for l in exec_jsonl.read_text().splitlines() if l.strip()]
        record = json.loads(lines[0])
        assert "agent" in record["command"]
        assert "-p" in record["command"]
        assert "--force" in record["command"]

    def test_exec_jsonl_claude_engine_command_format(self, tmp_path):
        """claude エンジン時の command が claude -p ... 形式。"""
        exec_jsonl = self._run(tmp_path, engine="claude", model="claude-sonnet-4-6")
        lines = [l for l in exec_jsonl.read_text().splitlines() if l.strip()]
        record = json.loads(lines[0])
        assert "claude" in record["command"]
        assert "--dangerously-skip-permissions" in record["command"]

    def test_idempotency_prevents_duplicate_submission(self, tmp_path):
        """同じ idempotency_key で 2 回呼ぶと exec.jsonl の行数が増えない。"""
        jobs_dir = _make_jobs_dir(tmp_path)
        key = "scheduler:diary_review:2026-05-08T23:00:00+09:00"

        for _ in range(2):
            try:
                with patch(_WAIT, return_value=("success", "")):
                    enqueue_and_wait(
                        prompt="prompt",
                        engine="cursor",
                        model="auto",
                        timeout=5.0,
                        idempotency_key=key,
                        jobs_dir=jobs_dir,
                        exec_done_dir=jobs_dir / "done",
                    )
            except (StopIteration, OSError):
                pass

        exec_jsonl = jobs_dir / "exec.jsonl"
        lines = [l for l in exec_jsonl.read_text().splitlines() if l.strip()]
        # Second call should be a no-op due to idempotency
        assert len(lines) == 1, (
            f"idempotency が機能していない: {len(lines)} 行書き込まれた"
        )

    def test_timeout_returns_false(self, tmp_path):
        """wait_for_result が TimeoutError を送出したとき (False, 'timeout ...') を返す。"""
        jobs_dir = _make_jobs_dir(tmp_path)

        with patch(_WAIT, side_effect=TimeoutError):
            ok, msg = enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=1.0,
                idempotency_key="scheduler:test:ts",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is False
        assert "timeout" in msg
        assert "1.0" in msg

    def test_timeout_exec_jsonl_still_valid(self, tmp_path):
        """タイムアウトしても exec.jsonl に書き込んだ行は valid JSON のまま。"""
        jobs_dir = _make_jobs_dir(tmp_path)

        with patch(_WAIT, side_effect=TimeoutError):
            enqueue_and_wait(
                prompt="prompt",
                engine="cursor",
                model="auto",
                timeout=1.0,
                idempotency_key="scheduler:test:ts",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        exec_jsonl = jobs_dir / "exec.jsonl"
        for line in exec_jsonl.read_text().splitlines():
            if line.strip():
                json.loads(line)  # must not raise
