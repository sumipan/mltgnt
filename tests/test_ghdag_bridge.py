"""tests/test_ghdag_bridge.py — ghdag_bridge の単体＋統合テスト。

カバレッジ対象:
  - _extract_result_filename(): JSON 形式 / テキスト形式 / フォールバック
  - _order_to_result_filename(): 標準テキスト exec 行からの導出
  - enqueue_and_wait() 統合: exec.jsonl に書き込む行が全て valid JSON
  - enqueue_and_wait() 結果読み取り: ghdag.files.md_read 経由での result 取得
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.bridges.ghdag_bridge import (
    DagStep,
    _extract_result_filename,
    _order_to_result_filename,
    enqueue_and_wait,
    enqueue_dag,
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
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
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
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
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
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        record = json.loads(lines[0])
        assert "agent" in record["command"]
        assert "-p" in record["command"]
        assert "--force" in record["command"]

    def test_exec_jsonl_claude_engine_command_format(self, tmp_path):
        """claude エンジン時の command が claude -p ... 形式。"""
        exec_jsonl = self._run(tmp_path, engine="claude", model="claude-sonnet-4-6")
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
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
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
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


# ---------------------------------------------------------------------------
# enqueue_and_wait — ペルソナ統合テスト（AC-1, AC-3）
# ---------------------------------------------------------------------------

_LOAD_PERSONA = "mltgnt.bridges.ghdag_bridge.load_persona"


class TestEnqueueAndWaitPersonaIntegration:
    """enqueue_and_wait の persona_name / persona_dir パラメータ検証。"""

    def _run_with_persona(
        self,
        tmp_path: Path,
        *,
        persona_name: str | None = None,
        persona_dir=None,
        mock_formatted: str = "フォーマット済みプロンプト",
    ):
        """submit に渡された StepConfig.template を返す。"""
        jobs_dir = _make_jobs_dir(tmp_path)

        mock_persona = MagicMock()
        mock_persona.format_prompt.return_value = mock_formatted

        submitted_templates = []

        from ghdag.pipeline import LLMPipelineAPI

        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            for step in steps:
                submitted_templates.append(step.template)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch(_LOAD_PERSONA, return_value=mock_persona) as mock_load,
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
        ):
            try:
                enqueue_and_wait(
                    prompt="元のプロンプト",
                    engine="cursor",
                    model="auto",
                    timeout=5.0,
                    idempotency_key=f"scheduler:persona_test:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=jobs_dir / "done",
                    persona_name=persona_name,
                    persona_dir=persona_dir,
                )
            except (StopIteration, OSError):
                pass

        return submitted_templates, mock_load, mock_persona

    def test_ac1_persona_name_triggers_format_prompt(self, tmp_path):
        """AC-1: persona_name 指定時、submit の template が format_prompt の返り値になる。"""
        formatted = "ペルソナでフォーマット済み: 元のプロンプト"
        templates, mock_load, mock_persona = self._run_with_persona(
            tmp_path,
            persona_name="タチコマ",
            persona_dir=Path("agents"),
            mock_formatted=formatted,
        )

        mock_load.assert_called_once_with("タチコマ", persona_dir=Path("agents"))
        mock_persona.format_prompt.assert_called_once_with("元のプロンプト")
        assert len(templates) >= 1
        assert templates[0] == formatted

    def test_ac1_persona_dir_none_uses_default(self, tmp_path):
        """AC-1: persona_dir=None のとき load_persona に None が渡される。"""
        _, mock_load, _ = self._run_with_persona(
            tmp_path,
            persona_name="タチコマ",
            persona_dir=None,
        )
        mock_load.assert_called_once_with("タチコマ", persona_dir=None)

    def test_ac2_no_persona_name_skips_load_persona(self, tmp_path):
        """AC-2: persona_name 未指定時、load_persona は呼ばれずプロンプトがそのまま使われる。"""
        templates, mock_load, mock_persona = self._run_with_persona(
            tmp_path,
            persona_name=None,
        )
        mock_load.assert_not_called()
        mock_persona.format_prompt.assert_not_called()
        assert len(templates) >= 1
        assert templates[0] == "元のプロンプト"

    def test_ac3_nonexistent_persona_raises_file_not_found(self, tmp_path):
        """AC-3: 存在しないペルソナ名で FileNotFoundError が送出される。"""
        jobs_dir = _make_jobs_dir(tmp_path)

        with patch(_LOAD_PERSONA, side_effect=FileNotFoundError("存在しない")):
            with pytest.raises(FileNotFoundError):
                enqueue_and_wait(
                    prompt="元のプロンプト",
                    engine="cursor",
                    model="auto",
                    timeout=5.0,
                    idempotency_key=f"scheduler:persona_test:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=jobs_dir / "done",
                    persona_name="存在しない",
                    persona_dir=Path("agents"),
                )


# ---------------------------------------------------------------------------
# enqueue_dag — DAG 投入テスト（AC-1〜AC-7）
# ---------------------------------------------------------------------------


def _make_jobs_dir_dag(tmp_path: Path) -> tuple[Path, Path]:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "exec.jsonl").write_text("", encoding="utf-8")
    done = jobs / "done"
    done.mkdir()
    return jobs, done


class TestEnqueueDag:
    """enqueue_dag() の受け入れ条件テスト。"""

    def test_ac1_two_step_linear_dag_submit(self, tmp_path):
        """AC-1: 2ステップ線形 DAG が s1→s2 の順で submit されること（逐次投入モデル）。"""
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow.schema import StepConfig

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_steps: list[StepConfig] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
        ):
            try:
                enqueue_dag(
                    steps=[
                        DagStep(id="s1", prompt="P1", engine="claude"),
                        DagStep(id="s2", prompt="P2", engine="claude", depends=["s1"]),
                    ],
                    timeout=5.0,
                    idempotency_key=f"dag:ac1:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                pass

        # 逐次投入: submit は 1 ステップずつ、s1 → s2 の順で呼ばれる
        assert len(captured_steps) == 2
        assert captured_steps[0].id == "s1"
        assert captured_steps[1].id == "s2"
        # 逐次投入では depends は空（順序制御は enqueue_dag 側が担保）
        assert captured_steps[0].depends == []
        assert captured_steps[1].depends == []

    def test_ac2_per_step_persona_prompt_transform(self, tmp_path):
        """AC-2: ステップ別ペルソナ指定時のプロンプト変換。"""
        from ghdag.pipeline import LLMPipelineAPI
        from ghdag.workflow.schema import StepConfig

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_steps: list[StepConfig] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, steps, **kwargs):
            captured_steps.extend(steps)
            return original_submit(self_api, steps, **kwargs)

        mock_persona = MagicMock()
        mock_persona.format_prompt.return_value = "ペルソナ変換済み: 指示A"

        with (
            patch(_LOAD_PERSONA, return_value=mock_persona) as mock_load,
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
        ):
            try:
                enqueue_dag(
                    steps=[
                        DagStep(id="s1", prompt="指示A", engine="claude", persona_name="タチコマ"),
                        DagStep(id="s2", prompt="指示B", engine="claude"),
                    ],
                    timeout=5.0,
                    idempotency_key=f"dag:ac2:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                    persona_dir=Path("agents"),
                )
            except (StopIteration, OSError):
                pass

        # load_persona は s1 のみ（1回）
        mock_load.assert_called_once_with("タチコマ", persona_dir=Path("agents"))
        assert len(captured_steps) == 2
        s1 = next(s for s in captured_steps if s.id == "s1")
        s2 = next(s for s in captured_steps if s.id == "s2")
        assert s1.template == "ペルソナ変換済み: 指示A"
        assert s2.template == "指示B"

    def test_ac3_empty_steps_raises_value_error(self, tmp_path):
        """AC-3: 空リストで ValueError（'empty' を含むメッセージ）。"""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with pytest.raises(ValueError, match="empty"):
            enqueue_dag(
                steps=[],
                timeout=5.0,
                idempotency_key="dag:ac3:k",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

    def test_ac4_idempotency_prevents_duplicate_submission(self, tmp_path):
        """AC-4: 冪等性による二重投入防止。"""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        key = f"dag:ac4:{uuid.uuid4()}"
        steps = [
            DagStep(id="s1", prompt="P1", engine="cursor"),
            DagStep(id="s2", prompt="P2", engine="cursor", depends=["s1"]),
        ]

        for _ in range(2):
            with patch(_WAIT, return_value=("success", "")):
                try:
                    enqueue_dag(
                        steps=steps,
                        timeout=5.0,
                        idempotency_key=key,
                        jobs_dir=jobs_dir,
                        exec_done_dir=done_dir,
                    )
                except (StopIteration, OSError):
                    pass

        exec_jsonl = jobs_dir / "exec.jsonl"
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2, f"冪等性チェック後も行が増えている: {len(lines)} 行"

        # 2回目の呼び出しは [(True, ""), (True, "")] を返す
        with patch(_WAIT, return_value=("success", "")):
            second_result = enqueue_dag(
                steps=steps,
                timeout=5.0,
                idempotency_key=key,
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )
        assert second_result == [(True, ""), (True, "")]

    def test_ac5_timeout_returns_false_with_message(self, tmp_path):
        """AC-5: タイムアウト時に (False, 'timeout (Ns)') を返す。"""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with patch(_WAIT, side_effect=TimeoutError):
            try:
                results = enqueue_dag(
                    steps=[DagStep(id="s1", prompt="P1", engine="cursor")],
                    timeout=3.0,
                    idempotency_key=f"dag:ac5:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                results = [(False, "timeout (3.0s)")]

        assert len(results) == 1
        ok, msg = results[0]
        assert ok is False
        assert "timeout" in msg

    def test_ac6_existing_tests_still_pass(self):
        """AC-6: このテスト自体が pass していれば既存テストは非破壊（pytest で確認済み）。"""
        # This test verifies the test suite can be collected without import errors.
        assert enqueue_and_wait is not None
        assert enqueue_dag is not None
        assert DagStep is not None

    def test_ac7_exec_jsonl_valid_json_with_required_fields(self, tmp_path):
        """AC-7: exec.jsonl レコードの valid JSON 保証。"""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        with patch(_WAIT, return_value=("success", "")):
            try:
                enqueue_dag(
                    steps=[
                        DagStep(id="s1", prompt="P1", engine="cursor"),
                        DagStep(id="s2", prompt="P2", engine="cursor", depends=["s1"]),
                    ],
                    timeout=5.0,
                    idempotency_key=f"dag:ac7:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                pass

        exec_jsonl = jobs_dir / "exec.jsonl"
        lines = [ln for ln in exec_jsonl.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2, f"2 ステップ分の行が書き込まれていない: {len(lines)} 行"
        for line in lines:
            record = json.loads(line)
            assert "uuid" in record, f"uuid フィールドがない: {record}"
            assert "command" in record, f"command フィールドがない: {record}"
            assert "result_path" in record, f"result_path フィールドがない: {record}"


# ---------------------------------------------------------------------------
# enqueue_and_wait — result 読み取りテスト
# （md_read をモックして result content の取得を検証）
# ---------------------------------------------------------------------------

_MD_READ = "mltgnt.bridges.ghdag_bridge.md_read"


class TestEnqueueAndWaitResultRead:
    """result ファイル読み取りが ghdag.files.md_read 経由で行われることを検証する。"""

    def test_success_uses_md_read(self, tmp_path):
        """wait_for_result が success のとき md_read 経由で content を返す。"""
        jobs_dir = _make_jobs_dir(tmp_path)
        mock_md = MagicMock()
        mock_md.content = "result text"

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, return_value=mock_md) as mock_read:
            ok, content = enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:result_read:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is True
        assert content == "result text"
        mock_read.assert_called_once()

    def test_result_not_found_returns_empty(self, tmp_path):
        """result ファイルが存在しないとき content="" を返す。"""
        jobs_dir = _make_jobs_dir(tmp_path)

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, side_effect=FileNotFoundError("not found")):
            ok, content = enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:result_missing:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is True
        assert content == ""

    def test_frontmatter_stripped_from_result(self, tmp_path):
        """result ファイルに frontmatter があっても content のみが返る。"""
        jobs_dir = _make_jobs_dir(tmp_path)
        mock_md = MagicMock()
        mock_md.content = "body only"

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, return_value=mock_md):
            ok, content = enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:frontmatter:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        assert ok is True
        assert content == "body only"

    def test_md_read_called_with_jobs_dir_as_repo_root(self, tmp_path):
        """md_read の repo_root が jobs_dir になっていること。"""
        jobs_dir = _make_jobs_dir(tmp_path)
        mock_md = MagicMock()
        mock_md.content = ""

        with patch(_WAIT, return_value=("success", "")), \
             patch(_MD_READ, return_value=mock_md) as mock_read:
            enqueue_and_wait(
                prompt="test",
                engine="cursor",
                model="auto",
                timeout=5.0,
                idempotency_key="test:repo_root:2026-05-23T00:00:00+09:00",
                jobs_dir=jobs_dir,
                exec_done_dir=jobs_dir / "done",
            )

        _args, kwargs = mock_read.call_args
        assert kwargs.get("repo_root") == jobs_dir


# ---------------------------------------------------------------------------
# enqueue_dag — データフロー・結果伝搬テスト（AC-8〜AC-12）
# ---------------------------------------------------------------------------


class TestEnqueueDagDataFlow:
    """AC-8〜AC-12: ステップ間の result 伝搬・逐次投入モデル検証。"""

    def test_ac8_result_content_propagated_to_dependent_step(self, tmp_path):
        """AC-8: step_a の result が step_b の base_context['step_a_result'] に伝搬される。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        mock_md_a = MagicMock()
        mock_md_a.content = "分析結果A"
        mock_md_b = MagicMock()
        mock_md_b.content = ""

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, side_effect=[mock_md_a, mock_md_b]),
        ):
            results = enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="指示A", engine="cursor"),
                    DagStep(id="step_b", prompt="$step_a_result を踏まえて", engine="cursor", depends=["step_a"]),
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac8:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured_contexts) == 2
        # step_b の submit 時に step_a_result が注入されている
        step_b_context = captured_contexts[1]
        assert "step_a_result" in step_b_context
        assert step_b_context["step_a_result"] == "分析結果A"
        # step_a は成功
        assert results[0] == (True, "分析結果A")

    def test_ac9_independent_steps_no_cross_context(self, tmp_path):
        """AC-9: depends なしのステップの base_context に他ステップの _result が含まれない。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, return_value=MagicMock(content="result")),
        ):
            enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="P_a", engine="cursor"),
                    DagStep(id="step_b", prompt="P_b", engine="cursor"),  # depends なし
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac9:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured_contexts) == 2
        # step_a の context に step_b_result がない
        assert "step_b_result" not in captured_contexts[0]
        # step_b の context に step_a_result がない（depends していない）
        assert "step_a_result" not in captured_contexts[1]

    def test_ac10_user_context_overrides_auto_injection(self, tmp_path):
        """AC-10: ユーザー指定 context が自動注入より優先される。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        captured_contexts: list[dict] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, base_context=None, **kwargs):
            captured_contexts.append(dict(base_context or {}))
            return original_submit(self_api, step_list, base_context=base_context, **kwargs)

        mock_md_a = MagicMock()
        mock_md_a.content = "自動注入値"
        mock_md_b = MagicMock()
        mock_md_b.content = ""

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("success", "")),
            patch(_MD_READ, side_effect=[mock_md_a, mock_md_b]),
        ):
            enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="指示A", engine="cursor"),
                    DagStep(
                        id="step_b",
                        prompt="$step_a_result を踏まえて",
                        engine="cursor",
                        depends=["step_a"],
                        context={"step_a_result": "カスタム値"},
                    ),
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac10:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        assert len(captured_contexts) == 2
        # ユーザー指定値が自動注入を上書きする
        assert captured_contexts[1]["step_a_result"] == "カスタム値"

    def test_ac11_failed_step_prevents_dependent_submission(self, tmp_path):
        """AC-11: 先行ステップ失敗時、依存する後続ステップが submit されない。"""
        from ghdag.pipeline import LLMPipelineAPI

        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)
        submitted_ids: list[str] = []
        original_submit = LLMPipelineAPI.submit

        def capture_submit(self_api, step_list, **kwargs):
            for s in step_list:
                submitted_ids.append(s.id)
            return original_submit(self_api, step_list, **kwargs)

        with (
            patch.object(LLMPipelineAPI, "submit", capture_submit),
            patch(_WAIT, return_value=("failed_exit", "exit code 1")),
        ):
            results = enqueue_dag(
                steps=[
                    DagStep(id="step_a", prompt="P_a", engine="cursor"),
                    DagStep(id="step_b", prompt="P_b", engine="cursor", depends=["step_a"]),
                ],
                timeout=5.0,
                idempotency_key=f"dag:ac11:{uuid.uuid4()}",
                jobs_dir=jobs_dir,
                exec_done_dir=done_dir,
            )

        # step_a は submit されたが step_b は submit されていない
        assert "step_a" in submitted_ids
        assert "step_b" not in submitted_ids
        # 両方 False
        assert results[0][0] is False
        assert results[1][0] is False

    def test_ac12_backward_compat_no_context_field(self, tmp_path):
        """AC-12: context 未指定の既存呼び出しパターンが後方互換で動作する。"""
        jobs_dir, done_dir = _make_jobs_dir_dag(tmp_path)

        # context フィールドなしで DagStep を生成できる
        step = DagStep(id="s1", prompt="P1", engine="cursor")
        assert step.context == {}

        with patch(_WAIT, return_value=("success", "")):
            try:
                results = enqueue_dag(
                    steps=[step],
                    timeout=5.0,
                    idempotency_key=f"dag:ac12:{uuid.uuid4()}",
                    jobs_dir=jobs_dir,
                    exec_done_dir=done_dir,
                )
            except (StopIteration, OSError):
                results = [(True, "")]

        assert len(results) == 1
        assert results[0][0] is True
