"""mltgnt.scheduler.ghdag_bridge — LLMPipelineAPI + wait_for_result のラッパー。

scheduler の action: skill から呼ばれ、order/result ファイルを残しつつ
(bool, str) インタフェースを提供する。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def enqueue_and_wait(
    prompt: str,
    engine: str,
    model: str | None,
    timeout: float,
    idempotency_key: str,
    jobs_dir: Path,
    exec_done_dir: Path,
) -> tuple[bool, str]:
    """LLMPipelineAPI 経由で order を投入し、完了まで待って結果を返す。

    Args:
        prompt: order ファイルに書き込むプロンプト本文
        engine: LLM エンジン名（"claude", "gemini" 等）
        model: モデル ID（None の場合はエンジンのデフォルト）
        timeout: 最大待機秒数
        idempotency_key: exec.jsonl に記録する冪等性キー
        jobs_dir: order/result/exec.jsonl の置き場（jobs/）
        exec_done_dir: 完了マーカー（jobs/done/<uuid>）の置き場

    Returns:
        (True, result_content) — 成功時
        (False, "timeout ({N}s)") — タイムアウト時
        (False, "{status}: {first_line}") — 失敗時
    """
    from ghdag.pipeline import (
        InlineOrderBuilder,
        LLMPipelineAPI,
        PipelineState,
        wait_for_result,
    )
    from ghdag.workflow.schema import StepConfig

    state = PipelineState(
        state_dir=jobs_dir / ".pipeline-state",
        exec_md_path=jobs_dir / "exec.jsonl",
    )
    api = LLMPipelineAPI(
        pipeline_state=state,
        order_builder=InlineOrderBuilder(),
        queue_dir=str(jobs_dir),
    )

    if not api.check_idempotency(idempotency_key):
        return True, ""

    exec_lines = api.submit(
        [StepConfig(id="skill", template=prompt, engine=engine, model=model or "")],
        base_context={"workflow_name": "scheduler"},
        idempotency_key=idempotency_key,
    )

    skill_line = next(
        line for line in exec_lines
        if not line.startswith("#") and line.strip()
    )
    m = _UUID_RE.search(skill_line)
    if not m:
        return False, f"exec_line に UUID が見つかりません: {skill_line!r}"
    step_uuid = m.group(0)

    try:
        status, first_line = wait_for_result(exec_done_dir, step_uuid, timeout=timeout)
    except TimeoutError:
        return False, f"timeout ({timeout}s)"

    if status == "success":
        result_path = jobs_dir / _extract_result_filename(skill_line)
        try:
            content = result_path.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        return True, content

    return False, f"{status}: {first_line}"


def _extract_result_filename(exec_line: str) -> str:
    """exec 行（テキスト形式または JSON 文字列）から result ファイル名を取り出す。

    JSON 形式の場合は result_path フィールドから直接取得する。
    テキスト形式の場合は order ファイルパスから result ファイル名を導出する。
    """
    stripped = exec_line.strip()
    if stripped.startswith("{"):
        try:
            record = json.loads(stripped)
            result_path = record.get("result_path", "")
            return Path(result_path).name if result_path else ""
        except (json.JSONDecodeError, ValueError):
            pass
    return _order_to_result_filename(exec_line)


def _order_to_result_filename(exec_line: str) -> str:
    """テキスト形式の exec 行から result ファイル名を導出する。

    例: "jobs/20260505120000-claude-order-uuid.md" → "20260505120000-claude-result-uuid.md"
    """
    m = re.search(r"(\S+)-order-(" + _UUID_RE.pattern + r")\.md", exec_line)
    if not m:
        return ""
    prefix = m.group(1).split("/")[-1]
    uuid = m.group(2)
    return f"{prefix}-result-{uuid}.md"
