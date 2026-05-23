"""mltgnt.scheduler.ghdag_bridge — LLMPipelineAPI + wait_for_result のラッパー。

scheduler の action: skill から呼ばれ、order/result ファイルを残しつつ
(bool, str) インタフェースを提供する。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from mltgnt.persona import load_persona

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@dataclass
class DagStep:
    """enqueue_dag に渡す 1 ステップの定義。"""

    id: str
    prompt: str
    engine: str
    model: str | None = None
    persona_name: str | None = None
    depends: list[str] = field(default_factory=list)


def enqueue_dag(
    steps: list[DagStep],
    timeout: float,
    idempotency_key: str,
    jobs_dir: Path,
    exec_done_dir: Path,
    persona_dir: Path | None = None,
) -> list[tuple[bool, str]]:
    """複数ステップを依存関係付きで投入し、全完了を待つ。

    Returns:
        入力ステップと同順の (bool, str) リスト。
        (True, content)       — ステップ成功
        (True, "")            — 冪等性チェックで既投入
        (False, "timeout Ns") — タイムアウト
        (False, "status: msg") — ステップ失敗
    """
    if not steps:
        raise ValueError("steps must not be empty")

    from ghdag.pipeline import (
        InlineOrderBuilder,
        LLMPipelineAPI,
        PipelineState,
        wait_for_result,
    )
    from ghdag.pipeline.audit import AuditContext
    from ghdag.workflow.schema import StepConfig

    step_configs: list[StepConfig] = []
    for step in steps:
        prompt = step.prompt
        if step.persona_name is not None:
            persona = load_persona(step.persona_name, persona_dir=persona_dir)
            prompt = persona.format_prompt(prompt)
        step_configs.append(
            StepConfig(
                id=step.id,
                template=prompt,
                engine=step.engine,
                model=step.model or "",
                depends=step.depends,
            )
        )

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
        return [(True, "")] * len(steps)

    exec_lines = api.submit(
        step_configs,
        base_context={"workflow_name": "scheduler"},
        idempotency_key=idempotency_key,
        audit_context=AuditContext(source="mltgnt-scheduler"),
    )

    data_lines = [ln for ln in exec_lines if not ln.startswith("#") and ln.strip()]
    step_exec: list[tuple[str, str]] = []  # (uuid, exec_line) per step in order
    for exec_line in data_lines:
        m = _UUID_RE.search(exec_line)
        step_exec.append((m.group(0) if m else "", exec_line))

    results: list[tuple[bool, str]] = []
    start = time.monotonic()
    timed_out = False

    for step_uuid, exec_line in step_exec:
        if timed_out:
            results.append((False, f"timeout ({timeout}s)"))
            continue
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            timed_out = True
            results.append((False, f"timeout ({timeout}s)"))
            continue
        try:
            status, first_line = wait_for_result(exec_done_dir, step_uuid, timeout=remaining)
        except TimeoutError:
            timed_out = True
            results.append((False, f"timeout ({timeout}s)"))
            continue

        if status == "success":
            from ghdag.files import md_read

            result_filename = _extract_result_filename(exec_line)
            try:
                md_file = md_read(str(jobs_dir / result_filename), repo_root=jobs_dir.parent)
                content = md_file.content.strip()
            except OSError:
                content = ""
            results.append((True, content))
        else:
            results.append((False, f"{status}: {first_line}"))

    return results


def enqueue_and_wait(
    prompt: str,
    engine: str,
    model: str | None,
    timeout: float,
    idempotency_key: str,
    jobs_dir: Path,
    exec_done_dir: Path,
    persona_name: str | None = None,
    persona_dir: Path | None = None,
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
        persona_name: ペルソナ名。指定時は load_persona → format_prompt でプロンプトを変換する
        persona_dir: ペルソナファイルのディレクトリ（None の場合は load_persona のデフォルト）

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
    from ghdag.pipeline.audit import AuditContext
    from ghdag.workflow.schema import StepConfig

    if persona_name is not None:
        persona = load_persona(persona_name, persona_dir=persona_dir)
        prompt = persona.format_prompt(prompt)

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
        audit_context=AuditContext(source="mltgnt-scheduler"),
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
        from ghdag.files import md_read

        result_filename = _extract_result_filename(skill_line)
        try:
            md_file = md_read(str(jobs_dir / result_filename), repo_root=jobs_dir.parent)
            content = md_file.content.strip()
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
