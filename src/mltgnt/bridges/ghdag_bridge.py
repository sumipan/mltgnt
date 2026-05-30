"""mltgnt.bridges.ghdag_bridge — LLMPipelineAPI + wait_for_result のラッパー。

scheduler の action: skill から呼ばれ、order/result ファイルを残しつつ
(bool, str) インタフェースを提供する。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ghdag.files import md_read

from mltgnt.skill.models import SkillMeta

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@dataclass
class DagStep:
    """enqueue_dag に渡す 1 ステップの定義。"""

    id: str
    prompt: str
    engine: str
    model: str | None = None
    depends: list[str] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)
    skill_name: str | None = None


class SkillIOTypeError(TypeError):
    """compose-time typecheck で検出されたパイプ型不整合。"""


def _scheduler_audit_context(
    correlation_id: str | None,
    parent_correlation_id: str | None,
    request_id: str | None = None,
):
    """mltgnt-scheduler 用 AuditContext。ghdag < v0.25.5 では parent_correlation_id を省略する。"""
    from ghdag.pipeline.audit import AuditContext

    kwargs: dict = {
        "source": "mltgnt-scheduler",
        "correlation_id": correlation_id,
        "request_id": request_id,
    }
    if parent_correlation_id is not None:
        kwargs["parent_correlation_id"] = parent_correlation_id
    return AuditContext(**kwargs)


def _topological_sort(steps: list[DagStep]) -> list[DagStep]:
    """Kahn's algorithm によるトポロジカルソート。循環依存時は ValueError を送出する。"""
    step_map = {s.id: s for s in steps}
    in_degree = {s.id: 0 for s in steps}
    adjacency: dict[str, list[str]] = {s.id: [] for s in steps}
    for step in steps:
        for dep_id in step.depends:
            if dep_id not in step_map:
                raise ValueError(f"Unknown dependency: {dep_id!r}")
            adjacency[dep_id].append(step.id)
            in_degree[step.id] += 1
    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    result: list[DagStep] = []
    while queue:
        node = queue.pop(0)
        result.append(step_map[node])
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) < len(steps):
        raise ValueError("circular dependency detected")
    return result


def _format_typecheck_error(
    downstream: DagStep,
    downstream_meta: SkillMeta,
    upstream: DagStep | None,
    upstream_meta: SkillMeta | None,
    *,
    field: str,
    expected: str,
    actual: str,
    consume_index: int | None = None,
) -> str:
    lines = [
        "SkillIOTypeError: pipe type mismatch in DAG step "
        f"'{downstream.id}' (skill: {downstream.skill_name})",
    ]
    if upstream is not None and upstream_meta is not None:
        lines.append(
            f"  upstream: '{upstream.id}' (skill: {upstream.skill_name})"
        )
    if consume_index is not None:
        lines.append(f"  field: {field}")
        lines.append(f"  expected: {expected} (downstream consumes[{consume_index}])")
        lines.append(f"  actual: {actual}")
        if upstream_meta is not None:
            lines.append(
                "  fix: align content_type in "
                f"skills/{upstream_meta.name}/SKILL.md produces section"
            )
        lines.append("  or: set skill_io: legacy on downstream to skip typecheck")
    else:
        lines.append(f"  field: {field}")
        lines.append(f"  expected producer: {expected}")
        lines.append(f"  actual upstream skill: {actual}")
    return "\n".join(lines)


def typecheck_dag(
    steps: list[DagStep],
    skills: dict[str, SkillMeta],
) -> None:
    """DAG エッジの produces/consumes 型整合を検証する。

    不整合があれば SkillIOTypeError を送出する。
    skill_name が None または skills に存在しないステップはスキップ。
    skill_io が "legacy" の下流もスキップ。
    """
    step_map = {s.id: s for s in steps}

    for downstream in steps:
        if downstream.skill_name is None or downstream.skill_name not in skills:
            continue
        downstream_meta = skills[downstream.skill_name]
        if downstream_meta.skill_io != "v1":
            continue

        if not downstream_meta.consumes:
            print(
                f"WARN: v1 skill '{downstream_meta.name}' (step '{downstream.id}') "
                "participates in pipe but declares no consumes",
                file=sys.stderr,
            )
            continue

        for i, req in enumerate(downstream_meta.consumes):
            resolvable: list[tuple[DagStep, SkillMeta]] = []
            for dep_id in downstream.depends:
                dep = step_map.get(dep_id)
                if dep is None or dep.skill_name is None or dep.skill_name not in skills:
                    continue
                resolvable.append((dep, skills[dep.skill_name]))

            upstream_step: DagStep | None = None
            upstream_meta: SkillMeta | None = None
            for dep, meta in resolvable:
                if meta.name == req.producer:
                    upstream_step = dep
                    upstream_meta = meta
                    break

            if upstream_step is None or upstream_meta is None:
                if not resolvable:
                    continue
                msg = _format_typecheck_error(
                    downstream,
                    downstream_meta,
                    None,
                    None,
                    field="producer",
                    expected=req.producer,
                    actual="(no matching upstream in depends)",
                    consume_index=i,
                )
                raise SkillIOTypeError(msg)

            upstream_produces = upstream_meta.produces
            actual_ct = (
                upstream_produces.content_type
                if upstream_produces is not None
                else "text/markdown"
            )
            if req.content_type != actual_ct:
                msg = _format_typecheck_error(
                    downstream,
                    downstream_meta,
                    upstream_step,
                    upstream_meta,
                    field="content_type",
                    expected=req.content_type,
                    actual=f"{actual_ct} (upstream produces.content_type)",
                    consume_index=i,
                )
                raise SkillIOTypeError(msg)


def enqueue_dag(
    steps: list[DagStep],
    timeout: float,
    idempotency_key: str,
    jobs_dir: Path,
    exec_done_dir: Path,
    persona_dir: Path | None = None,
    correlation_id: str | None = None,
    parent_correlation_id: str | None = None,
    request_id: str | None = None,
    skills: dict[str, SkillMeta] | None = None,
) -> list[tuple[bool, str]]:
    """複数ステップを依存関係付きで逐次投入し、全完了を待つ。

    各ステップを 1 つずつ投入・完了待ちし、前段の result を後段の base_context に注入する。

    Returns:
        入力ステップと同順の (bool, str) リスト。
        (True, content)       — ステップ成功
        (True, "")            — 冪等性チェックで既投入
        (False, "timeout Ns") — タイムアウト
        (False, "status: msg") — ステップ失敗
        (False, "dependency failed") — 先行ステップ失敗
    """
    if not steps:
        raise ValueError("steps must not be empty")

    from ghdag.pipeline import (
        InlineOrderBuilder,
        LLMPipelineAPI,
        PipelineState,
        wait_for_result,
    )
    from ghdag.workflow.schema import StepConfig

    state = PipelineState(
        state_dir=jobs_dir / ".pipeline-state",
        exec_jsonl_path=jobs_dir / "exec.jsonl",
    )
    api = LLMPipelineAPI(
        pipeline_state=state,
        order_builder=InlineOrderBuilder(),
        queue_dir=str(jobs_dir),
    )

    if not api.check_idempotency(idempotency_key):
        return [(True, "")] * len(steps)

    sorted_steps = _topological_sort(steps)

    if os.environ.get("SKILL_IO_TYPECHECK") == "1" and skills is not None:
        typecheck_dag(sorted_steps, skills)

    completed_results: dict[str, str] = {}
    failed_steps: set[str] = set()
    results_by_id: dict[str, tuple[bool, str]] = {}
    start = time.monotonic()
    first_submit = True

    for step in sorted_steps:
        if any(dep_id in failed_steps for dep_id in step.depends):
            results_by_id[step.id] = (False, "dependency failed")
            failed_steps.add(step.id)
            continue

        # コンテキストのマージ（優先度: 固定値 < 自動注入 < ユーザー指定）
        base_context: dict[str, str] = {"workflow_name": "scheduler"}
        for dep_id in step.depends:
            if dep_id in completed_results:
                base_context[f"{dep_id}_result"] = completed_results[dep_id]
        base_context.update(step.context)

        step_config = StepConfig(
            id=step.id,
            template=step.prompt,
            engine=step.engine,
            model=step.model or "",
            depends=[],  # 順序制御は enqueue_dag 側が担保するため不要
        )

        exec_lines = api.submit(
            [step_config],
            base_context=base_context,
            idempotency_key=idempotency_key if first_submit else None,
            audit_context=_scheduler_audit_context(
                correlation_id, parent_correlation_id, request_id
            ),
        )
        first_submit = False

        data_lines = [ln for ln in exec_lines if not ln.startswith("#") and ln.strip()]
        if not data_lines:
            results_by_id[step.id] = (False, "no exec line returned")
            failed_steps.add(step.id)
            continue

        exec_line = data_lines[0]
        m = _UUID_RE.search(exec_line)
        step_uuid = m.group(0) if m else ""

        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            results_by_id[step.id] = (False, f"timeout ({timeout}s)")
            failed_steps.add(step.id)
            continue

        try:
            status, first_line = wait_for_result(exec_done_dir, step_uuid, timeout=remaining)
        except TimeoutError:
            results_by_id[step.id] = (False, f"timeout ({timeout}s)")
            failed_steps.add(step.id)
            continue

        if status == "success":
            result_filename = _extract_result_filename(exec_line)
            try:
                content = md_read(result_filename, repo_root=jobs_dir).content.strip()
            except OSError:
                content = ""
            completed_results[step.id] = content
            results_by_id[step.id] = (True, content)
        else:
            results_by_id[step.id] = (False, f"{status}: {first_line}")
            failed_steps.add(step.id)

    return [results_by_id[step.id] for step in steps]


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
    correlation_id: str | None = None,
    parent_correlation_id: str | None = None,
    request_id: str | None = None,
) -> tuple[bool, str]:
    """LLMPipelineAPI 経由で order を投入し、完了まで待って結果を返す。

    Args:
        prompt: order ファイルに書き込むプロンプト本文（ペルソナ書式変換は呼び出し元が実施済みであること）
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
        exec_jsonl_path=jobs_dir / "exec.jsonl",
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
        audit_context=_scheduler_audit_context(
            correlation_id, parent_correlation_id, request_id
        ),
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
        result_filename = _extract_result_filename(skill_line)
        try:
            content = md_read(result_filename, repo_root=jobs_dir).content.strip()
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
