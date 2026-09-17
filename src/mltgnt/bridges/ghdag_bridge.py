"""mltgnt.bridges.ghdag_bridge — wrapper around LLMPipelineAPI + wait_for_result.

Called from scheduler action: skill; keeps order/result files and exposes
a (bool, str) interface.
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
from ghdag.pipeline.order import OrderBuilder

from mltgnt.skill.models import SkillMatchResult, SkillMeta, SkillRunResult
from mltgnt.skill.runner import write_result_frontmatter

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_PIPELINE_STATUS_RE = re.compile(r"^PIPELINE_STATUS:\s*(\S+)\s*$", re.MULTILINE)


@dataclass
class DagStep:
    """Definition of one step passed to enqueue_dag."""

    id: str
    prompt: str
    engine: str
    model: str | None = None
    depends: list[str] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)
    skill_name: str | None = None


class SkillIOTypeError(TypeError):
    """Pipe type mismatch detected by compose-time typecheck."""


def _extract_pipeline_status(content: str) -> str | None:
    """Extract PIPELINE_STATUS: <value> from result content (last match)."""
    matches = _PIPELINE_STATUS_RE.findall(content)
    return matches[-1] if matches else None


def compose_pipeline(
    match_results: list[SkillMatchResult],
    *,
    engine: str,
    model: str | None = None,
) -> list[DagStep]:
    """Convert a SkillMatchResult sequence into a linear DagStep pipe.

    - ValueError if any element has decisive is None
    - step_id is ``pipe_{i}_{skill_name}``; depends is the prior step_id
    - skill_io: for v1 downstream, if consumes.producer mismatches the prior
      skill name, raise SkillIOTypeError (fail fast before writing exec.jsonl).
      Skip for legacy.
    """
    if not match_results:
        raise ValueError("match_results must not be empty")

    steps: list[DagStep] = []
    for i, result in enumerate(match_results):
        if result.decisive is None:
            raise ValueError(
                f"compose_pipeline: match_results[{i}].decisive is None "
                f"(rationale={result.rationale!r})"
            )
        meta = result.decisive
        skill_name = meta.name

        if i > 0 and meta.skill_io == "v1" and meta.consumes:
            prev = match_results[i - 1].decisive
            assert prev is not None  # validated on prior iteration
            prev_name = prev.name
            if not any(req.producer == prev_name for req in meta.consumes):
                expected = ", ".join(req.producer for req in meta.consumes)
                raise SkillIOTypeError(
                    "SkillIOTypeError: pipe type mismatch in compose_pipeline "
                    f"step pipe_{i}_{skill_name} (skill: {skill_name})\n"
                    f"  field: producer\n"
                    f"  expected producer: {expected}\n"
                    f"  actual upstream skill: {prev_name}"
                )

        depends = [steps[i - 1].id] if i > 0 else []
        steps.append(
            DagStep(
                id=f"pipe_{i}_{skill_name}",
                prompt=result.arguments,
                engine=engine,
                model=model,
                depends=depends,
                skill_name=skill_name,
            )
        )
    return steps


def _scheduler_audit_context(
    correlation_id: str | None,
    parent_correlation_id: str | None,
    request_id: str | None = None,
):
    """AuditContext for mltgnt-scheduler. Omit parent_correlation_id on ghdag < v0.25.5."""
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
    """Topological sort via Kahn's algorithm. Raise ValueError on cycles."""
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
    """Validate produces/consumes type alignment on DAG edges.

    Raise SkillIOTypeError on mismatch.
    Skip steps with skill_name None or missing from skills.
    Also skip downstream with skill_io "legacy".
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
    permission: str | None = None,
    order_builder: OrderBuilder | None = None,
) -> list[tuple[bool, str]]:
    """Submit multiple steps with dependencies sequentially and wait for all.

    Submit and wait one step at a time; inject prior results into later base_context.

    Returns:
        (bool, str) list in the same order as input steps.
        (True, content)       — step succeeded
        (True, "")            — already submitted (idempotency)
        (False, "timeout Ns") — timeout
        (False, "status: msg") — step failed
        (False, "dependency failed") — upstream step failed
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
        order_builder=order_builder or InlineOrderBuilder(),
        queue_dir=str(jobs_dir),
    )

    if not api.check_idempotency(idempotency_key):
        return [(True, "")] * len(steps)

    sorted_steps = _topological_sort(steps)

    if os.environ.get("SKILL_IO_TYPECHECK") != "0" and skills is not None:
        typecheck_dag(sorted_steps, skills)

    completed_results: dict[str, str] = {}
    pipeline_statuses: dict[str, str] = {}
    failed_steps: set[str] = set()
    results_by_id: dict[str, tuple[bool, str]] = {}
    start = time.monotonic()
    first_submit = True

    for step in sorted_steps:
        if any(dep_id in failed_steps for dep_id in step.depends):
            results_by_id[step.id] = (False, "dependency failed")
            failed_steps.add(step.id)
            continue

        # Merge context (priority: fixed < auto-injected < user-specified)
        base_context: dict[str, str] = {"workflow_name": "scheduler"}
        for dep_id in step.depends:
            if dep_id in completed_results:
                base_context[f"{dep_id}_result"] = completed_results[dep_id]
            if dep_id in pipeline_statuses:
                base_context[f"{dep_id}_pipeline_status"] = pipeline_statuses[dep_id]
        base_context.update(step.context)

        step_config = StepConfig(
            id=step.id,
            template=step.prompt,
            engine=step.engine,
            model=step.model or "",
            depends=[],  # ordering is enforced by enqueue_dag; not needed here
            permission=permission,
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
            pipeline_status = _extract_pipeline_status(content)
            # Fail on INVALID_STATE even if ghdag exit succeeded (block downstream submit)
            if pipeline_status == "INVALID_STATE":
                results_by_id[step.id] = (False, "PIPELINE_STATUS: INVALID_STATE")
                failed_steps.add(step.id)
                continue
            completed_results[step.id] = content
            if pipeline_status is not None:
                pipeline_statuses[step.id] = pipeline_status
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
    permission: str | None = None,
    order_builder: OrderBuilder | None = None,
    run_result: SkillRunResult | None = None,
) -> tuple[bool, str]:
    """Submit an order via LLMPipelineAPI and wait for the result.

    Args:
        prompt: Prompt body written to the order file (caller must already apply persona formatting)
        engine: LLM engine name ("claude", "gemini", etc.)
        model: Model ID (engine default when None)
        timeout: Max wait seconds
        idempotency_key: Idempotency key recorded in exec.jsonl
        jobs_dir: Location for order/result/exec.jsonl (jobs/)
        exec_done_dir: Location for done markers (jobs/done/<uuid>)
        run_result: Skill runner result; write result frontmatter when skill_io=v1

    Returns:
        (True, result_content) — success
        (False, "timeout ({N}s)") — timeout
        (False, "{status}: {first_line}") — failure
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
        order_builder=order_builder or InlineOrderBuilder(),
        queue_dir=str(jobs_dir),
    )

    if not api.check_idempotency(idempotency_key):
        return True, ""

    exec_lines = api.submit(
        [StepConfig(id="skill", template=prompt, engine=engine, model=model or "", permission=permission)],
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
        return False, f"UUID not found in exec_line: {skill_line!r}"
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
        if run_result is not None and result_filename:
            write_result_frontmatter(jobs_dir / result_filename, run_result)
        return True, content

    return False, f"{status}: {first_line}"


def _extract_result_filename(exec_line: str) -> str:
    """Extract the result filename from an exec line (text or JSON string).

    For JSON, take result_path directly.
    For text, derive the result filename from the order file path.
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
    """Derive the result filename from a text-form exec line.

    Example: "jobs/20260505120000-claude-order-uuid.md" → "20260505120000-claude-result-uuid.md"
    """
    m = re.search(r"(\S+)-order-(" + _UUID_RE.pattern + r")\.md", exec_line)
    if not m:
        return ""
    prefix = m.group(1).split("/")[-1]
    uuid = m.group(2)
    return f"{prefix}-result-{uuid}.md"
