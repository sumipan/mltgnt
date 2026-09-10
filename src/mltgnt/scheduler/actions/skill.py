from __future__ import annotations

import fnmatch
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

from mltgnt.scheduler.fanout import _FANOUT_PROMPT_SUFFIX, _parse_fanout_steps
from mltgnt.scheduler.models import ScheduleJob
from mltgnt.skill.models import ExitStatus

_STATUS_MARKER_RE = re.compile(r"^PIPELINE_STATUS:\s*(\S+)\s*$")


def _snapshot_writes(patterns: list[str], repo_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    scanned: set[Path] = set()
    for pattern in patterns:
        # Find the non-wildcard prefix directory to scan broadly
        dir_parts: list[str] = []
        for part in Path(pattern).parts:
            if any(c in part for c in "*?["):
                break
            dir_parts.append(part)
        scan_dir = repo_root.joinpath(*dir_parts) if dir_parts else repo_root
        if scan_dir in scanned or not scan_dir.is_dir():
            continue
        scanned.add(scan_dir)
        for p in scan_dir.iterdir():
            if p.is_file():
                rel = str(p.relative_to(repo_root))
                result[rel] = p.stat().st_mtime
    return result


def _compute_write_diff(
    before: dict[str, float],
    after: dict[str, float],
) -> list[str]:
    changed = []
    for key, mtime in after.items():
        if key not in before or before[key] != mtime:
            changed.append(key)
    return changed


def _write_side_effect_audit(
    audit_path: Path,
    *,
    skill_name: str,
    job_id: str,
    declared_writes: list[str],
    actual_writes: list[str],
) -> None:
    all_covered = all(
        any(fnmatch.fnmatch(f, pat) for pat in declared_writes)
        for f in actual_writes
    ) if actual_writes else True

    record = {
        "schema_version": 1,
        "event_type": "side_effect_audit",
        "timestamp": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "skill_name": skill_name,
        "job_id": job_id,
        "declared_writes": declared_writes,
        "actual_writes": actual_writes,
        "all_declared_covered": all_covered,
    }
    try:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"side_effect_audit: write failed: {e}", file=sys.stderr)


def _audit_stats_from_extra_context(extra_context: str | None) -> tuple[int, int]:
    """extra_context から audit 用の knowledge パラグラフ数と memory バイト数を算出する。"""
    if not extra_context:
        return 0, 0
    knowledge_count = 0
    memory_bytes = 0
    remainder = extra_context
    mem_header = "### 記憶（末尾）\n\n"
    if mem_header in remainder:
        before, memory_text = remainder.split(mem_header, 1)
        memory_bytes = len(memory_text.encode("utf-8"))
        remainder = before.rstrip("\n")
    if remainder.startswith("### knowledge"):
        parts = remainder.split("\n\n", 1)
        if len(parts) == 2:
            knowledge_text = parts[1]
            knowledge_count = len([p for p in knowledge_text.split("\n\n") if p.strip()])
    return knowledge_count, memory_bytes


def _write_context_injection_audit(
    audit_path: Path,
    *,
    skill_name: str,
    job_id: str,
    knowledge_count: int,
    memory_bytes: int,
) -> None:
    record = {
        "schema_version": 1,
        "event_type": "context_injection",
        "timestamp": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "skill_name": skill_name,
        "job_id": job_id,
        "knowledge_count": knowledge_count,
        "memory_bytes": memory_bytes,
    }
    try:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"context_injection_audit: write failed: {e}", file=sys.stderr)


def _determine_exit_code(ok: bool, msg: str) -> int:
    if ok:
        if "PIPELINE_STATUS: ALREADY_APPLIED" in msg:
            return ExitStatus.ALREADY_APPLIED
        return ExitStatus.SUCCESS
    if "PIPELINE_STATUS: INVALID_STATE" in msg:
        return ExitStatus.INVALID_STATE
    return ExitStatus.USAGE_ERROR


def _extract_status_marker(
    msg: str, expected_markers: list[str]
) -> str | None:
    """msg 先頭 3 行 + 末尾 3 行から status marker を抽出する。

    Pass 1: ``PIPELINE_STATUS: <value>`` 形式（複数時は末尾寄りを採用）。
    Pass 2: Pass 1 未検出時のみ、宣言済み裸マーカー行を照合（複数時は先頭寄り）。
    """
    lines = msg.splitlines()
    if not lines:
        return None
    head = lines[:3]
    tail = lines[-3:] if len(lines) > 3 else []
    scan = head + tail

    found: str | None = None
    for line in scan:
        m = _STATUS_MARKER_RE.match(line)
        if m:
            found = m.group(1)
    if found is not None:
        return found

    if not expected_markers:
        return None
    for line in scan:
        stripped = line.strip()
        for decl in expected_markers:
            if decl.endswith(":"):
                if stripped.startswith(decl):
                    return stripped
            elif stripped == decl:
                return stripped
    return None


def _matches_expected(marker: str, expected_markers: list[str]) -> bool:
    """抽出 marker が expected_markers のいずれかに適合するか。"""
    for decl in expected_markers:
        if decl.endswith(":"):
            if marker.startswith(decl):
                return True
        elif marker == decl:
            return True
    return False


def _resolve_exit_code(
    ok: bool,
    msg: str,
    expected_markers: list[str],
    *,
    enforce: bool = False,
) -> tuple[int, list[str]]:
    """expected_markers があれば marker 突合。強制は enforce=True のときのみ。"""
    if expected_markers:
        marker = _extract_status_marker(msg, expected_markers)
        if marker is None:
            exit_code = (
                ExitStatus.CONTRACT_VIOLATION
                if enforce
                else _determine_exit_code(ok, msg)
            )
            diagnostics = [
                f"exit_code={exit_code}",
                "marker=<absent>",
                "contract_violation: PIPELINE_STATUS marker missing",
                f"expected_markers={expected_markers}",
            ]
            return exit_code, diagnostics
        if not _matches_expected(marker, expected_markers):
            exit_code = (
                ExitStatus.CONTRACT_VIOLATION
                if enforce
                else _determine_exit_code(ok, msg)
            )
            diagnostics = [
                f"exit_code={exit_code}",
                f"violated_marker={marker}",
                "contract_violation: undeclared PIPELINE_STATUS marker",
                f"expected_markers={expected_markers}",
            ]
            return exit_code, diagnostics
        exit_code = _determine_exit_code(ok, msg)
        return exit_code, [
            f"exit_code={exit_code}",
            f"marker={marker}",
        ]

    exit_code = _determine_exit_code(ok, msg)
    marker = _extract_status_marker(msg, expected_markers)
    diagnostics = [f"exit_code={exit_code}"]
    if marker is not None:
        diagnostics.append(f"marker={marker}")
    return exit_code, diagnostics


def _write_skill_result_audit(
    audit_path: Path,
    *,
    skill_name: str,
    job_id: str,
    exit_code: int,
    diagnostics: list[str],
) -> None:
    record = {
        "schema_version": 1,
        "event_type": "skill_result",
        "timestamp": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "skill_name": skill_name,
        "job_id": job_id,
        "exit_code": exit_code,
        "diagnostics": diagnostics,
    }
    try:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"skill_result_audit: write failed: {e}", file=sys.stderr)


def run_skill_action(
    job: ScheduleJob,
    *,
    persona_dir: Path,
    skill_registry: dict[str, Any],
    default_tz: str,
    repo_root: Path,
) -> tuple[bool, str]:
    """skill アクションを実行し (成功フラグ, メッセージ) を返す。"""
    aa = job.action_args
    skill_name = aa.get("skill")
    if not skill_name:
        return False, f"job {job.id}: action_args.skill が未指定です"
    persona_name = aa.get("persona")
    if not persona_name:
        return False, f"job {job.id}: action_args.persona が未指定です"

    from mltgnt.persona import load_persona

    try:
        persona = load_persona(persona_name, persona_dir=persona_dir)
    except FileNotFoundError as e:
        return False, f"ペルソナファイルが見つかりません: {e}"
    except Exception as e:
        return False, f"ペルソナ読込失敗 {persona_name}: {e}"

    engine = aa.get("engine") or (persona.fm.engine or None)
    model = aa.get("model") or (persona.fm.model or None)

    meta = skill_registry.get(skill_name)
    if meta is None:
        return False, f"スキルが見つかりません: {skill_name}"

    from mltgnt.skill import load
    from mltgnt.skill.context import build_extra_context

    skill_file = load(meta)

    argv_list = aa.get("argv", [])
    argv_str = " ".join(str(x) for x in argv_list) if argv_list else ""

    knowledge_count_cfg = aa.get("knowledge_count", 0)
    memory_max_bytes_cfg = aa.get("memory_max_bytes", 0)
    memory_exclude_source_tags_cfg = aa.get("memory_exclude_source_tags", None)
    extra_context = build_extra_context(
        meta,
        repo_root,
        persona_name,
        knowledge_count=knowledge_count_cfg,
        memory_max_bytes=memory_max_bytes_cfg,
        memory_exclude_source_tags=memory_exclude_source_tags_cfg,
    )
    knowledge_count_audit, memory_bytes_audit = _audit_stats_from_extra_context(
        extra_context
    )

    from mltgnt.interfaces.types import ChatInput, Message
    from mltgnt.skill import runner as skill_runner

    chat_input = ChatInput(
        source="scheduler",
        session_key=job.id,
        messages=[Message(role="user", content=argv_str or "")],
        persona_name=persona.name,
        model=model,
    )
    run_output = skill_runner.run(
        skill_file, persona, argv_str, chat_input, extra_context=extra_context
    )

    prompt = next(m["content"] for m in run_output.chat_input.messages if m["role"] == "system")
    resolved_model = run_output.chat_input.model

    # enable_pipeline 優先（fanout との同時指定時も pipeline を取る）
    if aa.get("enable_pipeline", False):
        return _run_pipeline_action(
            job,
            argv_str=argv_str,
            engine=engine,
            resolved_model=resolved_model,
            persona=persona,
            skill_registry=skill_registry,
            repo_root=repo_root,
            default_tz=default_tz,
            knowledge_count_cfg=knowledge_count_cfg,
            memory_max_bytes_cfg=memory_max_bytes_cfg,
            memory_exclude_source_tags_cfg=memory_exclude_source_tags_cfg,
            knowledge_count_audit=knowledge_count_audit,
            memory_bytes_audit=memory_bytes_audit,
            skill_name=skill_name,
            permission=aa.get("permission"),
        )

    if aa.get("enable_fanout", False):
        prompt = prompt + _FANOUT_PROMPT_SUFFIX

    from mltgnt.bridges.ghdag_bridge import enqueue_and_wait

    write_patterns = meta.side_effects.writes if meta.side_effects else []
    before = _snapshot_writes(write_patterns, repo_root) if write_patterns else {}

    fired_at = datetime.now(ZoneInfo(default_tz))
    request_id = str(uuid.uuid4())
    permission = aa.get("permission")
    ok, msg = enqueue_and_wait(
        prompt=prompt,
        engine=engine,
        model=resolved_model,
        timeout=job.timeout_seconds or 120,
        idempotency_key=f"scheduler:{job.id}:{fired_at.isoformat()}",
        jobs_dir=repo_root / "jobs",
        exec_done_dir=repo_root / "jobs" / "done",
        request_id=request_id,
        permission=permission,
        run_result=run_output,
    )

    if write_patterns:
        after = _snapshot_writes(write_patterns, repo_root)
        actual = _compute_write_diff(before, after)
        _write_side_effect_audit(
            repo_root / "jobs" / "audit.jsonl",
            skill_name=skill_name,
            job_id=job.id,
            declared_writes=write_patterns,
            actual_writes=actual,
        )

    if knowledge_count_audit > 0 or memory_bytes_audit > 0:
        _write_context_injection_audit(
            repo_root / "jobs" / "audit.jsonl",
            skill_name=skill_name,
            job_id=job.id,
            knowledge_count=knowledge_count_audit,
            memory_bytes=memory_bytes_audit,
        )

    if ok and aa.get("enable_fanout", False):
        fanout_steps = _parse_fanout_steps(msg, engine=engine, model=resolved_model)
        if fanout_steps:
            from mltgnt.bridges.ghdag_bridge import enqueue_dag

            permission = aa.get("permission")
            dag_results = enqueue_dag(
                fanout_steps,
                timeout=job.timeout_seconds or 120,
                idempotency_key=f"scheduler:{job.id}:{fired_at.isoformat()}:fanout",
                jobs_dir=repo_root / "jobs",
                exec_done_dir=repo_root / "jobs" / "done",
                request_id=request_id,
                permission=permission,
            )
            audit_path = repo_root / "jobs" / "audit.jsonl"
            first_failure: tuple[str, str] | None = None
            enforce = bool(aa.get("enforce_status_markers", False))
            for i, (step_ok, step_msg) in enumerate(dag_results):
                step_exit, step_diag = _resolve_exit_code(
                    step_ok, step_msg, [], enforce=enforce
                )
                _write_skill_result_audit(
                    audit_path,
                    skill_name=skill_name,
                    job_id=job.id,
                    exit_code=step_exit,
                    diagnostics=step_diag,
                )
                if not step_ok and first_failure is None:
                    step_id = fanout_steps[i].id
                    first_failure = (step_id, step_msg)
            if first_failure is not None:
                return False, (
                    f"fanout: step '{first_failure[0]}' failed: {first_failure[1]}"
                )
            return True, f"fanout: {len(dag_results)} steps completed"

    enforce = bool(aa.get("enforce_status_markers", False))
    exit_code, diagnostics = _resolve_exit_code(
        ok, msg, run_output.expected_markers, enforce=enforce
    )
    run_output.exit_code = exit_code
    run_output.diagnostics = diagnostics
    run_output.content = msg
    _write_skill_result_audit(
        repo_root / "jobs" / "audit.jsonl",
        skill_name=skill_name,
        job_id=job.id,
        exit_code=exit_code,
        diagnostics=diagnostics,
    )
    if run_output.exit_code == ExitStatus.SUCCESS:
        return True, msg
    if run_output.exit_code == ExitStatus.ALREADY_APPLIED:
        return True, "already_applied"
    if run_output.exit_code == ExitStatus.INVALID_STATE:
        return False, "invalid_state"
    return False, msg


def _run_pipeline_action(
    job: ScheduleJob,
    *,
    argv_str: str,
    engine: str,
    resolved_model: str | None,
    persona: Any,
    skill_registry: dict[str, Any],
    repo_root: Path,
    default_tz: str,
    knowledge_count_cfg: int,
    memory_max_bytes_cfg: int,
    memory_exclude_source_tags_cfg: list[str] | None,
    knowledge_count_audit: int,
    memory_bytes_audit: int,
    skill_name: str,
    permission: str | None,
) -> tuple[bool, str]:
    """enable_pipeline: match_pipeline → compose → typecheck → enqueue_dag。"""
    import asyncio

    from mltgnt.bridges.ghdag_bridge import compose_pipeline, enqueue_dag, typecheck_dag
    from mltgnt.interfaces.types import ChatInput, Message
    from mltgnt.skill import load
    from mltgnt.skill import runner as skill_runner
    from mltgnt.skill.context import build_extra_context
    from mltgnt.skill.matcher import match_pipeline

    skills = skill_registry
    persona_skills = persona.fm.skills or None
    match_results = asyncio.run(
        match_pipeline(
            argv_str,
            skills,
            persona_skills=persona_skills,
        )
    )
    steps = compose_pipeline(
        match_results, engine=engine, model=resolved_model
    )

    # 各段のプロンプトをスキル本文 + ペルソナで合成
    for step, mr in zip(steps, match_results):
        assert mr.decisive is not None
        skill_file = load(mr.decisive)
        extra_context = build_extra_context(
            mr.decisive,
            repo_root,
            persona.name,
            knowledge_count=knowledge_count_cfg,
            memory_max_bytes=memory_max_bytes_cfg,
            memory_exclude_source_tags=memory_exclude_source_tags_cfg,
        )
        chat_input = ChatInput(
            source="scheduler",
            session_key=job.id,
            messages=[Message(role="user", content=mr.arguments or "")],
            persona_name=persona.name,
            model=resolved_model,
        )
        run_out = skill_runner.run(
            skill_file, persona, mr.arguments, chat_input, extra_context=extra_context
        )
        step.prompt = next(
            m["content"] for m in run_out.chat_input.messages if m["role"] == "system"
        )
        if run_out.chat_input.model is not None:
            step.model = run_out.chat_input.model

    typecheck_dag(steps, skills)

    fired_at = datetime.now(ZoneInfo(default_tz))
    request_id = str(uuid.uuid4())
    dag_results = enqueue_dag(
        steps,
        timeout=job.timeout_seconds or 120,
        idempotency_key=f"scheduler:{job.id}:{fired_at.isoformat()}:pipeline",
        jobs_dir=repo_root / "jobs",
        exec_done_dir=repo_root / "jobs" / "done",
        request_id=request_id,
        skills=skills,
        permission=permission,
    )

    if knowledge_count_audit > 0 or memory_bytes_audit > 0:
        _write_context_injection_audit(
            repo_root / "jobs" / "audit.jsonl",
            skill_name=skill_name,
            job_id=job.id,
            knowledge_count=knowledge_count_audit,
            memory_bytes=memory_bytes_audit,
        )

    for i, (step_ok, step_msg) in enumerate(dag_results):
        if not step_ok:
            return False, f"pipeline: step '{steps[i].id}' failed: {step_msg}"
    return True, dag_results[-1][1] if dag_results else ""
