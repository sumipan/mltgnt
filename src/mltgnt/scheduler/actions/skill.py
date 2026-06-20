from __future__ import annotations

import fnmatch
import glob
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

from mltgnt.scheduler.fanout import _FANOUT_PROMPT_SUFFIX, _parse_fanout_steps
from mltgnt.scheduler.models import ScheduleJob
from mltgnt.skill.models import ExitStatus


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


def _determine_exit_code(ok: bool, msg: str) -> int:
    if ok:
        if "PIPELINE_STATUS: ALREADY_APPLIED" in msg:
            return ExitStatus.ALREADY_APPLIED
        return ExitStatus.SUCCESS
    if "PIPELINE_STATUS: INVALID_STATE" in msg:
        return ExitStatus.INVALID_STATE
    return ExitStatus.USAGE_ERROR


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

    skill_file = load(meta)

    argv_list = aa.get("argv", [])
    argv_str = " ".join(str(x) for x in argv_list) if argv_list else ""

    from mltgnt.interfaces.types import ChatInput, Message
    from mltgnt.skill import runner as skill_runner

    chat_input = ChatInput(
        source="scheduler",
        session_key=job.id,
        messages=[Message(role="user", content=argv_str or "")],
        persona_name=persona.name,
        model=model,
    )
    run_output = skill_runner.run(skill_file, persona, argv_str, chat_input)

    prompt = next(m["content"] for m in run_output.chat_input.messages if m["role"] == "system")
    resolved_model = run_output.chat_input.model

    if aa.get("enable_fanout", False):
        prompt = prompt + _FANOUT_PROMPT_SUFFIX

    from mltgnt.bridges.ghdag_bridge import enqueue_and_wait

    write_patterns = meta.side_effects.writes if meta.side_effects else []
    before = _snapshot_writes(write_patterns, repo_root) if write_patterns else {}

    fired_at = datetime.now(ZoneInfo(default_tz))
    request_id = str(uuid.uuid4())
    ok, msg = enqueue_and_wait(
        prompt=prompt,
        engine=engine,
        model=resolved_model,
        timeout=job.timeout_seconds or 120,
        idempotency_key=f"scheduler:{job.id}:{fired_at.isoformat()}",
        jobs_dir=repo_root / "jobs",
        exec_done_dir=repo_root / "jobs" / "done",
        request_id=request_id,
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

    if ok and aa.get("enable_fanout", False):
        fanout_steps = _parse_fanout_steps(msg, engine=engine, model=resolved_model)
        if fanout_steps:
            from mltgnt.bridges.ghdag_bridge import enqueue_dag

            dag_results = enqueue_dag(
                fanout_steps,
                timeout=job.timeout_seconds or 120,
                idempotency_key=f"scheduler:{job.id}:{fired_at.isoformat()}:fanout",
                jobs_dir=repo_root / "jobs",
                exec_done_dir=repo_root / "jobs" / "done",
                request_id=request_id,
            )
            for i, (step_ok, step_msg) in enumerate(dag_results):
                if not step_ok:
                    step_id = fanout_steps[i].id
                    return False, f"fanout: step '{step_id}' failed: {step_msg}"
            return True, f"fanout: {len(dag_results)} steps completed"

    run_output.exit_code = _determine_exit_code(ok, msg)
    run_output.content = msg
    if run_output.exit_code == ExitStatus.SUCCESS:
        return True, msg
    if run_output.exit_code == ExitStatus.ALREADY_APPLIED:
        return True, "already_applied"
    if run_output.exit_code == ExitStatus.INVALID_STATE:
        return False, "invalid_state"
    return False, msg
