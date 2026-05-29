from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

from mltgnt.scheduler.fanout import _FANOUT_PROMPT_SUFFIX, _parse_fanout_steps
from mltgnt.scheduler.models import ScheduleJob


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
    chat_input = skill_runner.run(skill_file, persona, argv_str, chat_input)

    prompt = next(m["content"] for m in chat_input.messages if m["role"] == "system")
    resolved_model = chat_input.model

    if aa.get("enable_fanout", False):
        prompt = prompt + _FANOUT_PROMPT_SUFFIX

    from mltgnt.bridges.ghdag_bridge import enqueue_and_wait

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

    return ok, msg
