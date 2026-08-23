"""mltgnt.loops.status — 人間向け status Markdown の生成・保存。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from mltgnt.bridges.files_adapter import md_write
from mltgnt.loops.models import LoopState, TERMINAL_STATUSES

_TZ = ZoneInfo("Asia/Tokyo")


def _parse_iso(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_TZ)
    return dt


def _format_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}時間{minutes}分"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def render_progress_summary(state: LoopState, now: datetime) -> str:
    """LLM を使わず、人間向けの進捗サマリを生成する。"""
    if now.tzinfo is None:
        now = now.replace(tzinfo=_TZ)
    updated = _parse_iso(state.updated_at)
    elapsed = _format_elapsed((now - updated).total_seconds()) if updated else "不明"
    lines = [
        f"**status**: `{state.status}`",
        f"**iteration**: {state.iteration}/{state.max_iterations}",
        f"**updated**: {elapsed}前",
    ]
    if state.subtasks:
        lines.append("")
        lines.append("**subtasks**:")
        for st in state.subtasks:
            lines.append(f"- `{st.id}` ({st.kind}): {st.status} — {st.title}")
    else:
        lines.append("")
        lines.append("**subtasks**: （なし）")
    return "\n".join(lines)


def render_status_markdown(state: LoopState) -> str:
    lines = [
        f"# {state.title}",
        "",
        f"- **loop_id**: `{state.loop_id}`",
        f"- **status**: `{state.status}`",
        f"- **iteration**: {state.iteration}/{state.max_iterations}",
        f"- **persona**: {state.persona}",
        f"- **clarify_round**: {state.clarify_round}",
        f"- **plan_approval**: {state.plan_approval}",
        f"- **plan_revision**: {state.plan_revision}",
        f"- **replan_count**: {state.replan_count}",
    ]
    if state.pending_question:
        lines.append(f"- **pending_question**: {state.pending_question.text!r}")
    if state.current_subtask_id:
        lines.append(f"- **current_subtask**: {state.current_subtask_id}")
    if state.content_change_warning:
        lines.append("")
        lines.append(f"> ⚠️ {state.content_change_warning}")
    if state.subtasks:
        lines.append("")
        lines.append("## Subtasks")
        for st in state.subtasks:
            dep = f" depends={st.depends}" if st.depends else ""
            lines.append(f"- `{st.id}` ({st.kind}): {st.status} — {st.title}{dep}")
    if state.status in TERMINAL_STATUSES:
        lines.append("")
        lines.append(f"**Terminal state**: {state.status}")
    lines.append("")
    lines.append("## Objective snapshot")
    lines.append("")
    lines.append(state.body)
    return "\n".join(lines) + "\n"


def write_status(
    status_dir: Path,
    state: LoopState,
    *,
    on_written: Callable[[Path], None] | None = None,
) -> Path:
    path = status_dir / f"{state.loop_id}.md"
    status_dir.mkdir(parents=True, exist_ok=True)
    md_write(path.name, render_status_markdown(state), repo_root=status_dir)
    if on_written is not None:
        on_written(path)
    return path
