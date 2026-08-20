"""mltgnt.loops.status — 人間向け status Markdown の生成・保存。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from mltgnt.bridges.files_adapter import md_write
from mltgnt.loops.models import LoopState, TERMINAL_STATUSES


def render_status_markdown(state: LoopState) -> str:
    lines = [
        f"# {state.title}",
        "",
        f"- **loop_id**: `{state.loop_id}`",
        f"- **status**: `{state.status}`",
        f"- **iteration**: {state.iteration}/{state.max_iterations}",
        f"- **persona**: {state.persona}",
        f"- **clarify_round**: {state.clarify_round}",
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
            lines.append(f"- `{st.id}` ({st.kind}): {st.status} — {st.title}")
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
