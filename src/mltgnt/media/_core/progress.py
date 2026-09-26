"""Summarize a job's events JSONL into a progress message and rewrite it in place.

Labels come from ``LanguagePack.status_labels``, progress lines in assistant text
are recognised with ``LanguagePack.progress_line_pattern`` and the update rate is
bounded by ``MediaConfig.progress_min_interval_sec``.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mltgnt.config.language import JA, LanguagePack
from mltgnt.interfaces.media import MediaClient, Status
from mltgnt.media._core.config import MediaConfig

__all__ = [
    "ProgressState",
    "finalize_progress",
    "status_for_done_marker",
    "status_label",
    "summarize_tool_use_block",
    "summarize_work_loop_step",
]

_log = logging.getLogger(__name__)

_TARGET_MAX_CHARS = 60
_LINE_MAX_CHARS = 120
_SHELL_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `cd /path && python x.py`: the `cd` says nothing about the work, so take the next segment's head
_SHELL_WRAPPER_COMMANDS = frozenset({"cd", "pushd", "popd", "export", "unset", "source", ".", "set", "true"})
_SHELL_SEGMENT_SEPARATORS = frozenset({"&&", "||", "|", ";", "(", ")", "{", "}"})


def _bash_command_name(command: str) -> str:
    """First command that does real work, skipping ``cd`` / assignments / ``export``."""
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return ""
    fallback = ""
    segment_head_seen = False
    for token in tokens:
        if token in _SHELL_SEGMENT_SEPARATORS:
            segment_head_seen = False
            continue
        if segment_head_seen or _SHELL_ASSIGNMENT_RE.match(token):
            continue
        segment_head_seen = True
        if token in _SHELL_WRAPPER_COMMANDS:
            fallback = fallback or token
            continue
        return token
    return fallback


def _tool_target(tool_name: str, inp: dict[str, Any]) -> str:
    if tool_name == "Bash":
        description = inp.get("description")
        if isinstance(description, str) and description.strip():
            return " ".join(description.split())[:_TARGET_MAX_CHARS]
        command = inp.get("command")
        if isinstance(command, str) and command.strip():
            return _bash_command_name(command)
        return ""
    for key in ("file_path", "path", "filePath", "command", "pattern"):
        val = inp.get(key)
        if isinstance(val, str) and val:
            return val[:_TARGET_MAX_CHARS]
    return ""


def summarize_tool_use_block(block: dict[str, Any]) -> str:
    """``<tool> <target>`` for an assistant ``tool_use`` content block."""
    name = block.get("name")
    if not isinstance(name, str) or not name:
        return ""
    inp = block.get("input")
    target = _tool_target(name, inp if isinstance(inp, dict) else {})
    return f"{name} {target}" if target else name


def summarize_work_loop_step(event: dict[str, Any]) -> str:
    """``<tool> <target> (plan d/t)`` for a ``work_loop_step`` event."""
    tool = event.get("tool")
    if not isinstance(tool, str) or not tool:
        return ""
    args = event.get("args")
    if not isinstance(args, dict):
        args = {}
    target = ""
    for key in ("path", "command", "url"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            target = " ".join(val.split())[:_TARGET_MAX_CHARS]
            break
    line = f"{tool} {target}" if target else tool
    plan = event.get("plan")
    if (
        isinstance(plan, list)
        and len(plan) == 2
        and all(isinstance(n, int) and not isinstance(n, bool) for n in plan)
        and plan[1] > 0
    ):
        line += f" (plan {plan[0]}/{plan[1]})"
    return line


def status_for_done_marker(marker: str) -> Status:
    """Map a job done marker (``"0"`` / ``"CANCELLED"`` / anything else) to a Status."""
    if marker == "0":
        return Status.DONE
    if marker == "CANCELLED":
        return Status.CANCELLED
    return Status.FAILED


def status_label(status: Status, language: LanguagePack = JA) -> str:
    return language.status_labels.get(status.value, status.value)


class ProgressState:
    """Events tail state and update rate limit for one job.

    The first update waits ``min_interval_sec`` after the first line, so a burst
    of events collapses into one update carrying the latest lines.
    """

    def __init__(
        self,
        *,
        min_interval_sec: float,
        max_lines: int = 1,
        language: LanguagePack = JA,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval_sec = min_interval_sec
        self._max_lines = max(1, max_lines)
        self._language = language
        self._clock = clock
        self._lines: list[str] = []
        self._offset = 0
        self._window_start: float | None = None
        self._rendered: str | None = None

    @classmethod
    def from_config(
        cls,
        config: MediaConfig,
        *,
        max_lines: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> ProgressState:
        return cls(
            min_interval_sec=config.progress_min_interval_sec,
            max_lines=max_lines,
            language=config.language,
            clock=clock,
        )

    def process_event(self, event: dict[str, Any]) -> list[str]:
        """Append the progress lines found in ``event`` and return them."""
        new_lines: list[str] = []
        event_type = event.get("type")
        if event_type == "assistant":
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        new_lines.append(summarize_tool_use_block(block))
                    elif block.get("type") == "text" and isinstance(block.get("text"), str):
                        new_lines.extend(
                            m.group("text").strip()
                            for m in self._language.progress_line_pattern.finditer(block["text"])
                        )
        elif event_type == "work_loop_step":
            new_lines.append(summarize_work_loop_step(event))
        added = [line[:_LINE_MAX_CHARS] for line in new_lines if line]
        for line in added:
            self._append_line(line)
        return added

    def read_new_lines(self, events_path: Path) -> bool:
        """Read events appended since the last call. Return True if lines were added."""
        if not events_path.is_file():
            return False
        changed = False
        try:
            with events_path.open(encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                while True:
                    raw = f.readline()
                    if not raw:
                        break
                    self._offset = f.tell()
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and self.process_event(event):
                        changed = True
        except OSError:
            _log.warning("[progress] failed to read events: %s", events_path, exc_info=True)
            return False
        return changed

    def render_text(self) -> str:
        return "\n".join(self._lines)

    def is_due(self, *, force: bool = False) -> bool:
        """True when there is unsent text and the rate limit allows an update."""
        text = self.render_text()
        if not text.strip() or text == self._rendered:
            return False
        if force or self._window_start is None:
            return True
        return self._clock() - self._window_start >= self._min_interval_sec

    def maybe_update(self, client: MediaClient, message_id: str, *, force: bool = False) -> bool:
        """Rewrite ``message_id`` with the latest lines when due. Return True if updated."""
        if not self.is_due(force=force):
            return False
        text = self.render_text()
        if not client.update(message_id, text):
            _log.warning("[progress] update failed message_id=%s", message_id)
            return False
        self.mark_sent(text)
        return True

    def mark_sent(self, text: str) -> None:
        """Record that ``text`` is now shown (after a post or an update)."""
        self._rendered = text
        self._window_start = self._clock()

    def _append_line(self, line: str) -> None:
        if self._window_start is None:
            self._window_start = self._clock()
        self._lines.append(line)
        del self._lines[: -self._max_lines]


def finalize_progress(
    client: MediaClient,
    message_id: str | None,
    done_marker: str,
    *,
    language: LanguagePack = JA,
) -> bool:
    """Replace the progress message with the final status label only."""
    if not message_id:
        return False
    return client.update(message_id, status_label(status_for_done_marker(done_marker), language))
