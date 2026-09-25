"""mltgnt.agent.plan — work plan data structure and JSON contract (#3855).

Standard library only; must not import ``mltgnt.agent._runner``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

_logger = logging.getLogger(__name__)

PlanStatus = Literal["pending", "done", "blocked"]
_VALID_STATUSES: frozenset[str] = frozenset({"pending", "done", "blocked"})


@dataclass
class PlanItem:
    """One item of a work plan."""
    id: str
    title: str
    depends: list[str] = field(default_factory=list)
    status: PlanStatus = "pending"
    note: str = ""


@dataclass
class Plan:
    """Work plan; updated in place by ``AgentRunner`` via ``plan_update``."""
    items: list[PlanItem]

    def apply(self, updates: object) -> None:
        """Apply ``[{"id", "status", "note"?}, ...]``; invalid entries are skipped."""
        if not isinstance(updates, list):
            _logger.warning("plan_update is not a list; ignored: %r", updates)
            return
        by_id = {item.id: item for item in self.items}
        for update in updates:
            if not isinstance(update, dict):
                _logger.warning("plan_update entry is not an object; skipped: %r", update)
                continue
            item = by_id.get(update.get("id"))  # type: ignore[arg-type]
            if item is None:
                _logger.warning("plan_update has unknown id; skipped: %r", update)
                continue
            status = update.get("status")
            if status not in _VALID_STATUSES:
                _logger.warning("plan_update has invalid status; skipped: %r", update)
                continue
            item.status = status
            if "note" in update:
                note = update["note"]
                if isinstance(note, str):
                    item.note = note
                else:
                    _logger.warning("plan_update note is not a string; ignored: %r", update)

    def progress(self) -> tuple[int, int]:
        """Return ``(done count, total count)``."""
        done = sum(1 for item in self.items if item.status == "done")
        return done, len(self.items)


def _extract_json(raw: str) -> object:
    # Same order as _parse_json_response: ```json block, then first { .. last }.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("plan response contains no JSON object")
        candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"plan response is not valid JSON: {exc}") from exc


def _check_acyclic(items: list[PlanItem]) -> None:
    deps = {item.id: item.depends for item in items}
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(node: str) -> None:
        state[node] = 1
        for dep in deps[node]:
            mark = state.get(dep)
            if mark == 1:
                raise ValueError(f"plan has a dependency cycle through {dep!r}")
            if mark is None:
                visit(dep)
        state[node] = 2

    for item in items:
        if item.id not in state:
            visit(item.id)


def parse_plan(raw: str) -> Plan:
    """Parse ``{"items": [{"id", "title", "depends"?}]}`` into a ``Plan``.

    Omitted ``depends`` defaults to the previous item's id (sequential).
    Raises ``ValueError`` on any contract violation.
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("plan JSON must be an object")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("plan 'items' must be a non-empty list")

    items: list[PlanItem] = []
    seen: set[str] = set()
    prev_id: str | None = None
    for entry in raw_items:
        if not isinstance(entry, dict):
            raise ValueError(f"plan item must be an object: {entry!r}")
        item_id = entry.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"plan item id must be a non-empty string: {entry!r}")
        if item_id in seen:
            raise ValueError(f"duplicate plan item id: {item_id!r}")
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"plan item title must be a non-empty string: {entry!r}")
        if "depends" in entry:
            depends = entry["depends"]
            if not isinstance(depends, list) or not all(isinstance(d, str) for d in depends):
                raise ValueError(f"plan item depends must be a list of strings: {entry!r}")
            depends = list(depends)
        else:
            depends = [prev_id] if prev_id is not None else []
        seen.add(item_id)
        items.append(PlanItem(id=item_id, title=title, depends=depends))
        prev_id = item_id

    for item in items:
        for dep in item.depends:
            if dep == item.id:
                raise ValueError(f"plan item {item.id!r} depends on itself")
            if dep not in seen:
                raise ValueError(f"plan item {item.id!r} depends on unknown id {dep!r}")
    _check_acyclic(items)
    return Plan(items=items)


def build_plan_prompt(prompt: str) -> str:
    """Build a request asking the LLM to decompose ``prompt`` into plan items."""
    return (
        "Decompose the following task into a short ordered list of concrete work items.\n"
        "\n"
        "## Task\n"
        f"{prompt}\n"
        "\n"
        "## Output format\n"
        "Return ONLY a JSON object of this shape, with no other text:\n"
        '{"items": [{"id": "<short unique id>", "title": "<what to do>", '
        '"depends": ["<id of a prerequisite item>"]}]}\n'
        "\n"
        "- `id` must be unique and non-empty; `title` must be non-empty.\n"
        "- `depends` is optional; when omitted the item depends on the previous item.\n"
        "- Dependencies must refer to other items in the list and must not form a cycle.\n"
    )
