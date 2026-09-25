"""tests/agent/test_plan.py -- Plan / parse_plan / build_plan_prompt (#3855)."""
from __future__ import annotations

import json

import pytest

from mltgnt.agent.plan import Plan, PlanItem, build_plan_prompt, parse_plan


def _raw(items: list) -> str:
    return json.dumps({"items": items})


def test_parse_plan_default_depends_is_sequential():
    plan = parse_plan(_raw([{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]))
    assert [i.id for i in plan.items] == ["a", "b"]
    assert plan.items[0].depends == []
    assert plan.items[1].depends == ["a"]
    assert all(i.status == "pending" for i in plan.items)


def test_parse_plan_explicit_depends_and_code_block():
    raw = "text\n```json\n" + _raw([
        {"id": "a", "title": "A"},
        {"id": "b", "title": "B", "depends": []},
        {"id": "c", "title": "C", "depends": ["a", "b"]},
    ]) + "\n```\ntrailer"
    plan = parse_plan(raw)
    assert plan.items[1].depends == []
    assert plan.items[2].depends == ["a", "b"]


def test_parse_plan_forward_reference_allowed():
    plan = parse_plan(_raw([
        {"id": "a", "title": "A", "depends": ["b"]},
        {"id": "b", "title": "B", "depends": []},
    ]))
    assert plan.items[0].depends == ["b"]


@pytest.mark.parametrize(
    "raw",
    [
        _raw([{"id": "a", "title": "A"}, {"id": "a", "title": "B"}]),
        _raw([{"id": "a", "title": ""}]),
        _raw([{"id": "", "title": "A"}]),
        _raw([{"id": "a", "title": "A", "depends": ["zzz"]}]),
        _raw([{"id": "a", "title": "A", "depends": ["a"]}]),
        _raw([
            {"id": "a", "title": "A", "depends": ["b"]},
            {"id": "b", "title": "B", "depends": ["a"]},
        ]),
        _raw([{"id": "a", "title": "A", "depends": "b"}]),
        _raw([{"id": "a", "title": "A", "depends": [1]}]),
        _raw([]),
        json.dumps({"items": "x"}),
        json.dumps({"other": []}),
        "not json at all",
        "{broken json",
    ],
    ids=[
        "dup-id", "empty-title", "empty-id", "unknown-dep", "self-dep", "cycle",
        "depends-not-list", "depends-not-str", "items-empty", "items-not-list",
        "no-items", "no-json", "broken-json",
    ],
)
def test_parse_plan_invalid_raises_value_error(raw):
    with pytest.raises(ValueError):
        parse_plan(raw)


def _plan() -> Plan:
    return Plan(items=[PlanItem(id="a", title="A"), PlanItem(id="b", title="B")])


def test_apply_and_progress():
    plan = _plan()
    assert plan.progress() == (0, 2)
    plan.apply([{"id": "a", "status": "done", "note": "ok"}])
    assert plan.progress() == (1, 2)
    assert plan.items[0].note == "ok"
    plan.apply([{"id": "b", "status": "blocked"}])
    assert plan.items[1].status == "blocked"
    assert plan.progress() == (1, 2)


def test_apply_ignores_invalid_entries():
    plan = _plan()
    plan.apply([
        {"id": "zzz", "status": "done"},
        {"id": "a", "status": "finished"},
        "not a dict",
        {"id": "b", "status": "done"},
    ])
    assert plan.items[0].status == "pending"
    assert plan.items[1].status == "done"


def test_apply_non_list_is_noop():
    plan = _plan()
    plan.apply({"id": "a", "status": "done"})
    plan.apply(None)
    assert plan.progress() == (0, 2)


def test_build_plan_prompt_embeds_prompt():
    text = build_plan_prompt("write the report")
    assert "write the report" in text
    assert '"items"' in text
