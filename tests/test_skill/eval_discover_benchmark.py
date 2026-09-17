"""
eval_discover_benchmark — compare old Step 5 (single-shot LLM) vs new Step 4 (AgenticSkillDiscoverer).

Design: Issue #1925
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from mltgnt.bridges.llm_adapter import call_llm
from mltgnt.routing.agentic_discover import AgenticSkillDiscoverer, DiscoverResult
from mltgnt.skill.models import SkillMeta

_DEFAULT_EVAL_MODEL = "claude-haiku-4-5-20251001"

_LLM_SYSTEM_PROMPT = """\
You are a skill matcher.
Decide whether the user input corresponds to one of the skills listed below.
If a skill matches, return only that skill name.
If none match, return only "none".
No extra explanation.
"""


@dataclass(frozen=True)
class EvalSample:
    user_input: str
    expected_skill: str | None


def _meta(
    name: str,
    description: str,
    *,
    triggers: list[str] | None = None,
) -> SkillMeta:
    return SkillMeta(
        name=name,
        description=description,
        argument_hint="",
        model=None,
        path=Path(f"/fake/skills/{name}/SKILL.md"),
        triggers=triggers or [],
    )


def _catalog() -> dict[str, SkillMeta]:
    return {
        "calendar": _meta(
            "calendar",
            "Check, add, and update calendar events",
            triggers=["schedule", "calendar", "agenda"],
        ),
        "diary-draft": _meta(
            "diary-draft",
            "Draft a diary entry from notes or source material on the user's behalf",
            triggers=["diary", "draft", "ghostwrite"],
        ),
        "diary-daily": _meta(
            "diary-daily",
            "Create or update today's diary file",
            triggers=["daily", "today's diary"],
        ),
        "diary-weekly": _meta(
            "diary-weekly",
            "Create this week's weekly retrospective file",
            triggers=["weekly retro", "weekly"],
        ),
        "diary-review": _meta(
            "diary-review",
            "Run a diary retrospective / review",
            triggers=["retrospective", "review"],
        ),
        "diary-callout": _meta(
            "diary-callout",
            "Extract important events from the diary as callouts",
            triggers=["callout extract", "callout"],
        ),
        "research": _meta(
            "research",
            "Own the flow from research-theme dialogue → Issue creation → workflow run → report",
            triggers=["research topic", "investigate", "research"],
        ),
        "asana": _meta(
            "asana",
            "Inventory / restructure Asana tasks and perform CRUD",
            triggers=["asana", "task"],
        ),
        "project": _meta(
            "project",
            "Manage project progress and organize tasks",
            triggers=["project progress", "project"],
        ),
        "mltgnt-persona": _meta(
            "mltgnt-persona",
            "Create or update persona definitions",
            triggers=["persona setup", "persona"],
        ),
        "okr-reflection": _meta(
            "okr-reflection",
            "Reflect on OKRs and check progress",
            triggers=["OKR", "goal"],
        ),
        "diary-wrapup": _meta(
            "diary-wrapup",
            "Wrap up the day by summarizing the diary",
            triggers=["wrapup", "wrap-up"],
        ),
    }


def _eval_samples() -> list[EvalSample]:
    return [
        # --- phrases where multiple skills' triggers overlap ---
        EvalSample("write the schedule into the diary", "diary-draft"),
        EvalSample("summarize today's schedule into the diary", "diary-draft"),
        EvalSample("use the schedule for the retrospective", "diary-review"),
        EvalSample("put calendar events into a diary draft", "diary-draft"),
        EvalSample("check the schedule and also write a diary", None),
        EvalSample("next week's schedule and retrospective", None),
        # --- direct trigger hits ---
        EvalSample("check the calendar", "calendar"),
        EvalSample("tell me today's schedule", "calendar"),
        EvalSample("make a diary draft", "diary-draft"),
        EvalSample("daily diary update", "diary-daily"),
        EvalSample("create the weekly retro file", "diary-weekly"),
        EvalSample("please review the diary", "diary-review"),
        EvalSample("do some research", "research"),
        EvalSample("asana task cleanup", "asana"),
        EvalSample("check project progress", "project"),
        EvalSample("change persona setup", "mltgnt-persona"),
        EvalSample("OKR retrospective", "okr-reflection"),
        EvalSample("please wrapup", "diary-wrapup"),
        EvalSample("extract callouts", "diary-callout"),
        # --- inferable from description (weak / no triggers) ---
        EvalSample("any free time next week?", "calendar"),
        EvalSample("I want to add tomorrow's schedule", "calendar"),
        EvalSample("write a diary from my notes", "diary-draft"),
        EvalSample("turn the material into a diary", "diary-draft"),
        EvalSample("create this week's retrospective file", "diary-weekly"),
        EvalSample("I want a diary retrospective", "diary-review"),
        EvalSample("I want to pick a research topic", "research"),
        EvalSample("I want to investigate a company", "research"),
        EvalSample("check overdue tasks", "asana"),
        EvalSample("update a task in Asana", "asana"),
        EvalSample("end-of-day wrap-up", "diary-wrapup"),
        EvalSample("pull out the important events", "diary-callout"),
        EvalSample("how is goal progress?", "okr-reflection"),
        EvalSample("update persona definition", "mltgnt-persona"),
        # --- non-existent skills (unresolved is correct) ---
        EvalSample("what's the weather", None),
        EvalSample("fix a Python bug", None),
        EvalSample("what should I eat for lunch", None),
        EvalSample("how to configure GitHub Actions", None),
        EvalSample("tell me the stock price", None),
        EvalSample("translate this to English", None),
        EvalSample("build a Docker image", None),
        EvalSample("change Slack notification color", None),
        EvalSample("book a meeting room", None),
        EvalSample("download the payslip", None),
    ]


def make_llm_fn(model: str | None = None) -> Callable[[str], str]:
    resolved_model = model or os.environ.get("MLTGNT_EVAL_MODEL", _DEFAULT_EVAL_MODEL)

    def llm_fn(prompt: str) -> str:
        result = call_llm(prompt, engine="claude", model=resolved_model, timeout=120)
        if not result.ok:
            return "none"
        return result.stdout.strip()

    return llm_fn


def old_llm_classify(
    user_input: str,
    catalog: dict[str, SkillMeta],
    llm_fn: Callable[[str], str],
) -> str | None:
    skill_list = "\n".join(f"- {m.name}: {m.description}" for m in catalog.values())
    prompt = f"{_LLM_SYSTEM_PROMPT}\n\nSkill list:\n{skill_list}\n\nUser input: {user_input}"
    response = llm_fn(prompt).strip().lower()
    if response == "none" or response not in catalog:
        return None
    return response


def extract_new_skill(result: DiscoverResult) -> str | None:
    if result.kind == "selected" and result.skill is not None:
        return result.skill.name
    return None


@dataclass
class BenchmarkMetrics:
    success_rate: float
    avg_rounds: float
    misclassification_rate: float
    unresolved_rate: float


@dataclass
class BenchmarkRun:
    metrics: BenchmarkMetrics
    misclassifications: list[tuple[str, str | None, str | None]]


def _is_misclassification(expected: str | None, actual: str | None) -> bool:
    return actual is not None and actual != expected


def _compute_metrics(
    *,
    expected: list[str | None],
    actual: list[str | None],
    rounds: list[int],
) -> BenchmarkMetrics:
    total = len(expected)
    successes = sum(1 for e, a in zip(expected, actual, strict=True) if e == a)
    misclassifications = sum(
        1 for e, a in zip(expected, actual, strict=True) if _is_misclassification(e, a)
    )
    unresolved = sum(1 for a in actual if a is None)
    return BenchmarkMetrics(
        success_rate=successes / total,
        avg_rounds=sum(rounds) / total,
        misclassification_rate=misclassifications / total,
        unresolved_rate=unresolved / total,
    )


def run_old_benchmark(
    samples: list[EvalSample],
    catalog: dict[str, SkillMeta],
    llm_fn: Callable[[str], str],
) -> BenchmarkRun:
    expected: list[str | None] = []
    actual: list[str | None] = []
    misclassifications: list[tuple[str, str | None, str | None]] = []

    for sample in samples:
        predicted = old_llm_classify(sample.user_input, catalog, llm_fn)
        expected.append(sample.expected_skill)
        actual.append(predicted)
        if _is_misclassification(sample.expected_skill, predicted):
            misclassifications.append((sample.user_input, sample.expected_skill, predicted))

    metrics = _compute_metrics(
        expected=expected,
        actual=actual,
        rounds=[1] * len(samples),
    )
    return BenchmarkRun(metrics=metrics, misclassifications=misclassifications)


def run_new_benchmark(
    samples: list[EvalSample],
    catalog: dict[str, SkillMeta],
    llm_fn: Callable[[str], str],
) -> BenchmarkRun:
    discoverer = AgenticSkillDiscoverer(llm_call=llm_fn, max_iterations=3)
    expected: list[str | None] = []
    actual: list[str | None] = []
    rounds: list[int] = []
    misclassifications: list[tuple[str, str | None, str | None]] = []

    for sample in samples:
        result = discoverer.discover(sample.user_input, catalog, persona_skills=None)
        predicted = extract_new_skill(result)
        expected.append(sample.expected_skill)
        actual.append(predicted)
        rounds.append(len(result.trace))
        if _is_misclassification(sample.expected_skill, predicted):
            misclassifications.append((sample.user_input, sample.expected_skill, predicted))

    metrics = _compute_metrics(expected=expected, actual=actual, rounds=rounds)
    return BenchmarkRun(metrics=metrics, misclassifications=misclassifications)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_report(old_run: BenchmarkRun, new_run: BenchmarkRun) -> str:
    lines = [
        "| Method | discover success | avg rounds | misclassification | unresolved |",
        "|---|---|---|---|---|",
        (
            f"| Old (single-shot LLM) | {_pct(old_run.metrics.success_rate)} | "
            f"{old_run.metrics.avg_rounds:.1f} | {_pct(old_run.metrics.misclassification_rate)} | "
            f"{_pct(old_run.metrics.unresolved_rate)} |"
        ),
        (
            f"| New (AgenticDiscoverer) | {_pct(new_run.metrics.success_rate)} | "
            f"{new_run.metrics.avg_rounds:.1f} | {_pct(new_run.metrics.misclassification_rate)} | "
            f"{_pct(new_run.metrics.unresolved_rate)} |"
        ),
        "",
        "## Misclassification details",
    ]

    for label, run in [("Old (single-shot LLM)", old_run), ("New (AgenticDiscoverer)", new_run)]:
        lines.append(f"\n### {label}")
        if not run.misclassifications:
            lines.append("(no misclassifications)")
            continue
        for user_input, expected, actual in run.misclassifications:
            lines.append(f"- input: {user_input!r} / expected: {expected!r} / actual: {actual!r}")

    return "\n".join(lines)


def run_benchmark(*, model: str | None = None) -> str:
    samples = _eval_samples()
    catalog = _catalog()
    llm_fn = make_llm_fn(model)
    old_run = run_old_benchmark(samples, catalog, llm_fn)
    new_run = run_new_benchmark(samples, catalog, llm_fn)
    report = format_report(old_run, new_run)
    print(report)
    return report


def test_eval_dataset_has_minimum_samples():
    assert len(_eval_samples()) >= 30
    assert all(isinstance(s.user_input, str) and s.user_input for s in _eval_samples())
    assert len(_catalog()) >= 10


def test_benchmark_metrics_with_mocked_llm():
    catalog = _catalog()
    samples = [
        EvalSample("check the calendar", "calendar"),
        EvalSample("what's the weather", None),
    ]

    def mock_llm(prompt: str) -> str:
        if "weather" in prompt:
            return "none"
        return "calendar"

    old_run = run_old_benchmark(samples, catalog, mock_llm)
    assert old_run.metrics.success_rate == 1.0
    assert old_run.metrics.avg_rounds == 1.0
    assert old_run.metrics.misclassification_rate == 0.0
    assert old_run.metrics.unresolved_rate == 0.5
    assert old_run.misclassifications == []


@pytest.mark.slow
def test_eval_discover_benchmark():
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()
