"""
eval_discover_benchmark — 旧 Step 5（単発 LLM）vs 新 Step 4（AgenticSkillDiscoverer）の比較ベンチマーク。

設計: Issue #1925
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
あなたはスキルマッチャーです。
ユーザー入力が以下のスキル一覧のどれかに対応するか判定してください。
対応するスキルがあればそのスキル名のみを返してください。
どれにも対応しない場合は "none" とだけ返してください。
余計な説明は不要です。
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
            "カレンダーの予定を確認・追加・更新する",
            triggers=["予定", "スケジュール", "カレンダー"],
        ),
        "diary-draft": _meta(
            "diary-draft",
            "ユーザーの代わりにメモや素材から日記を代筆する",
            triggers=["日記", "下書き", "代筆"],
        ),
        "diary-daily": _meta(
            "diary-daily",
            "今日の日記ファイルを作成・更新する",
            triggers=["daily", "今日の日記"],
        ),
        "diary-weekly": _meta(
            "diary-weekly",
            "今週の週の振り返りファイルを作成する",
            triggers=["週次", "weekly"],
        ),
        "diary-review": _meta(
            "diary-review",
            "日記の振り返り・レビューを実行する",
            triggers=["振り返り", "レビュー"],
        ),
        "diary-callout": _meta(
            "diary-callout",
            "日記から重要な出来事を抽出してコールアウトする",
            triggers=["コールアウト", "callout"],
        ),
        "research": _meta(
            "research",
            "リサーチテーマの対話→Issue作成→ワークフロー実行→結果報告の一連のフローを担当する",
            triggers=["リサーチ", "調査", "research"],
        ),
        "asana": _meta(
            "asana",
            "Asana タスクの棚卸し・再構成および CRUD 操作",
            triggers=["asana", "タスク"],
        ),
        "project": _meta(
            "project",
            "プロジェクトの進捗管理・タスク整理を行う",
            triggers=["プロジェクト", "project"],
        ),
        "mltgnt-persona": _meta(
            "mltgnt-persona",
            "ペルソナ定義の作成・更新を行う",
            triggers=["ペルソナ", "persona"],
        ),
        "okr-reflection": _meta(
            "okr-reflection",
            "OKR の振り返り・進捗確認を行う",
            triggers=["OKR", "目標"],
        ),
        "diary-wrapup": _meta(
            "diary-wrapup",
            "1日の締めくくりとして日記をまとめる",
            triggers=["wrapup", "締め"],
        ),
    }


def _eval_samples() -> list[EvalSample]:
    return [
        # --- 複数スキルの triggers が重複する表現 ---
        EvalSample("予定を日記に書いて", "diary-draft"),
        EvalSample("今日の予定を日記にまとめて", "diary-draft"),
        EvalSample("スケジュールを振り返りに使って", "diary-review"),
        EvalSample("カレンダーの予定を日記の下書きに", "diary-draft"),
        EvalSample("予定確認して日記も書いて", None),
        EvalSample("来週の予定と振り返り", None),
        # --- triggers 直接一致 ---
        EvalSample("カレンダー確認して", "calendar"),
        EvalSample("今日のスケジュール教えて", "calendar"),
        EvalSample("日記の下書き作って", "diary-draft"),
        EvalSample("daily 日記更新", "diary-daily"),
        EvalSample("週次振り返りファイル作って", "diary-weekly"),
        EvalSample("日記レビューお願い", "diary-review"),
        EvalSample("リサーチして", "research"),
        EvalSample("asana タスク整理", "asana"),
        EvalSample("プロジェクト進捗確認", "project"),
        EvalSample("ペルソナ設定変更", "mltgnt-persona"),
        EvalSample("OKR 振り返り", "okr-reflection"),
        EvalSample("wrapup お願い", "diary-wrapup"),
        EvalSample("コールアウト抽出", "diary-callout"),
        # --- description から推論可能（triggers 弱い / なし） ---
        EvalSample("来週の空き時間ある？", "calendar"),
        EvalSample("明日の予定追加したい", "calendar"),
        EvalSample("メモから日記書いて", "diary-draft"),
        EvalSample("素材を日記にして", "diary-draft"),
        EvalSample("今週の振り返りファイル作って", "diary-weekly"),
        EvalSample("日記の振り返りしたい", "diary-review"),
        EvalSample("調査テーマ決めたい", "research"),
        EvalSample("企業調査したい", "research"),
        EvalSample("タスクの期限切れチェック", "asana"),
        EvalSample("Asana でタスク更新", "asana"),
        EvalSample("1日の締めくくり", "diary-wrapup"),
        EvalSample("重要な出来事を抜き出して", "diary-callout"),
        EvalSample("目標の進捗どう？", "okr-reflection"),
        EvalSample("persona 定義更新", "mltgnt-persona"),
        # --- 存在しないスキル（unresolved が正解） ---
        EvalSample("天気教えて", None),
        EvalSample("Python のバグ直して", None),
        EvalSample("ランチ何食べよう", None),
        EvalSample("GitHub Actions の設定方法", None),
        EvalSample("株価教えて", None),
        EvalSample("英語に翻訳して", None),
        EvalSample("Docker イメージビルド", None),
        EvalSample("Slack 通知の色変更", None),
        EvalSample("会議室予約して", None),
        EvalSample("給与明細ダウンロード", None),
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
    prompt = f"{_LLM_SYSTEM_PROMPT}\n\nスキル一覧:\n{skill_list}\n\nユーザー入力: {user_input}"
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
        "| 方式 | discover 成功率 | 平均ラウンド数 | 誤選定率 | unresolved 率 |",
        "|---|---|---|---|---|",
        (
            f"| 旧（単発 LLM） | {_pct(old_run.metrics.success_rate)} | "
            f"{old_run.metrics.avg_rounds:.1f} | {_pct(old_run.metrics.misclassification_rate)} | "
            f"{_pct(old_run.metrics.unresolved_rate)} |"
        ),
        (
            f"| 新（AgenticDiscoverer） | {_pct(new_run.metrics.success_rate)} | "
            f"{new_run.metrics.avg_rounds:.1f} | {_pct(new_run.metrics.misclassification_rate)} | "
            f"{_pct(new_run.metrics.unresolved_rate)} |"
        ),
        "",
        "## 誤選定詳細",
    ]

    for label, run in [("旧（単発 LLM）", old_run), ("新（AgenticDiscoverer）", new_run)]:
        lines.append(f"\n### {label}")
        if not run.misclassifications:
            lines.append("（誤選定なし）")
            continue
        for user_input, expected, actual in run.misclassifications:
            lines.append(f"- 入力: {user_input!r} / 期待: {expected!r} / 実際: {actual!r}")

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


@pytest.mark.slow
def test_eval_discover_benchmark():
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()
