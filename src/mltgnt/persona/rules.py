"""mltgnt.persona.rules

ペルソナ別レビュールールの定義・検証。

FM 構造（ops.review）:
    ops:
      review:
        allowed_ops: [critique, debate]
        domain: ["哲学的な素朴な問い"]
        constraints:
          - "他のペルソナの担当領域に踏み込まない"
        max_chars: 500

公開 API:
    parse_review_rules(ops_dict)        -> PersonaReviewRules
    validate_review_request(rules, op)  -> RuleValidationResult
    build_constraints_prompt(rules)     -> str
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


VALID_OPS: frozenset[str] = frozenset({"critique", "edit", "debate", "system-debug"})

_KNOWN_REVIEW_KEYS: frozenset[str] = frozenset(
    {"allowed_ops", "domain", "constraints", "max_chars"}
)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class PersonaReviewRules:
    """ペルソナのレビュールール。FM 未設定時はデフォルト（全許可）。"""

    allowed_ops: list[str] = field(default_factory=lambda: list(VALID_OPS))
    domain: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    max_chars: int = 0  # 0 = 制限なし
    unknown_keys: list[str] = field(default_factory=list)


@dataclass
class RuleValidationResult:
    """ルール検証結果。"""

    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# パース
# ---------------------------------------------------------------------------


def parse_review_rules(ops_dict: dict[str, Any] | None) -> PersonaReviewRules:
    """ops dict から review ルールをパースする。"""
    if not ops_dict or not isinstance(ops_dict, dict):
        return PersonaReviewRules()

    review_ns = ops_dict.get("review")
    if not review_ns or not isinstance(review_ns, dict):
        return PersonaReviewRules()

    unknown: list[str] = []
    for k in review_ns:
        if k not in _KNOWN_REVIEW_KEYS:
            unknown.append(f"ops.review.{k}")

    raw_ops = review_ns.get("allowed_ops")
    if isinstance(raw_ops, list):
        allowed_ops = [str(o) for o in raw_ops if str(o) in VALID_OPS]
    else:
        allowed_ops = list(VALID_OPS)

    raw_domain = review_ns.get("domain")
    domain = list(raw_domain) if isinstance(raw_domain, list) else []

    raw_constraints = review_ns.get("constraints")
    constraints = list(raw_constraints) if isinstance(raw_constraints, list) else []

    raw_max = review_ns.get("max_chars")
    max_chars = int(raw_max) if raw_max is not None else 0

    return PersonaReviewRules(
        allowed_ops=allowed_ops,
        domain=domain,
        constraints=constraints,
        max_chars=max_chars,
        unknown_keys=unknown,
    )


# ---------------------------------------------------------------------------
# バリデーション（フック）
# ---------------------------------------------------------------------------


def validate_review_request(
    rules: PersonaReviewRules, op: str
) -> RuleValidationResult:
    """レビューリクエストがペルソナルールに準拠しているか検証する。"""
    errors: list[str] = []
    warnings: list[str] = []

    if rules.allowed_ops and op not in rules.allowed_ops:
        errors.append(
            f"このペルソナは op={op!r} を許可していません。"
            f"許可されている op: {rules.allowed_ops}"
        )

    for k in rules.unknown_keys:
        warnings.append(f"未定義のレビュールールキー: {k!r}")

    return RuleValidationResult(
        ok=len(errors) == 0,
        warnings=warnings,
        errors=errors,
    )


def validate_review_output(
    rules: PersonaReviewRules, text: str
) -> RuleValidationResult:
    """レビュー出力がペルソナルールに準拠しているか検証する。"""
    warnings: list[str] = []

    if rules.max_chars > 0 and len(text) > rules.max_chars:
        warnings.append(
            f"出力がペルソナの max_chars を超過しています"
            f"（{len(text)}字 > {rules.max_chars}字）"
        )

    return RuleValidationResult(ok=True, warnings=warnings)


# ---------------------------------------------------------------------------
# プロンプト生成
# ---------------------------------------------------------------------------


def build_constraints_prompt(rules: PersonaReviewRules) -> str:
    """ルールからプロンプトに注入する制約テキストを生成する。"""
    parts: list[str] = []

    if rules.domain:
        domains_str = "、".join(rules.domain)
        parts.append(f"あなたの担当領域: {domains_str}")
        parts.append("担当領域以外の話題には踏み込まないでください。")

    for constraint in rules.constraints:
        parts.append(f"制約: {constraint}")

    if rules.max_chars > 0:
        parts.append(f"回答は{rules.max_chars}字以内にしてください。")

    return "\n".join(parts)
