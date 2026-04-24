"""tests/test_routing_evaluate.py — mltgnt.routing.evaluate() の単体テスト"""
import pytest
from mltgnt.routing import RoutingRule, evaluate


def _make_rule(name: str, always: bool) -> RoutingRule:
    return RoutingRule(
        name=name,
        detector=lambda instruction, ctx: always,
        handler=f"handle_{name}",
    )


def test_returns_first_matching_rule():
    """最初にマッチしたルールを返す"""
    rule_a = _make_rule("a", True)
    rule_b = _make_rule("b", True)
    result = evaluate([rule_a, rule_b], "test", {})
    assert result is rule_a


def test_returns_none_when_no_match():
    """マッチなしで None を返す"""
    rule_a = _make_rule("a", False)
    result = evaluate([rule_a], "test", {})
    assert result is None


def test_returns_none_for_empty_rules():
    """空ルールリストで None を返す"""
    result = evaluate([], "test", {})
    assert result is None


def test_ctx_passed_to_detector():
    """ctx が detector に渡される"""
    received_ctx = {}

    def detector(instruction, ctx):
        received_ctx.update(ctx)
        return True

    rule = RoutingRule(name="r", detector=detector, handler="h")
    evaluate([rule], "test", {"key": "val"})
    assert received_ctx == {"key": "val"}


def test_returns_second_rule_when_first_does_not_match():
    """2番目のルールにマッチ"""
    rule_a = _make_rule("a", False)
    rule_b = _make_rule("b", True)
    result = evaluate([rule_a, rule_b], "test", {})
    assert result is rule_b


def test_instruction_passed_to_detector():
    """instruction が detector に渡される"""
    received = {}

    def detector(instruction, ctx):
        received["instruction"] = instruction
        return False

    rule = RoutingRule(name="r", detector=detector, handler="h")
    evaluate([rule], "hello world", {})
    assert received["instruction"] == "hello world"


def test_exception_propagates_from_detector():
    """detector の例外は呼び出し元に伝播する（silent fail しない）"""
    def bad_detector(instruction, ctx):
        raise ValueError("detector error")

    rule = RoutingRule(name="r", detector=bad_detector, handler="h")
    with pytest.raises(ValueError, match="detector error"):
        evaluate([rule], "test", {})
