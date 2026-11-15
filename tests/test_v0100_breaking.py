"""Acceptance tests for v0.10.0 breaking changes (Issue #1251)."""
from __future__ import annotations

import pytest

from mltgnt.agent._parse import _parse_json_response
from mltgnt.persona.schema import parse_fm, validate_fm


def test_parse_requires_args_key() -> None:
    assert _parse_json_response('{"tool":"x","a":1}') is None
    assert _parse_json_response('{"tool":"x","args":{"a":1}}') == {
        "tool": "x",
        "args": {"a": 1},
    }


def test_flat_fm_keys_rejected() -> None:
    fm = parse_fm(
        {"persona": {"name": "t"}, "chat_model": "x", "slack": {}},
        file_stem="t",
    )
    result = validate_fm(fm)
    assert not result.ok


def test_ops_chat_model_rejected() -> None:
    fm = parse_fm(
        {"persona": {"name": "t"}, "ops": {"chat_model": "gpt-4"}},
        file_stem="t",
    )
    result = validate_fm(fm)
    assert not result.ok


