"""Issue #910: ops.review 除去の受け入れ条件テスト。"""
from __future__ import annotations

import importlib
import sys


def test_review_not_in_known_ops_keys():
    """AC: _KNOWN_OPS_KEYS に "review" が含まれないこと。"""
    from mltgnt.persona.schema import _KNOWN_OPS_KEYS
    assert "review" not in _KNOWN_OPS_KEYS


def test_rules_module_not_importable():
    """AC: mltgnt.persona.rules が存在しないこと（ImportError になること）。"""
    # キャッシュを除去して再試行
    for key in list(sys.modules.keys()):
        if "persona.rules" in key:
            del sys.modules[key]
    try:
        importlib.import_module("mltgnt.persona.rules")
        raise AssertionError("mltgnt.persona.rules should not be importable")
    except ImportError:
        pass


def test_review_key_treated_as_unknown():
    """AC: ops.review を含む FM は unknown_keys に記録されること。"""
    from mltgnt.persona.schema import parse_fm

    fm = parse_fm(
        {
            "persona": {"name": "test"},
            "ops": {"review": {"allowed_ops": ["critique"]}},
        },
        file_stem="test",
    )
    assert any("review" in k for k in fm.unknown_keys)


def test_public_api_still_importable():
    """AC: 公開 API (load_persona, list_personas, validate_persona) に影響なし。"""
    from mltgnt.persona import load_persona, list_personas, validate_persona  # noqa: F401
