"""Issue #910: acceptance tests for removing ops.review."""
from __future__ import annotations

import importlib
import sys


def test_review_not_in_known_ops_keys():
    """AC: _KNOWN_OPS_KEYS must not contain \"review\"."""
    from mltgnt.persona.schema import _KNOWN_OPS_KEYS
    assert "review" not in _KNOWN_OPS_KEYS


def test_rules_module_not_importable():
    """AC: mltgnt.persona.rules must not exist (ImportError)."""
    # Clear cache and retry
    for key in list(sys.modules.keys()):
        if "persona.rules" in key:
            del sys.modules[key]
    try:
        importlib.import_module("mltgnt.persona.rules")
        raise AssertionError("mltgnt.persona.rules should not be importable")
    except ImportError:
        pass


def test_review_key_treated_as_unknown():
    """AC: FM containing ops.review is recorded in unknown_keys."""
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
    """AC: public API (load_persona, list_personas, validate_persona) unaffected."""
    from mltgnt.persona import load_persona, list_personas, validate_persona  # noqa: F401
