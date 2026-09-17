"""Issue #2928: acceptance tests for fully removing delegate_ack from PersonaFM."""
from __future__ import annotations

import importlib
import sys

from mltgnt.persona.schema import _KNOWN_OPS_SLACK_KEYS, parse_fm, validate_fm


def test_delegate_ack_not_in_known_ops_slack_keys() -> None:
    """AC: _KNOWN_OPS_SLACK_KEYS must not contain delegate_ack."""
    assert "delegate_ack" not in _KNOWN_OPS_SLACK_KEYS


def test_persona_fm_has_no_slack_delegate_ack_field() -> None:
    """AC: PersonaFM must not have a slack_delegate_ack field."""
    from mltgnt.persona.schema import PersonaFM

    assert "slack_delegate_ack" not in PersonaFM.__dataclass_fields__


def test_delegate_ack_from_meta_not_importable() -> None:
    """AC: delegate_ack_from_meta() is removed and not importable."""
    for key in list(sys.modules.keys()):
        if key == "mltgnt.persona.frontmatter":
            del sys.modules[key]
    mod = importlib.import_module("mltgnt.persona.frontmatter")
    assert not hasattr(mod, "delegate_ack_from_meta")


def test_ops_slack_delegate_ack_treated_as_unknown() -> None:
    """AC: ops.slack.delegate_ack is rejected as an unknown key."""
    fm = parse_fm(
        {
            "persona": {"name": "test"},
            "ops": {"slack": {"delegate_ack": "acknowledged"}},
        },
        file_stem="test",
    )
    assert "ops.slack.delegate_ack" in fm.unknown_keys
    result = validate_fm(fm)
    assert not result.ok
    assert any("delegate_ack" in e for e in result.errors)
