"""Validation behavior for ops.slack frontmatter keys."""
from __future__ import annotations

from mltgnt.persona.schema import parse_fm, validate_fm


def test_ops_slack_delegate_ack_rejected_as_unknown_key() -> None:
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
