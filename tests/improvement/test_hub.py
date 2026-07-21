"""Tests for ImprovementHub, ImprovementSource, and MltgntSource."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.improvement.hub import ImprovementHub, ImprovementSource, MltgntSource
from mltgnt.improvement.loop import CycleResult


# ---------------------------------------------------------------------------
# ImprovementSource protocol
# ---------------------------------------------------------------------------


class _FakeSource:
    """Minimal implementation of ImprovementSource for testing."""

    def __init__(self, name: str, result: CycleResult) -> None:
        self._name = name
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    def run_cycle(self) -> CycleResult:
        return self._result


def _make_cycle_result() -> CycleResult:
    from datetime import date

    return CycleResult(
        patterns=[],
        proposals=[],
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 8),
    )


def test_fake_source_satisfies_protocol() -> None:
    src = _FakeSource("test", _make_cycle_result())
    assert isinstance(src, ImprovementSource)


# ---------------------------------------------------------------------------
# ImprovementHub
# ---------------------------------------------------------------------------


def test_run_all_cycles_empty_returns_empty_list() -> None:
    hub = ImprovementHub()
    assert hub.run_all_cycles() == []


def test_register_and_run_all_cycles_returns_results() -> None:
    result = _make_cycle_result()
    src = _FakeSource("mltgnt", result)
    hub = ImprovementHub()
    hub.register(src)
    results = hub.run_all_cycles()
    assert results == [result]


def test_run_all_cycles_preserves_registration_order() -> None:
    r1 = _make_cycle_result()
    r2 = _make_cycle_result()
    hub = ImprovementHub()
    hub.register(_FakeSource("a", r1))
    hub.register(_FakeSource("b", r2))
    results = hub.run_all_cycles()
    assert results == [r1, r2]


def test_register_duplicate_name_raises_value_error() -> None:
    hub = ImprovementHub()
    hub.register(_FakeSource("mltgnt", _make_cycle_result()))
    with pytest.raises(ValueError, match="mltgnt"):
        hub.register(_FakeSource("mltgnt", _make_cycle_result()))


# ---------------------------------------------------------------------------
# MltgntSource
# ---------------------------------------------------------------------------


def test_mltgnt_source_name() -> None:
    src = MltgntSource(
        audit_path=Path("/tmp/audit.jsonl"),
        persona_dir=Path("/tmp/personas"),
        skills_dir=Path("/tmp/skills"),
    )
    assert src.name == "mltgnt"


def test_mltgnt_source_satisfies_protocol() -> None:
    src = MltgntSource(
        audit_path=Path("/tmp/audit.jsonl"),
        persona_dir=Path("/tmp/personas"),
        skills_dir=Path("/tmp/skills"),
    )
    assert isinstance(src, ImprovementSource)


def test_mltgnt_source_run_cycle_delegates_to_run_improvement_cycle() -> None:
    expected = _make_cycle_result()
    src = MltgntSource(
        audit_path=Path("/tmp/audit.jsonl"),
        persona_dir=Path("/tmp/personas"),
        skills_dir=Path("/tmp/skills"),
        since_days=14,
    )
    with patch(
        "mltgnt.improvement.hub.run_improvement_cycle", return_value=expected
    ) as mock_fn:
        result = src.run_cycle()

    assert result is expected
    mock_fn.assert_called_once_with(
        Path("/tmp/audit.jsonl"),
        Path("/tmp/personas"),
        Path("/tmp/skills"),
        since_days=14,
    )


def test_mltgnt_source_default_since_days() -> None:
    src = MltgntSource(
        audit_path=Path("/tmp/audit.jsonl"),
        persona_dir=Path("/tmp/personas"),
        skills_dir=Path("/tmp/skills"),
    )
    expected = _make_cycle_result()
    with patch(
        "mltgnt.improvement.hub.run_improvement_cycle", return_value=expected
    ) as mock_fn:
        src.run_cycle()

    mock_fn.assert_called_once_with(
        Path("/tmp/audit.jsonl"),
        Path("/tmp/personas"),
        Path("/tmp/skills"),
        since_days=7,
    )


def test_mltgnt_source_in_hub() -> None:
    expected = _make_cycle_result()
    src = MltgntSource(
        audit_path=Path("/tmp/audit.jsonl"),
        persona_dir=Path("/tmp/personas"),
        skills_dir=Path("/tmp/skills"),
    )
    hub = ImprovementHub()
    hub.register(src)

    with patch(
        "mltgnt.improvement.hub.run_improvement_cycle", return_value=expected
    ):
        results = hub.run_all_cycles()

    assert results == [expected]


def test_run_improvement_cycle_signature_unchanged() -> None:
    import inspect

    from mltgnt.improvement.loop import run_improvement_cycle

    sig = inspect.signature(run_improvement_cycle)
    params = list(sig.parameters.keys())
    assert "audit_path" in params
    assert "persona_dir" in params
    assert "skills_dir" in params
    assert "since_days" in params
    assert "eval_rollback" in params
    assert "repo_root" in params
