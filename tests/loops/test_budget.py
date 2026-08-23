"""tests/loops/test_budget.py — JST 日次共有 LLM 予算。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mltgnt.loops import budget

_TZ = ZoneInfo("Asia/Tokyo")


def test_per_loop_allows_limit_then_blocks(tmp_path: Path):
    state_dir = tmp_path / "state"
    now = datetime(2026, 8, 23, 12, 0, tzinfo=_TZ)
    count = 0
    for _ in range(3):
        r = budget.reserve_llm_call(
            state_dir,
            loop_id="a",
            loop_count=count,
            loop_limit=3,
            day_limit=100,
            now=now,
        )
        assert r.allowed
        count = r.loop_count
    denied = budget.reserve_llm_call(
        state_dir,
        loop_id="a",
        loop_count=count,
        loop_limit=3,
        day_limit=100,
        now=now,
    )
    assert not denied.allowed
    assert denied.reason == "loop_budget_exceeded"
    assert denied.loop_count == 3


def test_per_day_shared_across_loops(tmp_path: Path):
    state_dir = tmp_path / "state"
    now = datetime(2026, 8, 23, 12, 0, tzinfo=_TZ)
    r1 = budget.reserve_llm_call(
        state_dir, loop_id="a", loop_count=0, loop_limit=100, day_limit=2, now=now
    )
    r2 = budget.reserve_llm_call(
        state_dir, loop_id="b", loop_count=0, loop_limit=100, day_limit=2, now=now
    )
    assert r1.allowed and r2.allowed
    assert r2.day_count == 2
    denied = budget.reserve_llm_call(
        state_dir, loop_id="c", loop_count=0, loop_limit=100, day_limit=2, now=now
    )
    assert not denied.allowed
    assert denied.reason == "day_budget_exceeded"


def test_zero_limit_forbids_unless_override(tmp_path: Path):
    state_dir = tmp_path / "state"
    now = datetime(2026, 8, 23, 12, 0, tzinfo=_TZ)
    denied = budget.reserve_llm_call(
        state_dir, loop_id="a", loop_count=0, loop_limit=0, day_limit=10, now=now
    )
    assert not denied.allowed
    allowed = budget.reserve_llm_call(
        state_dir,
        loop_id="a",
        loop_count=0,
        loop_limit=0,
        day_limit=10,
        budget_override=True,
        now=now,
    )
    assert allowed.allowed
    assert allowed.loop_count == 1
    assert allowed.day_count == 1


def test_date_rollover_uses_new_counter(tmp_path: Path):
    state_dir = tmp_path / "state"
    day1 = datetime(2026, 8, 23, 23, 0, tzinfo=_TZ)
    day2 = datetime(2026, 8, 24, 1, 0, tzinfo=_TZ)
    r1 = budget.reserve_llm_call(
        state_dir, loop_id="a", loop_count=0, loop_limit=10, day_limit=1, now=day1
    )
    assert r1.allowed and r1.day_count == 1
    denied = budget.reserve_llm_call(
        state_dir, loop_id="a", loop_count=1, loop_limit=10, day_limit=1, now=day1
    )
    assert not denied.allowed
    r2 = budget.reserve_llm_call(
        state_dir, loop_id="a", loop_count=1, loop_limit=10, day_limit=1, now=day2
    )
    assert r2.allowed
    assert r2.day_count == 1
    assert budget.budget_path(state_dir, "2026-08-23").is_file()
    assert budget.budget_path(state_dir, "2026-08-24").is_file()


def test_lock_serializes_concurrent_reserves(tmp_path: Path):
    """同一ファイルへの連続予約が原子的に増えること。"""
    state_dir = tmp_path / "state"
    now = datetime(2026, 8, 23, 12, 0, tzinfo=_TZ)
    counts = []
    for i in range(5):
        r = budget.reserve_llm_call(
            state_dir,
            loop_id=f"l{i}",
            loop_count=0,
            loop_limit=100,
            day_limit=100,
            now=now,
        )
        assert r.allowed
        counts.append(r.day_count)
    assert counts == [1, 2, 3, 4, 5]


def test_negative_limits_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        budget.reserve_llm_call(
            tmp_path,
            loop_id="a",
            loop_count=0,
            loop_limit=-1,
            day_limit=1,
        )
