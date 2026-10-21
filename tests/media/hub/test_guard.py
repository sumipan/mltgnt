"""Guard checks for hub message posting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mltgnt.interfaces.hub import HubAddress, HubMessage
from mltgnt.media.hub.guard import GuardConfig, check

NOW = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
CFG = GuardConfig(hop_limit=3, max_agent_run=4, min_interval_sec=5.0, room_daily_limit=200)


def _msg(**kwargs: object) -> HubMessage:
    defaults = {
        "id": "m1",
        "ts": None,
        "room": "lobby",
        "thread": None,
        "sender": HubAddress("host-a", "bot"),
        "to": None,
        "text": "hi",
        "attachments": (),
        "hop": 0,
        "origin_id": None,
        "kind": "message",
        "status": None,
    }
    defaults.update(kwargs)
    return HubMessage(**defaults)  # type: ignore[arg-type]


def test_hop_exceeded_at_four() -> None:
    verdict = check(_msg(hop=4), [], 0, NOW, config=CFG)
    assert verdict.ok is False
    assert verdict.reason == "hop_exceeded"


def test_hop_allowed_at_limit() -> None:
    verdict = check(_msg(hop=3), [], 0, NOW, config=CFG)
    assert verdict.ok is True


def test_agent_run_exceeded_after_four_persona_rows() -> None:
    rows = [
        {"kind": "message", "sender": {"host": "host-a", "persona": "p1"}, "author": "host-a/p1"},
        {"kind": "message", "sender": {"host": "host-a", "persona": "p2"}, "author": "host-a/p2"},
        {"kind": "message", "sender": {"host": "host-a", "persona": "p3"}, "author": "host-a/p3"},
        {"kind": "message", "sender": {"host": "host-a", "persona": "p4"}, "author": "host-a/p4"},
    ]
    verdict = check(_msg(sender=HubAddress("host-a", "p5")), rows, 0, NOW, config=CFG)
    assert verdict.ok is False
    assert verdict.reason == "agent_run_exceeded"


def test_status_rows_do_not_count_toward_agent_run() -> None:
    rows = [
        {"kind": "status", "sender": {"host": "host-a", "persona": "p1"}, "author": "host-a/p1"},
    ] * 10
    verdict = check(_msg(sender=HubAddress("host-a", "p1")), rows, 0, NOW, config=CFG)
    assert verdict.ok is True


def test_too_fast_within_interval() -> None:
    prev_ts = (NOW - timedelta(seconds=4)).isoformat()
    rows = [
        {
            "kind": "message",
            "sender": {"host": "host-a", "persona": "bot"},
            "to": None,
            "ts": prev_ts,
        }
    ]
    verdict = check(_msg(sender=HubAddress("host-a", "bot")), rows, 0, NOW, config=CFG)
    assert verdict.ok is False
    assert verdict.reason == "too_fast"


def test_too_fast_allowed_at_exact_interval() -> None:
    prev_ts = (NOW - timedelta(seconds=5)).isoformat()
    rows = [
        {
            "kind": "message",
            "sender": {"host": "host-a", "persona": "bot"},
            "to": None,
            "ts": prev_ts,
        }
    ]
    verdict = check(_msg(sender=HubAddress("host-a", "bot")), rows, 0, NOW, config=CFG)
    assert verdict.ok is True


def test_room_daily_limit_at_200() -> None:
    verdict = check(_msg(), [], 200, NOW, config=CFG)
    assert verdict.ok is False
    assert verdict.reason == "room_daily_limit"


def test_ok_when_all_checks_pass() -> None:
    verdict = check(_msg(), [], 0, NOW, config=CFG)
    assert verdict.ok is True
    assert verdict.reason is None
