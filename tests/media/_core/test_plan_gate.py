"""mltgnt.media._core.plan_gate (#4030)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mltgnt.config.language import JA
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.pending import PendingStore
from mltgnt.media._core.plan_gate import PENDING_KEY, PlanGate, PlanState, expire_pending, is_approval


def _config(tmp_path: Path) -> MediaConfig:
    return MediaConfig(state_dir=tmp_path, pending_dir=tmp_path, events_dir=tmp_path, approval_ttl_sec=60.0)


def test_approval_word_approves(tmp_path: Path) -> None:
    gate = PlanGate.open(_config(tmp_path), now=1000.0)
    assert gate.state is PlanState.AWAITING
    assert gate.on_reply("OK!", now=1010.0).state is PlanState.APPROVED


def test_other_reply_rejects(tmp_path: Path) -> None:
    gate = PlanGate.open(_config(tmp_path), now=1000.0)
    assert gate.on_reply("please change step 2", now=1010.0).state is PlanState.REJECTED


def test_reply_after_ttl_expires_and_never_approves(tmp_path: Path) -> None:
    gate = PlanGate.open(_config(tmp_path), now=1000.0)
    expired = gate.on_reply("ok", now=1060.0)
    assert expired.state is PlanState.EXPIRED
    assert expired.on_reply("ok", now=1000.0).state is PlanState.EXPIRED
    assert expired.expire(2000.0).state is PlanState.EXPIRED


def test_terminal_states_do_not_change(tmp_path: Path) -> None:
    gate = PlanGate.open(_config(tmp_path), now=1000.0)
    approved = gate.on_reply("yes", now=1001.0)
    assert approved.on_reply("no", now=1002.0) is approved
    assert approved.expire(5000.0) is approved
    assert gate.expire(1001.0) is gate
    assert gate.expire(1060.0).state is PlanState.EXPIRED


def test_is_approval_word_boundaries_and_cancel() -> None:
    assert is_approval("Approve, thanks")
    assert not is_approval("look at this")
    assert not is_approval("ok but cancel")
    pack = replace(JA, approval_words=frozenset({"go ahead"}))
    assert is_approval("Go  ahead.", pack)
    assert not is_approval("ok", pack)


def test_round_trip_and_bad_dicts() -> None:
    gate = PlanGate(expires_at=5.0, state=PlanState.REJECTED)
    assert PlanGate.from_dict(gate.to_dict()) == gate
    assert PlanGate.from_dict(None) is None
    assert PlanGate.from_dict({"expires_at": True}) is None
    assert PlanGate.from_dict({"expires_at": 1, "state": "nope"}) is None
    assert PlanGate.from_dict({"expires_at": 1}) == PlanGate(expires_at=1.0)


def test_expire_pending_claims_once(tmp_path: Path) -> None:
    store = PendingStore(tmp_path)
    pending = {"state": "awaiting_plan_approval", PENDING_KEY: PlanGate(expires_at=100.0).to_dict()}
    store.save("u1", pending)
    assert expire_pending(store, "u1", pending, now=99.0) is None
    claimed = expire_pending(store, "u1", pending, now=100.0)
    assert claimed is not None
    assert claimed[PENDING_KEY]["state"] == "expired"
    assert store.load("u1") is None
    assert expire_pending(store, "u1", pending, now=100.0) is None
    assert expire_pending(store, "u2", {}, now=100.0) is None
