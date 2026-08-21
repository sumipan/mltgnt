"""tests/loops/test_requests.py — request JSON の検証・列挙・隔離・consume。"""
from __future__ import annotations

import json
from pathlib import Path

from mltgnt.loops.requests import StartRequest, consume_request, list_requests


def _write_request(requests_dir: Path, name: str, payload: dict | str) -> Path:
    requests_dir.mkdir(parents=True, exist_ok=True)
    path = requests_dir / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_payload(**overrides) -> dict:
    data = {
        "objective_path": "foo.md",
        "channel_id": "C0123",
        "thread_ts": "1234567890.123456",
        "persona": "mizuho",
        "requested_at": "2026-08-21T13:00:00+09:00",
    }
    data.update(overrides)
    return data


def test_list_requests_sorted_by_filename(tmp_path):
    state_dir = tmp_path / "state"
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    _write_request(req_dir, "b.json", _valid_payload(objective_path="b.md"))
    _write_request(req_dir, "a.json", _valid_payload(objective_path="a.md"))

    ok, errors = list_requests(state_dir, objectives_dir)

    assert errors == []
    assert [r.filename for r in ok] == ["a.json", "b.json"]
    assert isinstance(ok[0], StartRequest)
    assert ok[0].objective_path == "a.md"


def test_consume_moves_to_consumed(tmp_path):
    state_dir = tmp_path / "state"
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    _write_request(req_dir, "a.json", _valid_payload())

    assert consume_request(state_dir, "a.json") is True
    assert not (req_dir / "a.json").exists()
    assert (req_dir / "consumed" / "a.json").exists()


def test_consume_idempotent_noop(tmp_path):
    state_dir = tmp_path / "state"
    req_dir = state_dir / "requests"
    _write_request(req_dir, "a.json", _valid_payload())
    assert consume_request(state_dir, "a.json") is True
    assert consume_request(state_dir, "a.json") is False
    assert consume_request(state_dir, "a.json", corrupt=True) is False


def test_invalid_json_isolated_to_corrupt(tmp_path):
    state_dir = tmp_path / "state"
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    _write_request(req_dir, "bad.json", "{not json")

    ok, errors = list_requests(state_dir, objectives_dir)

    assert ok == []
    assert len(errors) == 1
    assert errors[0].filename == "bad.json"
    assert not (req_dir / "bad.json").exists()
    assert (req_dir / "corrupt" / "bad.json").exists()


def test_missing_key_and_wrong_types_go_corrupt(tmp_path):
    state_dir = tmp_path / "state"
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    _write_request(req_dir, "miss.json", {"channel_id": "C1"})
    _write_request(
        req_dir,
        "bool.json",
        _valid_payload(channel_id=True),  # type: ignore[arg-type]
    )
    _write_request(
        req_dir,
        "num.json",
        _valid_payload(thread_ts=1234567890.123456),  # type: ignore[arg-type]
    )

    ok, errors = list_requests(state_dir, objectives_dir)

    assert ok == []
    assert {e.filename for e in errors} == {"bool.json", "miss.json", "num.json"}
    for name in ("bool.json", "miss.json", "num.json"):
        assert (req_dir / "corrupt" / name).exists()


def test_path_traversal_and_subdir_rejected(tmp_path):
    state_dir = tmp_path / "state"
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    _write_request(req_dir, "abs.json", _valid_payload(objective_path="/tmp/foo.md"))
    _write_request(req_dir, "dot.json", _valid_payload(objective_path="../foo.md"))
    _write_request(req_dir, "sub.json", _valid_payload(objective_path="dir/foo.md"))
    _write_request(req_dir, "txt.json", _valid_payload(objective_path="foo.txt"))

    ok, errors = list_requests(state_dir, objectives_dir)

    assert ok == []
    assert len(errors) == 4
    assert all((req_dir / "corrupt" / e.filename).exists() for e in errors)


def test_requested_at_requires_timezone(tmp_path):
    state_dir = tmp_path / "state"
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    _write_request(req_dir, "naive.json", _valid_payload(requested_at="2026-08-21T13:00:00"))

    ok, errors = list_requests(state_dir, objectives_dir)

    assert ok == []
    assert len(errors) == 1
    assert (req_dir / "corrupt" / "naive.json").exists()


def test_consume_rejects_non_basename(tmp_path):
    state_dir = tmp_path / "state"
    assert consume_request(state_dir, "../a.json") is False
    assert consume_request(state_dir, "sub/a.json") is False


def test_empty_persona_allowed(tmp_path):
    state_dir = tmp_path / "state"
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    _write_request(req_dir, "a.json", _valid_payload(persona=""))

    ok, errors = list_requests(state_dir, objectives_dir)

    assert errors == []
    assert ok[0].persona == ""
