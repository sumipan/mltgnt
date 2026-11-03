"""mltgnt.media._core.id_map (#4029)."""

from __future__ import annotations

import pytest

from mltgnt.media._core import id_map


@pytest.mark.parametrize(
    ("space", "thread"),
    [("C123", "1700000000.000100"), ("room", "t:with:colons"), ("s", "1")],
)
def test_round_trip(space: str, thread: str) -> None:
    assert id_map.resolve(id_map.to_conversation_id(space, thread)) == (space, thread)


@pytest.mark.parametrize("bad", ["", "nosep", ":1", "C1:"])
def test_resolve_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        id_map.resolve(bad)


def test_storage_key() -> None:
    assert id_map.storage_key("C1", "1.2") == "C1-1.2"
    assert id_map.storage_key_from_conversation_id("C1:1.2") == "C1-1.2"
