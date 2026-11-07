"""matrix.inbound.to_media_event (#6101)."""

from __future__ import annotations

from pathlib import Path

from mltgnt.interfaces.turn import Attachment
from mltgnt.media._core import id_map
from mltgnt.media.matrix.config import MatrixMediaConfig
from mltgnt.media.matrix.inbound import to_media_event


def _config() -> MatrixMediaConfig:
    root = Path("/tmp/matrix-inbound")
    return MatrixMediaConfig(
        state_dir=root / "s",
        pending_dir=root / "p",
        events_dir=root / "e",
        homeserver_url="http://127.0.0.1:8008",
        user_id="@bot:x",
        rooms={"staff": "!a:x"},
        allowed_senders=("@u:x",),
    )


def _files(_url: str, _info: dict) -> Attachment | None:
    return Attachment(name="f.png", content_type="image/png", uri="/tmp/f.png")


def test_ignored_events_return_none() -> None:
    cfg = _config()
    base = {
        "type": "m.room.message",
        "event_id": "$e",
        "sender": "@u:x",
        "content": {"msgtype": "m.text", "body": "hi"},
    }
    assert to_media_event(base, room_id="!a:x", config=cfg, files=_files) is not None
    assert to_media_event({**base, "sender": "@other:x"}, room_id="!a:x", config=cfg, files=_files) is None
    assert to_media_event({**base, "sender": "@bot:x"}, room_id="!a:x", config=cfg, files=_files) is None
    assert to_media_event(base, room_id="!unknown:x", config=cfg, files=_files) is None
    assert (
        to_media_event(
            {**base, "content": {**base["content"], "m.relates_to": {"rel_type": "m.replace", "event_id": "$x"}}},
            room_id="!a:x",
            config=cfg,
            files=_files,
        )
        is None
    )
    assert to_media_event({**base, "content": {"msgtype": "m.notice", "body": "n"}}, room_id="!a:x", config=cfg, files=_files) is None
    assert to_media_event({"type": "m.room.encrypted", "event_id": "$e"}, room_id="!a:x", config=cfg, files=_files) is None


def test_conversation_id_thread_and_resolve() -> None:
    cfg = _config()
    plain = {
        "type": "m.room.message",
        "event_id": "$e1",
        "sender": "@u:x",
        "content": {"msgtype": "m.text", "body": "hi"},
    }
    ev = to_media_event(plain, room_id="!a:x", config=cfg, files=_files)
    assert ev is not None
    assert ev.conversation_id == "matrix.staff:$e1"
    threaded = {
        **plain,
        "event_id": "$e2",
        "content": {
            "msgtype": "m.text",
            "body": "reply",
            "m.relates_to": {"rel_type": "m.thread", "event_id": "$root"},
        },
    }
    ev2 = to_media_event(threaded, room_id="!a:x", config=cfg, files=_files)
    assert ev2 is not None
    assert ev2.conversation_id == "matrix.staff:$root"
    assert id_map.resolve(ev.conversation_id) == ("matrix.staff", "$e1")


def test_quote_strip_and_image_attachment() -> None:
    cfg = _config()
    event = {
        "type": "m.room.message",
        "event_id": "$e",
        "sender": "@u:x",
        "content": {
            "msgtype": "m.text",
            "body": "> <@u:x> old\n\nnew",
            "m.relates_to": {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": "$p"},
            },
        },
    }
    ev = to_media_event(event, room_id="!a:x", config=cfg, files=_files)
    assert ev is not None
    assert ev.text == "new"
    img = {
        "type": "m.room.message",
        "event_id": "$i",
        "sender": "@u:x",
        "content": {
            "msgtype": "m.image",
            "body": "pic.png",
            "url": "mxc://s/id",
            "info": {"mimetype": "image/png", "size": 3},
        },
    }
    ev2 = to_media_event(img, room_id="!a:x", config=cfg, files=_files)
    assert ev2 is not None
    assert ev2.attachments[0].name == "f.png"
