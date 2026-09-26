"""mltgnt.media.slack.history (#4029)."""

from __future__ import annotations

from mltgnt.media.slack import history
from tests.media.slack.fakes import FakeSlackApiError, FakeWebClient


def _rich(*elements: dict) -> dict:
    return {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": list(elements)}]}


def test_text_wins_over_blocks() -> None:
    assert history.extract_text_from_message({"text": "plain", "blocks": [_rich({"type": "text", "text": "b"})]}) == (
        "plain"
    )


def test_blocks_rich_text_and_section() -> None:
    message = {
        "text": "  ",
        "blocks": [
            _rich({"type": "text", "text": "hello "}, {"type": "link", "url": "https://example.test"}),
            {"type": "section", "text": {"type": "mrkdwn", "text": "section text"}},
            {"type": "section", "text": "not a dict"},
            {"type": "divider"},
            "junk",
        ],
    }
    assert history.extract_text_from_message(message) == "hello https://example.test\nsection text"


def test_table_block() -> None:
    table = {
        "type": "table",
        "rows": [
            [{"type": "raw_text", "text": "name"}, {"type": "raw_text", "text": "qty"}],
            [None, {"type": "raw_text", "text": ""}],
            [_rich({"type": "text", "text": "a|b"}), {"type": "raw_text", "text": "1\n2"}, {"type": "other"}],
            "junk",
            [],
        ],
    }
    assert history.table_block_to_markdown(table) == ("| name | qty |  |\n| --- | --- | --- |\n| a\\|b | 1<br>2 |  |")
    assert history.extract_text_from_message({"blocks": [table]}).startswith("| name")


def test_table_block_invalid() -> None:
    assert history.table_block_to_markdown({"type": "section"}) == ""
    assert history.table_block_to_markdown({"type": "table", "rows": []}) == ""
    assert history.table_block_to_markdown({"type": "table", "rows": [[None, "x"]]}) == ""


def test_attachment_blocks_then_fallback() -> None:
    with_blocks = {
        "attachments": [
            {"blocks": [_rich({"type": "text", "text": "att block"})]},
            {"fallback": "ignored when blocks exist"},
        ]
    }
    assert history.extract_text_from_message(with_blocks) == "att block"
    fallback_only = {"attachments": ["junk", {"fallback": " fb "}, {"pretext": "pre"}, {"text": ""}]}
    assert history.extract_text_from_message(fallback_only) == "fb\npre"
    assert history.extract_text_from_message({"attachments": []}) == ""
    assert history.extract_text_from_message("junk") == ""  # type: ignore[arg-type]


def test_fetch_thread_messages() -> None:
    web = FakeWebClient(
        replies=[
            {"user": "U1", "text": "root", "ts": "1.0"},
            {"bot_id": "B1", "text": "reply", "ts": "2.0"},
            {"user": "U1", "text": "trigger", "ts": "3.0"},
        ]
    )
    got = history.fetch_thread_messages(web, "C1", "1.0", limit=1, exclude_ts="3.0")
    assert web.calls_of("conversations_replies") == [{"channel": "C1", "ts": "1.0", "limit": 1}]
    assert got == [{"user": "", "text": "reply", "ts": "2.0", "is_bot": True, "bot_id": "B1"}]
    assert len(history.fetch_thread_messages(web, "C1", "1.0")) == 3


def test_fetch_thread_messages_failure_returns_empty() -> None:
    web = FakeWebClient(errors={"conversations_replies": FakeSlackApiError("thread_not_found")})
    assert history.fetch_thread_messages(web, "C1", "1.0") == []
