"""mltgnt.media.slack.mrkdwn (#4029)."""

from __future__ import annotations

from mltgnt.media.slack import mrkdwn


def test_basic_rules() -> None:
    text = "# Title\n**bold** and __under__ and ~~gone~~\n- item\n* star\n[site](https://example.test)"
    assert mrkdwn.markdown_to_mrkdwn(text) == (
        "*Title*\n*bold* and _under_ and ~gone~\n• item\n• star\n<https://example.test|site>"
    )


def test_code_is_preserved() -> None:
    text = "```python\n**x** [a](b)\n```\nuse `**y**` here"
    assert mrkdwn.markdown_to_mrkdwn(text) == "```\n**x** [a](b)\n```\nuse `**y**` here"


def test_table_becomes_bullets() -> None:
    text = "| a | b |\n|---|:-:|\n| 1 | 2 |\n|  |  |\nafter"
    assert mrkdwn.markdown_to_mrkdwn(text) == "• a / b\n• 1 / 2\nafter"


def test_wikilink_default_and_custom() -> None:
    assert mrkdwn.markdown_to_mrkdwn("see [[Note A|label]] and [[Note B]]") == "see label and Note B"
    got = mrkdwn.markdown_to_mrkdwn("[[Note A]]", wikilink=lambda target, label: f"<app://{target}|{label}>")
    assert got == "<app://Note A|Note A>"


def test_empty_and_empty_heading() -> None:
    assert mrkdwn.markdown_to_mrkdwn("   ") == ""
    assert mrkdwn.markdown_to_mrkdwn("#   \nx") == "#   \nx"


def test_normalize_markdown_residual() -> None:
    text = "## Head\n**b** *i* _u_\n- a\n1. one\n| x | y |\n|---|---|"
    assert mrkdwn.normalize_markdown_residual(text) == "Head\nb i u\n• a\n• one\n• x / y"
    assert mrkdwn.normalize_markdown_residual("") == ""
    assert mrkdwn.normalize_markdown_residual("#  \nz") == "#  \nz"
