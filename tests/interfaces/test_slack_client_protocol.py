"""SlackClientProtocol structural tests."""
from __future__ import annotations


def test_slack_protocol_no_kwargs() -> None:
    """SlackClientProtocol.post_message Home **kwargs There is no signature closure."""
    import inspect
    from mltgnt.interfaces import SlackClientProtocol
    sig = inspect.signature(SlackClientProtocol.post_message)
    var_keyword_params = [
        p for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    assert var_keyword_params == [], "**kwargs should not exist in SlackClientProtocol.post_message"


def test_slack_protocol_conforming_impl_basic() -> None:
    """SlackClientProtocol Meet implementation post_message(text, channel) Call only."""
    from mltgnt.interfaces import SlackClientProtocol

    class ConcreteSlack:
        def post_message(
            self,
            text: str,
            channel: str,
            thread_ts: str | None = None,
            blocks: list[dict] | None = None,
            reply_broadcast: bool = False,
        ) -> bool:
            return True

    client: SlackClientProtocol = ConcreteSlack()  # type: ignore[assignment]
    assert client.post_message("hello", "C123") is True


def test_slack_protocol_conforming_impl_full_kwargs() -> None:
    """SlackClientProtocol Complete implementation to meet optional Call with arguments."""
    from mltgnt.interfaces import SlackClientProtocol

    class ConcreteSlack:
        def post_message(
            self,
            text: str,
            channel: str,
            thread_ts: str | None = None,
            blocks: list[dict] | None = None,
            reply_broadcast: bool = False,
        ) -> bool:
            return True

    client: SlackClientProtocol = ConcreteSlack()  # type: ignore[assignment]
    assert client.post_message(
        "hello", "C123",
        thread_ts="ts001",
        blocks=[{"type": "section"}],
        reply_broadcast=True,
    ) is True
