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


