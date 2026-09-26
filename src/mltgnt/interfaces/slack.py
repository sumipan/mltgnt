from typing import Protocol


class SlackClientProtocol(Protocol):
    """Deprecated: use ``mltgnt.interfaces.media.MediaClient``.

    Kept as a compatibility alias; pass such clients through
    ``mltgnt.interfaces.media.adapt_client``. Scheduled for removal in the next
    minor (Y) bump.
    """

    def post_message(
        self,
        text: str,
        channel: str,
        thread_ts: str | None = None,
        blocks: list[dict] | None = None,
        reply_broadcast: bool = False,
    ) -> bool:
        """Post a message to Slack. Return False on failure (do not raise)."""
        ...

    def post_message_ts(
        self,
        text: str,
        channel: str,
        thread_ts: str | None = None,
    ) -> str | None:
        """Post a message and return its ts ("" when unknown). Return None on failure.

        Default: delegates to ``post_message`` and returns "" on success.
        """
        return "" if self.post_message(text, channel=channel, thread_ts=thread_ts) else None
