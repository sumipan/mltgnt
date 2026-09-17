from typing import Protocol


class SlackClientProtocol(Protocol):
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
