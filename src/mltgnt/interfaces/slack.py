from typing import Any, Protocol


class SlackClientProtocol(Protocol):
    def post_message(
        self,
        text: str,
        channel: str,
        thread_ts: str | None = None,
        blocks: list[dict] | None = None,
        reply_broadcast: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Slack にメッセージを投稿する。失敗時は False を返す（例外を送出しない）。"""
        ...
