from pathlib import Path
from typing import Protocol

from mltgnt.chat.models import ChatInput, ChatOutput


class ChatPipelineProtocol(Protocol):
    def run(self, inp: ChatInput, repo_root: Path) -> ChatOutput:
        """ChatInput を受け取りペルソナ応答を返す。書き戻しはホストの責務。"""
        ...
