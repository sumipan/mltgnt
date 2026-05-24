from pathlib import Path
from typing import Protocol, runtime_checkable

from mltgnt.interfaces.types import ChatInputBase, ChatOutputBase


@runtime_checkable
class ChatPipelineProtocol(Protocol):
    def run(self, inp: ChatInputBase, repo_root: Path) -> ChatOutputBase:
        """ChatInput を受け取りペルソナ応答を返す。書き戻しはホストの責務。"""
        ...
