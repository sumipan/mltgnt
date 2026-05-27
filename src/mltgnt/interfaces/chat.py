from pathlib import Path
from typing import Protocol, runtime_checkable

from mltgnt.interfaces.types import ChatInputBase, ChatOutputBase


@runtime_checkable
class ChatPipelineProtocol(Protocol):
    def run(self, inp: ChatInputBase, repo_root: Path) -> ChatOutputBase:
        """ホスト側が実装する L1 チャットパイプライン契約。

        ``ChatInputBase`` / ``ChatOutputBase`` を受け取り・返す。書き戻し（ファイル保存等）は
        ホストの責務。mltgnt 内部の ``run_chat``（L3 実行エンジン）とはレイヤが異なり、
        シグネチャも異なる。ホスト実装は通常 ``run_chat`` をラップして本 Protocol を満たす。
        """
        ...
