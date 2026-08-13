from typing import Any, Callable, Protocol, runtime_checkable

from mltgnt.interfaces.types import PersonaFMBase

PromptFilter = Callable[[str, dict[str, Any]], str]


@runtime_checkable
class PersonaProtocol(Protocol):
    name: str
    fm: PersonaFMBase

    def format_prompt(self, instruction: str) -> str:
        """システムプロンプト（人物像 + instruction）を返す。"""
        ...

    def register_prompt_filter(self, name: str, fn: PromptFilter) -> None:
        """名前付きフィルタを登録する。同名がある場合は置換。"""
        ...
