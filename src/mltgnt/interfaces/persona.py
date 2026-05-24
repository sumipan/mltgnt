from typing import Protocol, runtime_checkable

from mltgnt.interfaces.types import PersonaFMBase


@runtime_checkable
class PersonaProtocol(Protocol):
    name: str
    fm: PersonaFMBase

    def format_prompt(self, instruction: str) -> str:
        """システムプロンプト（人物像 + instruction）を返す。"""
        ...
