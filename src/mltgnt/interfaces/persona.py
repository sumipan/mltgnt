from typing import Protocol, runtime_checkable

from mltgnt.persona.schema import PersonaFM


@runtime_checkable
class PersonaProtocol(Protocol):
    name: str
    fm: PersonaFM

    def format_prompt(self, instruction: str) -> str:
        """システムプロンプト（人物像 + instruction）を返す。"""
        ...
