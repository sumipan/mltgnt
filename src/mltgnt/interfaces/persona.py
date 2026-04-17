from typing import Protocol


class PersonaProtocol(Protocol):
    name: str

    def format_prompt(self, instruction: str) -> str:
        """システムプロンプト（人物像 + instruction）を返す。"""
        ...
