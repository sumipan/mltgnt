from typing import Any, Callable, Protocol, runtime_checkable

from mltgnt.interfaces.types import PersonaFMBase

PromptFilter = Callable[[str, dict[str, Any]], str]


@runtime_checkable
class PersonaProtocol(Protocol):
    name: str
    fm: PersonaFMBase

    def format_prompt(self, instruction: str) -> str:
        """Return the system prompt (persona + instruction)."""
        ...

    def register_prompt_filter(self, name: str, fn: PromptFilter) -> None:
        """Register a named filter. Replaces an existing filter of the same name."""
        ...
