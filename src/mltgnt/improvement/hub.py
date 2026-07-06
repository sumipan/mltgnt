"""ImprovementHub: orchestrator for multi-source improvement loops."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from mltgnt.improvement.loop import CycleResult, run_improvement_cycle


@runtime_checkable
class ImprovementSource(Protocol):
    @property
    def name(self) -> str: ...

    def run_cycle(self) -> CycleResult: ...


class MltgntSource:
    def __init__(
        self,
        audit_path: Path,
        persona_dir: Path,
        skills_dir: Path,
        *,
        since_days: int = 7,
    ) -> None:
        self._audit_path = audit_path
        self._persona_dir = persona_dir
        self._skills_dir = skills_dir
        self._since_days = since_days

    @property
    def name(self) -> str:
        return "mltgnt"

    def run_cycle(self) -> CycleResult:
        return run_improvement_cycle(
            self._audit_path,
            self._persona_dir,
            self._skills_dir,
            since_days=self._since_days,
        )


class ImprovementHub:
    def __init__(self) -> None:
        self._sources: list[ImprovementSource] = []

    def register(self, source: ImprovementSource) -> None:
        for existing in self._sources:
            if existing.name == source.name:
                raise ValueError(f"Source '{source.name}' is already registered")
        self._sources.append(source)

    def run_all_cycles(self) -> list[CycleResult]:
        return [source.run_cycle() for source in self._sources]
