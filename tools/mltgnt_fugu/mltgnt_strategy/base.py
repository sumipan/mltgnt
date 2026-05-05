from __future__ import annotations

from abc import ABC, abstractmethod

from ..dag_bridge import DagBridge, DagJobSpec, DagJobResult
from ..types import StrategyInput, StrategyResult


class Strategy(ABC):

    @abstractmethod
    def execute(self, inp: StrategyInput) -> StrategyResult:
        ...

    def _persona_call(self, inp: StrategyInput, prompt: str, *, depends: tuple[str, ...] = ()) -> DagJobResult:
        """inp.state.metadata から engine/model/timeout_s を取得して DAG ジョブを投入・待機する"""
        meta = inp.state.metadata
        spec = DagJobSpec(
            prompt=prompt,
            persona_name=inp.persona_name,
            engine=meta.get("engine", "claude"),
            model=meta.get("model", "sonnet"),
            timeout_s=meta.get("timeout_s", 120),
            depends=depends,
        )
        return inp.dag.submit_and_wait(spec)

    def _judge_call(self, inp: StrategyInput, prompt: str, *, depends: tuple[str, ...] = ()) -> DagJobResult:
        """判定用: engine="claude", model="haiku", timeout_s=30 固定"""
        spec = DagJobSpec(
            prompt=prompt,
            persona_name=inp.persona_name,
            engine="claude",
            model="haiku",
            timeout_s=30,
            depends=depends,
        )
        return inp.dag.submit_and_wait(spec)
