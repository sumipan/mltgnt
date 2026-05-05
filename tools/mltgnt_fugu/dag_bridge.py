from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DagJobSpec:
    prompt: str
    persona_name: str
    engine: str
    model: str
    timeout_s: int = 120
    depends: tuple[str, ...] = ()


@dataclass(frozen=True)
class DagJobResult:
    uid: str
    status: str          # "ok" | "error" | "timeout"
    body: str
    cost_usd: float | None = None


class DagBridge(Protocol):
    """DAG ジョブの投入・待機を抽象化する Protocol"""

    def submit(self, spec: DagJobSpec) -> str:
        """ジョブを投入し、uid（追跡用一意文字列）を返す"""
        ...

    def wait(self, uid: str, *, timeout_s: int = 120) -> DagJobResult:
        """uid のジョブ完了を待ち、結果を返す。タイムアウト時は status="timeout" の DagJobResult を返す"""
        ...

    def submit_and_wait(self, spec: DagJobSpec) -> DagJobResult:
        """submit + wait のショートハンド。spec.timeout_s を wait の timeout_s に使用する"""
        ...


class GhdagDagBridge:
    """本番用: exec.md 経由で ghdag にジョブを投入する実装"""

    def __init__(self, exec_md_path: Path) -> None:
        self._exec_md_path = exec_md_path

    def submit(self, spec: DagJobSpec) -> str:
        raise NotImplementedError("Phase D で実装")

    def wait(self, uid: str, *, timeout_s: int = 120) -> DagJobResult:
        raise NotImplementedError("Phase D で実装")

    def submit_and_wait(self, spec: DagJobSpec) -> DagJobResult:
        raise NotImplementedError("Phase D で実装")


class FakeDagBridge:
    """テスト用: persona_name → body の固定マップで即座に応答する"""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self._counter = 0
        self._submitted: dict[str, DagJobSpec] = {}

    def submit(self, spec: DagJobSpec) -> str:
        self._counter += 1
        uid = f"fake-{self._counter}"
        self._submitted[uid] = spec
        return uid

    def wait(self, uid: str, *, timeout_s: int = 120) -> DagJobResult:
        spec = self._submitted.get(uid)
        if spec is None:
            return DagJobResult(uid=uid, status="error", body="unknown uid")
        body = self._responses.get(spec.persona_name)
        if body is None:
            return DagJobResult(uid=uid, status="error", body=f"no response registered for persona: {spec.persona_name}")
        return DagJobResult(uid=uid, status="ok", body=body)

    def submit_and_wait(self, spec: DagJobSpec) -> DagJobResult:
        uid = self.submit(spec)
        return self.wait(uid, timeout_s=spec.timeout_s)
