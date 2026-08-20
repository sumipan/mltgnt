"""mltgnt.loops.executor — GhdagSubtaskExecutor bridge ラッパ。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mltgnt.bridges.ghdag_bridge import enqueue_step, poll_step
from mltgnt.interfaces.loops import StepPoll, StepSubmission


@dataclass
class GhdagSubtaskExecutor:
    jobs_dir: Path
    exec_done_dir: Path
    engine: str
    model: str
    correlation_id: str | None = None

    def submit(
        self,
        *,
        prompt: str,
        idempotency_key: str,
        engine: str | None = None,
        model: str | None = None,
    ) -> StepSubmission:
        resolved_engine = engine if engine is not None else self.engine
        resolved_model = model if model is not None else self.model
        return enqueue_step(
            prompt=prompt,
            engine=resolved_engine,
            model=resolved_model or None,
            idempotency_key=idempotency_key,
            jobs_dir=self.jobs_dir,
            correlation_id=self.correlation_id,
        )

    def poll(self, *, uuid: str, result_filename: str) -> StepPoll:
        return poll_step(
            exec_done_dir=self.exec_done_dir,
            jobs_dir=self.jobs_dir,
            uuid=uuid,
            result_filename=result_filename,
        )
