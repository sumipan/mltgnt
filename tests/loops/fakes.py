"""tests/loops/fakes.py — FakeHumanChannel / FakeExecutor / fake LLM。"""
from __future__ import annotations

from dataclasses import dataclass, field

from mltgnt.interfaces.loops import HumanThreadRef, StepPoll, StepSubmission


@dataclass
class FakeHumanChannel:
    threads: list[HumanThreadRef] = field(default_factory=list)
    asks: list[dict] = field(default_factory=list)
    notifies: list[dict] = field(default_factory=list)
    fallbacks: list[dict] = field(default_factory=list)
    closes: list[dict] = field(default_factory=list)
    delivered: set[str] = field(default_factory=set)

    def open_thread(self, *, loop_id, persona, title, body, event_id):
        if event_id in self.delivered:
            return self.threads[-1] if self.threads else None
        ref = HumanThreadRef(channel_id="C1", thread_ts="123.456")
        self.threads.append(ref)
        self.delivered.add(event_id)
        return ref

    def ask(self, *, loop_id, persona, thread, question_id, text, event_id):
        if event_id in self.delivered:
            return True
        self.asks.append(
            {"loop_id": loop_id, "question_id": question_id, "text": text, "event_id": event_id}
        )
        self.delivered.add(event_id)
        return True

    def notify(self, *, loop_id, persona, thread, text, event_id):
        if event_id in self.delivered:
            return True
        self.notifies.append({"loop_id": loop_id, "text": text, "event_id": event_id})
        self.delivered.add(event_id)
        return True

    def notify_fallback(self, *, loop_id, text, event_id):
        if event_id in self.delivered:
            return True
        self.fallbacks.append({"loop_id": loop_id, "text": text, "event_id": event_id})
        self.delivered.add(event_id)
        return True

    def close_thread(self, *, loop_id, persona, thread, event_id):
        if event_id in self.delivered:
            return True
        self.closes.append({"loop_id": loop_id, "event_id": event_id})
        self.delivered.add(event_id)
        return True


@dataclass
class FakeExecutor:
    submissions: list[StepSubmission] = field(default_factory=list)
    poll_results: dict[str, StepPoll] = field(default_factory=dict)
    submit_calls: list[str] = field(default_factory=list)

    def submit(self, *, prompt: str, idempotency_key: str) -> StepSubmission:
        self.submit_calls.append(idempotency_key)
        if self.submissions:
            return self.submissions.pop(0)
        return StepSubmission(
            uuid="uuid-1",
            result_filename="result-1.md",
            submitted_at="2026-08-20T12:00:00+09:00",
            reused=False,
        )

    def poll(self, *, uuid: str, result_filename: str) -> StepPoll:
        return self.poll_results.get(uuid, StepPoll(status="pending", content=""))
