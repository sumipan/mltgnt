"""Plan approval state machine: ``awaiting -> approved | rejected | expired``.

Approval words come from ``LanguagePack.approval_words``, cancel words from
``LanguagePack.cancel_words`` and the deadline from ``MediaConfig.approval_ttl_sec``.
Terminal states never change again (``expired`` can not become ``approved``).
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from mltgnt.config.language import JA, LanguagePack
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.pending import PendingStore

__all__ = [
    "AWAITING_STATE",
    "PENDING_KEY",
    "PlanGate",
    "PlanState",
    "expire_pending",
    "is_approval",
]

# ``state`` value of a pending record whose plan waits for approval
AWAITING_STATE = "awaiting_plan_approval"
# pending record key holding ``PlanGate.to_dict()``
PENDING_KEY = "plan_gate"


class PlanState(str, Enum):
    AWAITING = "awaiting"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


def _normalize(text: str) -> str:
    """Casefold and turn punctuation / symbols (emoji included) into spaces."""
    out: list[str] = []
    for ch in (text or "").casefold():
        cat = unicodedata.category(ch)
        out.append(" " if cat.startswith(("P", "Z")) or cat in ("Sm", "Sc", "Sk", "So") else ch)
    return " ".join("".join(out).split())


def _contains_word(normalized: str, words: Iterable[str]) -> bool:
    # ASCII alphanumerics bound a word so that "ok" does not match "look"
    for word in words:
        w = _normalize(word)
        if w and re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", normalized):
            return True
    return False


def is_approval(text: str, language: LanguagePack = JA) -> bool:
    """True when ``text`` contains an approval word and no cancel word."""
    normalized = _normalize(text)
    return _contains_word(normalized, language.approval_words) and not _contains_word(normalized, language.cancel_words)


@dataclass(frozen=True)
class PlanGate:
    """One plan waiting for approval. ``expires_at`` is epoch seconds."""

    expires_at: float
    state: PlanState = PlanState.AWAITING

    @classmethod
    def open(cls, config: MediaConfig, *, now: float | None = None) -> PlanGate:
        base = time.time() if now is None else now
        return cls(expires_at=base + config.approval_ttl_sec)

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def expire(self, now: float) -> PlanGate:
        """``awaiting`` past the deadline becomes ``expired``; anything else is unchanged."""
        if self.state is PlanState.AWAITING and self.is_expired(now):
            return replace(self, state=PlanState.EXPIRED)
        return self

    def on_reply(self, text: str, *, now: float, language: LanguagePack = JA) -> PlanGate:
        """Apply a reply: approval words approve, anything else rejects, late replies expire."""
        if self.state is not PlanState.AWAITING:
            return self
        if self.is_expired(now):
            return replace(self, state=PlanState.EXPIRED)
        state = PlanState.APPROVED if is_approval(text, language) else PlanState.REJECTED
        return replace(self, state=state)

    def to_dict(self) -> dict[str, Any]:
        return {"expires_at": self.expires_at, "state": self.state.value}

    @classmethod
    def from_dict(cls, data: object) -> PlanGate | None:
        if not isinstance(data, dict):
            return None
        expires_at = data.get("expires_at")
        if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
            return None
        try:
            state = PlanState(data.get("state", PlanState.AWAITING.value))
        except ValueError:
            return None
        return cls(expires_at=float(expires_at), state=state)


def expire_pending(store: PendingStore, uid: str, pending: dict[str, Any], *, now: float) -> dict[str, Any] | None:
    """Claim an awaiting pending record whose gate has expired.

    Return the consumed record (with the gate marked ``expired``) to the single
    caller that won the claim, else None.
    """
    gate = PlanGate.from_dict(pending.get(PENDING_KEY))
    if gate is None or gate.expire(now).state is not PlanState.EXPIRED:
        return None
    claimed = store.consume(uid)
    if claimed is None:
        return None
    claimed[PENDING_KEY] = gate.expire(now).to_dict()
    return claimed
