"""In-memory stand-ins for the Slack WebClient and Bolt App (no unittest.mock)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["FakeBoltApp", "FakeSlackApiError", "FakeWebClient"]


class FakeSlackApiError(Exception):
    """Raised by FakeWebClient for methods registered in ``errors``."""

    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.response = {"ok": False, "error": error}


class FakeWebClient:
    """Record every API call as ``(method, kwargs)`` in ``calls``.

    ``errors`` maps a method name to the exception it raises. ``replies`` is the
    ``messages`` list returned by ``conversations_replies``.
    """

    def __init__(
        self,
        *,
        token: str | None = "xoxb-test",
        errors: dict[str, BaseException] | None = None,
        replies: list[dict[str, Any]] | None = None,
    ) -> None:
        self.token = token
        self.errors: dict[str, BaseException] = dict(errors or {})
        self.replies: list[dict[str, Any]] = list(replies or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_ts = 1

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, dict(kwargs)))
        error = self.errors.get(method)
        if error is not None:
            raise error

    def calls_of(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        self._record("chat_postMessage", kwargs)
        ts = f"{self._next_ts}.000100"
        self._next_ts += 1
        return {"ok": True, "channel": kwargs.get("channel"), "ts": ts}

    def chat_update(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        self._record("chat_update", kwargs)
        return {"ok": True, "ts": kwargs.get("ts")}

    def reactions_add(self, **kwargs: Any) -> dict[str, Any]:
        self._record("reactions_add", kwargs)
        return {"ok": True}

    def reactions_remove(self, **kwargs: Any) -> dict[str, Any]:
        self._record("reactions_remove", kwargs)
        return {"ok": True}

    def conversations_replies(self, **kwargs: Any) -> dict[str, Any]:
        self._record("conversations_replies", kwargs)
        return {"ok": True, "messages": list(self.replies)}


class FakeBoltApp:
    """Minimal Bolt ``App``: keeps constructor kwargs and registered event handlers."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.client = FakeWebClient(token=kwargs.get("token"))
        self.handlers: dict[str, list[Callable[..., Any]]] = {}

    def event(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(func: Callable[..., Any]) -> Callable[..., Any]:
            self.handlers.setdefault(name, []).append(func)
            return func

        return register

    def dispatch(self, name: str, event: dict[str, Any]) -> list[Any]:
        return [handler(event=event) for handler in self.handlers.get(name, [])]
