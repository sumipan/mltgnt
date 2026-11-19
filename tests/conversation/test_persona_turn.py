"""PersonaTurnHandler — standard one-turn flow (#7434)."""
from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

from ghdag.core import capabilities

from mltgnt.config import MemoryConfig
from mltgnt.conversation.persona_turn import PersonaTurnConfig, PersonaTurnHandler
from mltgnt.interfaces.turn import HistoryMessage, TurnInput, TurnResult
from mltgnt.memory import MemoryEntry, parse_jsonl, serialize_entry
from mltgnt.persona.memory import direct_dedupe_base

Permission = capabilities.Permission

PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: persona-a
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## Basic information

    persona-a is a test persona.

    ## Values

    Curious.

    ## Reaction patterns

    Answers questions.

    ## Tone

    Friendly.

    ## Output format

    Be concise.
""")


def _persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir(exist_ok=True)
    (d / "persona-a.md").write_text(PERSONA_CONTENT, encoding="utf-8")
    return d


def _memory_dir(tmp_path: Path) -> Path:
    d = tmp_path / "memory" / "persona-a"
    d.mkdir(parents=True)
    return d


def _turn_config(tmp_path: Path, **overrides: object) -> PersonaTurnConfig:
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    (jobs / "done").mkdir(exist_ok=True)
    base = dict(
        persona_dir=tmp_path / "agents",
        jobs_dir=jobs,
        exec_done_dir=jobs / "done",
        memory=MemoryConfig(chat_dir=tmp_path, chat_memory_dir=tmp_path / "memory"),
    )
    base.update(overrides)
    return PersonaTurnConfig(**base)


def _episode(
    space: str,
    thread: str,
    message_id: str,
    role: str,
    content: str,
    *,
    peer: str | None = None,
) -> MemoryEntry:
    key = f"{direct_dedupe_base(space, thread, message_id)}:{role}"
    return MemoryEntry(
        timestamp="2026-11-19 10:00",
        role=role,
        content=content,
        source_tag="[direct]",
        dedupe_key=key,
        peer=peer,
    )


def _write_episodes(path: Path, entries: list[MemoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(serialize_entry(e) + "\n" for e in entries), encoding="utf-8")


def _handler(
    tmp_path: Path,
    *,
    enqueue,
    instruction: str | None = None,
    extend_prompt=None,
    postprocess=None,
    **config_kw: object,
) -> PersonaTurnHandler:
    _persona_dir(tmp_path)
    _memory_dir(tmp_path)
    fixed = datetime(2026, 11, 19, 12, 0)
    return PersonaTurnHandler(
        _turn_config(tmp_path, **config_kw),
        resolve_persona=lambda _turn: "persona-a",
        instruction=instruction,
        extend_prompt=extend_prompt,
        postprocess=postprocess,
        enqueue=enqueue,
        now=lambda: fixed,
    )


def test_handle_builds_prompt_in_order_and_returns_reply(tmp_path: Path) -> None:
    captured: list[str] = []

    def enqueue(prompt: str, *_args, **_kwargs) -> tuple[bool, str]:
        captured.append(prompt)
        return True, "model reply"

    handler = _handler(tmp_path, enqueue=enqueue, instruction="Reply briefly.")
    turn = TurnInput(
        conversation_id="C1:1.0",
        text="current message",
        history=(
            HistoryMessage(role="user", text="history one"),
            HistoryMessage(role="assistant", text="history two"),
        ),
    )

    result = handler.handle(turn)

    assert result == TurnResult(kind="reply", text="model reply")
    prompt = captured[0]
    assert prompt.index("history one") < prompt.index("history two")
    assert prompt.index("history two") < prompt.index("current message")
    assert prompt.index("current message") < prompt.index("Reply briefly.")


def test_recent_scope_thread_peer_persona(tmp_path: Path) -> None:
    mem = _memory_dir(tmp_path)
    _write_episodes(
        mem / "episodes.jsonl",
        [
            _episode("C1", "thread-alice", "a1", "user", "alice-only-body", peer="alice"),
            _episode("C1", "thread-bob", "b1", "user", "bob-only-body", peer="bob"),
        ],
    )
    _persona_dir(tmp_path)
    fixed = datetime(2026, 11, 19, 12, 0)

    def make_handler(scope: str) -> tuple[PersonaTurnHandler, list[str]]:
        prompts: list[str] = []

        def enqueue(prompt: str, *_args, **_kwargs) -> tuple[bool, str]:
            prompts.append(prompt)
            return True, "ok"

        handler = PersonaTurnHandler(
            _turn_config(tmp_path, recent_scope=scope),
            resolve_persona=lambda _t: "persona-a",
            enqueue=enqueue,
            now=lambda: fixed,
        )
        return handler, prompts

    for scope, author, expect_alice, expect_bob in (
        ("thread", "alice", False, False),
        ("peer", "alice", True, False),
        ("persona", "alice", True, True),
    ):
        handler, prompts = make_handler(scope)
        handler.handle(
            TurnInput(conversation_id="C1:1.0", text="hello", author_id=author),
        )
        prompt = prompts[0]
        assert ("alice-only-body" in prompt) == expect_alice, scope
        assert ("bob-only-body" in prompt) == expect_bob, scope

    handler, prompts = make_handler("peer")
    handler.handle(TurnInput(conversation_id="C1:1.0", text="hello", author_id=None))
    prompt = prompts[0]
    assert "alice-only-body" not in prompt
    assert "bob-only-body" not in prompt


def test_episode_is_recorded_with_peer(tmp_path: Path) -> None:
    def enqueue(_prompt: str, *_args, **_kwargs) -> tuple[bool, str]:
        return True, "assistant line"

    handler = _handler(tmp_path, enqueue=enqueue)
    handler.handle(
        TurnInput(conversation_id="C1:1.0", text="user line", author_id="alice"),
    )

    path = tmp_path / "memory" / "persona-a" / "episodes.jsonl"
    entries = parse_jsonl(path)
    assert len(entries) == 2
    assert entries[0].content == "user line"
    assert entries[1].content == "assistant line"
    assert all(e.peer == "alice" for e in entries)


def test_failed_job_returns_error_and_records_nothing(tmp_path: Path) -> None:
    def enqueue(_prompt: str, *_args, **_kwargs) -> tuple[bool, str]:
        return False, "timeout (300s)"

    handler = _handler(tmp_path, enqueue=enqueue)
    result = handler.handle(
        TurnInput(conversation_id="C1:1.0", text="user line", author_id="alice"),
    )

    assert result == TurnResult(kind="reply", text="(error: timeout (300s))")
    assert not (tmp_path / "memory" / "persona-a" / "episodes.jsonl").exists()


def test_postprocess_can_replace_the_result(tmp_path: Path) -> None:
    def enqueue(_prompt: str, *_args, **_kwargs) -> tuple[bool, str]:
        return True, "raw reply"

    def postprocess(_turn: TurnInput, _text: str) -> TurnResult:
        return TurnResult(kind="reply", text="X", reaction="eyes")

    handler = _handler(tmp_path, enqueue=enqueue, postprocess=postprocess)
    result = handler.handle(
        TurnInput(conversation_id="C1:1.0", text="user line", author_id="alice"),
    )

    assert result == TurnResult(kind="reply", text="X", reaction="eyes")
    entries = parse_jsonl(tmp_path / "memory" / "persona-a" / "episodes.jsonl")
    assert entries[1].content == "X"


def test_permission_is_passed_to_enqueue(tmp_path: Path) -> None:
    perm = Permission(allowed_tools=("Read",))
    seen: list[object] = []

    def enqueue(_prompt: str, *_args, permission=None, **_kwargs) -> tuple[bool, str]:
        seen.append(permission)
        return True, "ok"

    handler = _handler(tmp_path, enqueue=enqueue, permission=perm)
    handler.handle(TurnInput(conversation_id="C1:1.0", text="hi"))

    assert seen == [perm]
