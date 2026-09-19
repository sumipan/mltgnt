"""Tests for persona phrases, resolution, types, and memory (#3318)."""
from __future__ import annotations

from pathlib import Path

from mltgnt.persona.memory import format_tool_trace_block
from mltgnt.persona.phrases import load_persona_phrases
from mltgnt.persona.resolve import build_persona_context, resolve_responder
from mltgnt.persona.types import PersonaContext


def _write_agent(tmp_path: Path, name: str, phrases: dict[str, str] | None) -> None:
    agents = tmp_path / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    if phrases is None:
        fm = "ops:\n  engine: cursor\n"
    else:
        lines = ["ops:", "  engine: cursor", "  phrases:"]
        for k, v in phrases.items():
            lines.append(f'    {k}: "{v}"')
        fm = "\n".join(lines) + "\n"
    (agents / f"{name}.md").write_text(
        f"---\nspec_version: 2.2.0\npersona:\n  name: {name}\n{fm}---\n# {name}\n",
        encoding="utf-8",
    )


def test_load_persona_phrases_returns_dict(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "persona-a test",
        {"approval_hint": "Say ok to proceed", "done": "✅ done"},
    )
    out = load_persona_phrases("persona-a test", persona_dir=tmp_path / "agents")
    assert out["approval_hint"] == "Say ok to proceed"
    assert out["done"] == "✅ done"


def test_load_persona_phrases_missing_file(tmp_path: Path) -> None:
    assert load_persona_phrases("nonexistent", persona_dir=tmp_path / "agents") == {}


def test_resolve_responder_uses_injected_fn() -> None:
    def _resolve(text: str, **kwargs: object) -> str | None:
        assert text == "persona-a, listen"
        assert kwargs["space_id"] == "space-1"
        return "persona-a"

    got = resolve_responder(
        "persona-a, listen",
        space_id="space-1",
        conversation_id="conv-1",
        persona_map={"space-1": []},
        pinned_personas={"conv-1": "persona-a"},
        resolve_fn=_resolve,
    )
    assert got == "persona-a"


def test_build_persona_context_uses_injected_loaders() -> None:
    ctx = build_persona_context(
        "persona-a",
        memory_excerpt="mem",
        observers=("obs1",),
        load_profile_fn=lambda _name, *, weight="heavy": ("profile text", None),
        engine_model_fn=lambda _name: ("claude", "sonnet"),
        skills_fn=lambda _name: ["s1"],
        phrases_fn=lambda _name: {"hi": "hi there"},
    )
    assert isinstance(ctx, PersonaContext)
    assert ctx.persona_id == "persona-a"
    assert ctx.engine == "claude"
    assert ctx.model == "sonnet"
    assert ctx.profile == "profile text"
    assert ctx.memory_excerpt == "mem"
    assert ctx.skills == ("s1",)
    assert ctx.phrases == ("hi there",)
    assert ctx.observers == ("obs1",)


def test_persona_context_has_no_media_fields() -> None:
    import dataclasses

    names = {f.name for f in dataclasses.fields(PersonaContext)}
    assert not names & {"channel", "thread_ts", "user_id", "blocks"}


def test_format_tool_trace_block() -> None:
    block = format_tool_trace_block(
        [{"tool": "read_file", "args": {"path": "a.md"}, "result": "ok"}]
    )
    assert "[tool: read_file(path='a.md')]" in block
    assert "[result: ok]" in block


def test_resolve_source_has_no_routing_import() -> None:
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mltgnt"
        / "persona"
        / "resolve.py"
    ).read_text(encoding="utf-8")
    assert "mltgnt.routing" not in src
    assert "slack_sdk" not in src
    assert "REPO_ROOT" not in src
