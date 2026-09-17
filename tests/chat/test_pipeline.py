"""Tests for mltgnt.chat.pipeline.run_pipeline"""
from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: persona-a
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## \u57fa\u672c\u60c5\u5831

    persona-a is a multi-legged tank-type AI robot from GHS.

    ## \u4fa1\u5024\u89b3

    Curious.

    ## \u53cd\u5fdc\u30d1\u30bf\u30fc\u30f3

    Answers questions.

    ## \u53e3\u8abf

    Friendly.

    ## \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

    Be concise.
""")


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    (d / "persona-a.md").write_text(PERSONA_CONTENT, encoding="utf-8")
    return d


def _load_persona(persona_dir: Path, persona_name: str):
    from mltgnt.persona.loader import load
    from mltgnt.persona.registry import resolve_with_alias
    from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, SYSTEM_DEFAULT_MODEL

    path = resolve_with_alias(persona_name, persona_dir)
    persona = load(path)
    engine = persona.fm.engine or SYSTEM_DEFAULT_ENGINE
    model = persona.fm.model or SYSTEM_DEFAULT_MODEL
    return persona, engine, model


def _make_llm_result(ok: bool = True, stdout: str = "response", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.success = ok
    r.body = stdout
    r.stderr = stderr
    return r


def test_run_pipeline_returns_chat_output(persona_dir: Path) -> None:
    """run_pipeline must return ChatOutput."""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout="test response")):
        out = run_pipeline("hello", persona, engine=engine, model=model)

    from mltgnt.interfaces.types import ChatOutput
    assert isinstance(out, ChatOutput)


def test_run_pipeline_content_has_llm_response(persona_dir: Path) -> None:
    """ChatOutput.content stores the LLM response text."""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout="test response")):
        out = run_pipeline("hello", persona, engine=engine, model=model)

    assert out.content == "test response"


def test_run_pipeline_persona_name_matches(persona_dir: Path) -> None:
    """ChatOutput.persona_name matches the given persona name."""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()):
        out = run_pipeline("test", persona, engine=engine, model=model)

    assert out.persona_name == "persona-a"


def test_run_pipeline_timestamp_is_asia_tokyo(persona_dir: Path) -> None:
    """ChatOutput.timestamp is a datetime in Asia/Tokyo."""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()):
        out = run_pipeline("test", persona, engine=engine, model=model)

    assert isinstance(out.timestamp, datetime)
    assert out.timestamp.tzinfo is not None
    assert out.timestamp.utcoffset().total_seconds() == 9 * 3600


def test_run_pipeline_memory_prepended(persona_dir: Path) -> None:
    """When memory is not None, it is prepended to the prompt."""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()) as mock_call:
        run_pipeline("test", persona, engine=engine, model=model, memory="memory content")

    called_prompt: str = mock_call.call_args[0][0]
    assert "memory content\n\n" in called_prompt


def test_run_pipeline_ok_false_returns_error_content(persona_dir: Path) -> None:
    """When LLM returns ok=False, content includes the error marker."""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=False, stderr="engine error")):
        out = run_pipeline("test", persona, engine=engine, model=model)

    assert "\u30a8\u30e9\u30fc" in out.content
    assert "engine error" in out.content


def test_run_pipeline_exception_returns_error_content(persona_dir: Path) -> None:
    """When LLM raises RuntimeError, content includes failure marker and no exception escapes."""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", side_effect=RuntimeError("connection refused")):
        out = run_pipeline("test", persona, engine=engine, model=model)

    assert "\u5b9f\u884c\u5931\u6557" in out.content
    assert "connection refused" in out.content


def test_run_pipeline_rejects_audit_writer_kwarg(persona_dir: Path) -> None:
    """audit_writer \u30ad\u30fc\u30ef\u30fc\u30c9\u5f15\u6570\u306f TypeError \u306b\u306a\u308b\u3053\u3068。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=True, stdout="ok")):
        with pytest.raises(TypeError):
            run_pipeline("test", persona, engine=engine, model=model, audit_writer=MagicMock())


def test_run_pipeline_no_ghdag_import_in_pipeline() -> None:
    """chat/pipeline.py \u5185\u306b ghdag \u306e\u76f4\u63a5 import \u304c\u306a\u3044\u3053\u3068（L3→L0 \u4f9d\u5b58\u6392\u9664）。"""
    import inspect
    import mltgnt.chat.pipeline as mod

    source = inspect.getsource(mod)
    assert "from ghdag" not in source
    assert "import ghdag" not in source


def test_run_pipeline_records_with_orchestration_context(persona_dir: Path, tmp_path: Path) -> None:
    """orchestration_ctx + audit_path \u6307\u5b9a\u3067 persona_call \u304c\u8a18\u9332\u3055\u308c\u308b。"""
    from mltgnt.bridges.audit_adapter import OrchestrationContext
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "persona-a")
    audit_path = tmp_path / "audit.jsonl"
    ctx = OrchestrationContext(orchestration_id="orch-1", source="test")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=True, stdout="ok")):
        run_pipeline(
            "test",
            persona,
            engine=engine,
            model=model,
            orchestration_ctx=ctx,
            audit_path=audit_path,
        )

    record = __import__("json").loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["event_type"] == "persona_call"
    assert record["orchestration_id"] == "orch-1"


def test_bridges_llm_adapter_importable() -> None:
    """from mltgnt.bridges.llm_adapter import call_llm \u304c\u6210\u529f\u3059\u308b\u3053\u3068。"""
    from mltgnt.bridges.llm_adapter import call_llm  # noqa: F401


def test_chat_pipeline_importable_via_init() -> None:
    """from mltgnt.chat import run_pipeline \u304c\u6210\u529f\u3059\u308b\u3053\u3068。"""
    from mltgnt.chat import run_pipeline  # noqa: F401


def test_run_chat_raises_import_error() -> None:
    """v0.10.0: run_chat \u306f\u524a\u9664\u6e08\u307f — ImportError \u304c\u767a\u751f\u3059\u308b\u3053\u3068。"""
    import pytest
    with pytest.raises(ImportError):
        from mltgnt.chat.pipeline import run_chat  # noqa: F401


def test_run_chat_not_importable_via_init() -> None:
    """v0.10.0: from mltgnt.chat import run_chat \u306f ImportError \u306b\u306a\u308b\u3053\u3068。"""
    import pytest
    with pytest.raises(ImportError):
        from mltgnt.chat import run_chat  # noqa: F401


# ---------------------------------------------------------------------------
# codex JSONL leak regression test
# ---------------------------------------------------------------------------

_CODEX_JSONL = (
    '{"type":"thread.started","thread_id":"01a04d63-3798-7280-8248-cfe66b80d1c4"}\n'
    '{"type":"turn.started"}\n'
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
    '"text":"\u3084\u3063\u307b\u30fc。\u65b0\u3057\u3044\u9774、\u3044\u3044\u306d\u30fc。"}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":36377,"output_tokens":95}}'
)


def test_run_pipeline_does_not_leak_codex_jsonl(persona_dir: Path) -> None:
    """codex \u306f EngineSpec \u304c\u5e38\u306b --json \u3092\u4ed8\u3051\u308b\u305f\u3081 raw stdout \u306f JSONL \u306b\u306a\u308b。

    llm_adapter \u304c call_text \u7d4c\u7531（engine output adapter \u9069\u7528）\u3067\u3042\u308b\u3053\u3068\u3092\u62c5\u4fdd\u3057、
    JSONL \u304c ChatOutput.content \u306b\u305d\u306e\u307e\u307e\u51fa\u306a\u3044\u3053\u3068\u3092\u56fa\u5b9a\u3059\u308b。
    """
    from ghdag.llm.engines import LLMResult, TextResult
    from ghdag.llm.adapters import get_output_adapter
    from mltgnt.chat import run_pipeline

    persona, _engine, model = _load_persona(persona_dir, "persona-a")

    raw = LLMResult(stdout=_CODEX_JSONL, stderr="", returncode=0)
    body = get_output_adapter("codex").extract_result_text(
        raw.stdout.encode("utf-8"), b""
    ).decode("utf-8")
    text_result = TextResult(body=body, success=True, raw=raw)

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=text_result):
        out = run_pipeline("hello", persona, engine="codex", model=model)

    assert out.content == "\u3084\u3063\u307b\u30fc。\u65b0\u3057\u3044\u9774、\u3044\u3044\u306d\u30fc。"
    assert "thread.started" not in out.content
    assert "turn.completed" not in out.content


def test_llm_adapter_delegates_to_call_text() -> None:
    """llm_adapter.call_llm \u304c ghdag.llm.call \u3067\u306f\u306a\u304f call_text \u3092\u547c\u3076\u3053\u3068。"""
    from mltgnt.bridges import llm_adapter

    with patch("ghdag.llm.call_text") as mock_call_text:
        mock_call_text.return_value = "sentinel"
        assert llm_adapter.call_llm("p", engine="codex", model="m", timeout=5) == "sentinel"

    mock_call_text.assert_called_once_with("p", engine="codex", model="m", timeout=5)
