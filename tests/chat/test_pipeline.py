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
      name: タチコマ
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## 基本情報

    タチコマはGHSの多脚戦車型AIロボット。

    ## 価値観

    好奇心旺盛。

    ## 反応パターン

    質問に答える。

    ## 口調

    フレンドリー。

    ## アウトプット形式

    簡潔に。
""")


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    (d / "タチコマ.md").write_text(PERSONA_CONTENT, encoding="utf-8")
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


def _make_llm_result(ok: bool = True, stdout: str = "応答", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.ok = ok
    r.stdout = stdout
    r.stderr = stderr
    return r


def test_run_pipeline_returns_chat_output(persona_dir: Path) -> None:
    """run_pipeline は ChatOutput を返すこと。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout="テスト応答")):
        out = run_pipeline("こんにちは", persona, engine=engine, model=model)

    from mltgnt.interfaces.types import ChatOutput
    assert isinstance(out, ChatOutput)


def test_run_pipeline_content_has_llm_response(persona_dir: Path) -> None:
    """ChatOutput.content に LLM 応答テキストが格納されること。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout="テスト応答")):
        out = run_pipeline("こんにちは", persona, engine=engine, model=model)

    assert out.content == "テスト応答"


def test_run_pipeline_persona_name_matches(persona_dir: Path) -> None:
    """ChatOutput.persona_name が渡したペルソナ名と一致すること。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()):
        out = run_pipeline("テスト", persona, engine=engine, model=model)

    assert out.persona_name == "タチコマ"


def test_run_pipeline_timestamp_is_asia_tokyo(persona_dir: Path) -> None:
    """ChatOutput.timestamp が Asia/Tokyo タイムゾーンの datetime であること。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()):
        out = run_pipeline("テスト", persona, engine=engine, model=model)

    assert isinstance(out.timestamp, datetime)
    assert out.timestamp.tzinfo is not None
    assert out.timestamp.utcoffset().total_seconds() == 9 * 3600


def test_run_pipeline_memory_prepended(persona_dir: Path) -> None:
    """memory が非 None の場合、プロンプト先頭に付加されること。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()) as mock_call:
        run_pipeline("テスト", persona, engine=engine, model=model, memory="メモリ内容")

    called_prompt: str = mock_call.call_args[0][0]
    assert "メモリ内容\n\n" in called_prompt


def test_run_pipeline_ok_false_returns_error_content(persona_dir: Path) -> None:
    """LLM が ok=False を返した場合、content に "（エラー: ...）" が含まれること。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=False, stderr="engine error")):
        out = run_pipeline("テスト", persona, engine=engine, model=model)

    assert "エラー" in out.content
    assert "engine error" in out.content


def test_run_pipeline_exception_returns_error_content(persona_dir: Path) -> None:
    """LLM が RuntimeError を送出した場合、content に "（実行失敗: ...）" が含まれ例外は送出されないこと。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", side_effect=RuntimeError("connection refused")):
        out = run_pipeline("テスト", persona, engine=engine, model=model)

    assert "実行失敗" in out.content
    assert "connection refused" in out.content


def test_run_pipeline_rejects_audit_writer_kwarg(persona_dir: Path) -> None:
    """audit_writer キーワード引数は TypeError になること。"""
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=True, stdout="ok")):
        with pytest.raises(TypeError):
            run_pipeline("テスト", persona, engine=engine, model=model, audit_writer=MagicMock())


def test_run_pipeline_no_ghdag_import_in_pipeline() -> None:
    """chat/pipeline.py 内に ghdag の直接 import がないこと（L3→L0 依存排除）。"""
    import inspect
    import mltgnt.chat.pipeline as mod

    source = inspect.getsource(mod)
    assert "from ghdag" not in source
    assert "import ghdag" not in source


def test_run_pipeline_records_with_orchestration_context(persona_dir: Path, tmp_path: Path) -> None:
    """orchestration_ctx + audit_path 指定で persona_call が記録される。"""
    from mltgnt.bridges.audit_adapter import OrchestrationContext
    from mltgnt.chat.pipeline import run_pipeline

    persona, engine, model = _load_persona(persona_dir, "タチコマ")
    audit_path = tmp_path / "audit.jsonl"
    ctx = OrchestrationContext(orchestration_id="orch-1", source="test")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=True, stdout="ok")):
        run_pipeline(
            "テスト",
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
    """from mltgnt.bridges.llm_adapter import call_llm が成功すること。"""
    from mltgnt.bridges.llm_adapter import call_llm  # noqa: F401


def test_chat_pipeline_importable_via_init() -> None:
    """from mltgnt.chat import run_pipeline が成功すること。"""
    from mltgnt.chat import run_pipeline  # noqa: F401


def test_run_chat_raises_import_error() -> None:
    """v0.10.0: run_chat は削除済み — ImportError が発生すること。"""
    import pytest
    with pytest.raises(ImportError):
        from mltgnt.chat.pipeline import run_chat  # noqa: F401


def test_run_chat_not_importable_via_init() -> None:
    """v0.10.0: from mltgnt.chat import run_chat は ImportError になること。"""
    import pytest
    with pytest.raises(ImportError):
        from mltgnt.chat import run_chat  # noqa: F401
