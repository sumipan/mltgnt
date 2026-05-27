"""DeprecationWarning 発火の統合テスト（Issue #1250）。"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.agent._parse import _parse_json_response
from mltgnt.config import MemoryConfig
from mltgnt.memory import memory_file_path, normalize_source_prefix, read_memory_agentic
from mltgnt.persona import validate_persona
from mltgnt.persona.loader import Persona
from mltgnt.persona.schema import PersonaFM, parse_fm

_PERSONA_MD = textwrap.dedent("""\
    ---
    persona:
      name: タチコマ
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## 基本情報

    テスト用ペルソナ。
""")


def test_run_chat_deprecation(tmp_path: Path) -> None:
    """#2: run_chat() は DeprecationWarning を発する。"""
    from mltgnt.chat.pipeline import run_chat

    persona_dir = tmp_path / "agents"
    persona_dir.mkdir()
    (persona_dir / "タチコマ.md").write_text(_PERSONA_MD, encoding="utf-8")

    with patch("mltgnt.bridges.llm_adapter.call_llm") as mock_llm:
        mock_llm.return_value = MagicMock(ok=True, stdout="ok", stderr="")
        with pytest.warns(DeprecationWarning, match="run_chat\\(\\) is deprecated"):
            run_chat("hi", "タチコマ", persona_dir)


def test_read_memory_agentic_deprecation(tmp_path: Path) -> None:
    """#3: read_memory_agentic は DeprecationWarning を発する。"""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    config = MemoryConfig(chat_dir=tmp_path, chat_memory_dir=mem_dir)
    memory_file_path(config, "persona").write_text("", encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="read_memory_agentic is deprecated"):
        read_memory_agentic(
            config,
            "persona",
            "query",
            max_bytes=1000,
            max_entries=5,
            llm_call=lambda p: "SUFFICIENT",
        )


def test_flat_chat_model_deprecation() -> None:
    """#4: トップレベル chat_model で DeprecationWarning が出る。"""
    with pytest.warns(DeprecationWarning, match="chat_model"):
        parse_fm({"chat_model": "gpt-4", "persona": {"name": "t"}}, file_stem="t")


def test_flat_slack_deprecation() -> None:
    """#4: トップレベル slack で DeprecationWarning が出る。"""
    with pytest.warns(DeprecationWarning, match="slack"):
        parse_fm({"slack": {"username": "bot"}, "persona": {"name": "t"}}, file_stem="t")


def test_ops_chat_model_deprecation() -> None:
    """#4: ops.chat_model で DeprecationWarning が出る。"""
    with pytest.warns(DeprecationWarning, match="ops.chat_model"):
        parse_fm(
            {"persona": {"name": "t"}, "ops": {"chat_model": "gpt-4"}},
            file_stem="t",
        )


def test_chat_models_reexport_deprecation() -> None:
    """#5: mltgnt.chat.models からの import で DeprecationWarning。"""
    with pytest.warns(DeprecationWarning, match="mltgnt.chat.models.ChatInput"):
        from mltgnt.chat.models import ChatInput  # noqa: F401

    assert ChatInput is not None


def test_scheduler_ghdag_bridge_deprecation() -> None:
    """#6: scheduler.ghdag_bridge からの import で DeprecationWarning。"""
    with pytest.warns(DeprecationWarning, match="mltgnt.scheduler.ghdag_bridge.enqueue_dag"):
        from mltgnt.scheduler.ghdag_bridge import enqueue_dag  # noqa: F401

    assert callable(enqueue_dag)


def test_agent_parse_flat_json_deprecation() -> None:
    """#7: フラット JSON で DeprecationWarning。"""
    with pytest.warns(DeprecationWarning, match="フラット JSON"):
        result = _parse_json_response('{"tool": "search", "query": "x"}')
    assert result == {"tool": "search", "args": {"query": "x"}}


def test_memory_compaction_shim_deprecation() -> None:
    """#8: memory._compaction からの import で DeprecationWarning。"""
    with pytest.warns(DeprecationWarning, match="mltgnt.memory._compaction.compact"):
        from mltgnt.memory._compaction import compact  # noqa: F401

    assert callable(compact)


def test_normalize_source_prefix_deprecation() -> None:
    """#9: normalize_source_prefix([file-chat]) で DeprecationWarning。"""
    with pytest.warns(DeprecationWarning, match="normalize_source_prefix"):
        out = normalize_source_prefix("[file-chat]\nbody")
    assert out.startswith("[file]")


def test_persona_ops_config_deprecation() -> None:
    """#11: Persona.ops_config で DeprecationWarning。"""
    persona = Persona(
        name="t",
        fm=PersonaFM(name="t"),
        sections={},
        body="",
        path=Path("t.md"),
    )
    with pytest.warns(DeprecationWarning, match="Persona.ops_config"):
        _ = persona.ops_config


def test_persona_slack_post_kwargs_deprecation() -> None:
    """#11: Persona.slack_post_kwargs() で DeprecationWarning。"""
    persona = Persona(
        name="t",
        fm=PersonaFM(name="t", slack_username="bot"),
        sections={},
        body="",
        path=Path("t.md"),
    )
    with pytest.warns(DeprecationWarning, match="slack_post_kwargs"):
        persona.slack_post_kwargs()


def test_persona_delegate_ack_deprecation() -> None:
    """#11: Persona.delegate_ack() で DeprecationWarning。"""
    persona = Persona(
        name="t",
        fm=PersonaFM(name="t", slack_delegate_ack="ok"),
        sections={},
        body="",
        path=Path("t.md"),
    )
    with pytest.warns(DeprecationWarning, match="delegate_ack"):
        assert persona.delegate_ack() == "ok"


def test_validate_persona_legacy_keys_deprecation() -> None:
    """#12: validate_persona で legacy_keys 非空時に DeprecationWarning。"""
    persona = Persona(
        name="t",
        fm=PersonaFM(name="t", legacy_keys=["chat_model"]),
        sections={},
        body="",
        path=Path("t.md"),
    )
    with pytest.warns(DeprecationWarning, match="旧形式の FM キー"):
        msgs = validate_persona(persona)
    assert any("旧形式" in m for m in msgs)


def test_mltgnt_all_excludes_deprecated() -> None:
    """__all__ に run_chat / read_memory_agentic が含まれない。"""
    import mltgnt

    assert "run_chat" not in mltgnt.__all__
    assert "read_memory_agentic" not in mltgnt.__all__
    assert "run_pipeline" in mltgnt.__all__
