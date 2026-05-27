"""v0.10.0 破壊的変更の受け入れ条件テスト（Issue #1251）。"""
from __future__ import annotations

import pytest

from mltgnt.agent._parse import _parse_json_response
from mltgnt.persona.loader import Persona
from mltgnt.persona.schema import parse_fm, validate_fm


def test_parse_requires_args_key() -> None:
    assert _parse_json_response('{"tool":"x","a":1}') is None
    assert _parse_json_response('{"tool":"x","args":{"a":1}}') == {
        "tool": "x",
        "args": {"a": 1},
    }


def test_flat_fm_keys_rejected() -> None:
    fm = parse_fm(
        {"persona": {"name": "t"}, "chat_model": "x", "slack": {}},
        file_stem="t",
    )
    result = validate_fm(fm)
    assert not result.ok


def test_ops_chat_model_rejected() -> None:
    fm = parse_fm(
        {"persona": {"name": "t"}, "ops": {"chat_model": "gpt-4"}},
        file_stem="t",
    )
    result = validate_fm(fm)
    assert not result.ok


def test_persona_compat_attrs_removed() -> None:
    for attr in ("WEIGHT_MAP", "ops_config", "slack_post_kwargs", "delegate_ack"):
        assert not hasattr(Persona, attr)


def test_removed_imports() -> None:
    with pytest.raises(ImportError):
        from mltgnt.chat.models import ChatInput  # noqa: F401
    with pytest.raises(ImportError):
        from mltgnt.chat.pipeline import run_chat  # noqa: F401
    with pytest.raises(ImportError):
        from mltgnt.memory import read_memory_agentic  # noqa: F401
    with pytest.raises(ImportError):
        from mltgnt.memory._compaction import compact  # noqa: F401
    with pytest.raises(ImportError):
        from mltgnt.memory import normalize_source_prefix  # noqa: F401
    with pytest.raises(ImportError):
        from mltgnt.scheduler.ghdag_bridge import enqueue_and_wait  # noqa: F401


def test_canonical_imports() -> None:
    from mltgnt.interfaces.types import ChatInput, ChatOutput, Message  # noqa: F401
    from mltgnt.chat import ChatInput as CI, run_pipeline  # noqa: F401
    from mltgnt.memory import read_memory_iterative  # noqa: F401
    from mltgnt.memory.compaction import compact, CompactionResult  # noqa: F401
    from mltgnt.bridges.ghdag_bridge import enqueue_and_wait  # noqa: F401

    assert CI is ChatInput
    assert callable(run_pipeline)
