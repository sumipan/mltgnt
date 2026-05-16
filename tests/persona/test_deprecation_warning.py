import warnings
import pytest
from mltgnt.persona.schema import parse_fm


def test_flat_chat_model_emits_deprecation():
    """トップレベル chat_model で DeprecationWarning が出ること。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_fm({"chat_model": "gpt-4", "persona": {"name": "t"}}, file_stem="t")
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deps) >= 1
        assert "chat_model" in str(deps[0].message)


def test_flat_slack_emits_deprecation():
    """トップレベル slack で DeprecationWarning が出ること。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_fm({"slack": {"username": "bot"}, "persona": {"name": "t"}}, file_stem="t")
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deps) >= 1
        assert "slack" in str(deps[0].message)


def test_ops_chat_model_emits_deprecation():
    """ops.chat_model で DeprecationWarning が出ること。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_fm(
            {"persona": {"name": "t"}, "ops": {"chat_model": "gpt-4"}},
            file_stem="t",
        )
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deps) >= 1
        assert "ops.chat_model" in str(deps[0].message)


def test_no_deprecation_when_clean():
    """旧キー未使用時は DeprecationWarning が出ないこと。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_fm(
            {"persona": {"name": "t"}, "ops": {"engine": "claude", "model": "opus"}},
            file_stem="t",
        )
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deps) == 0
