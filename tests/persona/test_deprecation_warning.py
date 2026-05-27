"""v0.10.0: 旧 FM キーは validate_fm で unknown key エラーになる。"""

from mltgnt.persona.schema import parse_fm, validate_fm


def test_flat_chat_model_is_unknown_key():
    fm = parse_fm({"chat_model": "gpt-4", "persona": {"name": "t"}}, file_stem="t")
    result = validate_fm(fm)
    assert not result.ok
    assert any("chat_model" in err for err in result.errors)


def test_flat_slack_is_unknown_key():
    fm = parse_fm({"slack": {"username": "bot"}, "persona": {"name": "t"}}, file_stem="t")
    result = validate_fm(fm)
    assert not result.ok
    assert any("slack" in err for err in result.errors)


def test_ops_chat_model_is_unknown_key():
    fm = parse_fm(
        {"persona": {"name": "t"}, "ops": {"chat_model": "gpt-4"}},
        file_stem="t",
    )
    result = validate_fm(fm)
    assert not result.ok
    assert any("ops.chat_model" in err for err in result.errors)


def test_clean_fm_validates():
    fm = parse_fm(
        {"persona": {"name": "t"}, "ops": {"engine": "claude", "model": "opus"}},
        file_stem="t",
    )
    result = validate_fm(fm)
    assert result.ok
