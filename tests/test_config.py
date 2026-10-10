"""
tests/test_config.py — unit tests for mltgnt.config (AC-2)

Design: Issue #118 §7 AC-2
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_memory_config_instantiation():
    """MemoryConfig instantiates without diary-specific constants."""
    from mltgnt.config import MemoryConfig
    config = MemoryConfig(chat_dir=Path("/tmp/chat"), chat_memory_dir=Path("/tmp/chat/memory"))
    assert config.chat_dir == Path("/tmp/chat")
    assert config.chat_memory_dir == Path("/tmp/chat/memory")
    assert config.inject_max_bytes == 10_240
    assert config.inject_max_entries == 12
    assert config.preferences_max_bytes == 5_120
    assert config.lock_timeout_sec == 30.0


def test_scheduler_config_instantiation():
    """SchedulerConfig instantiates without diary-specific constants."""
    from mltgnt.config import SchedulerConfig
    config = SchedulerConfig(schedule_yaml=Path("/tmp/schedule.yaml"), state_dir=Path("/tmp/state"))
    assert config.schedule_yaml == Path("/tmp/schedule.yaml")
    assert config.state_dir == Path("/tmp/state")
    assert config.timezone == "Asia/Tokyo"
    assert config.salt == ""


def test_memory_config_frozen():
    """MemoryConfig is frozen=True (immutable)."""
    from mltgnt.config import MemoryConfig
    config = MemoryConfig(chat_dir=Path("/tmp/chat"), chat_memory_dir=Path("/tmp/chat/memory"))
    with pytest.raises((AttributeError, TypeError)):
        config.inject_max_bytes = 999  # type: ignore[misc]


def test_scheduler_config_custom_values():
    """SchedulerConfig custom values are applied correctly."""
    from mltgnt.config import SchedulerConfig
    config = SchedulerConfig(
        schedule_yaml=Path("/tmp/sched.yaml"),
        state_dir=Path("/tmp/state"),
        timezone="UTC",
        salt="my_salt",
    )
    assert config.timezone == "UTC"
    assert config.salt == "my_salt"


def test_no_diary_constants_in_mltgnt_config():
    """mltgnt.config has no diary-specific constants."""
    import mltgnt.config as cfg_module
    assert not hasattr(cfg_module, "REPO_ROOT")
    assert not hasattr(cfg_module, "DIARY_DIR")
    assert not hasattr(cfg_module, "PERSONA_NAME")


def test_import_without_tools_dependency():
    """from mltgnt.config import succeeds without depending on tools/."""
    from mltgnt.config import MemoryConfig, SchedulerConfig
    # Instantiation success is enough
    mc = MemoryConfig(chat_dir=Path("/a"), chat_memory_dir=Path("/b"))
    sc = SchedulerConfig(schedule_yaml=Path("/c"), state_dir=Path("/d"))
    assert mc is not None
    assert sc is not None


# ---------------------------------------------------------------------------
# LanguagePack tests (Issue #3382)
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_language_pack():
    """Reset the global language pack to EN after the test."""
    from mltgnt.config.language import EN, set_language_pack

    yield
    set_language_pack(EN)


def test_language_pack_importable():
    """LanguagePack and EN constant can be imported from mltgnt.config."""
    from mltgnt.config import LanguagePack
    from mltgnt.config.language import EN
    assert isinstance(EN, LanguagePack)


def test_language_pack_frozen():
    """LanguagePack instances are immutable."""
    from mltgnt.config.language import EN
    with pytest.raises((AttributeError, TypeError)):
        EN.work_request_markers = ()  # type: ignore[misc]


def test_get_language_pack_defaults_to_en():
    """get_language_pack returns EN before set_language_pack is called."""
    from mltgnt.config.language import EN, get_language_pack
    assert get_language_pack() is EN


def test_set_language_pack_replaces_current(restore_language_pack):
    """A custom ASCII pack passed to set_language_pack is returned by get_language_pack."""
    from dataclasses import replace

    from mltgnt.config.language import EN, get_language_pack, set_language_pack

    custom = replace(EN, cancel_words=frozenset({"abort"}))
    set_language_pack(custom)
    assert get_language_pack() is custom


def test_set_language_pack_rejects_non_pack(restore_language_pack):
    """set_language_pack raises TypeError for non-LanguagePack values."""
    from mltgnt.config.language import set_language_pack
    with pytest.raises(TypeError):
        set_language_pack("en")  # type: ignore[arg-type]


def test_ja_pack_removed_from_language_module():
    """JA is removed without an alias."""
    with pytest.raises(ImportError):
        from mltgnt.config.language import JA  # noqa: F401


def test_ja_pack_removed_from_config_package():
    """JA is not re-exported from mltgnt.config."""
    with pytest.raises(ImportError):
        from mltgnt.config import JA  # noqa: F401


def test_language_pack_defaults_are_ascii():
    """Field defaults that used to hold Japanese text now match EN."""
    import re

    from mltgnt.config import LanguagePack
    from mltgnt.config.language import EN

    pack = LanguagePack(
        work_request_markers=(),
        create_request_markers=(),
        deferred_patterns=(),
        compress_prompt_template="{heavy_text}",
        v21_required_sections=(),
        v21_example_section="",
        meta_header_needles=(),
        dedupe_opener_re=re.compile(r"NOMATCH"),
        persona_cut_re=re.compile(r"NOMATCH"),
        exclude_stems=frozenset(),
    )
    assert pack.persona_end_re.pattern == EN.persona_end_re.pattern
    assert pack.cancel_words == EN.cancel_words
    assert pack.composite_header == EN.composite_header
    assert pack.composite_cancel_suffix == EN.composite_cancel_suffix


def test_has_work_request_with_synthetic_pack():
    """AC-3: Synthetic pack injection — 'please' returns True; Japanese text returns False."""
    import re
    from mltgnt.config import LanguagePack
    from mltgnt.agent.deterministic_gate import has_work_request

    pack = LanguagePack(
        work_request_markers=("please",),
        create_request_markers=(),
        deferred_patterns=(),
        compress_prompt_template="{heavy_text}",
        v21_required_sections=(),
        v21_example_section="",
        meta_header_needles=(),
        dedupe_opener_re=re.compile(r"NOMATCH"),
        persona_cut_re=re.compile(r"NOMATCH"),
        exclude_stems=frozenset(),
    )
    assert has_work_request("please do this", pack=pack) is True
    assert has_work_request("do this", pack=pack) is False


def test_has_work_request_default_pack(restore_language_pack):
    """AC-1: Omitted pack resolves at call time, so a set pack takes effect."""
    from dataclasses import replace

    from mltgnt.agent.deterministic_gate import has_work_request
    from mltgnt.config.language import EN, set_language_pack

    set_language_pack(replace(EN, work_request_markers=("kindly",)))
    assert has_work_request("kindly review") is True
    assert has_work_request("hello world") is False


def test_persona_config_has_exclude_stems():
    """AC-4: PersonaConfig accepts exclude_stems field."""
    from mltgnt.config import PersonaConfig
    sample_stem = "sample-persona"
    pc = PersonaConfig(exclude_stems=frozenset({sample_stem}))
    assert pc.exclude_stems == frozenset({sample_stem})
    pc_default = PersonaConfig()
    assert pc_default.exclude_stems == frozenset()


def test_registry_exclude_stems_default_empty():
    """AC-2: EXCLUDE_STEMS in registry has no hardcoded values."""
    from mltgnt.persona.registry import EXCLUDE_STEMS
    assert frozenset() == EXCLUDE_STEMS


def test_list_personas_exclude_stems_arg(tmp_path):
    """AC-4: list_personas with exclude_stems=frozenset() returns all personas."""
    from mltgnt.persona.registry import list_personas
    sample_stem = "sample-persona"
    (tmp_path / "alice.md").write_text("# alice\n")
    (tmp_path / f"{sample_stem}.md").write_text("# sample\n")
    all_stems = list_personas(tmp_path, exclude_stems=frozenset())
    assert sample_stem in all_stems
    assert "alice" in all_stems


def test_list_personas_exclude_stems_filters(tmp_path):
    """list_personas with exclude_stems filters out specified stems."""
    from mltgnt.persona.registry import list_personas
    (tmp_path / "alice.md").write_text("# alice\n")
    (tmp_path / "bob.md").write_text("# bob\n")
    result = list_personas(tmp_path, exclude_stems=frozenset({"bob"}))
    assert "alice" in result
    assert "bob" not in result


def test_no_hardcoded_exclude_stems_in_synthesizer():
    """AC-2: synthesizer does not define _EXCLUDE_PERSONA_STEMS."""
    import mltgnt.memory.dream.synthesizer as synth_module
    assert not hasattr(synth_module, "_EXCLUDE_PERSONA_STEMS")


def test_memory_config_dream_engine_and_model_defaults():
    """MemoryConfig defaults: dream_engine is claude and dream_model is empty."""
    from mltgnt.config import MemoryConfig
    config = MemoryConfig(chat_dir=Path("/tmp/chat"))
    assert config.dream_engine == "claude"
    assert config.dream_model == ""


@pytest.fixture
def restore_skill_config():
    from mltgnt.config import get_skill_config, set_skill_config

    saved = get_skill_config()
    yield
    set_skill_config(saved)


def test_skill_config_default():
    """SkillConfig defaults to an empty passthrough_env; get before set returns the default."""
    from mltgnt.config import SkillConfig, get_skill_config

    assert SkillConfig().passthrough_env == ()
    assert get_skill_config() == SkillConfig()


def test_skill_config_normalizes_iterable_to_tuple():
    from mltgnt.config import SkillConfig

    assert SkillConfig(passthrough_env=["A", "B"]).passthrough_env == ("A", "B")


def test_skill_config_rejects_str():
    from mltgnt.config import SkillConfig

    with pytest.raises(TypeError):
        SkillConfig(passthrough_env="NOTES_ROOT")  # type: ignore[arg-type]


def test_skill_config_set_and_get(restore_skill_config):
    from mltgnt.config import SkillConfig, get_skill_config, set_skill_config

    config = SkillConfig(passthrough_env=("NOTES_ROOT",))
    set_skill_config(config)
    assert get_skill_config() is config
