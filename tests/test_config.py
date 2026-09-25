"""
tests/test_config.py — unit tests for mltgnt.config (AC-2)

Design: Issue #118 §7 AC-2
"""
from __future__ import annotations

from pathlib import Path


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
    import pytest
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


def test_language_pack_importable():
    """LanguagePack and JA constant can be imported from mltgnt.config."""
    from mltgnt.config import LanguagePack
    from mltgnt.config.language import JA
    assert isinstance(JA, LanguagePack)


def test_language_pack_frozen():
    """LanguagePack instances are immutable."""
    import pytest
    from mltgnt.config.language import JA
    with pytest.raises((AttributeError, TypeError)):
        JA.work_request_markers = ()  # type: ignore[misc]


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


def test_has_work_request_default_ja():
    """AC-1: Default JA pack works for Japanese request phrasing."""
    from mltgnt.agent.deterministic_gate import has_work_request
    from mltgnt.config.language import JA

    assert has_work_request(JA.work_request_markers[0]) is True
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
