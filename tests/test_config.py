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
