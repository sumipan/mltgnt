"""
tests/test_mltgnt_config.py — mltgnt.config のユニットテスト（AC-2）

設計: Issue #118 §7 AC-2
"""
from __future__ import annotations

from pathlib import Path


def test_memory_config_instantiation():
    """MemoryConfig が diary 固有定数なしでインスタンス化できる。"""
    from mltgnt.config import MemoryConfig
    config = MemoryConfig(chat_dir=Path("/tmp/chat"), chat_memory_dir=Path("/tmp/chat/memory"))
    assert config.chat_dir == Path("/tmp/chat")
    assert config.chat_memory_dir == Path("/tmp/chat/memory")
    assert config.inject_max_bytes == 10_240
    assert config.inject_max_entries == 12
    assert config.preferences_max_bytes == 5_120
    assert config.lock_timeout_sec == 30.0


def test_scheduler_config_instantiation():
    """SchedulerConfig が diary 固有定数なしでインスタンス化できる。"""
    from mltgnt.config import SchedulerConfig
    config = SchedulerConfig(schedule_yaml=Path("/tmp/schedule.yaml"), state_dir=Path("/tmp/state"))
    assert config.schedule_yaml == Path("/tmp/schedule.yaml")
    assert config.state_dir == Path("/tmp/state")
    assert config.timezone == "Asia/Tokyo"
    assert config.salt == ""


def test_memory_config_frozen():
    """MemoryConfig は frozen=True で immutable。"""
    from mltgnt.config import MemoryConfig
    import pytest
    config = MemoryConfig(chat_dir=Path("/tmp/chat"), chat_memory_dir=Path("/tmp/chat/memory"))
    with pytest.raises((AttributeError, TypeError)):
        config.inject_max_bytes = 999  # type: ignore[misc]


def test_scheduler_config_custom_values():
    """SchedulerConfig のカスタム値が正しく設定される。"""
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
    """mltgnt.config モジュール内に diary 固有定数が存在しない。"""
    import mltgnt.config as cfg_module
    assert not hasattr(cfg_module, "REPO_ROOT")
    assert not hasattr(cfg_module, "DIARY_DIR")
    assert not hasattr(cfg_module, "PERSONA_NAME")


def test_import_without_tools_dependency():
    """from mltgnt.config import が tools/ への依存なしに成功する。"""
    from mltgnt.config import MemoryConfig, SchedulerConfig
    # インスタンス化できればOK
    mc = MemoryConfig(chat_dir=Path("/a"), chat_memory_dir=Path("/b"))
    sc = SchedulerConfig(schedule_yaml=Path("/c"), state_dir=Path("/d"))
    assert mc is not None
    assert sc is not None
