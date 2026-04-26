"""
Tests for SkillRegistry and SkillWatcherComponent.
"""
from __future__ import annotations

import textwrap
import time
import threading
from pathlib import Path

import pytest

from mltgnt.skill._registry import SkillRegistry
from mltgnt.daemon._skill_watcher import SkillWatcherComponent


def _write_skill(base: Path, skill_name: str, description: str = "A test skill") -> Path:
    skill_dir = base / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(textwrap.dedent(f"""
        ---
        name: {skill_name}
        description: {description}
        ---
        # {skill_name}
        本文。
    """).lstrip(), encoding="utf-8")
    return skill_file


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class TestSkillRegistry:
    def test_initial_empty(self, tmp_path):
        registry = SkillRegistry(paths=[tmp_path])
        assert registry.get() == {}

    def test_reload_discovers_skills(self, tmp_path):
        _write_skill(tmp_path, "skill-a")
        _write_skill(tmp_path, "skill-b")
        registry = SkillRegistry(paths=[tmp_path])
        result = registry.reload()
        assert "skill-a" in result
        assert "skill-b" in result

    def test_get_returns_copy(self, tmp_path):
        _write_skill(tmp_path, "skill-a")
        registry = SkillRegistry(paths=[tmp_path])
        registry.reload()
        d1 = registry.get()
        d1["injected"] = None  # type: ignore
        assert "injected" not in registry.get()

    def test_reload_reflects_new_skill(self, tmp_path):
        _write_skill(tmp_path, "skill-a")
        registry = SkillRegistry(paths=[tmp_path])
        registry.reload()
        assert "skill-b" not in registry.get()

        _write_skill(tmp_path, "skill-b")
        registry.reload()
        assert "skill-b" in registry.get()

    def test_thread_safe_concurrent_access(self, tmp_path):
        _write_skill(tmp_path, "skill-a")
        registry = SkillRegistry(paths=[tmp_path])
        registry.reload()
        errors = []

        def reader():
            for _ in range(50):
                try:
                    registry.get()
                except Exception as e:
                    errors.append(e)

        def reloader():
            for _ in range(10):
                try:
                    registry.reload()
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads += [threading.Thread(target=reloader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []


# ---------------------------------------------------------------------------
# SkillWatcherComponent
# ---------------------------------------------------------------------------

class TestSkillWatcherComponent:
    def test_name(self, tmp_path):
        registry = SkillRegistry(paths=[tmp_path])
        watcher = SkillWatcherComponent(registry=registry, interval=1.0)
        assert watcher.name == "skill_watcher"

    def test_detects_new_file(self, tmp_path):
        _write_skill(tmp_path, "skill-a")
        registry = SkillRegistry(paths=[tmp_path])
        registry.reload()

        watcher = SkillWatcherComponent(registry=registry, interval=0.1)
        watcher.start()

        time.sleep(0.05)
        _write_skill(tmp_path, "skill-new")
        time.sleep(0.5)  # 少なくとも 1 ポーリングサイクル待つ

        watcher.stop()
        assert "skill-new" in registry.get()

    def test_detects_modified_file(self, tmp_path):
        skill_file = _write_skill(tmp_path, "skill-a", "original")
        registry = SkillRegistry(paths=[tmp_path])
        registry.reload()
        assert registry.get()["skill-a"].description == "original"

        watcher = SkillWatcherComponent(registry=registry, interval=0.1)
        watcher.start()

        time.sleep(0.05)
        skill_file.write_text(skill_file.read_text().replace("original", "updated"), encoding="utf-8")
        time.sleep(0.5)

        watcher.stop()
        assert registry.get()["skill-a"].description == "updated"

    def test_stop_is_idempotent(self, tmp_path):
        registry = SkillRegistry(paths=[tmp_path])
        watcher = SkillWatcherComponent(registry=registry, interval=0.1)
        watcher.start()
        watcher.stop()
        watcher.stop()  # 二度呼んでもエラーにならない
