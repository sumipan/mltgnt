"""SkillMeta / I/O dataclass のユニットテスト（Issue #1382 U1）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.interfaces.types import ChatInput
from mltgnt.skill import (
    ArtifactSpec,
    ConsumesSpec,
    ProducesSpec,
    SkillMeta,
    SkillRunResult,
)
from mltgnt.skill.loader import _build_meta
from mltgnt.skill.models import (
    ArtifactSpec as ArtifactSpecFromModels,
    ConsumesSpec as ConsumesSpecFromModels,
    ProducesSpec as ProducesSpecFromModels,
    SideEffectsSpec,
    SkillRunResult as SkillRunResultFromModels,
)


class TestArtifactSpec:
    def test_default_role_is_primary(self) -> None:
        spec = ArtifactSpec(path="report.md")
        assert spec.path == "report.md"
        assert spec.role == "primary"

    def test_explicit_role(self) -> None:
        spec = ArtifactSpec(path="debug.log", role="log")
        assert spec.role == "log"


class TestProducesSpec:
    def test_defaults(self) -> None:
        spec = ProducesSpec()
        assert spec.content_type == "text/markdown"
        assert spec.artifacts == []
        assert spec.status_markers == []

    def test_all_fields(self) -> None:
        artifact = ArtifactSpec(path="out.txt")
        spec = ProducesSpec(
            content_type="text/plain",
            artifacts=[artifact],
            status_markers=["ACCEPTED"],
        )
        assert spec.content_type == "text/plain"
        assert spec.artifacts == [artifact]
        assert spec.status_markers == ["ACCEPTED"]


class TestConsumesSpec:
    def test_default_content_type(self) -> None:
        spec = ConsumesSpec(producer="upstream-skill")
        assert spec.producer == "upstream-skill"
        assert spec.content_type == "text/markdown"


class TestSideEffectsSpec:
    def test_defaults(self) -> None:
        spec = SideEffectsSpec()
        assert spec.writes == []
        assert spec.network == []
        assert spec.mutates == []
        assert spec.conditional == []

    def test_all_fields(self) -> None:
        spec = SideEffectsSpec(
            writes=["jobs/*.jsonl"],
            network=["api.github.com"],
            mutates=["git", "github"],
            conditional=["--force 時のみ git push"],
        )
        assert spec.writes == ["jobs/*.jsonl"]
        assert spec.network == ["api.github.com"]
        assert spec.mutates == ["git", "github"]
        assert spec.conditional == ["--force 時のみ git push"]


class TestSkillRunResult:
    def _make_chat_input(self) -> ChatInput:
        return ChatInput(
            source="test",
            session_key="session-1",
            messages=[{"role": "user", "content": "hello"}],
            model=None,
        )

    def test_defaults(self) -> None:
        ci = self._make_chat_input()
        result = SkillRunResult(chat_input=ci, expected_markers=[], skill_io="v1", content="done")
        assert result.content == "done"
        assert result.exit_code == 0
        assert result.diagnostics == []
        assert result.artifacts == []
        assert result.status_markers == []

    def test_diagnostics_defaults_to_empty_list(self) -> None:
        ci = self._make_chat_input()
        result = SkillRunResult(chat_input=ci, expected_markers=[], skill_io="v1")
        assert result.diagnostics == []


class TestSkillMetaBackwardCompat:
    def test_legacy_construction_without_new_fields(self) -> None:
        meta = SkillMeta(
            name="test",
            description="desc",
            argument_hint="",
            model=None,
            path=Path("."),
        )
        assert meta.skill_io == "legacy"
        assert meta.input_schema == {}
        assert meta.produces is None
        assert meta.consumes == []
        assert meta.side_effects is None

    def test_v1_fields(self) -> None:
        produces = ProducesSpec()
        meta = SkillMeta(
            name="test",
            description="desc",
            argument_hint="",
            model=None,
            path=Path("."),
            skill_io="v1",
            produces=produces,
        )
        assert meta.skill_io == "v1"
        assert meta.produces is produces


class TestBuildMetaSideEffects:
    def test_build_meta_parses_side_effects(self) -> None:
        path = Path("/skills/audit/SKILL.md")
        fm = {
            "name": "audit",
            "description": "desc",
            "side_effects": {"writes": ["jobs/*.jsonl"]},
        }
        meta = _build_meta(fm, path)
        assert meta.side_effects == SideEffectsSpec(
            writes=["jobs/*.jsonl"],
            network=[],
            mutates=[],
            conditional=[],
        )

    def test_build_meta_without_side_effects_key(self) -> None:
        path = Path("/skills/plain/SKILL.md")
        fm = {"name": "plain", "description": "desc"}
        meta = _build_meta(fm, path)
        assert meta.side_effects is None


class TestImportPaths:
    @pytest.mark.parametrize(
        "cls, expected",
        [
            (ArtifactSpecFromModels, ArtifactSpec),
            (ProducesSpecFromModels, ProducesSpec),
            (ConsumesSpecFromModels, ConsumesSpec),
            (SkillRunResultFromModels, SkillRunResult),
        ],
    )
    def test_models_and_package_exports_match(self, cls, expected) -> None:
        assert cls is expected
