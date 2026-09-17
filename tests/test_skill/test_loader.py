"""
tests/test_skill/test_loader.py — unit tests for loader.discover / loader.load.

Design: Issue #124 §8 AC-1, AC-2
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mltgnt.skill.loader import discover, load
from mltgnt.skill.models import SkillMeta


# --- helpers ---

def _write_skill(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


FULL_SKILL_MD = """\
---
name: review
description: Review a diary file
argument_hint: "[target-file]"
model: null
---

body here
"""

NO_NAME_SKILL_MD = """\
---
description: summary skill
---

body
"""

NO_DESCRIPTION_SKILL_MD = """\
---
name: bad
---

body
"""

INVALID_YAML_MD = """\
---
name: broken
description: [unclosed
---

body
"""

NO_FRONTMATTER_MD = "body only\n"


# --- AC-1: SkillMeta parse ---

class TestSkillMetaParse:
    def test_full_fields(self, tmp_path: Path) -> None:
        """AC-1-1: SKILL.md with all fields present"""
        p = _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([tmp_path])
        assert "review" in skills
        meta = skills["review"]
        assert meta.name == "review"
        assert meta.description == "Review a diary file"
        assert meta.argument_hint == "[target-file]"
        assert meta.model is None
        assert meta.path == p.resolve()

    def test_name_fallback_to_dir(self, tmp_path: Path) -> None:
        """AC-1-2: missing name falls back to directory name"""
        _write_skill(tmp_path, "summarize/SKILL.md", NO_NAME_SKILL_MD)
        skills = discover([tmp_path])
        assert "summarize" in skills
        assert skills["summarize"].name == "summarize"

    def test_missing_description_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """AC-1-3: missing description is skipped as a parse error"""
        _write_skill(tmp_path, "bad/SKILL.md", NO_DESCRIPTION_SKILL_MD)
        with caplog.at_level(logging.WARNING, logger="mltgnt.skill.loader"):
            skills = discover([tmp_path])
        assert skills == {}
        # Japanese text intentionally kept for CJK processing test
        assert any("Parse error" in r.message for r in caplog.records)

    def test_invalid_yaml_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """AC-1-4: invalid YAML is skipped"""
        _write_skill(tmp_path, "broken/SKILL.md", INVALID_YAML_MD)
        with caplog.at_level(logging.WARNING, logger="mltgnt.skill.loader"):
            skills = discover([tmp_path])
        assert skills == {}
        # Japanese text intentionally kept for CJK processing test
        assert any("Parse error" in r.message for r in caplog.records)

    def test_no_frontmatter_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """AC-1-5: missing frontmatter is skipped"""
        _write_skill(tmp_path, "nofront/SKILL.md", NO_FRONTMATTER_MD)
        with caplog.at_level(logging.WARNING, logger="mltgnt.skill.loader"):
            skills = discover([tmp_path])
        assert skills == {}
        # Japanese text intentionally kept for CJK processing test
        assert any("Parse error" in r.message for r in caplog.records)


# --- AC-2: discover ---

class TestDiscover:
    def test_multiple_skills(self, tmp_path: Path) -> None:
        """AC-2-1: multiple skills can be discovered"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        _write_skill(tmp_path, "summarize/SKILL.md", NO_NAME_SKILL_MD)
        skills = discover([tmp_path])
        assert set(skills.keys()) == {"review", "summarize"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        """AC-2-2: empty directory → {}"""
        skills = discover([tmp_path])
        assert skills == {}

    def test_nonexistent_path(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """AC-2-3: missing path → {} + warning"""
        with caplog.at_level(logging.WARNING, logger="mltgnt.skill.loader"):
            skills = discover([tmp_path / "nonexistent"])
        assert skills == {}
        # Japanese text intentionally kept for CJK processing test
        assert any("Path does not exist" in r.message for r in caplog.records)

    def test_duplicate_name_first_wins(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """AC-2-4: same skill name on multiple paths → first wins"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_skill(dir_a, "review/SKILL.md", FULL_SKILL_MD)
        _write_skill(dir_b, "review/SKILL.md", FULL_SKILL_MD)
        with caplog.at_level(logging.WARNING, logger="mltgnt.skill.loader"):
            skills = discover([dir_a, dir_b])
        assert len(skills) == 1
        assert skills["review"].path.parent.parent == dir_a.resolve()
        # Japanese text intentionally kept for CJK processing test
        assert any("Duplicate" in r.message or "duplicate" in r.message.lower() for r in caplog.records)

    def test_ignores_non_skill_files(self, tmp_path: Path) -> None:
        """AC-2-5: non-SKILL.md files are ignored"""
        _write_skill(tmp_path, "review/skill.yaml", "name: review\n")
        skills = discover([tmp_path])
        assert skills == {}

    def test_load_returns_skill_file(self, tmp_path: Path) -> None:
        """load() returns a SkillFile"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([tmp_path])
        sf = load(skills["review"])
        assert sf.meta.name == "review"
        assert "body here" in sf.body

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError"""
        meta = SkillMeta(
            name="ghost",
            description="ghost",
            argument_hint="",
            model=None,
            path=tmp_path / "ghost" / "SKILL.md",
        )
        with pytest.raises(FileNotFoundError):
            load(meta)


# --- Issue #1383 U2: skill_io / produces / consumes parse ---

V1_SKILL_MD = """\
---
name: v1-skill
description: v1 skill
skill_io: v1
input_schema:
  type: object
  properties:
    target:
      type: string
produces:
  content_type: text/plain
  artifacts:
    - path: output.txt
      role: log
  status_markers:
    - ACCEPTED
consumes:
  - producer: upstream-skill
    content_type: text/markdown
---

body
"""


def _write_v1_skill(tmp_path: Path) -> Path:
    """Write V1_SKILL_MD to disk."""
    return _write_skill(tmp_path, "v1-skill/SKILL.md", V1_SKILL_MD)


class TestSkillIoParse:
    def test_skill_io_v1(self, tmp_path: Path) -> None:
        """AC1: skill_io: v1 is parsed"""
        _write_v1_skill(tmp_path)
        skills = discover([tmp_path])
        meta = skills["v1-skill"]
        assert meta.skill_io == "v1"

    def test_skill_io_default_legacy(self, tmp_path: Path) -> None:
        """AC1: missing skill_io defaults to legacy"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([tmp_path])
        assert skills["review"].skill_io == "legacy"

    def test_produces_parsed(self, tmp_path: Path) -> None:
        """AC1: produces dict → ProducesSpec"""
        _write_v1_skill(tmp_path)
        meta = discover([tmp_path])["v1-skill"]
        assert meta.produces is not None
        assert meta.produces.content_type == "text/plain"
        assert len(meta.produces.artifacts) == 1
        assert meta.produces.artifacts[0].path == "output.txt"
        assert meta.produces.artifacts[0].role == "log"
        assert meta.produces.status_markers == ["ACCEPTED"]

    def test_produces_none_when_missing(self, tmp_path: Path) -> None:
        """AC1: missing produces is None"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        assert discover([tmp_path])["review"].produces is None

    def test_consumes_parsed(self, tmp_path: Path) -> None:
        """AC1: consumes list → list[ConsumesSpec]"""
        _write_v1_skill(tmp_path)
        meta = discover([tmp_path])["v1-skill"]
        assert len(meta.consumes) == 1
        assert meta.consumes[0].producer == "upstream-skill"
        assert meta.consumes[0].content_type == "text/markdown"

    def test_input_schema_parsed(self, tmp_path: Path) -> None:
        """AC1: input_schema dict is set as-is"""
        _write_v1_skill(tmp_path)
        meta = discover([tmp_path])["v1-skill"]
        assert meta.input_schema["type"] == "object"
        assert "target" in meta.input_schema["properties"]

    def test_input_schema_default_empty(self, tmp_path: Path) -> None:
        """AC1: missing input_schema is {}"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        assert discover([tmp_path])["review"].input_schema == {}


# --- Issue #1403: mltgnt-create-skill / create-persona v1 template examples ---

CREATE_SKILL_V1_TEMPLATE_MD = """\
---
name: example-skill
description: >
  Example skill (mltgnt-create-skill §3 v1 template equivalent)
argument_hint: "[target-file]"
skill_io: v1
input_schema:
  target_file:
    type: string
    description: "target file path"
produces:
  content_type: text/markdown
  status_markers: []
model: null
---

body
"""

CREATE_PERSONA_META_V1_MD = """\
---
name: mltgnt-create-persona
description: >
  Persona-generation meta skill (v1 frontmatter example)
skill_io: v1
input_schema:
  persona_name:
    type: string
    description: "name of the persona to generate"
produces:
  content_type: text/markdown
  artifacts:
    - path: "agents/example-persona.md"
      role: primary
  status_markers: []
model: null
---

body
"""


class TestCreateSkillV1TemplateParse:
    """Issue #1403 AC-3: v1 generation template examples parse correctly"""

    def test_create_skill_template_build_meta(self, tmp_path: Path) -> None:
        """Validate mltgnt-create-skill §3 v1-template SKILL.md via _build_meta"""
        from mltgnt.bridges.files_adapter import md_read
        from mltgnt.skill.loader import _build_meta

        skill_path = tmp_path / "example-skill" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(CREATE_SKILL_V1_TEMPLATE_MD, encoding="utf-8")
        md = md_read("SKILL.md", repo_root=skill_path.parent)
        meta = _build_meta(md.frontmatter, skill_path)
        assert meta.skill_io == "v1"
        assert meta.produces is not None
        assert isinstance(meta.input_schema, dict)
        assert meta.input_schema["target_file"]["type"] == "string"
        assert meta.produces.content_type == "text/markdown"
        assert meta.produces.status_markers == []

    def test_create_skill_template_discover(self, tmp_path: Path) -> None:
        """v1 template example is still obtainable after discover + lint integration"""
        _write_skill(tmp_path, "example-skill/SKILL.md", CREATE_SKILL_V1_TEMPLATE_MD)
        meta = discover([tmp_path])["example-skill"]
        assert meta.skill_io == "v1"
        assert meta.produces is not None
        assert isinstance(meta.input_schema, dict)

    def test_create_persona_meta_build_meta(self, tmp_path: Path) -> None:
        """Validate mltgnt-create-persona v1 frontmatter example via _build_meta"""
        from mltgnt.bridges.files_adapter import md_read
        from mltgnt.skill.loader import _build_meta

        skill_path = tmp_path / "mltgnt-create-persona" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(CREATE_PERSONA_META_V1_MD, encoding="utf-8")
        md = md_read("SKILL.md", repo_root=skill_path.parent)
        meta = _build_meta(md.frontmatter, skill_path)
        assert meta.skill_io == "v1"
        assert meta.produces is not None
        assert isinstance(meta.input_schema, dict)
        assert meta.input_schema["persona_name"]["description"] == "name of the persona to generate"
        assert len(meta.produces.artifacts) == 1
        assert meta.produces.artifacts[0].path == "agents/example-persona.md"
        assert meta.produces.artifacts[0].role == "primary"


# --- Issue #1383 U4: discover lint integration ---

INVALID_SKILL_IO_MD = """\
---
name: bad-io
description: bad skill_io
skill_io: v2
---

body
"""

NAME_MISMATCH_MD = """\
---
name: wrong-name
description: name mismatch
---

body
"""


class TestDiscoverLintIntegration:
    def test_v4_fail_excluded_from_discover(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC3: V4 FAIL skills are excluded from discover results"""
        _write_skill(tmp_path, "bad-io/SKILL.md", INVALID_SKILL_IO_MD)
        with caplog.at_level(logging.WARNING, logger="mltgnt.skill.loader"):
            skills = discover([tmp_path])
        assert skills == {}
        assert any("skill lint failed" in r.message for r in caplog.records)

    def test_v3_fail_excluded_from_discover(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC3: V3 FAIL skills are excluded from discover results"""
        _write_skill(tmp_path, "bad-name/SKILL.md", NAME_MISMATCH_MD)
        with caplog.at_level(logging.WARNING, logger="mltgnt.skill.loader"):
            skills = discover([tmp_path])
        assert skills == {}
        assert any("skill lint failed" in r.message for r in caplog.records)

    def test_legacy_skill_still_discovered(self, tmp_path: Path) -> None:
        """AC3: legacy skills remain included in discover"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([tmp_path])
        assert "review" in skills
        assert skills["review"].skill_io == "legacy"


# --- Issue #3030: knowledge index during discover ---

class TestDiscoverKnowledgePaths:
    def test_discover_knowledge_md(self, tmp_path: Path) -> None:
        skill = _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        knowledge = skill.parent / "knowledge.md"
        knowledge.write_text("k1\n\nk2", encoding="utf-8")
        skills = discover([tmp_path])
        assert skills["review"].knowledge_paths == [knowledge]

    def test_discover_knowledge_subdir(self, tmp_path: Path) -> None:
        skill = _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        sub = skill.parent / "knowledge"
        sub.mkdir()
        a = sub / "a.md"
        b = sub / "b.md"
        a.write_text("a", encoding="utf-8")
        b.write_text("b", encoding="utf-8")
        skills = discover([tmp_path])
        assert skills["review"].knowledge_paths == [a, b]

    def test_discover_no_knowledge(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([tmp_path])
        assert skills["review"].knowledge_paths == []

    def test_discover_both_patterns(self, tmp_path: Path) -> None:
        skill = _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        single = skill.parent / "knowledge.md"
        single.write_text("root", encoding="utf-8")
        sub = skill.parent / "knowledge"
        sub.mkdir()
        extra = sub / "extra.md"
        extra.write_text("extra", encoding="utf-8")
        skills = discover([tmp_path])
        assert skills["review"].knowledge_paths == [single, extra]



# --- Issue #3041 / #3179: discover diagnostics_dir opt-in ---

INVALID_SKILL_IO_MD_FOR_UNRESOLVED = """\
---
name: bad-io
description: bad skill_io
skill_io: v2
---

body
"""

VALID_LEGACY_AFTER_FIX = """\
---
name: bad-io
description: fixed skill
---

body
"""


class TestUnresolved:
    def test_default_diagnostics_dir_writes_nothing(self, tmp_path: Path) -> None:
        """AC-3: diagnostics_dir=None (default) writes no diagnostic JSON"""
        _write_skill(tmp_path, "bad-io/SKILL.md", INVALID_SKILL_IO_MD_FOR_UNRESOLVED)
        skills = discover([tmp_path])
        assert skills == {}
        assert not (tmp_path / "_unresolved").exists()
        assert list(tmp_path.glob("*.json")) == []

    def test_lint_fail_writes_to_diagnostics_dir(self, tmp_path: Path) -> None:
        """AC-3: with diagnostics_dir set, write <tmp>/{name}.json (no _unresolved subdir)"""
        diag_dir = tmp_path / "diag-out"
        diag_dir.mkdir()
        _write_skill(tmp_path, "bad-io/SKILL.md", INVALID_SKILL_IO_MD_FOR_UNRESOLVED)
        skills = discover([tmp_path], diagnostics_dir=diag_dir)
        assert skills == {}
        diag = diag_dir / "bad-io.json"
        assert diag.is_file()
        assert not (tmp_path / "_unresolved").exists()
        data = json.loads(diag.read_text(encoding="utf-8"))
        assert data["skill_name"] == "bad-io"
        assert data["path"] == "bad-io/SKILL.md"
        assert any(e["id"] == "V4" for e in data["errors"])
        assert any("V4:" in e["message"] for e in data["errors"])

    def test_lint_pass_deletes_from_diagnostics_dir(self, tmp_path: Path) -> None:
        """AC-3: after lint passes, diagnostics_dir/{name}.json is deleted"""
        diag_dir = tmp_path / "diag-out"
        diag_dir.mkdir()
        stale = diag_dir / "bad-io.json"
        stale.write_text('{"skill_name":"bad-io"}', encoding="utf-8")
        _write_skill(tmp_path, "bad-io/SKILL.md", VALID_LEGACY_AFTER_FIX)
        skills = discover([tmp_path], diagnostics_dir=diag_dir)
        assert "bad-io" in skills
        assert not stale.exists()

    def test_unresolved_dir_skipped_by_discover(self, tmp_path: Path) -> None:
        """SKILL.md under underscore-prefixed dirs is skipped by discover scan"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        _write_skill(
            tmp_path,
            "_unresolved/fake-skill/SKILL.md",
            FULL_SKILL_MD.replace("name: review", "name: fake-skill"),
        )
        skills = discover([tmp_path])
        assert "review" in skills
        assert "fake-skill" not in skills
