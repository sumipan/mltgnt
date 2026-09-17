"""Tests for mltgnt.persona.loader — AC-3 / AC-4 / AC-5 / BC."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.persona.loader import Persona, load, _parse_sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_persona(body: str, sections: dict[str, str] | None = None) -> Persona:
    """Build a Persona instance directly for tests."""
    from mltgnt.persona.schema import PersonaFM

    if sections is None:
        sections = _parse_sections(body)

    fm = PersonaFM(name="test")
    return Persona(
        name="test",
        fm=fm,
        sections=sections,
        body=body,
        path=Path("test.md"),
    )


FULL_BODY = textwrap.dedent("""\
    ## 1. \u57fa\u672c\u60c5\u5831

    Basic info content.

    ## 2. \u4fa1\u5024\u89b3

    Values content.

    ## 3. \u53cd\u5fdc\u30d1\u30bf\u30fc\u30f3

    Reaction pattern content.

    ## 4. \u53e3\u8abf

    Tone content.

    ## 5. \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

    #### critique

    Critique format.

    #### edit

    Edit format.
""")

OUTPUT_FORMAT_BODY = textwrap.dedent("""\
    ## 1. \u57fa\u672c\u60c5\u5831

    persona-a.

    ## 5. \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

    #### critique

    Critique format content.
    Multiple lines.

    #### edit

    Edit format content.
""")


# ---------------------------------------------------------------------------
# AC-3: exclude section 0
# ---------------------------------------------------------------------------


class TestParseSection0Exclusion:
    def test_3_1_section0_excluded(self):
        body = textwrap.dedent("""\
            ## 0. \u30d5\u30a1\u30a4\u30eb\u306e\u4f7f\u3044\u65b9

            This usage note is excluded.

            ## 1. \u57fa\u672c\u60c5\u5831

            persona-a.
        """)
        sections = _parse_sections(body)
        assert "\u30d5\u30a1\u30a4\u30eb\u306e\u4f7f\u3044\u65b9" not in sections
        assert "\u57fa\u672c\u60c5\u5831" in sections

    def test_3_2_no_section0_regression(self):
        body = textwrap.dedent("""\
            ## 1. \u57fa\u672c\u60c5\u5831

            persona-a.

            ## 2. \u4fa1\u5024\u89b3

            Curious.
        """)
        sections = _parse_sections(body)
        assert "\u57fa\u672c\u60c5\u5831" in sections
        assert "\u4fa1\u5024\u89b3" in sections
        assert len(sections) == 2

    def test_3_3_all_persona_files(self):
        from mltgnt.persona import PersonaValidationError

        repo_root = Path(__file__).parent.parent.parent
        persona_dir = repo_root / "personas"
        if not persona_dir.exists():
            pytest.skip("personas/ directory does not exist")

        files = list(persona_dir.glob("*.md"))
        if not files:
            pytest.skip("no files in personas/")

        for path in files:
            try:
                p = load(path)
                assert p is not None
            except PersonaValidationError:
                pytest.fail(f"PersonaValidationError for {path.name}")


# ---------------------------------------------------------------------------
# AC-4: H2 block selection
# ---------------------------------------------------------------------------


class TestFormatPromptWeightSelection:
    def test_4_1_weight_heavy(self):
        persona = _make_persona(FULL_BODY)
        result = persona.format_prompt("instruction", weight="heavy")
        assert "Basic info content" in result
        assert "Values content" in result
        assert "Reaction pattern content" in result
        assert "Tone content" in result
        assert "Critique format" not in result

    def test_4_2_weight_light(self):
        persona = _make_persona(FULL_BODY)
        result = persona.format_prompt("instruction", weight="light")
        assert "instruction" in result
        assert "Basic info content" not in result

    def test_4_3_default_is_heavy(self):
        persona = _make_persona(FULL_BODY)
        result_default = persona.format_prompt("instruction")
        result_heavy = persona.format_prompt("instruction", weight="heavy")
        assert result_default == result_heavy

    def test_4_4_unknown_section_warns_and_fallbacks(self):
        body_with_unknown = textwrap.dedent("""\
            ## 1. \u672a\u77e5\u306e\u30bb\u30af\u30b7\u30e7\u30f3

            Unknown content.

            ## 2. \u57fa\u672c\u60c5\u5831

            Basic info content.
        """)
        persona = _make_persona(body_with_unknown)
        with patch("mltgnt.persona.loader.logger") as mock_logger:
            result = persona.format_prompt("instruction", weight="heavy")
            mock_logger.warning.assert_called()
        assert "Unknown content" in result
        assert "Basic info content" in result


# ---------------------------------------------------------------------------
# AC-5: output format extraction
# ---------------------------------------------------------------------------


class TestExtractOutputFormat:
    def test_5_1_extract_critique(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "Critique format content" in result
        assert "Edit format content" not in result

    def test_5_2_default_op_mode(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result_none = persona.extract_output_format(None)
        result_critique = persona.extract_output_format("critique")
        assert result_none == result_critique

    def test_5_3_nonexistent_op_mode_returns_none(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.extract_output_format("nonexistent")
        assert result is None

    def test_5_4_critique_does_not_include_edit(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "Edit format content" not in result

    def test_5_5_no_output_format_section(self):
        body = textwrap.dedent("""\
            ## 1. \u57fa\u672c\u60c5\u5831

            persona-a.
        """)
        persona = _make_persona(body)
        result = persona.extract_output_format("critique")
        assert result is None


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_bc_1_format_prompt_no_weight(self):
        persona = _make_persona(FULL_BODY)
        result = persona.format_prompt("test instruction")
        assert "test instruction" in result

    def test_bc_2_build_review_prompt_with_output(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.build_review_prompt("critique")
        assert "persona-a" in result
        assert "Critique format content" in result

    def test_bc_2_build_review_prompt_no_output_section(self):
        body = "## 1. \u57fa\u672c\u60c5\u5831\n\npersona-a."
        persona = _make_persona(body)
        result = persona.build_review_prompt()
        assert "persona-a" in result


# ---------------------------------------------------------------------------
# AC-1: v2 _parse_sections H3 expansion
# ---------------------------------------------------------------------------


import textwrap as _textwrap  # noqa: E402
from unittest.mock import patch as _patch  # noqa: E402


class TestParseSectionsV2:
    def test_1_1_v2_expands_h3_heavy_and_reference(self):
        """AC 1-1: v2 H3 expansion. Heavy/Reference H3s flatten into the dict."""
        body = _textwrap.dedent("""\
            ## \u8efd\u91cf

            summary

            ## \u91cd\u91cf

            ### \u57fa\u672c\u60c5\u5831

            content A

            ### \u4fa1\u5024\u89b3

            content B

            ## \u53c2\u7167

            ### \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

            #### critique

            template

            ### \u30e1\u30e2\u30fb\u66f4\u65b0\u5c65\u6b74

            update notes
        """)
        sections = _parse_sections(body)
        assert sections.get("\u8efd\u91cf") == "summary"
        assert sections.get("\u57fa\u672c\u60c5\u5831") == "content A"
        assert sections.get("\u4fa1\u5024\u89b3") == "content B"
        assert "\u91cd\u91cf" not in sections
        assert "\u53c2\u7167" not in sections
        assert "\u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f" in sections
        assert "#### critique" in sections["\u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f"]
        assert sections.get("\u30e1\u30e2\u30fb\u66f4\u65b0\u5c65\u6b74") == "update notes"

    def test_1_2_format_prompt_v2_no_unknown_warning(self):
        """AC 1-2: v2 (heavy only) format_prompt must not warn about undefined WEIGHT_MAP."""
        body = _textwrap.dedent("""\
            ## \u91cd\u91cf

            ### \u57fa\u672c\u60c5\u5831

            Basic info content.

            ### \u4fa1\u5024\u89b3

            Values content.

            ### \u53cd\u5fdc\u30d1\u30bf\u30fc\u30f3

            Reaction pattern content.

            ### \u53e3\u8abf

            Tone content.
        """)
        persona = _make_persona(body)
        with _patch("mltgnt.persona.loader.logger") as mock_logger:
            persona.format_prompt("instruction", weight="heavy")
            for call_args in mock_logger.warning.call_args_list:
                args = call_args[0]
                assert "WEIGHT_MAP \u306b\u672a\u5b9a\u7fa9" not in str(args), \
                    f"must not warn about undefined WEIGHT_MAP: {args}"

    def test_1_3_reference_block_h3_expansion(self):
        """AC 1-3: Reference block H3 expansion."""
        body = _textwrap.dedent("""\
            ## \u53c2\u7167

            ### \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

            #### critique

            template

            ### \u30e1\u30e2\u30fb\u66f4\u65b0\u5c65\u6b74

            update notes
        """)
        sections = _parse_sections(body)
        assert "\u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f" in sections
        assert "#### critique" in sections["\u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f"]
        assert "template" in sections["\u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f"]
        assert sections.get("\u30e1\u30e2\u30fb\u66f4\u65b0\u5c65\u6b74") == "update notes"
        assert "\u53c2\u7167" not in sections

    def test_1_4_v1_backward_compat(self):
        """AC 1-4: v1 backward compat. Direct H2 sections work as before."""
        body = _textwrap.dedent("""\
            ## \u57fa\u672c\u60c5\u5831

            content

            ## \u4fa1\u5024\u89b3

            content B
        """)
        sections = _parse_sections(body)
        assert sections == {"\u57fa\u672c\u60c5\u5831": "content", "\u4fa1\u5024\u89b3": "content B"}

    def test_1_5_v1_numbered_heading(self):
        """AC 1-5: v1 numbered heading normalization is preserved."""
        body = "## 1. \u57fa\u672c\u60c5\u5831\ncontent"
        sections = _parse_sections(body)
        assert "\u57fa\u672c\u60c5\u5831" in sections
        assert sections["\u57fa\u672c\u60c5\u5831"] == "content"

    def test_1_6_section0_exclusion_maintained(self):
        """AC 1-6: section 0 exclusion is preserved."""
        body = _textwrap.dedent("""\
            ## 0. \u30d5\u30a1\u30a4\u30eb\u306e\u4f7f\u3044\u65b9

            content

            ## \u57fa\u672c\u60c5\u5831

            content B
        """)
        sections = _parse_sections(body)
        assert "\u30d5\u30a1\u30a4\u30eb\u306e\u4f7f\u3044\u65b9" not in sections
        assert sections.get("\u57fa\u672c\u60c5\u5831") == "content B"

    def test_1_7_v1_full_section_names_no_warning(self):
        """AC 1-7: full v1 section names must not warn about undefined WEIGHT_MAP."""
        body = _textwrap.dedent("""\
            ## \u57fa\u672c\u60c5\u5831

            Basic info content.

            ## \u4fa1\u5024\u89b3\u30fb\u512a\u5148\u9806\u4f4d

            Values content.

            ## \u53cd\u5fdc\u30d1\u30bf\u30fc\u30f3

            Reaction pattern content.

            ## \u53e3\u8abf\u30fb\u8a9e\u308a\u65b9

            Tone content.

            ## \u8efd\u91cf

            Light text.
        """)
        persona = _make_persona(body)
        with _patch("mltgnt.persona.loader.logger") as mock_logger:
            persona.format_prompt("instruction", weight="heavy")
            for call_args in mock_logger.warning.call_args_list:
                args = call_args[0]
                assert "WEIGHT_MAP \u306b\u672a\u5b9a\u7fa9" not in str(args), \
                    f"must not warn about undefined WEIGHT_MAP: {args}"

    def test_1_8_v1_full_section_names_heavy_excludes_light(self):
        """AC 1-8: v1 weight=heavy must exclude the light section."""
        body = _textwrap.dedent("""\
            ## \u57fa\u672c\u60c5\u5831

            Basic info content.

            ## \u4fa1\u5024\u89b3\u30fb\u512a\u5148\u9806\u4f4d

            Values content.

            ## \u8efd\u91cf

            Light text.
        """)
        persona = _make_persona(body)
        result = persona.format_prompt("instruction", weight="heavy")
        assert "Basic info content" in result
        assert "Light text" not in result

    def test_1_9_v1_full_section_names_light_weight(self):
        """AC 1-9: v1 weight=light must include only the light section."""
        body = _textwrap.dedent("""\
            ## \u57fa\u672c\u60c5\u5831

            Basic info content.

            ## \u8efd\u91cf

            Light text.
        """)
        persona = _make_persona(body)
        result = persona.format_prompt("instruction", weight="light")
        assert "Light text" in result
        assert "Basic info content" not in result


# ---------------------------------------------------------------------------
# AC-2: v2 extract_output_format
# ---------------------------------------------------------------------------


class TestExtractOutputFormatV2:
    def test_2_1_v2_extract_critique(self):
        """AC 2-1: extract critique from v2 (Reference > Output format)."""
        body = _textwrap.dedent("""\
            ## \u53c2\u7167

            ### \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

            #### critique

            【\u6240\u898b】

            #### edit

            【\u4fee\u6b63\u6848】
        """)
        persona = _make_persona(body)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "【\u6240\u898b】" in result
        assert "【\u4fee\u6b63\u6848】" not in result

    def test_2_2_v2_unknown_mode_returns_none(self):
        """AC 2-2: unknown mode returns None (silent skip)."""
        body = _textwrap.dedent("""\
            ## \u53c2\u7167

            ### \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

            #### critique

            【\u6240\u898b】
        """)
        persona = _make_persona(body)
        assert persona.extract_output_format("debate") is None

    def test_2_3_v1_extract_output_format(self):
        """AC 2-3: extract from v1 (direct H2 Output format)."""
        body = _textwrap.dedent("""\
            ## \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

            #### critique

            content
        """)
        persona = _make_persona(body)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "content" in result


# ---------------------------------------------------------------------------
# AC-3: v2 extract_triage_section
# ---------------------------------------------------------------------------


from mltgnt.routing.triage import extract_triage_section as _extract_triage_section  # noqa: E402


class TestExtractTriageSectionV2:
    def test_3_1_v2_returns_light_section(self):
        """AC 3-1: return triage section from v2 light heading."""
        md = _textwrap.dedent("""\
            ## \u8efd\u91cf

            logical and frank

            ## \u91cd\u91cf

            ### \u57fa\u672c\u60c5\u5831

            content
        """)
        result = _extract_triage_section(md)
        assert result is not None
        assert "logical and frank" in result

    def test_3_2_v1_fallback(self):
        """AC 3-2: fallback to v1 triage heading."""
        md = _textwrap.dedent("""\
            ## \u30c8\u30ea\u30a2\u30fc\u30b8\u7528

            triage content

            ## \u57fa\u672c\u60c5\u5831

            content
        """)
        result = _extract_triage_section(md)
        assert result is not None
        assert "triage content" in result

    def test_3_3_neither_returns_none(self):
        """AC 3-3: neither present → return None."""
        md = "## \u57fa\u672c\u60c5\u5831\ncontent only"
        assert _extract_triage_section(md) is None

    def test_3_4_both_present_v2_wins(self):
        """AC 3-4: when both exist, light heading wins."""
        md = _textwrap.dedent("""\
            ## \u8efd\u91cf

            v2 content

            ## \u30c8\u30ea\u30a2\u30fc\u30b8\u7528

            v1 content
        """)
        result = _extract_triage_section(md)
        assert result is not None
        assert "v2 content" in result
        assert "v1 content" not in result


# ---------------------------------------------------------------------------
# Issue-919: DEFAULT_WEIGHT_MAP / PersonaConfig / English persona support
# ---------------------------------------------------------------------------

import textwrap as _tw  # noqa: E402

from mltgnt.config import DEFAULT_WEIGHT_MAP, PersonaConfig  # noqa: E402


_ENGLISH_BODY = _tw.dedent("""\
    ## Light

    Light content.

    ## Heavy

    ### Background

    Background content.

    ### Values

    Values content.

    ### Tone

    Tone content.

    ## Reference

    ### Output format

    #### critique

    Critique template.
""")


class TestPersonaConfig:
    def test_ac1_default_weight_map(self):
        """AC-1: default construction uses DEFAULT_WEIGHT_MAP."""
        cfg = PersonaConfig()
        assert cfg.weight_map == DEFAULT_WEIGHT_MAP

    def test_ac1_weight_map_is_copy(self):
        """AC-1: weight_map is a copy of DEFAULT_WEIGHT_MAP."""
        cfg = PersonaConfig()
        assert cfg.weight_map is not DEFAULT_WEIGHT_MAP

    def test_ac2_custom_weight_map(self):
        """AC-2: custom weight_map can override."""
        custom = {"Custom": "heavy"}
        cfg = PersonaConfig(weight_map=custom)
        assert cfg.weight_map == custom


class TestEnglishPersona:
    def test_ac4_english_heavy_sections(self):
        """AC-4: English persona weight=heavy includes Background/Values/Tone only."""
        persona = _make_persona(_ENGLISH_BODY)
        result = persona.format_prompt("instruction", weight="heavy")
        assert "Background content" in result
        assert "Values content" in result
        assert "Tone content" in result
        assert "Light content" not in result

    def test_ac5_english_light_section(self):
        """AC-5: English persona weight=light includes Light only."""
        persona = _make_persona(_ENGLISH_BODY)
        result = persona.format_prompt("instruction", weight="light")
        assert "Light content" in result
        assert "Background content" not in result

    def test_ac6_parse_sections_expands_heavy_h3(self):
        """AC-6: _parse_sections flattens ## Heavy > ### X / ### Y."""
        body = _tw.dedent("""\
            ## Heavy

            ### Background

            Background content.

            ### Values

            Values content.
        """)
        sections = _parse_sections(body)
        assert "Background" in sections
        assert "Values" in sections
        assert "Heavy" not in sections
        assert sections["Background"] == "Background content."
        assert sections["Values"] == "Values content."

    def test_ac7_parse_sections_expands_reference_h3(self):
        """AC-7: Reference > Output format expands and extract_output_format works."""
        persona = _make_persona(_ENGLISH_BODY)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "Critique template" in result

    def test_ac4_no_unknown_warning_english(self):
        """AC-4: English persona must not warn about undefined WEIGHT_MAP."""
        persona = _make_persona(_ENGLISH_BODY)
        with _patch("mltgnt.persona.loader.logger") as mock_logger:
            persona.format_prompt("instruction", weight="heavy")
            for call_args in mock_logger.warning.call_args_list:
                args = call_args[0]
                assert "WEIGHT_MAP \u306b\u672a\u5b9a\u7fa9" not in str(args), \
                    f"must not warn about undefined WEIGHT_MAP: {args}"


class TestLoadWithConfig:
    def test_ac8_load_with_custom_config(self, tmp_path):
        """AC-8: load(..., config=PersonaConfig(weight_map=custom)) uses custom map."""
        md = _tw.dedent("""\
            ---
            persona:
              name: test
            ---
            ## Custom

            Custom content.
        """)
        p = tmp_path / "test.md"
        p.write_text(md, encoding="utf-8")
        custom = {"Custom": "heavy"}
        cfg = PersonaConfig(weight_map=custom)
        persona = load(p, config=cfg)
        assert persona.weight_map == custom

    def test_ac9_load_without_config(self, tmp_path):
        """AC-9: without config, weight_map equals DEFAULT_WEIGHT_MAP."""
        md = _tw.dedent("""\
            ---
            persona:
              name: test
            ---
            ## \u57fa\u672c\u60c5\u5831

            content.
        """)
        p = tmp_path / "test.md"
        p.write_text(md, encoding="utf-8")
        persona = load(p)
        assert persona.weight_map == DEFAULT_WEIGHT_MAP

    def test_ac10_unknown_section_warns_and_fallbacks(self):
        """AC-10: unknown sections warn and fall back to all sections."""
        body = _tw.dedent("""\
            ## UnknownSection

            Unknown content.
        """)
        persona = _make_persona(body)
        with _patch("mltgnt.persona.loader.logger") as mock_logger:
            result = persona.format_prompt("instruction", weight="heavy")
            mock_logger.warning.assert_called()
        assert "Unknown content" in result

    def test_ac11_empty_weight_map_fallback(self):
        """AC-11: empty weight_map falls back to all sections."""
        body = _tw.dedent("""\
            ## \u57fa\u672c\u60c5\u5831

            content.
        """)
        from mltgnt.persona.schema import PersonaFM
        fm = PersonaFM(name="test")
        persona = Persona(
            name="test",
            fm=fm,
            sections=_parse_sections(body),
            body=body,
            path=Path("test.md"),
            weight_map={},
        )
        with _patch("mltgnt.persona.loader.logger") as mock_logger:
            result = persona.format_prompt("instruction", weight="heavy")
            mock_logger.warning.assert_called()
        assert "content" in result

    def test_ac12_class_var_weight_map_removed(self):
        """AC-12: Persona.WEIGHT_MAP ClassVar removed in v0.9 (use instance weight_map)."""
        assert not hasattr(Persona, "WEIGHT_MAP")

    def test_ac13_make_persona_without_weight_map(self):
        """AC-13: constructing Persona without weight_map still works (helper compat)."""
        persona = _make_persona("## \u57fa\u672c\u60c5\u5831\n\n\u5185\u5bb9")
        assert persona is not None
        assert persona.weight_map == DEFAULT_WEIGHT_MAP


# ---------------------------------------------------------------------------
# Issue-1034: load() error handling after md_read migration
# ---------------------------------------------------------------------------


class TestLoadMdRead:
    """AC: load() error handling after md_read migration."""

    def test_yaml_error_raises_persona_validation_error(self, tmp_path):
        """AC: PersonaValidationError on YAML parse errors."""
        from mltgnt.persona import PersonaValidationError

        p = tmp_path / "broken.md"
        p.write_text("---\ndescription: [unclosed\n---\n\nbody text\n", encoding="utf-8")
        with pytest.raises(PersonaValidationError):
            load(p)

    def test_no_persona_namespace_raises_persona_validation_error(self, tmp_path):
        """AC: PersonaValidationError when persona namespace is missing."""
        from mltgnt.persona import PersonaValidationError

        p = tmp_path / "nons.md"
        p.write_text(
            "---\nops:\n  engine: claude\n---\n\n## \u57fa\u672c\u60c5\u5831\n\n\u5185\u5bb9。\n",
            encoding="utf-8",
        )
        with pytest.raises(PersonaValidationError):
            load(p)

    def test_no_frontmatter_raises_persona_validation_error(self, tmp_path):
        """AC: PersonaValidationError when frontmatter is missing."""
        from mltgnt.persona import PersonaValidationError

        p = tmp_path / "nofront.md"
        p.write_text("## \u57fa\u672c\u60c5\u5831\n\ncontent only.\n", encoding="utf-8")
        with pytest.raises(PersonaValidationError):
            load(p)


class TestFrontmatterDeadCode:
    """AC: confirm dead code removal."""

    def test_read_persona_markdown_deleted(self):
        """AC: read_persona_markdown has been removed."""
        import mltgnt.persona.frontmatter as m

        assert not hasattr(m, "read_persona_markdown")
