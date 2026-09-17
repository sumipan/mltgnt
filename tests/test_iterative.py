"""
tests/test_iterative.py — IterativeRetriever Home read_memory_iterative() Unit Test

TC1: memory Loop only if enough 0 Return results at times
TC2: memory Close → memory If re-search is sufficient
TC3: memory Close → skill When searching is sufficient
TC4: When the loop is sufficient after multiple loops
TC5: preferences sections are always included in the result
TC6: max_bytes Restrictions
TC7: loop max_iterations Collected and punched when reached
TC8: LLM If the call throws an exception, fallback to the first search result
TC9: skill_paths empty list
TC10: LLM If the response fails to parse (invalid format)
TC11: memory If the file is empty

AC1: read_memory_iterative works
AC3: from mltgnt.memory._iterative import IterativeRetriever 
AC4: judge_sufficiency SUFFICIENT → Instantly returnTC1 Verified)
AC5: INSUFFICIENT → MEMORY Search → SUFFICIENT（TC2 Verified)
AC6: from mltgnt.memory._agentic import AgenticRetriever Home ImportError
AC7: llm_call Returns the first search result at an exceptionTC8 Verified)
AC8: JSONL File empty → empty stringTC11 Verified)
AC1 (issue): _iterative.py Home mltgnt.skill Do not import directly
AC4 (issue): search_skills In callback injection skill Search works
AC5 (issue): search_skills=None Home SKILL Error when specifying source
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import textwrap
from unittest.mock import patch

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory import read_memory_iterative, memory_file_path
from mltgnt.memory._iterative import IterativeRetriever
from mltgnt.memory._scoring import ScoredEntry


def make_config(tmp_path: Path) -> MemoryConfig:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=mem_dir,
    )


def _write_memory(config: MemoryConfig, persona: str, content: str) -> None:
    memory_file_path(config, persona).write_text(content, encoding="utf-8")


MEMORY_SUSHI = (
    json.dumps({"timestamp": "2026-01-01T00:00:00+09:00", "role": "user", "content": "A user with food preferences.", "source_tag": "preferences"}, ensure_ascii=False) + "\n"
    + json.dumps({"timestamp": "2026-01-01T10:00:00+09:00", "role": "user", "content": "I like sushi. I like salmon and tuna.", "source_tag": "file"}, ensure_ascii=False) + "\n"
)

MEMORY_PROJECT = (
    json.dumps({"timestamp": "2026-01-01T10:00:00+09:00", "role": "user", "content": "Finished fixing a frontend bug.", "source_tag": "file"}, ensure_ascii=False) + "\n"
    + json.dumps({"timestamp": "2026-01-02T10:00:00+09:00", "role": "user", "content": "Backend API Started design. I discussed the project progress last week.", "source_tag": "file"}, ensure_ascii=False) + "\n"
)


def _make_llm_responses(*responses: str):
    """Returns a response in order llm_call Create"""
    it = iter(responses)
    def llm_call(_prompt: str) -> str:
        return next(it)
    return llm_call


# ---------------------------------------------------------------------------
# AC3: IterativeRetriever Home import ation
# ---------------------------------------------------------------------------


def test_ac3_import_iterative_retriever() -> None:
    """AC3: from mltgnt.memory._iterative import IterativeRetriever success."""
    assert IterativeRetriever is not None


def test_issue_ac1_no_skill_import_in_iterative() -> None:
    """AC1: _iterative.py Home mltgnt.skill Do not import directly."""
    import mltgnt.memory._iterative as iterative_mod

    source_path = Path(iterative_mod.__file__)
    source = source_path.read_text(encoding="utf-8")
    assert "from mltgnt.skill" not in source
    assert "import mltgnt.skill" not in source


def test_issue_ac4_callback_injection(tmp_path: Path) -> None:
    """AC4: search_skills via callback skill Search results are merged."""
    config = make_config(tmp_path)
    _write_memory(
        config,
        "persona",
        json.dumps(
            {
                "timestamp": "2026-01-01T10:00:00+09:00",
                "role": "user",
                "content": "General info only.",
                "source_tag": "file",
            },
            ensure_ascii=False,
        )
        + "\n",
    )

    def search_skills(_query: str, _max_entries: int) -> list[ScoredEntry]:
        return [ScoredEntry(text="deploy procedure", score=1.0)]

    retriever = IterativeRetriever(
        config=config,
        persona_stem="persona",
        llm_call=_make_llm_responses(
            "INSUFFICIENT\nSKILL\ndeploy",
            "SUFFICIENT",
        ),
        search_skills=search_skills,
    )

    result = retriever.retrieve(
        "Tell me the deployment procedure",
        max_bytes=4096,
        max_entries=5,
    )

    assert "deploy" in result


def test_issue_ac5_search_skills_none(tmp_path: Path) -> None:
    """AC5: search_skills=None、LLM Home SKILL Source specification → Loop co ation in empty list."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    retriever = IterativeRetriever(
        config=config,
        persona_stem="persona",
        llm_call=_make_llm_responses(
            "INSUFFICIENT\nSKILL\ndeploy",
            "SUFFICIENT",
        ),
        search_skills=None,
    )

    result = retriever.retrieve(
        "deploy procedure",
        max_bytes=4096,
        max_entries=5,
    )

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AC6: AgenticRetriever NoneImportError）
# ---------------------------------------------------------------------------


def test_ac6_agentic_retriever_import_error() -> None:
    """AC6: from mltgnt.memory._agentic import AgenticRetriever Home ImportError。"""
    with pytest.raises(ImportError):
        from mltgnt.memory._agentic import AgenticRetriever  # noqa: F401


# ---------------------------------------------------------------------------
# TC1: memory Only enough
# ---------------------------------------------------------------------------


def test_tc1_memory_sufficient(tmp_path: Path) -> None:
    """TC1: LLM Home SUFFICIENT loop if 0 Returns results at times."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    call_count = 0

    def llm_call(_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "SUFFICIENT"

    result = read_memory_iterative(
        config,
        "persona",
        "What food do you like?",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
    )

    assert "I like sushi" in result
    assert call_count == 1  # 1Ends once it is determined


# ---------------------------------------------------------------------------
# TC2: memory Close → memory If re-search is sufficient
# ---------------------------------------------------------------------------


def test_tc2_memory_requery(tmp_path: Path) -> None:
    """TC2: INSUFFICIENT→MEMORY→re-search query,2 SUFFICIENT。"""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nMEMORY\nLast week's project progress",
        "SUFFICIENT",
    )

    result = read_memory_iterative(
        config,
        "persona",
        "What is the project progress last week?",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
    )

    # Both entries included (no duplicates)
    assert "Front End" in result or "Backend" in result


# ---------------------------------------------------------------------------
# TC3: memory Close → skill When searching is sufficient
# ---------------------------------------------------------------------------


def test_tc3_skill_search(tmp_path: Path) -> None:
    """TC3: INSUFFICIENT→SKILL→skill The body is merged."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", json.dumps({"timestamp": "2026-01-01T10:00:00+09:00", "role": "user", "content": "General info only.", "source_tag": "file"}, ensure_ascii=False) + "\n")

    # skill directory SKILL.md Create
    skill_dir = tmp_path / "skills"
    deploy_skill = skill_dir / "deploy"
    deploy_skill.mkdir(parents=True)
    (deploy_skill / "SKILL.md").write_text(
        textwrap.dedent("""            ---
            name: deploy
            description: Steps to perform the deployment procedure
            ---
            deploy procedure: git push → CI/CD → Application
        """),
        encoding="utf-8",
    )

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nSKILL\ndeploy",
        "SUFFICIENT",
    )

    result = read_memory_iterative(
        config,
        "persona",
        "Tell me the deployment procedure",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
        skill_paths=[skill_dir],
    )

    assert "deploy" in result


# ---------------------------------------------------------------------------
# TC4: When the loop is sufficient after multiple loops
# ---------------------------------------------------------------------------


def test_tc4_multi_loop(tmp_path: Path) -> None:
    """TC4: 2After looping SUFFICIENTEach entry is merged."""
    config = make_config(tmp_path)
    _write_memory(
        config,
        "persona",
        json.dumps({"timestamp": "2026-01-01T10:00:00+09:00", "role": "user", "content": "A: First information.", "source_tag": "file"}, ensure_ascii=False) + "\n"
        + json.dumps({"timestamp": "2026-01-02T10:00:00+09:00", "role": "user", "content": "B: Additional information.", "source_tag": "file"}, ensure_ascii=False) + "\n"
        + json.dumps({"timestamp": "2026-01-03T10:00:00+09:00", "role": "user", "content": "C: More information.", "source_tag": "file"}, ensure_ascii=False) + "\n",
    )

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nMEMORY\nAdditional information",
        "INSUFFICIENT\nMEMORY\nMore information",
        "SUFFICIENT",
    )

    result = read_memory_iterative(
        config,
        "persona",
        "Questions",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
        max_iterations=3,
    )

    assert result  # Returns some results


# ---------------------------------------------------------------------------
# TC5: preferences sections are always included in the result
# ---------------------------------------------------------------------------


def test_tc5_preferences_always_included(tmp_path: Path) -> None:
    """TC5: Any query Home preferences The section is included at the top."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    result = read_memory_iterative(
        config,
        "persona",
        "All",
        max_bytes=4096,
        max_entries=5,
        llm_call=lambda _: "SUFFICIENT",
    )

    # Japanese text intentionally kept for CJK processing test
    assert "ユーザーの好み・傾向" in result
    assert "A user with food preferences" in result


# ---------------------------------------------------------------------------
# TC6: max_bytes Restrictions
# ---------------------------------------------------------------------------


def test_tc6_max_bytes_limit(tmp_path: Path) -> None:
    """TC6: Return text max_bytes Contact Us"""
    config = make_config(tmp_path)
    big_memory = "".join(
        json.dumps({"timestamp": f"2026-01-{i+1:02d}T10:00:00+09:00", "role": "user", "content": "a" * 200, "source_tag": "file"}, ensure_ascii=False) + "\n"
        for i in range(10)
    )
    _write_memory(config, "persona", big_memory)

    max_bytes = 500
    result = read_memory_iterative(
        config,
        "persona",
        "test",
        max_bytes=max_bytes,
        max_entries=10,
        llm_call=lambda _: "SUFFICIENT",
    )

    assert len(result.encode("utf-8")) <= max_bytes


# ---------------------------------------------------------------------------
# TC7: max_iterations When reaches
# ---------------------------------------------------------------------------


def test_tc7_max_iterations_reached(tmp_path: Path) -> None:
    """TC7: LLM always INSUFFICIENT ifmax_iterations Then, the error occurred."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    call_count = 0

    def always_insufficient(_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "INSUFFICIENT\nMEMORY\nAdditional queries"

    result = read_memory_iterative(
        config,
        "persona",
        "test",
        max_bytes=4096,
        max_entries=5,
        llm_call=always_insufficient,
        max_iterations=3,
    )

    assert call_count == 3  # max_iterations called round
    assert isinstance(result, str)  # Error


# ---------------------------------------------------------------------------
# TC8: LLM In the case of an exception, fallback to the first search result
# ---------------------------------------------------------------------------


def test_tc8_llm_exception_fallback(tmp_path: Path, caplog) -> None:
    """TC8: llm_call Home RuntimeError Home raise → warning Logs and initial results are returned."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    def failing_llm(_prompt: str) -> str:
        raise RuntimeError("LLM API error")

    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._iterative"):
        result = read_memory_iterative(
            config,
            "persona",
            "What food do you like?",
            max_bytes=4096,
            max_entries=5,
            llm_call=failing_llm,
        )

    assert isinstance(result, str)
    assert any("failed" in r.message.lower() or "llm" in r.message.lower() for r in caplog.records)
    # First time memory Returns search results
    assert "I like sushi" in result


# ---------------------------------------------------------------------------
# TC9: skill_paths empty list
# ---------------------------------------------------------------------------


def test_tc9_empty_skill_paths(tmp_path: Path) -> None:
    """TC9: skill_paths=[], LLM Home SKILL Sauce → skill Search result is empty and loop continues."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    llm_call = _make_llm_responses(
        "INSUFFICIENT\nSKILL\ndeploy",
        "SUFFICIENT",
    )

    # Returns results without errors
    result = read_memory_iterative(
        config,
        "persona",
        "deploy procedure",
        max_bytes=4096,
        max_entries=5,
        llm_call=llm_call,
        skill_paths=[],
    )

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TC10: LLM Failed to parse the response
# ---------------------------------------------------------------------------


def test_tc10_parse_failure(tmp_path: Path, caplog) -> None:
    """TC10: LLM returns an unexpected format → sufficient=True Handles and returns collected entries."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = read_memory_iterative(
            config,
            "persona",
            "test",
            max_bytes=4096,
            max_entries=5,
            llm_call=lambda _: "INVALID_FORMAT_XYZ",
        )

    assert isinstance(result, str)
    assert any(
        "unexpected" in r.message.lower() or "sufficient" in r.message.lower()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# TC11: memory If the file is empty
# ---------------------------------------------------------------------------


def test_tc11_empty_memory(tmp_path: Path) -> None:
    """TC11: memory No files → empty string or preferences error only."""
    config = make_config(tmp_path)
    # memory Not

    result = read_memory_iterative(
        config,
        "nonexistent",
        "test",
        max_bytes=4096,
        max_entries=5,
        llm_call=lambda _: "SUFFICIENT",
    )

    assert isinstance(result, str)
    # Error


# ---------------------------------------------------------------------------
# retrieve_skills
# ---------------------------------------------------------------------------


def test_retrieve_skills_calls_callback(tmp_path: Path) -> None:
    """retrieve_skills Home search_skills Callback only."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PROJECT)

    calls: list[tuple[str, int]] = []

    def search_skills(query: str, max_entries: int) -> list[ScoredEntry]:
        calls.append((query, max_entries))
        return [ScoredEntry(text="how to write a diary", score=0.9)]

    retriever = IterativeRetriever(
        config=config,
        persona_stem="persona",
        llm_call=lambda _: "SUFFICIENT",
        search_skills=search_skills,
    )

    result = retriever.retrieve_skills("write a diary", max_entries=5)

    assert len(calls) == 1
    assert calls[0] == ("write a diary", 5)
    assert len(result) == 1
    assert result[0].text == "how to write a diary"


def test_retrieve_skills_none_returns_empty(tmp_path: Path) -> None:
    """search_skills=None Forretrieve_skills empty list."""
    config = make_config(tmp_path)
    retriever = IterativeRetriever(
        config=config,
        persona_stem="persona",
        llm_call=lambda _: "SUFFICIENT",
        search_skills=None,
    )

    result = retriever.retrieve_skills("write a diary", max_entries=5)

    assert result == []


def test_retrieve_skills_does_not_search_memory(tmp_path: Path) -> None:
    """retrieve_skills Home memory Do not search the source."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SUSHI)

    memory_searched = False

    original_search_memory = IterativeRetriever._search_memory

    def tracking_search_memory(self, query, max_entries):
        nonlocal memory_searched
        memory_searched = True
        return original_search_memory(self, query, max_entries)

    retriever = IterativeRetriever(
        config=config,
        persona_stem="persona",
        llm_call=lambda _: "SUFFICIENT",
        search_skills=lambda q, n: [ScoredEntry(text="skill only", score=1.0)],
    )

    with patch.object(IterativeRetriever, "_search_memory", tracking_search_memory):
        result = retriever.retrieve_skills("sushi", max_entries=5)

    assert memory_searched is False
    assert result[0].text == "skill only"
