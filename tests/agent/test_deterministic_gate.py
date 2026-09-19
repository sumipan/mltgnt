"""Tests for mltgnt.agent.deterministic_gate using ASCII LanguagePack."""
from __future__ import annotations

from mltgnt.agent.deterministic_gate import (
    extract_artifact_references,
    has_work_request,
    is_create_request,
    match_deferred_promise,
)


def test_work_request_with_artifact(ascii_pack) -> None:
    text = "please polish the report.md"
    assert has_work_request(text, pack=ascii_pack) is True
    assert extract_artifact_references(text) == ("report.md",)


def test_no_work_request_no_artifact(ascii_pack) -> None:
    assert has_work_request("how are you today", pack=ascii_pack) is False


def test_deferred_promise_detected(ascii_pack) -> None:
    assert match_deferred_promise("will do later", pack=ascii_pack) is not None


def test_deferred_promise_none_when_absent(ascii_pack) -> None:
    assert match_deferred_promise("sounds good", pack=ascii_pack) is None


def test_create_request(ascii_pack) -> None:
    assert is_create_request("please create a new-file for me", pack=ascii_pack) is True


def test_no_create_request(ascii_pack) -> None:
    assert is_create_request("please revise the existing file", pack=ascii_pack) is False


def test_extract_artifact_references_url_and_file() -> None:
    text = "see https://example.com/doc and also notes.md"
    refs = extract_artifact_references(text)
    assert "https://example.com/doc" in refs
    assert "notes.md" in refs


def test_extract_artifact_references_dedup() -> None:
    refs = extract_artifact_references("look at notes.md and notes.md again")
    assert refs.count("notes.md") == 1


def test_extract_artifact_references_empty() -> None:
    assert extract_artifact_references("") == ()


def test_has_work_request_empty(ascii_pack) -> None:
    assert has_work_request("", pack=ascii_pack) is False


def test_match_deferred_promise_empty(ascii_pack) -> None:
    assert match_deferred_promise("", pack=ascii_pack) is None
