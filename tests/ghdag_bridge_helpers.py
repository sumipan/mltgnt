"""Shared helpers for ghdag_bridge tests."""
from __future__ import annotations

from pathlib import Path

def _make_jobs_dir(tmp_path: Path) -> Path:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "exec.jsonl").write_text("", encoding="utf-8")
    return jobs


_WAIT = "ghdag.pipeline.wait_for_result"

def _make_jobs_dir_dag(tmp_path: Path) -> tuple[Path, Path]:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "exec.jsonl").write_text("", encoding="utf-8")
    done = jobs / "done"
    done.mkdir()
    return jobs, done

_MD_READ = "mltgnt.bridges.ghdag_bridge.md_read"
