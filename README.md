# mltgnt

**L1 agent runtime for host-integrated operations.** In the three-layer stack (**L0 ghdag / L1 mltgnt / L2 host**), mltgnt defines agent contracts, loop state transitions, and orchestration boundaries, while model execution and DAG transport stay in ghdag.

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.22.0)-orange)

## Not (what this is not)

| Item | Why |
|------|-----|
| Not an LLM SDK | mltgnt does not call model providers directly; LLM calls are delegated through ghdag adapters. |
| Not a DAG engine | Step scheduling and queue execution are owned by ghdag. |
| Not a Slack (or host) runtime | Channel delivery, daemon wiring, and infrastructure details are owned by your L2 host. |

Objective execution is request-driven: writing only an objective file does not start a loop. A host must produce start-request JSON files.

## Installation

| Item | Value |
|------|-------|
| Package | `pip install mltgnt` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ v0.33.0` |
| Entry point | `mltgnt = mltgnt.cli.main:main` |
| License | MIT |

Dependency pin source: `pyproject.toml` (`ghdag @ git+https://github.com/sumipan/ghdag.git@v0.33.0`).

## Quick Start

### 1) Parse an objective markdown file

```python
from pathlib import Path
import tempfile

from mltgnt.loops.objective import Objective, ObjectiveError, parse_objective

text = """```yaml
id: release-note-refresh
title: Refresh the release objective
agent: operator
max_iterations: 3
status: active
```

Rewrite README sections and submit a PR.
"""

with tempfile.TemporaryDirectory() as tmp:
    objective_path = Path(tmp) / "release-note-refresh.md"
    objective_path.write_text(text, encoding="utf-8")

    parsed = parse_objective(
        objective_path,
        default_persona="operator",
        default_max_iterations=5,
    )
    if isinstance(parsed, Objective):
        print(parsed.loop_id, parsed.agent, parsed.max_iterations)
    else:
        assert isinstance(parsed, ObjectiveError)
        print(parsed.message)
```

### 2) Validate start-request JSON (`list_requests` contract)

```python
import json
from pathlib import Path
import tempfile

from mltgnt.loops.requests import StartRequest, RequestError, list_requests


def validate_start_request(state_dir: Path, objectives_dir: Path) -> tuple[list[StartRequest], list[RequestError]]:
    # list_requests performs schema and safety validation and isolates corrupt files.
    return list_requests(state_dir, objectives_dir)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    objectives_dir = root / "objectives"
    state_dir = root / "state"
    request_dir = state_dir / "requests"
    objectives_dir.mkdir()
    request_dir.mkdir(parents=True)

    payload = {
        "objective_path": "release-note-refresh.md",
        "channel_id": "C123456",
        "thread_ts": "1725252000.000100",
        "persona": "operator",
        "requested_at": "2026-09-02T10:00:00+09:00",
    }
    (request_dir / "start.json").write_text(json.dumps(payload), encoding="utf-8")

    ok, errors = validate_start_request(state_dir, objectives_dir)
    assert len(ok) == 1 and not errors
    print(ok[0].filename)
```

### 3) Load a persona

```python
from pathlib import Path
import tempfile

from mltgnt import load_persona

persona_text = """```yaml
persona:
  name: Tachikoma
ops:
  engine: claude
  model: claude-sonnet-4-6
```

## Background
A curious multi-legged AI tank.
"""

with tempfile.TemporaryDirectory() as tmp:
    persona_dir = Path(tmp)
    (persona_dir / "Tachikoma.md").write_text(persona_text, encoding="utf-8")
    persona = load_persona("Tachikoma", persona_dir=persona_dir)
    print(persona.name, persona.fm.engine, persona.fm.model)
```

## CLI Reference

The package exposes `mltgnt` and module entry points for KPI and improvement workflows.

| Command | Description | Required arguments |
|---------|-------------|--------------------|
| `mltgnt run` | Start daemon runner with user-provided component factory | `--components MODULE:FUNCTION` |
| `mltgnt memory dream show` | Show dream summary sections for one persona | `persona`, `--chat-dir` |
| `mltgnt memory dream forget` | Remove one dream summary category | `persona`, `--category`, `--chat-dir` |
| `python -m mltgnt.kpi` | Compute KPI report from `audit.jsonl` | `audit_path` |
| `python -m mltgnt.improvement` | Run improvement cycle and print report | `--audit`, `--persona-dir`, `--skills-dir` |

## Public API

### `mltgnt.__all__` (24 symbols)

`run_pipeline`, `read_memory_iterative`, `read_memory_by_relevance`, `read_memory_with_sufficiency_check`, `DreamSection`, `DreamSummary`, `read_dream`, `write_dream`, `Persona`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt`, `ChatInput`, `ChatOutput`, `Message`, `PersonaProtocol`, `AgentResult`, `AgentRunner`, `enqueue_dag`, `enqueue_and_wait`, `PersonaScheduler`, `ScheduleJob`, `__version__`

### Major subpackage exports

| Package | Symbols in `__all__` |
|---------|------------------------|
| `mltgnt.loops` | `GhdagSubtaskExecutor`, `LoopState`, `LoopsComponent`, `LoopsEngine`, `Objective`, `ObjectiveError`, `Subtask`, `TERMINAL_STATUSES`, `list_objective_files`, `parse_objective` |
| `mltgnt.bridges` | `DagStep`, `MltgntHooks`, `call_llm`, `create_audit_writer`, `enqueue_and_wait`, `enqueue_dag`, `files_adapter`, `ghdag_bridge`, `hooks_adapter`, `llm_adapter`, `md_read`, `md_write` |
| `mltgnt.chat` | `ChatInput`, `ChatOutput`, `Message`, `run_pipeline` |
| `mltgnt.config` | `DEFAULT_WEIGHT_MAP`, `MemoryConfig`, `PersonaConfig`, `SchedulerConfig`, `ChatConfig` |
| `mltgnt.daemon` | `DaemonComponent`, `DaemonRunner`, `PidLock`, `SkillWatcherComponent` |
| `mltgnt.execution` | `BaseRunner` |
| `mltgnt.improvement` | `FailurePattern`, `analyze_failures`, `ImprovementHub`, `ImprovementSource`, `MltgntSource`, `ImprovementProposal`, `generate_proposals`, `PatchResult`, `RollbackDecision`, `evaluate_cycle_outcome`, `evaluate_rollback`, `execute_rollback`, `CycleResult`, `run_improvement_cycle` |
| `mltgnt.interfaces` | `SlackClientProtocol`, `PersonaProtocol`, `ChatPipelineProtocol`, `PersonaFMBase`, `Message`, `ChatInput`, `ChatOutput`, `ChatInputBase`, `ChatOutputBase` |
| `mltgnt.memory` | `persona_memory_lock`, `append_memory_entry`, `read_memory_preferences`, `read_memory_tail_text`, `read_memory_by_relevance`, `read_memory_with_sufficiency_check`, `read_memory_iterative`, `memory_file_path`, `LlmCallError`, `CompactionResult`, `MemoryEntry`, `parse_jsonl`, `serialize_entry`, `assemble_entries_text`, `get_collection`, `query_similar`, `upsert_entry`, `tail_utf8_bytes`, `LlmCall`, `MEMORY_CORRUPT_THRESHOLD_BYTES`, `MEMORY_DEDUPE_SCAN_BYTES`, `MEMORY_DEDUPE_SCAN_LINES`, `_ensure_jsonl`, `_resolve_memory_dir`, `_scan_tail_for_dedupe_key`, `_tail_utf8_bytes`, `_search_and_score` |
| `mltgnt.memory.dream` | `DreamSection`, `DreamSummary`, `read_dream`, `write_dream`, `read_global`, `write_global`, `read_global_summary`, `DreamSelector`, `Synthesizer` |
| `mltgnt.ooda` | `OODARunner`, `OODAConfig`, `OODATickResult` |
| `mltgnt.persona` | `Persona`, `PersonaValidationError`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt`, `compress_heavy_to_light`, `regenerate_light_block` |
| `mltgnt.routing` | `ChannelPersonaEntry`, `RoutingRule`, `TRIAGE_PROFILE_MAX_CHARS`, `detect_nickname`, `evaluate`, `extract_json_object`, `extract_triage_section`, `find_observers`, `load_channel_persona_map`, `prepare_profile_for_triage`, `resolve_responding_persona` |
| `mltgnt.scheduler` | `ScheduleJob`, `PersonaScheduler`, `SchedulePaths`, `load_schedule_jobs`, `atomic_write_text`, `_hash_offset` |
| `mltgnt.skill` | `discover`, `discover_bodies`, `load`, `match`, `resolve_skill`, `run`, `SkillMeta`, `SkillFile`, `SkillRegistry`, `ArtifactSpec`, `ProducesSpec`, `ConsumesSpec`, `SkillRunResult`, `SkillMatchResult`, `lint_skill_meta` |

### Protocols and extension points

| Contract | Module | Purpose |
|----------|--------|---------|
| `HumanChannel` | `mltgnt.interfaces.loops` | Host-side thread open/ask/notify/progress/deliverable callbacks |
| `SubtaskExecutor` | `mltgnt.interfaces.loops` | Async submit/poll abstraction for auto subtasks |
| `ConditionEvaluator` | `mltgnt.interfaces.loops` | Host-defined watch condition evaluation |
| `ActionExecutor` | `mltgnt.interfaces.loops` | Deterministic side-effect execution boundary |
| `MemoryAppender` | `mltgnt.interfaces.loops` | Optional host memory append sink |

## Objective Loops

`mltgnt.loops` orchestrates objective progression across deterministic state transitions:

`clarifying -> awaiting_answer -> decomposing -> awaiting_plan_approval (optional) -> executing / awaiting_human / replanning -> evaluating -> done|failed` (plus `paused` and `cancelled`)

Core behavior:

- Request-driven start via `state_dir/requests/*.json`.
- Typed subtask kinds: `auto`, `human`, `watch`, `action`.
- Optional plan approval gate (`ok`, `承認`, `進めて`, `go` as exact approval answers).
- Replanning budgets and watch budgets.
- LLM call budgets per loop/day with pause/resume behavior.

## Architecture

Top-level modules under `src/mltgnt/`:

| Path | Responsibility |
|------|----------------|
| `agent/` | Generic agent loop primitives |
| `bridges/` | ghdag adapters (LLM, files, hooks, DAG boundaries) |
| `chat/` | One-shot chat pipeline |
| `cli/` | `mltgnt` command implementations |
| `config/` | Dataclasses for runtime configs |
| `daemon/` | Component lifecycle runner and PID lock |
| `execution/` | Shared runner interfaces |
| `improvement/` | KPI-driven improvement and rollback cycle |
| `interfaces/` | Protocols and typed DTO contracts |
| `kpi/` | KPI computation and CLI module |
| `loops/` | Objective loop engine and request handling |
| `memory/` | Memory retrieval, compaction, dream summaries |
| `ooda/` | OODA runner |
| `persona/` | Persona loading and validation |
| `routing/` | Channel routing and triage |
| `scheduler/` | Persona scheduling |
| `skill/` | Skill discovery, matching, linting, execution |

## Configuration

### Environment variables

| Variable | Used in | Meaning |
|----------|---------|---------|
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Root path fallback for diary-style substitutions |
| `REPO_ROOT` | `mltgnt.skill.runner` | Repository root fallback for substitutions |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge` | Enables skill I/O type-checking unless set to `"0"` |
| `MLTGNT_AS_OF_DATE` | `mltgnt.improvement.loop` | Default `today` override (`YYYY-MM-DD`) for improvement cycle |

### Config dataclasses

| Dataclass | Module | Role |
|-----------|--------|------|
| `PersonaConfig` | `mltgnt.config` | Persona section weighting defaults |
| `MemoryConfig` | `mltgnt.config` | Memory retention/search/compaction behavior |
| `SchedulerConfig` | `mltgnt.config` | Schedule file path, state path, timezone/salt |
| `ChatConfig` | `mltgnt.config` | Persona directory and matcher model defaults |
| `LoopsConfig` | `mltgnt.config` | Objective loop paths, limits, budgets, host adapters |

## Error Reference

### `mltgnt.exceptions` hierarchy

```text
MltgntError
|- ConfigError
`- DependencyError
```

### Publicly relevant error and exception types

| Type | Module | Notes |
|------|--------|-------|
| `MltgntError` | `mltgnt.exceptions` | Base package exception |
| `ConfigError` | `mltgnt.exceptions` | Invalid configuration / component wiring |
| `DependencyError` | `mltgnt.exceptions` | Missing/blocked external dependency (for example PID lock) |
| `PersonaValidationError` | `mltgnt.persona` | Persona frontmatter validation failure |
| `LlmCallError` | `mltgnt.memory.compaction` | Memory compaction LLM call failure |
| `LlmCallError` | `mltgnt.loops.prompts` | Loop prompt/JSON contract failure |
| `ObjectiveError` | `mltgnt.loops.objective` | Dataclass result for invalid objective files (not raised) |
| `RequestError` | `mltgnt.loops.requests` | Dataclass result for invalid start requests (not raised) |

## Public API Stability

mltgnt is **pre-1.0** (`0.Y.Z`).

- `Y` increments may include breaking changes.
- `Z` increments are intended to be backward compatible.
- The documented public surface is primarily `mltgnt.__all__`, `mltgnt.loops.__all__`, and host contracts in `mltgnt.interfaces.loops`.

## License

MIT (matches `license = "MIT"` in `pyproject.toml`).

- Source: https://github.com/sumipan/mltgnt
- Issues: https://github.com/sumipan/mltgnt/issues
- L0 runtime: https://github.com/sumipan/ghdag
