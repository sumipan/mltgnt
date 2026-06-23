# mltgnt

**L1: persona-driven multi-agent orchestration layer.** mltgnt sits in the middle tier of a three-layer architecture — **L0 [ghdag](https://github.com/sumipan/ghdag)** / **L1 mltgnt** / **L2 host app** — and owns persona loading, skill matching, memory management, channel routing, and scheduling. LLM calls, file I/O, and DAG enqueuing are delegated to L0 ghdag via the `bridges` package.

**Status:** Pre-1.0 (`v0.16.1`)

---

## Not

mltgnt is not:

| What | Why |
|------|-----|
| An LLM library | LLM calls go through ghdag (L0); mltgnt does not perform model inference itself |
| A general-purpose chatbot framework | It is purpose-built for persona-driven multi-agent orchestration |
| A DAG execution engine | DAG execution is ghdag's responsibility; mltgnt enqueues work via `enqueue_dag` / `enqueue_and_wait` |
| A host application | Slack integration, file persistence, and channel-specific logic belong to the L2 host |

---

## Installation

```bash
pip install mltgnt
```

| Requirement | Value |
|-------------|-------|
| Python | ≥ 3.10 |
| Core dependencies | [ghdag](https://github.com/sumipan/ghdag) v0.30.4, PyYAML ≥ 6.0, scikit-learn ≥ 1.0, numpy ≥ 1.21 |

ghdag is a non-optional prerequisite. All LLM calls and DAG submissions flow through it.

---

## Quick Start

Load a persona with `load_persona()`, then run a single-turn exchange with `run_pipeline()`.
The pattern below matches the test structure in `tests/chat/test_pipeline.py`.

```python
from pathlib import Path
from mltgnt import load_persona, run_pipeline

persona = load_persona("my-persona", persona_dir=Path("agents"))

output = run_pipeline(
    "Hello!",
    persona,
    engine=persona.fm.engine,
    model=persona.fm.model,
)

print(output.content)       # LLM response text
print(output.persona_name)  # "my-persona"
```

`run_pipeline` does not raise on LLM errors; failures are surfaced in `output.content` as an error string.

---

## CLI Reference

### `mltgnt run`

Start the mltgnt daemon process.

```bash
mltgnt run --components myhost.daemon:build_components [--pid-file /tmp/mltgnt_daemon.pid]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--components MODULE:FUNCTION` | Yes | — | Component factory spec loaded via `importlib`; the callable must return a list of `DaemonComponent` |
| `--pid-file PATH` | No | `/tmp/mltgnt_daemon.pid` | Path to the PID lock file |

**Exit codes**

| Code | Meaning | Exception |
|------|---------|-----------|
| 0 | Normal exit (including help display and signal-based shutdown) | — |
| 1 | General error | `MltgntError` and other unhandled exceptions |
| 2 | Configuration error | `ConfigError` |
| 3 | Dependency error | `DependencyError` |

### `mltgnt memory dream show`

Print the dream summary for a persona.

```bash
mltgnt memory dream show PERSONA --chat-dir DIR
```

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `persona` | Yes | Persona name or stem |
| `--chat-dir PATH` | Yes | Parent path of persona directories |

Exit code is always `0` (including when no dream summary exists).

### `mltgnt memory dream forget`

Remove a category from a persona's dream summary.

```bash
mltgnt memory dream forget PERSONA --category CATEGORY --chat-dir DIR
```

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `persona` | Yes | Persona name or stem |
| `--category NAME` | Yes | Category name to remove |
| `--chat-dir PATH` | Yes | Parent path of persona directories |

| Exit code | Meaning |
|-----------|---------|
| 0 | Category removed successfully |
| 1 | Dream summary not found, or category not present |

---

## Public API

All symbols exported from `mltgnt.__all__` (26 total). Import directly from the top-level package:

```python
from mltgnt import run_pipeline, load_persona, Persona
```

| Category | Symbol | Type | Notes |
|----------|--------|------|-------|
| Chat | `run_pipeline` | function | Single-turn LLM pipeline |
| Memory | `read_memory_iterative` | function | Iterative memory retrieval |
| Memory | `read_memory_by_relevance` | function | TF-IDF relevance search |
| Memory | `read_memory_with_sufficiency_check` | function | Retrieval with sufficiency guard |
| Memory | `compact` | function | **Deprecated** — use `PersonaScheduler` dream action |
| Memory | `needs_compaction` | function | **Deprecated** — use `PersonaScheduler` dream action |
| Memory | `DreamSection` | dataclass | One section of a dream summary |
| Memory | `DreamSummary` | dataclass | Full dream summary for a persona |
| Memory | `read_dream` | function | Read dream summary from disk |
| Memory | `write_dream` | function | Write dream summary to disk |
| Persona | `Persona` | dataclass | Loaded persona (body + frontmatter) |
| Persona | `load_persona` | function | Load a persona by name from a directory |
| Persona | `list_personas` | function | List available persona names |
| Persona | `validate_persona` | function | Validate persona frontmatter |
| Persona | `run_persona_prompt` | function | Run a raw prompt through a persona |
| Types | `ChatInput` | dataclass | Pipeline input (source, session_key, messages, persona_name) |
| Types | `ChatOutput` | dataclass | Pipeline output (content, persona_name, timestamp, session_key) |
| Types | `Message` | TypedDict | `{role: str, content: str}` chat message |
| Types | `PersonaProtocol` | Protocol | L1 contract for persona objects |
| Agent | `AgentResult` | dataclass | Result of a single agent loop tick |
| Agent | `AgentRunner` | class | Generic agent loop (LLM + tool execution) |
| Bridge | `enqueue_dag` | function | Enqueue a DAG order to ghdag |
| Bridge | `enqueue_and_wait` | function | Enqueue a DAG order and block until result |
| Scheduler | `PersonaScheduler` | class | Cron-style job scheduler for personas |
| Scheduler | `ScheduleJob` | dataclass | A single scheduler job entry |
| Version | `__version__` | str | Installed package version string |

### Public API Stability

mltgnt follows pre-1.0 SemVer (`0.Y.Z`):

- **Y bump** — breaking change: public API removed, renamed, or signature changed
- **Z bump** — backward-compatible: bug fixes, new opt-in parameters, new symbols added to `__all__`

Only symbols listed in `__all__` are covered by this stability policy.

---

## Deprecated API

The following symbols remain in `__all__` for backward compatibility but should not be used in new code.

| Symbol | Deprecated since | Replacement |
|--------|-----------------|-------------|
| `compact` | v0.15.x | `PersonaScheduler` dream action (scheduler-driven memory compaction) |
| `needs_compaction` | v0.15.x | `PersonaScheduler` dream action (scheduler-driven compaction check) |

These will be removed in a future version.

---

## Architecture

All packages live under `src/mltgnt/`.

| Module | Responsibility |
|--------|---------------|
| `agent/` | Generic agent loop — `AgentRunner` drives LLM + tool execution cycles; `AgentResult` carries tick output |
| `bridges/` | Adapter layer — `ghdag_bridge` (DAG enqueue/wait), `audit_adapter` (orchestration audit), `files_adapter` (file read/write), `hooks_adapter` (lifecycle hooks), `llm_adapter` (LLM call shim) |
| `chat/` | Single round-trip chat pipeline — `run_pipeline` formats prompt, calls LLM via bridges, records audit |
| `cli/` | CLI entry point — `mltgnt run` daemon launcher, `mltgnt memory dream show/forget` subcommands |
| `config/` | Configuration dataclasses — `MemoryConfig`, `PersonaConfig`, `SchedulerConfig`, `ChatConfig` |
| `daemon/` | Daemon lifecycle — PID lock, signal handling, component start/stop coordination |
| `exceptions.py` | Shared error hierarchy — `MltgntError`, `ConfigError`, `DependencyError` |
| `execution/` | Runner base ABC — `BaseRunner.tick()` defines the common interface for all tick-based runners *(new in v0.16.0)* |
| `improvement/` | Failure analysis cycle — analyzes audit logs, generates improvement proposals, applies patches |
| `interfaces/` | L1 Protocol and DTO definitions — `chat.py`, `persona.py`, `slack.py`, `types.py`, `ooda.py`, `dispatch.py` |
| `kpi/` | KPI aggregation — parses `audit.jsonl` for response failure rate, re-question rate |
| `memory/` | Memory API — `read_memory_iterative`, `read_memory_by_relevance`, `read_memory_with_sufficiency_check`; dream summary synthesis in `memory/dream/` |
| `ooda/` | OODA loop runtime — `OODARunner` drives observe → orient → decide → act cycles |
| `persona/` | Persona management — YAML frontmatter loading, alias registry, schema validation |
| `routing/` | Channel routing — maps channels to personas; triage via TF-IDF classifier |
| `scheduler/` | Persona scheduler — `scheduled`, `interval`, `fuzzy_window`, `chained` job modes; `side_effect_audit` hook; interval state persistence |
| `skill/` | Skill system — YAML skill loading, semantic matching, execution, lint |

---

## Protocols / Extension Points

These Protocols define the L2 host implementation contracts. mltgnt calls into them; the L2 host provides the concrete implementations.

| Protocol | Module | Key interface |
|----------|--------|---------------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | `name: str`, `fm: PersonaFMBase`, `format_prompt(instruction: str) -> str` |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | `run(inp: ChatInputBase, repo_root: Path) -> ChatOutputBase` |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, thread_ts, blocks, reply_broadcast) -> bool` — returns `False` on failure, never raises |
| `ObserveSource` | `mltgnt.interfaces.ooda` | `observe(*, since: str | None) -> list[ObservationEvent]` — returns events since the given ISO 8601 timestamp |
| `ActDispatcher` | `mltgnt.interfaces.dispatch` | `dispatch(action: str, args: dict) -> ActResult` — executes an OODA action *(moved to `interfaces/dispatch.py` in v0.16.0)* |
| `PersonaFMBase` | `mltgnt.interfaces.types` | `name: str` — base Protocol for persona frontmatter; hosts extend with engine, model, etc. |
| `ChatInputBase` | `mltgnt.interfaces.types` | `source`, `session_key`, `messages`, `persona_name` — L1 input contract |
| `ChatOutputBase` | `mltgnt.interfaces.types` | `content`, `persona_name`, `timestamp`, `session_key` — L1 output contract |

Also available as data companions for the OODA Protocols:

| Type | Module | Description |
|------|--------|-------------|
| `ActResult` | `mltgnt.interfaces.dispatch` | `action: str`, `success: bool`, `detail: str` — result of a dispatched action |
| `ObservationEvent` | `mltgnt.interfaces.ooda` | `event_id`, `event_type`, `status`, `timestamp`, `payload` — single observed event |
| `OODAConfig` | `mltgnt.interfaces.ooda` | `max_recovery_attempts`, `escalate_after`, `observe_filter` |
| `OODATickResult` | `mltgnt.interfaces.ooda` | `observed_events`, `actions_taken`, `escalated` |

---

## Configuration

### Environment variables

| Variable | Defined in | Purpose |
|----------|-----------|---------|
| `SKILL_IO_TYPECHECK` | `bridges/ghdag_bridge.py` | Set to `"0"` to disable skill I/O type checking. Enabled by default (opt-out). |
| `NIKKI_ROOT` | `skill/runner.py` | Diary/memory root path; used for `$NIKKI_ROOT` variable substitution in skill bodies |
| `REPO_ROOT` | `skill/runner.py` | Repository root fallback; used for `$REPO_ROOT` variable substitution in skill bodies |
| `MLTGNT_AS_OF_DATE` | `improvement/loop.py` | Analysis period end date in ISO 8601 format (`YYYY-MM-DD`). Defaults to `date.today()`. |

### Configuration dataclasses (`mltgnt.config`)

| Dataclass | Key fields | Purpose |
|-----------|-----------|---------|
| `MemoryConfig` | `chat_dir`, `inject_max_bytes` (10240), `inject_max_entries` (12), `preferences_max_bytes` (5120), `compact_threshold_bytes` (40960), `compact_target_bytes` (25600), `timezone` ("Asia/Tokyo"), `dream_model`, `use_dream_summary`, `dream_dir_name`, `raw_days` (7), `mid_weeks` (3) | Memory JSONL paths, thresholds, compaction, dream settings |
| `PersonaConfig` | `weight_map` | Section weight map for persona Markdown (`light` / `heavy` / `reference`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir`, `timezone` ("Asia/Tokyo"), `salt` | Schedule YAML path and job state directory |
| `ChatConfig` | `persona_dir`, `memory_dir`, `matcher_model` ("claude-haiku-4-5-20251001") | Chat pipeline paths and matcher model |

---

## Error Reference

### MltgntError hierarchy

```
Exception
└── MltgntError          (mltgnt.exceptions)  — common base; catch with `except MltgntError`
    ├── ConfigError       (mltgnt.exceptions)  — config file read/parse failure
    └── DependencyError   (mltgnt.exceptions)  — external dependency call failure
```

### All public exception types

| Class | Module | Base | Raised when |
|-------|--------|------|-------------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base class for all mltgnt-hierarchy errors |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Config YAML is malformed or `--components` spec is invalid |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | External callable, subprocess, API, or PID lock fails |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | Persona frontmatter is invalid (outside MltgntError hierarchy) |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | LLM call during memory compaction fails (outside MltgntError hierarchy) |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | Skill I/O type mismatch between DAG steps (outside MltgntError hierarchy) |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | Skill file fails to load or references an unknown tool (outside MltgntError hierarchy) |

---

## License

MIT — SPDX identifier: `MIT`

Matches `license = "MIT"` in `pyproject.toml`.
