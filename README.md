# mltgnt

**L1: persona-driven multi-agent orchestration layer.** In the three-tier architecture (**L0 [ghdag](https://github.com/sumipan/ghdag)** / **L1 mltgnt** / **L2 host application**), mltgnt sits in the middle: it handles persona definition, skill matching, memory management, channel routing, and scheduling. LLM inference, file I/O, and DAG submission are all delegated to L0 ghdag via `bridges`.

**Status:** Pre-1.0 (`v0.17.0`)

---

## Not (what this is not)

| Item | Explanation |
|------|-------------|
| Not an LLM library | LLM calls are made via ghdag (L0). mltgnt does not perform model inference itself. |
| Not a general-purpose chatbot framework | Specialized for persona-driven multi-agent orchestration. |
| Not a DAG execution engine | DAG execution is ghdag's responsibility. mltgnt submits jobs via `enqueue_dag` / `enqueue_and_wait`. |
| Not a host application | Slack integration, file storage, and channel-specific logic are implemented by L2 hosts. |

---

## Installation

| Item | Value |
|------|-------|
| Install | `pip install mltgnt` |
| Python | 3.10 or later (`requires-python = ">=3.10"`) |
| Core dependency | [ghdag](https://github.com/sumipan/ghdag) (pinned in `pyproject.toml`), `PyYAML`, `scikit-learn`, `numpy` |

ghdag is a non-replaceable prerequisite. All LLM calls and DAG submissions are routed through ghdag.

---

## Quick Start

Load a persona with `load_persona()` and run a single round-trip with `run_pipeline()`. The following example matches the pattern in `tests/chat/test_pipeline.py`:

```python
from pathlib import Path

from mltgnt import load_persona, run_pipeline

persona_dir = Path("agents")
persona = load_persona("Tachikoma", persona_dir=persona_dir)

output = run_pipeline(
    "Hello",
    persona,
    engine=persona.fm.engine,
    model=persona.fm.model,
)

print(output.content)       # LLM response text
print(output.persona_name)  # "Tachikoma"
```

`run_pipeline` does not raise exceptions on failure; errors are stored in `ChatOutput.content`.

---

## CLI Reference

### `mltgnt run`

Start the daemon process.

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--components MODULE:FUNCTION` | Yes | — | Component factory. Imports the module via `importlib` and calls the callable to obtain a list of `DaemonComponent`. |
| `--pid-file PATH` | No | `/tmp/mltgnt_daemon.pid` | Path to the PID lock file. |

```bash
mltgnt run --components myhost.daemon:build_components --pid-file /tmp/mltgnt.pid
```

| Exit code | Meaning | Exception |
|-----------|---------|-----------|
| 0 | Normal exit (help display or signal shutdown) | — |
| 1 | General error | `MltgntError` or other |
| 2 | Configuration error | `ConfigError` |
| 3 | Dependency error | `DependencyError` |

### `mltgnt memory dream show`

Display the dream summary for a persona.

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `persona` | Yes | Persona name (stem) |
| `--chat-dir PATH` | Yes | Parent path containing persona directories |

```bash
mltgnt memory dream show Tachikoma --chat-dir /path/to/chat
```

| Exit code | Meaning |
|-----------|---------|
| 0 | Normal exit (also 0 when no dream summary exists) |

### `mltgnt memory dream forget`

Remove a category from the dream summary.

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `persona` | Yes | Persona name (stem) |
| `--category NAME` | Yes | Category name to remove |
| `--chat-dir PATH` | Yes | Parent path containing persona directories |

```bash
mltgnt memory dream forget Tachikoma --category "Conversation tendencies" --chat-dir /path/to/chat
```

| Exit code | Meaning |
|-----------|---------|
| 0 | Category removed successfully |
| 1 | Dream summary not found, or category not found |

---

## Public API

Public symbols exported from `mltgnt.__all__`. Import directly from the top-level package:

```python
from mltgnt import run_pipeline, Persona, load_persona
```

| Category | Symbol | Type | Description |
|----------|--------|------|-------------|
| Chat | `run_pipeline` | function | Execute a single round-trip chat pipeline |
| Memory | `read_memory_iterative` | function | Iterative memory retrieval |
| Memory | `read_memory_by_relevance` | function | Relevance-scored memory retrieval |
| Memory | `read_memory_with_sufficiency_check` | function | Memory retrieval with sufficiency gating |
| Memory | `DreamSection` | dataclass | One section of a dream summary |
| Memory | `DreamSummary` | dataclass | Full dream summary for a persona |
| Memory | `read_dream` | function | Read the dream summary from disk |
| Memory | `write_dream` | function | Write the dream summary to disk |
| Persona | `Persona` | class | Persona data object |
| Persona | `load_persona` | function | Load a persona from a Markdown file |
| Persona | `list_personas` | function | List available persona names |
| Persona | `validate_persona` | function | Validate persona frontmatter against available skills |
| Persona | `run_persona_prompt` | function | Run a raw prompt against a named persona |
| Types | `ChatInput` | dataclass | Pipeline input (source, session_key, messages, …) |
| Types | `ChatOutput` | dataclass | Pipeline output (content, persona_name, timestamp, …) |
| Types | `Message` | TypedDict | Chat message with `role` and `content` |
| Types | `PersonaProtocol` | Protocol | L2 host contract for persona objects |
| Agent | `AgentResult` | dataclass | Result object from an agent run |
| Agent | `AgentRunner` | class | LLM + tool execution loop |
| Bridge | `enqueue_dag` | function | Submit a DAG job to ghdag (fire-and-forget) |
| Bridge | `enqueue_and_wait` | function | Submit a DAG job and wait for completion |
| Scheduler | `PersonaScheduler` | class | Scheduled and interval job runner for personas |
| Scheduler | `ScheduleJob` | dataclass | Definition of a single scheduled job |
| Version | `__version__` | str | Package version string |

### Public API Stability

This library is pre-1.0 (`0.Y.Z`). Breaking changes may be introduced in minor or patch releases. The stable surface is limited to symbols listed in `__all__`. Under the SemVer `0.Y.Z` policy, a Y bump indicates a breaking change and a Z bump indicates a backward-compatible change.

---

## Architecture

Module layout under `src/mltgnt/`:

| Module | Responsibility |
|--------|----------------|
| `agent/` | Generic agent loop (LLM + tool execution) |
| `bridges/` | Adapter layer (file I/O, audit, hooks, LLM, ghdag integration) |
| `chat/` | Single round-trip chat pipeline |
| `cli/` | CLI entry points (`mltgnt run` / `mltgnt memory dream`) |
| `config/` | Configuration dataclasses (`MemoryConfig`, `PersonaConfig`, `SchedulerConfig`, `ChatConfig`) |
| `daemon/` | Daemon runner (PID lock, skill watcher) |
| `exceptions.py` | `MltgntError` exception hierarchy |
| `execution/` | Execution infrastructure (`BaseRunner` ABC, `ActDispatcher`) |
| `improvement/` | Improvement analysis, proposal, patch application, and rollback judgement |
| `interfaces/` | Protocol definitions (chat, persona, slack, types, ooda) |
| `kpi/` | KPI aggregation from `audit.jsonl` |
| `memory/` | Memory search, compaction, and dream summary API |
| `ooda/` | OODA loop execution (audit_source, exec_dispatcher, runner) |
| `persona/` | Persona loading, registry, and schema validation |
| `routing/` | Channel routing and triage |
| `scheduler/` | Persona scheduler (scheduled, interval, fuzzy_window, chained) |
| `skill/` | Skill loading, matching, execution, and lint |

---

## Protocols / Extension Points

Extension points that L2 host applications implement. mltgnt uses these Protocols on the caller side.

| Protocol | Module | Responsibility |
|----------|--------|----------------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | `name`, `fm` (frontmatter), `format_prompt(instruction) -> str` — persona contract |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | `run(inp, repo_root) -> ChatOutputBase` — host-side chat pipeline override |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, ...) -> bool` — Slack posting; returns `False` on failure (no exception) |
| `ChatInputBase` | `mltgnt.interfaces.types` | L1 Protocol for chat pipeline input |
| `ChatOutputBase` | `mltgnt.interfaces.types` | L1 Protocol for chat pipeline output |
| `PersonaFMBase` | `mltgnt.interfaces.types` | L1 Protocol for persona frontmatter (`name` required) |

---

## Configuration

### Environment variables

| Variable | Defined in | Purpose |
|----------|------------|---------|
| `SKILL_IO_TYPECHECK` | `bridges/ghdag_bridge.py` | Set to `"0"` to disable skill I/O type checking. Enabled by default (opt-out). |
| `NIKKI_ROOT` | `skill/runner.py` | Root path for nikki (diary/memory). Used for `$NIKKI_ROOT` variable substitution in skill bodies. |
| `REPO_ROOT` | `skill/runner.py` | Repository root fallback. Used for `$REPO_ROOT` variable substitution in skill bodies. |
| `MLTGNT_AS_OF_DATE` | `improvement/loop.py` | Reference date for the improvement cycle (`YYYY-MM-DD`). Defaults to `date.today()` if unset. |

### Configuration dataclasses (`mltgnt.config`)

| Dataclass | Key fields | Purpose |
|-----------|------------|---------|
| `MemoryConfig` | `chat_dir`, `inject_max_bytes`, `compact_threshold_bytes`, `timezone`, `dream_model`, … | Memory JSONL path, thresholds, compaction, and dream settings |
| `PersonaConfig` | `weight_map` | Weighting of persona Markdown sections (`light` / `heavy` / `reference`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir`, `timezone`, `salt` | Schedule YAML and state directory |
| `ChatConfig` | `persona_dir`, `memory_dir`, `matcher_model` | Chat pipeline paths and matcher model |

---

## Error Reference

### MltgntError hierarchy

```
MltgntError (mltgnt.exceptions)  ← base
├── ConfigError (mltgnt.exceptions)
└── DependencyError (mltgnt.exceptions)
```

### All public exception types

| Class | Module | Inherits | Purpose |
|-------|--------|----------|---------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Common base. Catch with `except MltgntError` for unified handling. |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Configuration file read/parse error; malformed `--components` argument. |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | External dependency failure (callable, subprocess, API, PID lock). |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | Invalid persona frontmatter (outside the `MltgntError` hierarchy). |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | LLM call failure during memory compaction (outside the `MltgntError` hierarchy). |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | Skill I/O type mismatch between DAG steps (outside the `MltgntError` hierarchy). |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | Skill load failure or unknown tool reference (outside the `MltgntError` hierarchy). |

---

## License

MIT — SPDX: `MIT` (matches `license = "MIT"` in `pyproject.toml`)
