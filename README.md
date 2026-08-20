# mltgnt

**L1: persona, memory, skill, routing, scheduler, and Objective-loop orchestration.** In the three-tier stack (**L0 [ghdag](https://github.com/sumipan/ghdag)** / **L1 mltgnt** / **L2 host**), mltgnt owns agent contracts and loops; LLM inference, file I/O, and DAG execution stay in ghdag.

**Status:** Pre-1.0 (`v0.18.0`)

---

## Not (what this is not)

| Item | Explanation |
|------|-------------|
| Not an LLM SDK | Inference goes through ghdag (`mltgnt.bridges.llm_adapter.call_llm`). mltgnt does not talk to model APIs itself. |
| Not a DAG engine | Job submission and completion markers are ghdag's. mltgnt calls `enqueue_dag` / `enqueue_and_wait` / `enqueue_step` / `poll_step`. |
| Not a Slack (or other) host | Channel posting, thread IDs, and daemon wiring belong to the L2 host. `HumanChannel` and `SubtaskExecutor` are Protocols the host implements. |

v0.18.0 keeps existing scheduler, chat, and OODA APIs backward compatible (see `CHANGELOG.md` for v0.18.0).

---

## Installation

| Item | Value |
|------|-------|
| Install | `pip install mltgnt` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, [ghdag](https://github.com/sumipan/ghdag) `v0.30.12` (pinned in `pyproject.toml`) |
| Console script | `mltgnt` → `mltgnt.cli.main:main` |
| License | MIT (`license = "MIT"` in `pyproject.toml`) |

ghdag is required. LLM calls and DAG/step enqueue go through it.

---

## Quick Start

**Prerequisites:** Python 3.10+, `mltgnt` installed (pulls in ghdag and PyYAML). No Slack host, daemon, or live LLM is required for the examples below.

These snippets follow `tests/loops/test_objective.py` and `tests/chat/test_pipeline.py` (persona Markdown + `ops.engine` / `ops.model`).

### Parse an Objective Markdown file

`parse_objective` returns `Objective` on success or `ObjectiveError` (a dataclass, not an exception) on invalid input.

```python
from pathlib import Path
import tempfile

from mltgnt.loops.objective import Objective, ObjectiveError, parse_objective

text = """\
---
id: hp-renewal
title: Renew the homepage
agent: operator
max_iterations: 5
status: active
---

Ship a clearer homepage for the public docs.
"""

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "hp-renewal.md"
    path.write_text(text, encoding="utf-8")
    result = parse_objective(
        path,
        default_persona="operator",
        default_max_iterations=5,
    )
    assert isinstance(result, Objective)
    print(result.loop_id, result.title, result.agent, result.max_iterations, result.status)

    bad = parse_objective(
        path,
        default_persona="operator",
        default_max_iterations=5,
        known_ids={"hp-renewal"},
    )
    assert isinstance(bad, ObjectiveError)
```

### Load a persona

```python
from pathlib import Path
import tempfile

from mltgnt import load_persona

persona_md = """\
---
persona:
  name: Tachikoma
ops:
  engine: claude
  model: claude-sonnet-4-6
---

## Background

A curious multi-legged AI tank.

## Values

Ask questions.

## Tone

Friendly.

## Output format

Keep it short.
"""

with tempfile.TemporaryDirectory() as tmp:
    persona_dir = Path(tmp)
    (persona_dir / "Tachikoma.md").write_text(persona_md, encoding="utf-8")
    persona = load_persona("Tachikoma", persona_dir=persona_dir)
    print(persona.name)
    print(persona.fm.engine, persona.fm.model)
```

`run_pipeline(prompt, persona, engine=..., model=...)` (same fixture pattern as `tests/chat/test_pipeline.py`) returns `ChatOutput`. A live ghdag LLM engine is required; failures are stored in `ChatOutput.content` and are not raised.

---

## CLI Reference

Console script: `mltgnt` (`mltgnt.cli.main:main`). Also `python -m mltgnt` (`mltgnt.__main__` → the same `main()`).

### `mltgnt run`

Start the daemon (`mltgnt.cli.run.execute` → `DaemonRunner`).

| Argument / option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `--components MODULE:FUNCTION` | Yes | — | Import `MODULE` and call `FUNCTION()` to get a `list` of `DaemonComponent`. |
| `--pid-file PATH` | No | `/tmp/mltgnt_daemon.pid` | PID lock file. |
| `-h, --help` | No | — | Help; process exits `0`. |

```bash
mltgnt run --help
```

Operational form (host must provide `MODULE:FUNCTION` returning `list[DaemonComponent]`):

`mltgnt run --components myhost.daemon:build_components --pid-file /tmp/mltgnt.pid`

| Exit code | Meaning |
|-----------|---------|
| 0 | Help, or clean shutdown after SIGINT/SIGTERM |
| 1 | `MltgntError` (other than Config/Dependency) or unexpected error during component start |
| 2 | `ConfigError` (bad `--components` shape, missing module, missing/non-callable factory) |
| 3 | `DependencyError` (PID lock held: another instance is running) |

Argparse itself exits **2** when required flags are missing (for example `mltgnt run` with no `--components`).

### `mltgnt memory dream show`

Print dream-summary sections for a persona (`read_dream(chat_dir / persona)`).

| Argument / option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `persona` | Yes | — | Persona name / directory stem |
| `--chat-dir PATH` | Yes | — | Parent of persona directories |
| `-h, --help` | No | — | Help |

```bash
mltgnt memory dream show --help
```

Operational form: `mltgnt memory dream show Tachikoma --chat-dir /path/to/chat`

| Exit code | Meaning |
|-----------|---------|
| 0 | Printed sections, or no dream file (`No dream summary found for …`) |
| 2 | Argparse error (missing `persona` or `--chat-dir`) |

### `mltgnt memory dream forget`

Drop one category from the dream summary and rewrite `dream.json`.

| Argument / option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `persona` | Yes | — | Persona name / directory stem |
| `--category NAME` | Yes | — | Section `category` to remove |
| `--chat-dir PATH` | Yes | — | Parent of persona directories |
| `-h, --help` | No | — | Help |

```bash
mltgnt memory dream forget --help
```

Operational form: `mltgnt memory dream forget Tachikoma --category "Conversation tendencies" --chat-dir /path/to/chat`

| Exit code | Meaning |
|-----------|---------|
| 0 | Category removed |
| 1 | No dream summary, or category not found |
| 2 | Argparse error (missing `persona`, `--category`, or `--chat-dir`) |

### `python -m mltgnt.kpi`

`prog` is `mltgnt.kpi`. Reads `audit.jsonl` via `compute_kpis`.

| Argument / option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `audit_path` | Yes | — | Path to `audit.jsonl` |
| `--since YYYY-MM-DD` | No | unset | Include events on/after this date |
| `--until YYYY-MM-DD` | No | unset | Include events on/before this date |
| `--format {text,json}` | No | `text` | Output format |
| `-h, --help` | No | — | Help |

| Exit code | Meaning |
|-----------|---------|
| 0 | Report written to stdout |
| 1 | `FileNotFoundError` (`audit file not found: …`) |
| 2 | Argparse error (missing `audit_path`, invalid date, or invalid `--format`) |

### `python -m mltgnt.improvement`

Runs `run_improvement_cycle` and prints `format_cycle_report`.

| Argument / option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `--audit PATH` | Yes | — | Audit log path |
| `--persona-dir PATH` | Yes | — | Persona Markdown directory |
| `--skills-dir PATH` | Yes | — | Skills directory |
| `--since INT` | No | `7` | Lookback days (`since_days`) |
| `--today YYYY-MM-DD` | No | unset (`None`) | Period end; if omitted, `MLTGNT_AS_OF_DATE` or `date.today()` |
| `-h, --help` | No | — | Help |

| Exit code | Meaning |
|-----------|---------|
| 0 | Report printed |
| 1 | `FileNotFoundError` from the cycle |
| 2 | Argparse error (missing required flags or invalid `--since` / `--today`) |

---

## Public API

Stable import surface is `mltgnt.__all__` plus the `mltgnt.loops` package exports below. Other modules exist for hosts and internals; they are not implied stable unless listed.

```python
from mltgnt import run_pipeline, Persona, load_persona
from mltgnt.loops import LoopsComponent, parse_objective
```

### `mltgnt.__all__`

| Symbol | Kind | Import path | Inputs | Returns / meaning |
|--------|------|-------------|----------------|-------------------|
| `run_pipeline` | function | `mltgnt` / `mltgnt.chat.pipeline` | `prompt: str`, `persona: PersonaProtocol`, `engine=""`, `model=""`, `timeout=300`, `memory=None`, `orchestration_ctx=None`, `audit_path=None` | `ChatOutput`. On LLM failure, `content` holds the error string; no exception. |
| `read_memory_iterative` | function | `mltgnt` / `mltgnt.memory.search` | `MemoryConfig`, `persona_stem`, `query`, `max_bytes`, `max_entries`, `llm_call`, optional `skill_paths`, `max_iterations=3` | Memory excerpt `str` after iterative sufficiency search. |
| `read_memory_by_relevance` | function | `mltgnt` / `mltgnt.memory.search` | `MemoryConfig`, `persona_stem`, `query`, `max_bytes`, `max_entries`, optional `layers` | Relevance-ranked memory excerpt `str`. |
| `read_memory_with_sufficiency_check` | function | `mltgnt` / `mltgnt.memory.search` | `MemoryConfig`, `persona_stem`, `query`, `max_bytes`, `max_entries`, optional `llm_call` | Excerpt `str`; without `llm_call` delegates to `read_memory_by_relevance`. |
| `DreamSection` | dataclass | `mltgnt` / `mltgnt.memory.dream` | `category`, `content`, `source_entries` | One dream-summary section. |
| `DreamSummary` | dataclass | `mltgnt` / `mltgnt.memory.dream` | `persona`, `sections`, `updated_at` | Full dream summary. |
| `read_dream` | function | `mltgnt` / `mltgnt.memory.dream.api` | `persona_dir: Path`, `memory_dir_name="memory"` | `DreamSummary \| None` (`None` if missing or unreadable). |
| `write_dream` | function | `mltgnt` / `mltgnt.memory.dream.api` | `persona_dir`, `summary: DreamSummary`, `memory_dir_name="memory"` | Writes `persona_dir/memory/dream.json`. |
| `Persona` | class | `mltgnt` / `mltgnt.persona.loader` | Loaded from Markdown | Persona body + frontmatter (`fm`). |
| `load_persona` | function | `mltgnt` / `mltgnt.persona` | `name`, optional `persona_dir` (default `./agents/`), `config` | `Persona`. Raises `FileNotFoundError` or `PersonaValidationError`. |
| `list_personas` | function | `mltgnt` / `mltgnt.persona` | optional `persona_dir` | `list[str]` of persona names. |
| `validate_persona` | function | `mltgnt` / `mltgnt.persona` | persona + `available_skills` | `list[str]` of validation messages. |
| `run_persona_prompt` | function | `mltgnt` / `mltgnt.persona.runner` | persona name, prompt, `persona_dir`, LLM kwargs | `str` model output. |
| `ChatInput` | dataclass | `mltgnt` / `mltgnt.interfaces.types` | `source`, `session_key`, `messages`, optional persona/model/context | Pipeline input DTO. |
| `ChatOutput` | dataclass | `mltgnt` / `mltgnt.interfaces.types` | `content`, `persona_name`, `timestamp`, `session_key` | Pipeline output DTO. |
| `Message` | TypedDict | `mltgnt` / `mltgnt.interfaces.types` | `role: str`, `content: str` | Chat message. |
| `PersonaProtocol` | Protocol | `mltgnt` / `mltgnt.interfaces.persona` | `name`, `fm`, `format_prompt(instruction) -> str` | Host/persona structural contract. |
| `AgentResult` | dataclass | `mltgnt` / `mltgnt.agent` | `tool`, `args`, `raw_response`, optional `tool_trace`, `reflexion_count` | One agent-loop result. |
| `AgentRunner` | class | `mltgnt` / `mltgnt.agent` | `llm_call`, `tool_executor`, `terminal_tools`, optional iteration/retry/audit hooks | LLM + tool loop. |
| `enqueue_dag` | function | `mltgnt` / `mltgnt.bridges.ghdag_bridge` | `steps`, `timeout`, `idempotency_key`, `jobs_dir`, `exec_done_dir`, optional persona/skills/permission | `list[tuple[bool, str]]` per step (success flag + content or error). **Blocks** until done or timeout. |
| `enqueue_and_wait` | function | `mltgnt` / `mltgnt.bridges.ghdag_bridge` | `prompt`, `engine`, `model`, `timeout`, `idempotency_key`, `jobs_dir`, `exec_done_dir`, optional persona/permission | `tuple[bool, str]`. **Blocks**. |
| `PersonaScheduler` | class | `mltgnt` / `mltgnt.scheduler` | schedule jobs + host callbacks | Interval / scheduled / fuzzy / chained persona jobs. |
| `ScheduleJob` | dataclass | `mltgnt` / `mltgnt.scheduler.models` | `id`, `mode`, `action`, `notify`, plus schedule fields | One scheduler job. |
| `__version__` | str | `mltgnt` | — | Installed package version (`importlib.metadata`), or `"0.0.0"` if not installed. |

### `mltgnt.loops.__all__`

| Symbol | Kind | Import path | Inputs | Returns / meaning |
|--------|------|-------------|----------------|-------------------|
| `LoopsComponent` | class | `mltgnt.loops.component` | `LoopsConfig`, `HumanChannel`, `SubtaskExecutor` | `DaemonComponent`: `name == "loops"`, `start()` / `stop()`. Polls `objectives_dir` every `poll_interval_sec`. |
| `LoopsEngine` | class (`BaseRunner`) | `mltgnt.loops.engine` | `LoopsConfig`, `HumanChannel`, `SubtaskExecutor`, optional objective callbacks | One tick = at most one state transition per loop. |
| `GhdagSubtaskExecutor` | class | `mltgnt.loops.executor` | `jobs_dir`, `exec_done_dir`, `engine`, `model`, optional `correlation_id` | Host-facing `SubtaskExecutor`: `submit` / `poll`. |
| `Objective` | dataclass | `mltgnt.loops.objective` | parsed fields | Valid Objective: `loop_id`, `title`, `body`, `agent`, `max_iterations`, `status`, `path`, `content_hash`. |
| `ObjectiveError` | dataclass | `mltgnt.loops.objective` | `loop_id`, `message`, `path` | Parse/validation failure. **Not** an `Exception`. |
| `parse_objective` | function | `mltgnt.loops.objective` | `path: Path`, `default_persona`, `default_max_iterations`, optional `known_ids` | `Objective \| ObjectiveError`. |
| `list_objective_files` | function | `mltgnt.loops.objective` | `objectives_dir: Path` | Non-recursive `*.md` paths, sorted. Missing dir → `[]`. |
| `LoopState` | dataclass | `mltgnt.loops.models` | loop identity, `status: LoopStatus`, iteration, persona, thread, subtasks, … | Persisted loop state (`schema_version = 1`). |
| `Subtask` | dataclass | `mltgnt.loops.models` | `id`, `title`, `kind` (`auto` \| `human`), `prompt`, `status`, `result`, optional `submission` | One decomposed step. |
| `TERMINAL_STATUSES` | `frozenset[str]` | `mltgnt.loops.models` | — | `{"done", "failed", "cancelled"}`. |

---

## Objective Loops

`mltgnt.loops` drives an Objective Markdown file through **clarify → decompose → execute → evaluate**. The host injects channel and executor implementations; mltgnt does not implement Slack.

### Host → `LoopsComponent`

| Dependency | Role |
|------------|------|
| `LoopsConfig` | Paths, persona default, poll interval, iteration/clarify/subtask limits, timeouts, LLM/subtask engine+model |
| `HumanChannel` | Open/ask/notify/close human threads |
| `SubtaskExecutor` | Non-blocking auto-subtask submit + poll |

`LoopsComponent.start()` launches a daemon thread; `stop()` joins it. Each watch cycle refreshes Objective snapshots and calls `LoopsEngine.tick()`.

### Objective file → `parse_objective`

YAML keys allowed: `id`, `title`, `agent`, `max_iterations`, `status`. Other keys are ignored with a warning.

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `id` | str | file stem | `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`; unique among loaded files |
| `title` | str | first non-empty body line, else `id` | Must be a string if present |
| `agent` | str | `LoopsConfig.default_persona` | Persona stem (`{persona_dir}/{agent}.md`) |
| `max_iterations` | int | `LoopsConfig.max_iterations` | `1..10`; bool is rejected |
| `status` | str | `active` | `active` or `cancelled` |

Body must be non-empty. `cancelled` files are not started. Duplicate `id` values become `ObjectiveError` and a failed status Markdown file.

### State machine

`LoopStatus` (`mltgnt.interfaces.loops`): `clarifying` → `awaiting_answer` (if a clarify question is asked) → `decomposing` → `executing` / `awaiting_human` → `evaluating` → `done` \| `failed`, or `cancelled` from inbox/`status: cancelled`/file removal.

| Limit | Source | Range / value |
|-------|--------|----------------|
| Poll interval | `LoopsConfig.poll_interval_sec` | `> 0`, default `10.0` |
| Iterations | Objective / `LoopsConfig.max_iterations` | `1..10`, default `5` |
| Clarify rounds | `LoopsConfig.max_clarify_rounds` | `1..3`, default `3` |
| Subtasks per iteration | `LoopsConfig.max_subtasks_per_iteration` | `1..5`, default `5` |
| Subtask timeout | `LoopsConfig.subtask_timeout_sec` | `> 0`, default `1800.0` (30 minutes) |
| Consecutive engine errors | `mltgnt.loops.engine._MAX_CONSECUTIVE_ERRORS` | `3` then `failed` |

Status Markdown is written to `<status_dir>/<loop_id>.md`.

### `HumanChannel` (host Protocol)

| Method | Returns |
|--------|---------|
| `open_thread(*, loop_id, persona, title, body, event_id)` | `HumanThreadRef \| None` (`channel_id`, `thread_ts`) |
| `ask(*, loop_id, persona, thread, question_id, text, event_id)` | `bool` |
| `notify(*, loop_id, persona, thread, text, event_id)` | `bool` |
| `notify_fallback(*, loop_id, text, event_id)` | `bool` |
| `close_thread(*, loop_id, persona, thread, event_id)` | `bool` |

### `SubtaskExecutor` and ghdag step boundary

Protocol:

| Method | Returns |
|--------|---------|
| `submit(*, prompt: str, idempotency_key: str)` | `StepSubmission` |
| `poll(*, uuid: str, result_filename: str)` | `StepPoll` |

`GhdagSubtaskExecutor` maps those onto:

```text
enqueue_step(*, prompt, engine, model, idempotency_key, jobs_dir, correlation_id=None, order_builder=None) -> StepSubmission
poll_step(*, exec_done_dir, jobs_dir, uuid, result_filename) -> StepPoll
```

`StepSubmission`: `uuid`, `result_filename`, `submitted_at`, `reused` (True if the idempotency key already had an exec record).

`StepPoll.status`: `pending` \| `success` \| `failed_exit` \| `rejected` \| `empty_result` \| `other`. `poll_step` does **not** wait; missing done markers yield `pending`.

`enqueue_dag` / `enqueue_and_wait` remain the **blocking** DAG helpers on the package root.

---

## Architecture

Layout of `src/mltgnt/` (packages and modules, including `loops/` added in v0.18.0):

| Path | Responsibility |
|------|----------------|
| `__init__.py` | Package root exports (`__all__`, `__version__`) |
| `__main__.py` | `python -m mltgnt` → `cli.main.main` |
| `py.typed` | PEP 561 marker |
| `agent/` | Generic LLM + tool agent loop (`AgentRunner`) |
| `bridges/` | ghdag adapters: files, LLM, audit, hooks, DAG/step enqueue |
| `chat/` | Single round-trip `run_pipeline` |
| `cli/` | `mltgnt run` and `mltgnt memory dream` |
| `config/` | `MemoryConfig`, `PersonaConfig`, `SchedulerConfig`, `ChatConfig`, `LoopsConfig` |
| `daemon/` | `DaemonRunner`, `PidLock`, `DaemonComponent`, skill watcher |
| `exceptions.py` | `MltgntError` / `ConfigError` / `DependencyError` |
| `execution/` | `BaseRunner` ABC (`LoopsEngine` subclasses it) |
| `improvement/` | Failure analysis, proposals, patch/rollback, `python -m mltgnt.improvement` |
| `interfaces/` | Protocols and DTOs (chat, persona, Slack, OODA, loops host types) |
| `kpi/` | `compute_kpis` / `python -m mltgnt.kpi` |
| `loops/` | Objective-driven clarify/decompose/execute/evaluate |
| `memory/` | JSONL memory search, compaction, dream summaries |
| `ooda/` | OODA runner (`OODARunner`, `OODAConfig`) |
| `persona/` | Load, validate, and prompt personas |
| `routing/` | Channel routing and triage |
| `scheduler/` | `PersonaScheduler` job runner |
| `skill/` | Skill discover/match/run/lint |

---

## Protocols / host extension points

| Protocol | Module | Contract |
|----------|--------|----------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | `name`, `fm`, `format_prompt(instruction) -> str` |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | `run(inp, repo_root) -> ChatOutputBase` |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(...) -> bool` (False on failure, no raise) |
| `ChatInputBase` / `ChatOutputBase` / `PersonaFMBase` | `mltgnt.interfaces.types` | Structural chat/persona DTOs |
| `HumanChannel` / `SubtaskExecutor` | `mltgnt.interfaces.loops` | Objective-loop host boundary (see above) |
| `DaemonComponent` | `mltgnt.daemon` | `name`, `start()`, `stop()` — includes `LoopsComponent` |

`mltgnt.interfaces.__all__` does not export the loops Protocols; import them from `mltgnt.interfaces.loops`.

---

## Configuration

### Environment variables

Every `os.environ` / `os.getenv` use in `src/mltgnt/**/*.py`:

| Variable | Defined in | Purpose |
|----------|------------|---------|
| `SKILL_IO_TYPECHECK` | `bridges/ghdag_bridge.py` | Skill I/O typecheck is on unless this is `"0"`. |
| `NIKKI_ROOT` | `skill/runner.py` | Diary/memory root for `$NIKKI_ROOT` substitution in skill bodies. |
| `REPO_ROOT` | `skill/runner.py` | Repo-root fallback for `$REPO_ROOT` substitution. |
| `MLTGNT_AS_OF_DATE` | `improvement/loop.py` | `YYYY-MM-DD` period end for `run_improvement_cycle` when `today` is omitted. |

### Config dataclasses

| Dataclass | Module | Fields (type, default, constraints) |
|-----------|--------|-------------------------------------|
| `PersonaConfig` | `mltgnt.config` | `weight_map: dict[str, str]` default `DEFAULT_WEIGHT_MAP` (`light` / `heavy` / `reference` section weights). |
| `MemoryConfig` | `mltgnt.config` | `chat_dir: Path` (required); `chat_memory_dir: Path \| None = None`; `inject_max_bytes=10240`; `inject_max_entries=12`; `preferences_max_bytes=5120`; `lock_timeout_sec=30.0`; `lock_stale_threshold_sec=300.0`; `raw_days=7`; `mid_weeks=3`; `compact_threshold_bytes=40960`; `compact_target_bytes=25600`; `preferences_section_name="ユーザーの好み・傾向"`; `protected_layers=("caveat",)`; `timezone="Asia/Tokyo"`; `dream_model="claude-haiku-4-5-20251001"`; `use_dream_summary=False`; `dream_dir_name="memory"`; `global_dream_exclude_personas=()`. |
| `SchedulerConfig` | `mltgnt.config` | `schedule_yaml: Path`; `state_dir: Path`; `timezone="Asia/Tokyo"`; `salt=""`. |
| `ChatConfig` | `mltgnt.config` | `persona_dir: Path`; `memory_dir: Path \| None = None`; `matcher_model="claude-haiku-4-5-20251001"`. |
| `LoopsConfig` | `mltgnt.config` | Required: `objectives_dir`, `state_dir`, `status_dir`, `jobs_dir`, `exec_done_dir`, `persona_dir`, `default_persona`, `fallback_channel` (all paths except the two strings). Optional: `poll_interval_sec=10.0` (`> 0`); `max_iterations=5` (`1..10`); `max_clarify_rounds=3` (`1..3`); `max_subtasks_per_iteration=5` (`1..5`); `subtask_timeout_sec=1800.0` (`> 0`); `llm_engine="claude"`; `llm_model=""`; `subtask_engine="claude"`; `subtask_model=""`; `on_status_written: Callable[[Path], None] \| None = None`. Empty `default_persona` raises `ValueError`. **Not** listed in `mltgnt.config.__all__`; import `from mltgnt.config import LoopsConfig`. |
| `RetryConfig` | `mltgnt.agent._runner` | `max_retries=2`; `base_delay_s=1.0`; `max_delay_s=30.0`. Optional retry policy accepted by `AgentRunner(retry_config=...)`; it is not exported from `mltgnt.agent.__all__`. |
| `OODAConfig` | `mltgnt.interfaces.ooda` / `mltgnt.ooda` | `max_recovery_attempts=3`; `escalate_after=2`; `observe_filter: str \| None = None`. |

`mltgnt.config.__all__` is `DEFAULT_WEIGHT_MAP`, `MemoryConfig`, `PersonaConfig`, `SchedulerConfig`, `ChatConfig`.

---

## Error Reference

### `MltgntError` hierarchy (`mltgnt.exceptions`)

```
MltgntError
├── ConfigError      YAML/config parse; malformed --components
└── DependencyError  External callable/subprocess/API; PID lock busy
```

### Public exception and error types reachable from published APIs

| Type | Module | Inherits | When | CLI exit |
|------|--------|----------|------|----------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base for unified `except MltgntError` | `mltgnt run` → **1** |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | `--components` not `module:function`; import/callable failures | `mltgnt run` → **2** |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | Daemon PID lock held | `mltgnt run` → **3** |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | Invalid persona frontmatter (`load_persona`) | not mapped (library) |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | Compaction LLM failure | not mapped |
| `LlmCallError` | `mltgnt.loops.prompts` | `ValueError` | Clarify/decompose/evaluate LLM/JSON contract failure (carries `LlmTrace`) | loop → `failed` after 3 consecutive errors |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | Skill I/O type mismatch on DAG steps | not mapped |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | Skill load / unknown tool | not mapped |
| `ObjectiveError` | `mltgnt.loops.objective` | *(dataclass)* | Bad Objective YAML/body/id/status | status file `failed`; **not raised** |
| `FileNotFoundError` | stdlib | `OSError` | Missing audit file (`kpi` / `improvement`); missing persona file | `python -m mltgnt.kpi` / `improvement` → **1** |
| `ValueError` | stdlib | `Exception` | `LoopsConfig.__post_init__` limits; corrupt `LoopState` JSON (`schema_version`, types) | library / loop isolate+fallback |
| argparse | stdlib | `SystemExit` | Missing CLI args, invalid `--format` | **2** |

`ObjectiveError` is exported from `mltgnt.loops` and is the invalid-Objective result type. It is not a subclass of `Exception`.

---

## Public API Stability

This package is pre-1.0 (`0.Y.Z`). Y bumps may be breaking; Z bumps are intended to be backward compatible. Treat **`mltgnt.__all__` and `mltgnt.loops.__all__`** as the documented export surface. Host Protocols in `mltgnt.interfaces.loops` are part of the v0.18.0 loops contract even though they are not on the root `__all__`.

Removed in v0.10.0 and **not** part of this API: `run_chat`, `read_memory_agentic`, persona flat keys (`chat_model`, `slack`) and `ops.chat_model`, `Persona.WEIGHT_MAP` / `ops_config` / `slack_post_kwargs` / `delegate_ack`, and `mltgnt.scheduler.ghdag_bridge` (use `mltgnt.bridges.ghdag_bridge`).

---

## License

MIT — SPDX: `MIT` (matches `license = "MIT"` in [`pyproject.toml`](https://github.com/sumipan/mltgnt/blob/v0.18.0/pyproject.toml)).

| Link | Target |
|------|--------|
| Source | https://github.com/sumipan/mltgnt |
| Issues | https://github.com/sumipan/mltgnt/issues |
| ghdag (L0) | https://github.com/sumipan/ghdag |
| Changelog | [CHANGELOG.md](./CHANGELOG.md) |
| License file | [LICENSE](./LICENSE) |
