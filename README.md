# mltgnt

**L1 agent runtime:** persona, memory, skill, routing, scheduler, and Objective-loop orchestration. In the three-tier stack (**L0 [ghdag](https://github.com/sumipan/ghdag)** / **L1 mltgnt** / **L2 host**), mltgnt owns agent contracts and loops; LLM inference, file I/O, and DAG execution stay in ghdag. Hosts (Slack daemons, diary wiring) live at L2.

**Status:** Pre-1.0 (`v0.20.0`)

---

## Not (what this is not)

| Item | Explanation |
|------|-------------|
| Not an LLM SDK | Inference goes through ghdag (`mltgnt.bridges.llm_adapter.call_llm`). mltgnt does not call model APIs itself. |
| Not a DAG engine | Job submission and completion markers are ghdag's. mltgnt uses `enqueue_dag` / `enqueue_and_wait` / `enqueue_step` / `poll_step`. |
| Not a Slack (or other) host | Channel posting, thread IDs, and daemon wiring belong to the L2 host. `HumanChannel`, `SubtaskExecutor`, `ConditionEvaluator`, `ActionExecutor`, and `MemoryAppender` are Protocols the host implements. |

Objective **startup** is request-driven (see [Objective loops in v0.20.0](#objective-loops-in-v0200)). Placing an Objective file alone does not start a loop.

---

## Installation

| Item | Value |
|------|-------|
| Install | `pip install mltgnt` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, [ghdag](https://github.com/sumipan/ghdag) `v0.30.12` (pinned in [`pyproject.toml`](https://github.com/sumipan/mltgnt/blob/v0.20.0/pyproject.toml)) |
| Console script | `mltgnt` → `mltgnt.cli.main:main` |
| License | MIT (`license = "MIT"` in `pyproject.toml`) |

ghdag is required. LLM calls and DAG/step enqueue go through it.

---

## Quick Start

**Prerequisites:** Python 3.10+, `mltgnt` installed (pulls in ghdag and PyYAML). No Slack host, daemon, or live LLM is required for the examples below.

These snippets follow `tests/loops/test_objective.py`, `tests/loops/test_requests.py`, and persona fixtures used in `tests/chat/test_pipeline.py`.

### Parse an Objective Markdown file

`parse_objective` returns `Objective` on success or `ObjectiveError` (a dataclass, **not** an exception) on invalid input. `max_iterations` must be an integer in `1..10` (booleans are rejected).

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

### Validate a start-request JSON

Invalid request files are moved to `state_dir/requests/corrupt/` by `list_requests` (same behavior as `tests/loops/test_requests.py`).

```python
import json
from pathlib import Path
import tempfile

from mltgnt.loops.requests import StartRequest, list_requests

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    state_dir = root / "state"
    objectives_dir = root / "objectives"
    objectives_dir.mkdir()
    req_dir = state_dir / "requests"
    req_dir.mkdir(parents=True)

    payload = {
        "objective_path": "hp-renewal.md",
        "channel_id": "C0123",
        "thread_ts": "1234567890.123456",
        "persona": "operator",
        "requested_at": "2026-08-21T13:00:00+09:00",
    }
    (req_dir / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    (req_dir / "bad.json").write_text("{not json", encoding="utf-8")

    ok, errors = list_requests(state_dir, objectives_dir)
    assert len(ok) == 1 and isinstance(ok[0], StartRequest)
    assert ok[0].objective_path == "hp-renewal.md"
    assert len(errors) == 1
    assert (req_dir / "corrupt" / "bad.json").is_file()
    print(ok[0].filename, errors[0].message)
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

`run_pipeline(prompt, persona, engine=..., model=...)` returns `ChatOutput`. A live ghdag LLM engine is required; failures are stored in `ChatOutput.content` and are not raised.

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

## Public API / Protocols

Stable import surface is `mltgnt.__all__` plus package `__all__` lists below and `mltgnt.loops.__all__`. Other modules exist for hosts and internals; they are not implied stable unless listed.

```python
from mltgnt import run_pipeline, Persona, load_persona
from mltgnt.loops import LoopsComponent, parse_objective
from mltgnt.interfaces.loops import HumanChannel, ActionExecutor, MemoryAppender
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
| `LoopsComponent` | class | `mltgnt.loops.component` | `LoopsConfig`, `HumanChannel`, `SubtaskExecutor`, optional `ActionExecutor` | `DaemonComponent`: `name == "loops"`, `start()` / `stop()`. Refreshes Objective snapshots, consumes `state_dir/requests/*.json`, then calls `LoopsEngine.tick()`. |
| `LoopsEngine` | class (`BaseRunner`) | `mltgnt.loops.engine` | `LoopsConfig`, `HumanChannel`, `SubtaskExecutor`, optional objective callbacks, `ConditionEvaluator`, `ActionExecutor` | One tick = at most one state transition per loop. `start_loop(objective, *, thread=None)` inherits optional `HumanThreadRef`. |
| `GhdagSubtaskExecutor` | class | `mltgnt.loops.executor` | `jobs_dir`, `exec_done_dir`, `engine`, `model`, optional `correlation_id` | Host-facing `SubtaskExecutor`: `submit` / `poll`. |
| `Objective` | dataclass | `mltgnt.loops.objective` | parsed fields | Valid Objective: `loop_id`, `title`, `body`, `agent`, `max_iterations`, `status`, `path`, `content_hash`, `plan_approval`. |
| `ObjectiveError` | dataclass | `mltgnt.loops.objective` | `loop_id`, `message`, `path` | Parse/validation failure. **Not** an `Exception`. |
| `parse_objective` | function | `mltgnt.loops.objective` | `path: Path`, `default_persona`, `default_max_iterations`, optional `known_ids`, `plan_approval_default=True` | `Objective \| ObjectiveError`. |
| `list_objective_files` | function | `mltgnt.loops.objective` | `objectives_dir: Path` | Non-recursive `*.md` paths, sorted. Missing dir → `[]`. |
| `LoopState` | dataclass | `mltgnt.loops.models` | loop identity, `status: LoopStatus`, iteration, persona, thread, subtasks, budgets, … | Persisted loop state (`schema_version = 1`). |
| `Subtask` | dataclass | `mltgnt.loops.models` | `id`, `title`, `kind` (`auto` \| `human` \| `watch` \| `action`), `prompt`, `status`, `result`, optional `condition` / `action` / `depends` / `submission` | One decomposed step. |
| `TERMINAL_STATUSES` | `frozenset[str]` | `mltgnt.loops.models` | — | `{"done", "failed", "cancelled"}`. |

### Other package `__all__` (symbol sets)

| Package | Symbols (sorted; match `tests/test_all_snapshot.py`) |
|---------|------------------------------------------------------|
| `mltgnt.agent` | `AgentResult`, `AgentRunner` |
| `mltgnt.bridges` | `DagStep`, `MltgntHooks`, `call_llm`, `create_audit_writer`, `enqueue_and_wait`, `enqueue_dag`, `files_adapter`, `ghdag_bridge`, `hooks_adapter`, `llm_adapter`, `md_read`, `md_write` |
| `mltgnt.chat` | `ChatInput`, `ChatOutput`, `Message`, `run_pipeline` |
| `mltgnt.config` | `ChatConfig`, `DEFAULT_WEIGHT_MAP`, `MemoryConfig`, `PersonaConfig`, `SchedulerConfig` |
| `mltgnt.daemon` | `DaemonComponent`, `DaemonRunner`, `PidLock`, `SkillWatcherComponent` |
| `mltgnt.exceptions` | `ConfigError`, `DependencyError`, `MltgntError` |
| `mltgnt.interfaces` | `ChatInput`, `ChatInputBase`, `ChatOutput`, `ChatOutputBase`, `ChatPipelineProtocol`, `Message`, `PersonaFMBase`, `PersonaProtocol`, `SlackClientProtocol` |
| `mltgnt.memory` | `CompactionResult`, `LlmCall`, `LlmCallError`, `MEMORY_CORRUPT_THRESHOLD_BYTES`, `MEMORY_DEDUPE_SCAN_BYTES`, `MEMORY_DEDUPE_SCAN_LINES`, `MemoryEntry`, `_ensure_jsonl`, `_resolve_memory_dir`, `_scan_tail_for_dedupe_key`, `_search_and_score`, `_tail_utf8_bytes`, `append_memory_entry`, `assemble_entries_text`, `get_collection`, `memory_file_path`, `parse_jsonl`, `persona_memory_lock`, `query_similar`, `read_memory_by_relevance`, `read_memory_iterative`, `read_memory_preferences`, `read_memory_tail_text`, `read_memory_with_sufficiency_check`, `serialize_entry`, `tail_utf8_bytes`, `upsert_entry` |
| `mltgnt.persona` | `Persona`, `PersonaValidationError`, `compress_heavy_to_light`, `list_personas`, `load_persona`, `regenerate_light_block`, `run_persona_prompt`, `validate_persona` |
| `mltgnt.routing` | `ChannelPersonaEntry`, `RoutingRule`, `TRIAGE_PROFILE_MAX_CHARS`, `detect_nickname`, `evaluate`, `extract_json_object`, `extract_triage_section`, `find_observers`, `load_channel_persona_map`, `prepare_profile_for_triage`, `resolve_responding_persona` |
| `mltgnt.scheduler` | `PersonaScheduler`, `ScheduleJob`, `SchedulePaths`, `_hash_offset`, `atomic_write_text`, `load_schedule_jobs` |
| `mltgnt.skill` | `ArtifactSpec`, `ConsumesSpec`, `ProducesSpec`, `SkillFile`, `SkillMatchResult`, `SkillMeta`, `SkillRegistry`, `SkillRunResult`, `discover`, `discover_bodies`, `lint_skill_meta`, `load`, `match`, `resolve_skill`, `run` |

`LoopsConfig` is defined in `mltgnt.config` but is **not** in `mltgnt.config.__all__`; import `from mltgnt.config import LoopsConfig`.

### Host Protocols (L2)

| Protocol / type | Module | Contract |
|-----------------|--------|----------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | `name`, `fm`, `format_prompt(instruction) -> str` |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | `run(inp, repo_root) -> ChatOutputBase` |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(...) -> bool` (False on failure, no raise) |
| `HumanChannel` | `mltgnt.interfaces.loops` | Open/ask/notify/progress/deliverable/fallback/close (returns `HumanThreadRef \| None` or `bool`) |
| `SubtaskExecutor` | `mltgnt.interfaces.loops` | `submit(*, prompt, idempotency_key, engine=None, model=None) -> StepSubmission`; `poll(*, uuid, result_filename) -> StepPoll` |
| `ConditionEvaluator` | `mltgnt.interfaces.loops` | `evaluate(condition, *, previous_token) -> WatchVerdict \| None` (host-owned conditions; local `path_*` use built-in evaluator) |
| `ActionExecutor` | `mltgnt.interfaces.loops` | `execute(*, request: ActionRequest, idempotency_key: str) -> ActionResult` (synchronous; host owns side effects) |
| `MemoryAppender` | `mltgnt.interfaces.loops` | `(*, persona, content, timestamp, dedupe_key) -> bool` (optional; host owns persistence) |
| `DaemonComponent` | `mltgnt.daemon` | `name`, `start()`, `stop()` — includes `LoopsComponent` |

`mltgnt.interfaces.__all__` does not export the loops Protocols; import them from `mltgnt.interfaces.loops`.

`ActionRequest` / `ActionResult` / `WatchVerdict` / `HumanThreadRef` / `StepSubmission` / `StepPoll` are dataclasses in the same module.

---

## Objective loops in v0.20.0

`mltgnt.loops` drives an Objective Markdown file through clarify → decompose → execute → evaluate, with watch/replan, plan approval, comment dialogue, deterministic actions, optional persona memory append, and LLM/watch/replan budgets. The host injects channel and executor implementations; mltgnt does not implement Slack.

### Request-driven start (since v0.19.0)

| Behavior | Detail |
|----------|--------|
| Objective file alone | Does **not** create loop state |
| Start | Consume `state_dir/requests/*.json` only |
| Migration | Deploy mltgnt consumer first; switch the host producer afterward |

`LoopsComponent` still polls `objectives_dir` (refresh snapshots, `ensure_frontmatter`, cancel/hash changes), then processes the request inbox, then calls `LoopsEngine.tick()`.

### Start-request JSON (`mltgnt.loops.requests`)

Files live in `state_dir/requests/*.json` (basename only; sorted by filename). Required keys (exact set, all strings):

| Key | Constraint |
|-----|------------|
| `objective_path` | Basename ending in `.md` (no `/`, `\`, `..`, or absolute paths) |
| `channel_id` | Non-empty |
| `thread_ts` | Non-empty |
| `persona` | String (empty allowed) |
| `requested_at` | ISO-8601 datetime **with timezone** |

| Outcome | Directory |
|---------|-----------|
| Valid request, handled | moved to `requests/consumed/` |
| Invalid JSON / keys / types / path / naive timestamp | isolated to `requests/corrupt/` by `list_requests` (`RequestError` dataclass, not raised) |

Helpers: `list_requests(state_dir, objectives_dir)`, `consume_request(state_dir, filename, *, corrupt=False)`. Importable from `mltgnt.loops.requests`; not in `mltgnt.loops.__all__`.

### Frontmatter completion (`ensure_frontmatter`)

On Objective refresh, `LoopsComponent` calls `ensure_frontmatter(path, default_max_iterations=LoopsConfig.max_iterations)`. It fills **only missing** keys deterministically:

| Key | Completed when missing | Not completed |
|-----|------------------------|---------------|
| `id` | from file stem (sanitized) | — |
| `title` | from first body heading/line, else `id` | — |
| `status` | `"active"` | — |
| `max_iterations` | `default_max_iterations` | — |
| `agent` | — | **never** auto-filled |

Returns `True` only when the file was rewritten.

### Objective file → `parse_objective`

YAML keys allowed: `id`, `title`, `agent`, `max_iterations`, `status`, `plan_approval`. Other keys are ignored with a warning.

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `id` | str | file stem | `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`; unique among loaded files |
| `title` | str | first non-empty body line, else `id` | Must be a string if present |
| `agent` | str | `LoopsConfig.default_persona` (or request `persona` at start) | Persona stem (`{persona_dir}/{agent}.md`) |
| `max_iterations` | int | `LoopsConfig.max_iterations` | `1..10`; bool is rejected |
| `status` | str | `active` | `active` or `cancelled` |
| `plan_approval` | bool | `plan_approval_default` (`LoopsConfig.plan_approval_default`, default `True`) | Must be bool if present |

Body must be non-empty. `cancelled` files are not started. Duplicate `id` values become `ObjectiveError` and a failed status Markdown file. Invalid YAML / out-of-range `max_iterations` also yield `ObjectiveError` (never raised).

### Single deliverable

Canonical artifact: `state_dir/<loop_id>/deliverable.md`, initialized from the Objective body at `start_loop`. Auto subtasks edit that file in stages; evaluate uses `result_summary` plus a deliverable excerpt (`LoopsConfig.deliverable_excerpt_chars`, default `4000`).

### Subtask kinds (`Subtask.kind`)

| Kind | Meaning | Host / engine contract |
|------|---------|------------------------|
| `auto` | LLM/agent step via `SubtaskExecutor.submit` / `poll` | Prompt required; optional `depends` |
| `human` | Ask human via `HumanChannel.ask`; wait for inbox answer | Prompt required |
| `watch` | Poll a condition until satisfied/failed | `condition` object; local types `path_exists` / `path_changed` under `watch_root`, or host `ConditionEvaluator` for other types |
| `action` | Deterministic host action | `action: {name, args}` must match `LoopsConfig.action_schemas`; executed synchronously with idempotency key via `ActionExecutor` |

`depends` forms a DAG among subtasks in the same iteration. Missing `depends` keys in old state are normalized to sequential dependencies.

Watch failure (when configured) can enter `replanning`: keep running/success subtasks, add replacements, bounded by `max_replans_per_iteration` (default `3`) and `max_replans_per_loop` (default `20`).

### Plan approval

When `plan_approval` is true, after decompose the loop enters `awaiting_plan_approval`. Approval answers are **exact full-string** matches: `ok` / `承認` / `進めて` / `go`. Human revisions are allowed up to `max_plan_revisions` (default `3`) and do **not** consume `replan_count`. Events include `plan_proposed` / `plan_approved` / `plan_revised`.

### Comment dialogue (inbox `kind=comment`)

Comments are classified (deterministic status inquiry, else LLM: `status` / `instruction` / `question` / `chitchat`) and answered within budgets:

| Setting | Default | Range |
|---------|---------|-------|
| `max_comments_per_tick` | `10` | `1..100` |
| `comment_reply_budget_per_hour` | `10` | `0..100` |
| `comment_reply_max_chars` | `800` | `1..4000` |
| `comment_model` | `""` (falls back to loop LLM model) | string |

Status inquiries can use `render_progress_summary` (no LLM) and `HumanChannel.post_progress`. Instructions route to replan; questions get a persona reply; chitchat may append to `clarification_context`.

### Action execution and persona memory

| Feature | Behavior |
|---------|----------|
| Action | Synchronous `ActionExecutor.execute(*, request, idempotency_key) -> ActionResult`. Host owns side effects. Event: `action_executed`. |
| Memory | Optional `LoopsConfig.memory_append: MemoryAppender`. Short summaries on plan approval, iteration complete, and done/failed. Dedupe via `dedupe_key`; returns `bool`. Events: `memory_appended` / `memory_append_failed`. Cap: `memory_summary_max_chars` (default `500`, must be `> 0`). |

`ActionExecutor` / `MemoryAppender` may be omitted; existing hosts keep working.

### LLM / watch / replan budgets

| Field | Default | Constraint | Scope |
|-------|---------|------------|-------|
| `llm_call_budget_per_loop` | `200` | `>= 0` | Per-loop LLM calls |
| `llm_call_budget_per_day` | `1000` | `>= 0` | Shared JST calendar day |
| `max_watch_subtasks_per_loop` | `50` | `>= 0` | Watch subtasks per loop |
| `max_replans_per_loop` | `20` | `>= 0` | Replans per loop |

Budget `0` is allowed and immediately blocks the corresponding resource. On exceed, status becomes `paused` (previous status stored in `paused_from_status`). An inbox message whose text is exactly `再開` sets `budget_override` and resumes. Events: `budget_resumed`.

### Schema compatibility

`schema_version: 1` is unchanged. Older state files without newer fields restore defaults (`plan_approval=True`, budget counters `0`, `budget_override=False`, empty `memory_dedupe_keys`, etc.). Unsupported `schema_version` raises `ValueError` when loading.

### Host → `LoopsComponent` / `LoopsEngine`

| Dependency | Role |
|------------|------|
| `LoopsConfig` | Paths, limits, budgets, `action_schemas`, optional `memory_append`, `watch_root` |
| `HumanChannel` | Open/ask/notify/progress/deliverable/fallback/close |
| `SubtaskExecutor` | Non-blocking auto-subtask submit + poll |
| `ActionExecutor` (optional) | Synchronous actions |
| `ConditionEvaluator` (optional on engine) | Non-local watch conditions |

`LoopsComponent.start()` launches a daemon thread; `stop()` joins it.

### Thread inheritance and re-start

| Behavior | Detail |
|----------|--------|
| `LoopsEngine.start_loop(objective, *, thread=None)` | Optional `HumanThreadRef` on initial `LoopState` |
| Request start | Component passes `HumanThreadRef(channel_id, thread_ts)` from the request |
| Non-terminal restore | Existing `state_dir/<loop_id>/state.json` continues; a new request while running is rejected (`already_running`) and consumed |
| Terminal re-request | Terminal state archived under `state_dir/archive/`, then a new loop starts |
| Cancel | Objective removal / `status: cancelled` / inbox cancel still cancel non-terminal loops |

### State machine

`LoopStatus`: `clarifying` → `awaiting_answer` → `decomposing` → `awaiting_plan_approval` (optional) → `executing` / `awaiting_human` / `replanning` → `evaluating` → `done` \| `failed`, plus `paused` / `cancelled`.

| Limit | Source | Range / value |
|-------|--------|----------------|
| Poll interval | `LoopsConfig.poll_interval_sec` | `> 0`, default `10.0` |
| Iterations | Objective / `LoopsConfig.max_iterations` | `1..10`, default `5` |
| Clarify rounds | `LoopsConfig.max_clarify_rounds` | `1..3`, default `3` |
| Subtasks per iteration | `LoopsConfig.max_subtasks_per_iteration` | `1..5`, default `5` |
| Subtask timeout | `LoopsConfig.subtask_timeout_sec` | `> 0`, default `1800.0` |
| Consecutive engine errors | `mltgnt.loops.engine._MAX_CONSECUTIVE_ERRORS` | `3` then `failed` |

Status Markdown: `<status_dir>/<loop_id>.md`.

### `HumanChannel` methods

| Method | Returns |
|--------|---------|
| `open_thread(*, loop_id, persona, title, body, event_id)` | `HumanThreadRef \| None` |
| `ask(*, loop_id, persona, thread, question_id, text, event_id)` | `bool` |
| `notify(*, loop_id, persona, thread, text, event_id)` | `bool` |
| `post_progress(*, loop_id, persona, thread, text, event_id)` | `bool` |
| `post_deliverable(*, loop_id, persona, thread, deliverable_path, summary, event_id)` | `bool` |
| `notify_fallback(*, loop_id, text, event_id)` | `bool` |
| `close_thread(*, loop_id, persona, thread, event_id)` | `bool` |

`progress_notify=False` on `LoopsConfig` suppresses progress posts only.

### `SubtaskExecutor` and ghdag step boundary

| Method | Returns |
|--------|---------|
| `submit(*, prompt, idempotency_key, engine=None, model=None)` | `StepSubmission` |
| `poll(*, uuid, result_filename)` | `StepPoll` |

`GhdagSubtaskExecutor` maps onto `enqueue_step` / `poll_step`. `StepPoll.status`: `pending` \| `success` \| `failed_exit` \| `rejected` \| `empty_result` \| `other`. `poll_step` does **not** wait.

`enqueue_dag` / `enqueue_and_wait` remain the **blocking** DAG helpers on the package root.

---

## Architecture

Layout of `src/mltgnt/`:

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
| `loops/` | Objective-driven loop engine, requests, watch/action/budget, status |
| `memory/` | JSONL memory search, compaction, dream summaries |
| `ooda/` | OODA runner (`OODARunner`, `OODAConfig`) |
| `persona/` | Load, validate, and prompt personas |
| `routing/` | Channel routing and triage |
| `scheduler/` | `PersonaScheduler` job runner |
| `skill/` | Skill discover/match/run/lint |

---

## Configuration

### Environment variables

Every `os.environ` / `os.getenv` use in `src/mltgnt/**/*.py`:

| Variable | Defined in | Purpose |
|----------|------------|---------|
| `SKILL_IO_TYPECHECK` | [`bridges/ghdag_bridge.py`](https://github.com/sumipan/mltgnt/blob/v0.20.0/src/mltgnt/bridges/ghdag_bridge.py) | Skill I/O typecheck is on unless this is `"0"`. |
| `NIKKI_ROOT` | [`skill/runner.py`](https://github.com/sumipan/mltgnt/blob/v0.20.0/src/mltgnt/skill/runner.py) | Diary/memory root for `$NIKKI_ROOT` substitution in skill bodies. |
| `REPO_ROOT` | [`skill/runner.py`](https://github.com/sumipan/mltgnt/blob/v0.20.0/src/mltgnt/skill/runner.py) | Repo-root fallback for `$REPO_ROOT` substitution. |
| `MLTGNT_AS_OF_DATE` | [`improvement/loop.py`](https://github.com/sumipan/mltgnt/blob/v0.20.0/src/mltgnt/improvement/loop.py) | `YYYY-MM-DD` period end for `run_improvement_cycle` when `today` is omitted. |

### Config dataclasses

| Dataclass | Module | Fields (type, default, constraints) |
|-----------|--------|-------------------------------------|
| `PersonaConfig` | `mltgnt.config` | `weight_map: dict[str, str]` default `DEFAULT_WEIGHT_MAP` (`light` / `heavy` / `reference` section weights). |
| `MemoryConfig` | `mltgnt.config` | `chat_dir: Path` (required); `chat_memory_dir: Path \| None = None`; `inject_max_bytes=10240`; `inject_max_entries=12`; `preferences_max_bytes=5120`; `lock_timeout_sec=30.0`; `lock_stale_threshold_sec=300.0`; `raw_days=7`; `mid_weeks=3`; `compact_threshold_bytes=40960`; `compact_target_bytes=25600`; `preferences_section_name="ユーザーの好み・傾向"`; `protected_layers=("caveat",)`; `timezone="Asia/Tokyo"`; `dream_model="claude-haiku-4-5-20251001"`; `use_dream_summary=False`; `dream_dir_name="memory"`; `global_dream_exclude_personas=()`. |
| `SchedulerConfig` | `mltgnt.config` | `schedule_yaml: Path`; `state_dir: Path`; `timezone="Asia/Tokyo"`; `salt=""`. |
| `ChatConfig` | `mltgnt.config` | `persona_dir: Path`; `memory_dir: Path \| None = None`; `matcher_model="claude-haiku-4-5-20251001"`. |
| `LoopsConfig` | `mltgnt.config` | **Required:** `objectives_dir`, `state_dir`, `status_dir`, `jobs_dir`, `exec_done_dir`, `persona_dir`, `default_persona`, `fallback_channel`. **Optional:** `poll_interval_sec=10.0` (`> 0`); `max_iterations=5` (`1..10`); `max_clarify_rounds=3` (`1..3`); `max_subtasks_per_iteration=5` (`1..5`); `subtask_timeout_sec=1800.0` (`> 0`); `llm_engine="claude"`; `llm_model=""`; `subtask_engine="claude"`; `subtask_model=""`; `on_status_written: Callable[[Path], None] \| None = None`; `progress_notify=True`; `deliverable_excerpt_chars=4000` (`> 0`); `result_summary_chars=1000` (`> 0`); `watch_root: Path \| None = None`; `max_replans_per_iteration=3` (`0..10`); `max_plan_revisions=3` (`0..10`); `plan_approval_default=True`; `comment_model=""`; `max_comments_per_tick=10` (`1..100`); `comment_reply_budget_per_hour=10` (`0..100`); `comment_reply_max_chars=800` (`1..4000`); `action_schemas: Mapping[str, Mapping[str, object]] = {}`; `memory_append: MemoryAppender \| None = None`; `memory_summary_max_chars=500` (`> 0`); `llm_call_budget_per_loop=200` (`>= 0`); `llm_call_budget_per_day=1000` (`>= 0`); `max_watch_subtasks_per_loop=50` (`>= 0`); `max_replans_per_loop=20` (`>= 0`). Empty `default_persona` or invalid ranges raise `ValueError` in `__post_init__`. **Not** in `mltgnt.config.__all__`. |
| `RetryConfig` | `mltgnt.agent._runner` | `max_retries=2`; `base_delay_s=1.0`; `max_delay_s=30.0`. Optional for `AgentRunner`; not in `mltgnt.agent.__all__`. |
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

### Public exception and error-result types

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
| `ObjectiveError` | `mltgnt.loops.objective` | *(dataclass)* | Bad Objective YAML/body/id/status/`max_iterations` | status file `failed`; **not raised** |
| `RequestError` | `mltgnt.loops.requests` | *(dataclass)* | Bad start-request JSON | isolated to `corrupt/`; **not raised** |
| `FileNotFoundError` | stdlib | `OSError` | Missing audit file (`kpi` / `improvement`); missing persona file | `python -m mltgnt.kpi` / `improvement` → **1** |
| `ValueError` | stdlib | `Exception` | `LoopsConfig.__post_init__` limits (including budget `>= 0`); corrupt `LoopState` JSON | library / loop isolate+fallback |
| argparse | stdlib | `SystemExit` | Missing CLI args, invalid `--format`, unknown dream category path via missing args | **2** |

`ObjectiveError` / `RequestError` are result dataclasses. They are not subclasses of `Exception`.

---

## Public API Stability / removed API

This package is pre-1.0 (`0.Y.Z`). Y bumps may be breaking; Z bumps are intended to be backward compatible. Treat **`mltgnt.__all__` and `mltgnt.loops.__all__`** as the documented export surface. Host Protocols in `mltgnt.interfaces.loops` remain part of the loops contract even though they are not on the root `__all__`.

Removed in **v0.10.0** and **not** part of this API (do not use; covered by `tests/test_v0100_breaking.py`):

| Removed | Use instead |
|---------|-------------|
| `run_chat` | `run_pipeline` |
| `read_memory_agentic` | `read_memory_iterative` / `read_memory_by_relevance` / `read_memory_with_sufficiency_check` |
| Persona flat keys (`chat_model`, `slack`) and `ops.chat_model` | `ops.engine` / `ops.model` frontmatter |
| `Persona.WEIGHT_MAP` / `ops_config` / `slack_post_kwargs` / `delegate_ack` | Current `Persona` + `PersonaConfig.weight_map` |
| `mltgnt.scheduler.ghdag_bridge` | `mltgnt.bridges.ghdag_bridge` |
| Root `compact` / `needs_compaction` | `mltgnt.memory.compaction.compact` (and related helpers) |

Behavioral break since **v0.19.0:** Objective placement no longer auto-starts loops; hosts must produce `state_dir/requests/*.json`.

---

## License

MIT — SPDX: `MIT` (matches `license = "MIT"` in [`pyproject.toml`](https://github.com/sumipan/mltgnt/blob/v0.20.0/pyproject.toml)).

| Link | Target |
|------|--------|
| Source | https://github.com/sumipan/mltgnt |
| Issues | https://github.com/sumipan/mltgnt/issues |
| ghdag (L0) | https://github.com/sumipan/ghdag |
| Changelog | [CHANGELOG.md](./CHANGELOG.md) |
| License file | [LICENSE](./LICENSE) |
| Tag `v0.20.0` | https://github.com/sumipan/mltgnt/tree/v0.20.0 |
