# mltgnt

## Title / Positioning

**Typed building blocks for multi-agent chat hosts: personas, memory, skills, an agent loop and work loop, routing, scheduling, conversation state, and a media I/O contract.**

mltgnt is a Python library. It sits between an execution engine and the application that hosts your agents:

| Layer | Owner | Responsibility |
|-------|-------|----------------|
| Engine | [ghdag](https://github.com/sumipan/ghdag) | DAG execution, LLM engines, VCS sinks |
| Library | **mltgnt** | Domain types and behavior: personas, memory, skills, agent loop, routing, scheduler, conversation state, media contract |
| Host | Your application | Processes, credentials, filesystem paths, deployment |

All LLM calls and git commits go out through `mltgnt.bridges`, which is the only package that imports ghdag. Chat media (Slack, WebChat, or your own) connect through a single Protocol, `MediaClient`.

## Status

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.134.0)-orange)

Pre-1.0. The current release is `0.134.0` (tag `v0.134.0`). A minor release may change the API. See [Public API Stability](#public-api-stability).

## Source inventory (`v0.134.0`)

Regenerated from the `v0.134.0` tree (`package 0.134.0`, ghdag pin `v0.104.5`) on 2026-10-26. Row counts below match the tables in this file.

| Inventory item | Count |
|----------------|------:|
| Python modules under `src/mltgnt/` | 145 |
| `mltgnt.__all__` exports | 23 |
| `mltgnt.agent.__all__` exports | 38 |
| `mltgnt.media.__all__` exports | 27 |
| Direct `os.environ` / `os.getenv` keys in library code | 4 (`REPO_ROOT`, `SKILL_IO_TYPECHECK`, `THREAD_PERSONA_TTL_DAYS`, `MLTGNT_DEFAULT_ENGINE`) |
| `SkillConfig.passthrough_env` keys | host-declared at startup |
| Exceptions in [Error Reference](#error-reference) | 9 |
| Rows in [Deprecated API](#deprecated-api) | 2 |
| CLI commands documented | 4 (`run`, `memory dream show`, `memory dream forget`, `hub serve`) |

## Not

| mltgnt is not | Use instead |
|---------------|-------------|
| An LLM SDK | mltgnt ships no model client. `call_llm`, `enqueue_and_wait`, and `enqueue_dag` in `mltgnt.bridges` hand the work to ghdag. |
| A DAG engine | Queueing, dependency resolution, cancellation, and job state belong to ghdag. mltgnt builds the steps and waits for the results. |
| A host runtime | The host supplies paths, secrets, and the process layout. `mltgnt run` only starts the components that a host factory returns. |
| A Slack bot | Slack and WebChat are optional adapters behind `MediaClient`. Core packages never import `mltgnt.media`. |

## Installation

mltgnt is distributed as git tags. It is not on PyPI.

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.134.0"

# Slack medium (mltgnt.media.slack)
pip install "mltgnt[slack] @ git+https://github.com/sumipan/mltgnt.git@v0.134.0"

# WebChat medium (mltgnt.media.webchat)
pip install "mltgnt[webchat] @ git+https://github.com/sumipan/mltgnt.git@v0.134.0"

# WebChat Web Push (includes the webchat dependencies and pywebpush)
pip install "mltgnt[webchat-push] @ git+https://github.com/sumipan/mltgnt.git@v0.134.0"

# Hub relay server (mltgnt.media.hub; mltgnt hub serve)
pip install "mltgnt[hub] @ git+https://github.com/sumipan/mltgnt.git@v0.134.0"

# Development tools (pytest, ruff, mypy, import-linter)
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.134.0"
```

See [TESTING.md](./TESTING.md) for what tests should and should not pin.

| Item | Value |
|------|-------|
| Distribution | `mltgnt` `0.134.0` |
| Python | `>=3.10` |
| Dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.104.5` |
| Extra `slack` | `slack_sdk>=3.0`, `slack_bolt>=1.18` |
| Extra `webchat` | `fastapi>=0.100`, `uvicorn>=0.20` |
| Extra `webchat-push` | `fastapi>=0.100`, `uvicorn>=0.20`, `pywebpush` (enables VAPID push when `WebChatMediaConfig.push` is configured) |
| Extra `hub` | `fastapi>=0.100`, `uvicorn>=0.20`, `PyYAML>=6.0` |
| Extra `dev` | `pytest>=7.0`, `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `freezegun>=1.2`, `import-linter>=2.0`, `mypy>=1.10`, `ruff>=0.4` |
| Console script | `mltgnt` -> `mltgnt.cli.main:main` (`python -m mltgnt` does the same) |
| Typing | Ships `py.typed` |
| Optional, undeclared | `chromadb`: if it can be imported, memory search also queries a Chroma collection. Without it, search uses TF-IDF only |

The Slack and WebChat apps import their third-party packages lazily. The rest of `mltgnt.media` imports without the extras.

## Quick Start

This example needs no LLM and no network. It loads a persona, saves a dream summary, and drives the agent loop with a scripted LLM.

```python
import json
import tempfile
from pathlib import Path

from mltgnt import (
    AgentRunner,
    DreamSection,
    DreamSummary,
    list_personas,
    load_persona,
    read_dream,
    validate_persona,
    write_dream,
)

root = Path(tempfile.mkdtemp())
agents = root / "agents"
agents.mkdir()
(agents / "guide.md").write_text(
    "---\npersona:\n  name: guide\n---\n## Background\n\nA calm, concise guide.\n",
    encoding="utf-8",
)

# A persona is a Markdown file with YAML frontmatter.
print(list_personas(agents))                     # ['guide']
persona = load_persona("guide", persona_dir=agents)
print(persona.name, validate_persona(persona))   # guide []

# Dream summaries are stored in <chat_dir>/<persona>/memory/dream.json.
write_dream(
    root / "chat" / "guide",
    DreamSummary(
        persona="guide",
        sections=[DreamSection(category="style", content="Prefers bullet points.", source_entries=2)],
        updated_at="2026-01-01T00:00:00+00:00",
    ),
)
print(read_dream(root / "chat" / "guide").sections[0].category)  # style

# The agent loop receives the LLM and the tools as plain callables.
replies = iter([
    json.dumps({"thought": "look it up", "tool": "lookup", "args": {"key": "tz"}}),
    json.dumps({"thought": "done", "tool": "answer", "args": {"text": "UTC"}}),
])
runner = AgentRunner(
    llm_call=lambda prompt, *, tool_result=None: next(replies),
    tool_executor=lambda name, args: "UTC",
    terminal_tools=frozenset({"answer"}),
)
result = runner.run("Which timezone?")
print(result.tool, result.args)                  # answer {'text': 'UTC'}
```

To read the saved summary from the shell:

```bash
mltgnt memory dream show guide --chat-dir /path/to/root/chat
```

A custom chat medium only needs to satisfy the `MediaClient` Protocol:

```python
from collections.abc import Mapping
from typing import Any

from mltgnt.interfaces.media import MediaClient, Status


class ConsoleMedia:
    def post(
        self,
        text: str,
        space: str,
        thread: str | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> str | None:
        print(f"[{space}] {text}")
        return "m-1"

    def update(self, message_id: str, text: str) -> bool:
        return True

    def set_status(self, message_id: str, status: Status) -> bool:
        return True

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        return False

    def react(self, message_id: str, name: str) -> bool:
        return False


media = ConsoleMedia()
assert isinstance(media, MediaClient)            # runtime-checkable
media.set_status(media.post("hello", "general"), Status.DONE)
```

## CLI Reference

Inventory source: `src/mltgnt/cli/main.py`, `src/mltgnt/cli/memory.py`, and `src/mltgnt/cli/hub.py` (`argparse`) at `v0.134.0`.

The CLI is defined in `src/mltgnt/cli/main.py` (`run`), `src/mltgnt/cli/memory.py` (`memory dream show`, `memory dream forget`), and `src/mltgnt/cli/hub.py` (`hub serve`). Running `mltgnt` with no subcommand prints help and exits with `0`.

| Command | Purpose |
|---------|---------|
| `mltgnt run` | Start host-provided daemon components under a PID lock |
| `mltgnt memory dream show` | Print a persona's dream summary |
| `mltgnt memory dream forget` | Delete one category from a persona's dream summary |
| `mltgnt hub serve` | Run the multi-host hub relay server from a YAML config |

### `mltgnt run`

Imports `MODULE` and calls `FUNCTION()`, which returns a list of `DaemonComponent` objects. `DaemonRunner` then runs them under a PID lock until SIGINT or SIGTERM arrives.

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--components MODULE:FUNCTION` | yes | - | Component factory, for example `myhost.daemon:components` |
| `--pid-file PATH` | no | `/tmp/mltgnt_daemon.pid` | PID lock file |

| Exit code | Meaning |
|-----------|---------|
| `0` | Normal shutdown |
| `1` | Any other `MltgntError` |
| `2` | `ConfigError`: `--components` is not in `module:function` form, the module cannot be imported, or the attribute is missing or not callable |
| `3` | `DependencyError`: another process holds the PID lock |

### `mltgnt memory dream show`

Prints each section of `<chat-dir>/<persona>/memory/dream.json` as `=== <category> (source_entries: N) ===`, followed by the section content.

| Argument / option | Required | Description |
|-------------------|----------|-------------|
| `persona` | yes | Persona name (a directory name under `--chat-dir`) |
| `--chat-dir PATH` | yes | Parent directory of the persona directories |

Exits with `0`. If no summary exists, it prints a notice and still exits with `0`.

### `mltgnt memory dream forget`

Deletes one category from the dream summary and rewrites `dream.json`.

| Argument / option | Required | Description |
|-------------------|----------|-------------|
| `persona` | yes | Persona name (a directory name under `--chat-dir`) |
| `--category NAME` | yes | Category to delete |
| `--chat-dir PATH` | yes | Parent directory of the persona directories |

Exits with `0` on success. Exits with `1` if the summary or the category does not exist.

### `mltgnt hub serve`

Loads a hub server YAML config and runs the FastAPI relay (requires the `hub` extra).

| Argument | Required | Description |
|----------|----------|-------------|
| `config` | yes | Path to hub server YAML (`store_dir`, `rooms`, `token_env`, optional `guard` / `ttl_days`) |

Loads YAML via `load_server_config` (see `HubServerConfig`: `store_dir`, `host` default `127.0.0.1`, `port` default `8790`, `rooms`, `token_env` host→env-var map, optional `guard`, `ttl_days` default `7`). Bearer tokens are read from the environment variables named in `token_env` (values are never stored in the YAML).

| Exit code | Meaning |
|-----------|---------|
| `0` | Normal shutdown (SIGINT / SIGTERM) |
| `2` | `ConfigError` (missing token env, invalid rooms, bad YAML, and similar) |
| `3` | `ImportError` when `fastapi` / `uvicorn` are not installed, or `DependencyError` from the server (`pip install 'mltgnt[hub]'`) |

## Public API

Inventory source: `__all__` and public re-exports under `src/mltgnt/` at `v0.134.0` (23 top-level names).

### Top level: `mltgnt.__all__`

The top level exports these 23 names. Import everything else from its subpackage.

| Name | Kind | Signature / fields | Defined in |
|------|------|--------------------|------------|
| `read_memory_by_relevance` | function | `(config, persona_stem, query, *, max_bytes, max_entries, layers=None) -> str` | `mltgnt.memory.search` |
| `read_memory_with_sufficiency_check` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call=None) -> str` | `mltgnt.memory.search` |
| `read_memory_iterative` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call, skill_paths=None, max_iterations=3) -> str` | `mltgnt.memory.search` |
| `DreamSection` | frozen dataclass | `category`, `content`, `source_entries` | `mltgnt.memory.dream` |
| `DreamSummary` | frozen dataclass | `persona`, `sections`, `updated_at` | `mltgnt.memory.dream` |
| `read_dream` | function | `(persona_dir, *, memory_dir_name="memory") -> DreamSummary \| None` | `mltgnt.memory.dream` |
| `write_dream` | function | `(persona_dir, summary, *, memory_dir_name="memory") -> None` | `mltgnt.memory.dream` |
| `Persona` | dataclass | `name`, `fm`, `sections`, `body`, `path`, `weight_map` | `mltgnt.persona.loader` |
| `load_persona` | function | `(name, *, persona_dir=None, config=None) -> Persona` | `mltgnt.persona` |
| `list_personas` | function | `(persona_dir=None) -> list[str]` | `mltgnt.persona` |
| `validate_persona` | function | `(persona, *, available_skills=None) -> list[str]` (warnings; an empty list means valid) | `mltgnt.persona` |
| `run_persona_prompt` | function | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str` | `mltgnt.persona` |
| `ChatInput` | dataclass | `source`, `session_key`, `messages`, `persona_name=""`, `model=None`, `context_files=[]`, `context_memory_excerpt=None`, `context_memory_preferences=None` | `mltgnt.interfaces.types` |
| `ChatOutput` | dataclass | `content`, `persona_name`, `timestamp`, `session_key` | `mltgnt.interfaces.types` |
| `Message` | TypedDict | `role`, `content` | `mltgnt.interfaces.types` |
| `PersonaProtocol` | Protocol | Structural persona type. `Persona` satisfies it | `mltgnt.interfaces.persona` |
| `AgentRunner` | class | See [`mltgnt.agent`](#mltgntagent) | `mltgnt.agent._runner` |
| `AgentResult` | dataclass | `tool`, `args`, `raw_response`, `tool_trace=None`, `reflexion_count=0`, `plan=None` | `mltgnt.agent._runner` |
| `enqueue_and_wait` | function | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_name=None, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, permission=None, order_builder=None, run_result=None, task_timeout_sec=None) -> tuple[bool, str]` | `mltgnt.bridges.ghdag_bridge` |
| `enqueue_dag` | function | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, skills=None, permission=None, order_builder=None, task_timeout_sec=None) -> list[tuple[bool, str]]` | `mltgnt.bridges.ghdag_bridge` |
| `PersonaScheduler` | class | `(slack, *, config=None, state_dir=None, yaml_path=None, salt="", jobs=None, notify_channel_resolver=None, default_slack_post_kwargs=None, persona_post_kwargs_resolver=None, repo_root=None, persona_dir=None, append_memory_fn=None, actions=None, memory_config=None, state_keep_days=30)`. `slack` is a `MediaClient` | `mltgnt.scheduler.runner` |
| `ScheduleJob` | dataclass | See [Scheduler jobs](#scheduler-jobs). Build one with `ScheduleJob.from_dict(raw)` | `mltgnt.scheduler.models` |
| `__version__` | `str` | Installed distribution version. `"0.0.0"` when package metadata is unavailable | `mltgnt` |

`run_persona_prompt` uses the persona's `engine`. If the persona does not set one, it uses `mltgnt.persona.schema.system_default_engine()`.

`enqueue_and_wait` and `enqueue_dag` return `(True, result)` on success and `(False, "<status>: <first line>")` on failure. When `timeout` expires, the result is `(False, "timeout (<N>s)")`. mltgnt also asks ghdag to cancel the task by creating the empty file `<jobs>/cancel/<uuid>` (`<jobs>` is the parent of `exec_done_dir`). If that file cannot be written, `; cancel failed: <error>` is appended to the message. `task_timeout_sec` is passed to ghdag as the per-task execution limit. `None` keeps the host default.

### `mltgnt.agent`

`AgentRunner(*, llm_call, tool_executor, terminal_tools, max_iterations=3, max_iterations_fn=None, evaluator=None, retry_config=None, logger=None, audit_writer=None, classifier=None, history_mode="last_result", history_max_chars=24000, plan=None, max_reflexions=None, step_hook=None)`.
`runner.run(prompt)` returns an `AgentResult` once the LLM picks a terminal tool. It returns `None` if the LLM call or response parsing fails.

| Option | Effect |
|--------|--------|
| `history_mode="full_trace"` | Sends the numbered tool trace to the LLM instead of only the last result. Older entries are truncated once the trace exceeds `history_max_chars` |
| `evaluator` | A `ReflexionEvaluator`. `DefaultReflexionEvaluator(failure_markers=(), repeat_window=3)` retries on `[ERROR]` results, on host failure markers, and on a repeated `(tool, args)` |
| `max_reflexions=N` | Limits Reflexion retries. When the limit is exceeded, `AgentResult.tool` is `REFLEXION_EXHAUSTED_TOOL` (`"__reflexion_exhausted__"`) |
| `plan=Plan(...)` | Updated from the `plan_update` key of LLM responses and returned as `AgentResult.plan` |
| `step_hook(entry)` | Called after each tool-trace entry. Exceptions are logged and ignored |

#### Work loop

`run_work_loop` runs a multi-step task. It first asks the LLM for a plan, then runs an `AgentRunner` with `history_mode="full_trace"`, `DefaultReflexionEvaluator`, and the terminal tool `FINISH_TOOL` (`"finish"`).

```python
from mltgnt.agent import WorkLoopConfig, run_work_loop

outcome = run_work_loop(
    order_text,
    llm_call=my_llm_call,                 # LLMCaller; may raise WorkLoopDeadline
    tools=my_tools,                       # ToolExecutor
    cfg=WorkLoopConfig(max_iterations=40, max_reflexions=5, tool_skills=("my-skill",)),
    events_sink=print,                    # optional; receives event dicts
    skill_runner=my_skill_runner,         # optional SkillRunner, e.g. GhdagSkillRunner
    parent_correlation_id="loop-uuid",
)
print(outcome.status, outcome.reason, outcome.message, outcome.artifacts)
```

| Name | Kind | Contract |
|------|------|----------|
| `run_work_loop` | function | `(order_text, *, llm_call, tools, cfg, events_sink=None, skill_runner=None, parent_correlation_id=None) -> WorkLoopOutcome` |
| `WorkLoopConfig` | frozen dataclass | `max_iterations=40`, `max_reflexions=5`, `tool_skills=()` (skills that the `run_skill` tool may call) |
| `WorkLoopOutcome` | dataclass | `status`, `reason`, `message`, `artifacts`, `plan`, `trace` |
| `WorkLoopDeadline` | exception | The host's `llm_call` raises it when the time budget runs out. The loop then returns `BLOCKED` |
| `FINISH_TOOL` | constant | `"finish"`. Its args `message` and `artifacts` become `WorkLoopOutcome.message` and `WorkLoopOutcome.artifacts` |
| `SkillRunner` | Protocol | `(skill_name, arguments, *, parent_correlation_id) -> tuple[bool, str]` |
| `GhdagSkillRunner` | class | `(skills, persona, *, engine, model, jobs_dir, exec_done_dir, timeout, permission_by_skill=None, persona_dir=None, enqueue=enqueue_and_wait)`. A `SkillRunner` that renders the skill with `mltgnt.skill.run` and submits it with `enqueue_and_wait` |
| `make_skill_tool` | function | `(tools, runner, *, allowed, parent_correlation_id) -> ToolExecutor`. Adds a `run_skill` tool (args `{"skill": ..., "arguments": ...}`) that accepts only skills in `allowed` |
| `run_skill_contract` | function | `(skills) -> str`. Prompt text that describes the `run_skill` tool. Returns `""` for an empty list |
| `RepeatGuard` | class | `(inner)`. A `ToolExecutor` wrapper that refuses the third identical `(tool, args)` call in a row with an `[ERROR]` result |
| `TrackingCaller` | class | `(inner, plan)`. An `LLMCaller` wrapper that appends plan progress to prompts and records `last_ok`, `last_raw`, and `llm_calls` |
| `is_plan_prompt` | function | `(prompt) -> bool`. `True` for the planning prompt that `build_plan_prompt` produces |

| `status` | `reason` |
|----------|----------|
| `IMPL_DONE` | `finished` |
| `IMPL_FAILED` | `llm_failed` |
| `BLOCKED` | `deadline_before_plan`, `deadline`, `max_iterations`, `reflexion_exhausted`, `unparseable_response` |

`events_sink` receives `work_loop_plan_failed` and `work_loop_step` events (`step`, `tool`, `args` and `result_head` truncated to 200 characters, `plan` progress). If the sink raises, the error is logged and ignored.

#### Other agent names

| Names | Purpose |
|-------|---------|
| `AgentRunner`, `AgentResult`, `DefaultReflexionEvaluator`, `LLMCaller`, `ToolExecutor`, `RetryConfig`, `REFLEXION_EXHAUSTED_TOOL` | Tool-calling loop, its result, evaluator, and exported Protocol / config types (`REFLEXION_EXHAUSTED_TOOL` is `"__reflexion_exhausted__"`) |
| `Plan`, `PlanItem`, `parse_plan`, `build_plan_prompt` | Plan tracking. `PlanItem(id, title, depends=[], status="pending", note="")`. `Plan.apply(updates)` and `Plan.progress() -> (done, total)`. `parse_plan` raises `ValueError` |
| `DispatchDecision`, `make_dispatch_decision`, `MODE_REPLY`, `MODE_DELEGATE` | Decide whether to reply or delegate (`"reply"` / `"delegate"`) |
| `should_force_delegate`, `should_preempt_delegate`, `has_work_request`, `is_create_request`, `match_deferred_promise`, `extract_artifact_references` | Deterministic request gates. Their vocabulary comes from the current `LanguagePack` |
| `PreflightContext`, `run_preflight`, `DirectAgentResult`, `SkillWorkerResult`, `MemoryWorkerResult` | Preflight before dispatch, and worker result types |

### `mltgnt.bridges`

| Names | Purpose |
|-------|---------|
| `enqueue_dag`, `enqueue_and_wait`, `DagStep` | Build ghdag steps, submit them, and wait for the results |
| `call_llm` | `(prompt, *, engine="", model="", timeout=120)`. One text call through ghdag |
| `MltgntHooks`, `create_audit_writer` | ghdag DAG hooks, and the audit writer for `AgentRunner(audit_writer=...)` |
| `md_read`, `md_write` | Wrappers around `ghdag.files` |
| `ghdag_bridge`, `llm_adapter`, `hooks_adapter`, `files_adapter` | Submodules. `files_adapter.commit(paths, message, *, sink="memory", trailers=None)` commits through a ghdag VCS sink |

### `mltgnt.persona`

| Names | Purpose |
|-------|---------|
| `Persona`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt` | Same as the top level |
| `PersonaContext` | Resolved persona context for one turn |
| `PersonaValidationError` | See [Error Reference](#error-reference) |
| `format_persona_body`, `format_result_for_persona` | Tone formatting, and result text in the persona's voice |
| `compress_heavy_to_light`, `regenerate_light_block` | LLM compression of the heavy persona block into the light block |

Persona section headings are canonical English (`Background`, `Values`, `Reaction patterns`, `Tone`, `Output format`, `Light`, `Heavy`, `Reference`, `Triage`). Localized legacy headings are recognized only through `LanguagePack.persona_section_aliases`. The default `EN` pack defines no aliases. See [LanguagePack](#languagepack).

### `mltgnt.memory`

| Group | Names |
|-------|-------|
| Episodic log | `append_memory_entry`, `read_memory_preferences`, `read_memory_tail_text`, `memory_file_path`, `persona_memory_lock`, `tail_utf8_bytes`, `assemble_entries_text` |
| Search | `read_memory_by_relevance`, `read_memory_with_sufficiency_check`, `read_memory_iterative` |
| Entries | `MemoryEntry`, `parse_jsonl`, `serialize_entry` |
| Compaction | `compact`, `needs_compaction`, `CompactionResult`, `LlmCall`, `LlmCallError` |
| Commits | `flush_memory_commits` |
| Chroma (optional) | `get_collection`, `query_similar`, `upsert_entry`. `get_collection` returns `None` when `chromadb` is not installed |
| Constants | `MEMORY_CORRUPT_THRESHOLD_BYTES` (`10`), `MEMORY_DEDUPE_SCAN_BYTES` (`32768`), `MEMORY_DEDUPE_SCAN_LINES` (`200`) |
| Internal compatibility exports | `_ensure_jsonl`, `_resolve_memory_dir`, `_scan_tail_for_dedupe_key`, `_search_and_score` (underscore-prefixed and unsupported) |

`mltgnt.memory.__all__` includes the internal compatibility exports above so the table remains an exact inventory. They are not supported public API despite appearing in `__all__`.

Compaction drops meta lines from the Phase 1 LLM output: lines that start with `Understood`, `Analyzing`, `Analysis`, or `Below`, `## ` headings, and lines that contain `**Size**`, `**Statistics**`, or `**Analysis**`. `LanguagePack.phase1_meta_prefixes` and `LanguagePack.phase1_meta_markers` add localized prefixes and markers to these.

Memory submodules that declare their own `__all__`:

| Module | Names | Purpose |
|--------|-------|---------|
| `mltgnt.memory.dream` | `DreamSection`, `DreamSummary`, `read_dream`, `write_dream`, `read_global`, `write_global`, `read_global_summary`, `DreamSelector`, `Synthesizer` | Per-persona `dream.json` and cross-persona `global.json` summaries |
| `mltgnt.memory.semantic` | `SemanticStore`, `SemanticEntry`, `KINDS`, `STATUSES`, `validate_entry`, `normalize_content` | `SemanticStore(path, *, persona_stem, commit_debounce_sec=None)` over `semantic.jsonl`. Kinds: `fact`, `preference`, `commitment`, `caveat`, `self`, `reflection`. Statuses: `active`, `superseded` |
| `mltgnt.memory.tools` | `MemoryToolExecutor`, `MemoryGate`, `MEMORY_TOOL_SPECS`, `MEMORY_TOOL_NAMES`, `query_terms` | `remember` / `recall` / `forget` tools in `ToolExecutor` form. `MemoryGate(max_per_turn=3, max_chars=300, secret_check=None, allowed_subject_prefixes=None)` |
| `mltgnt.memory.reflection` | `build_reflection_prompt`, `parse_reflection`, `apply_reflection`, `ReflectionResult`, `ReflectionAdd`, `ApplyReport`, `ReflectionParseError` | Reflection prompt, parser, and applier for a `SemanticStore` |
| `mltgnt.memory.core_render` | `render_core`, `MANDATORY_KINDS`, `OPTIONAL_KINDS` | `render_core(entries, *, max_bytes=4096, heading="## Memory")`. `caveat` and `commitment` entries are always kept |
| `mltgnt.memory.archive` | `archive_episodes` | `archive_episodes(persona_dir, *, now, keep_days=90) -> int`. Moves old episodes into monthly files |

### `mltgnt.skill`

| Names | Purpose |
|-------|---------|
| `discover`, `discover_bodies`, `load` | Find and parse skill files |
| `match` | `(user_input, skills, persona_skills=None, model=None, *, engine="claude") -> SkillMatchResult` |
| `resolve_skill` | `(user_input, skill_paths, persona_skills=None, entry_file="SKILL.md", matcher_model=None, *, matcher_engine="claude")`. Returns `(SkillFile, arguments)` or `None` |
| `run` | `(skill, persona, arguments, chat_input, extra_context=None) -> SkillRunResult`. Substitutes `$ARGUMENTS`, `$PERSONA`, `$SKILL_DIR`, `$REPO_ROOT`, `$0`, `$1`, ..., plus `$KEY` for each name in `get_skill_config().passthrough_env` (environment value, empty when unset; built-ins win). Other `$KEY` stay as is |
| `build_extra_context` | Extra prompt context from skill knowledge files and persona memory |
| `lint_skill_meta` | Validates skill frontmatter |
| `SkillMeta`, `SkillFile`, `SkillRegistry`, `SkillRunResult`, `SkillMatchResult`, `ArtifactSpec`, `ProducesSpec`, `ConsumesSpec` | Data types |

When `engine="claude"` and no model is given, the LLM matcher uses `claude-haiku-4-5-20251001`. Other engines use their ghdag default model.

### `mltgnt.routing`

| Names | Purpose |
|-------|---------|
| `resolve_persona` | Pick the persona that responds to a message in a space or conversation |
| `find_observers_in_space` | The other personas present in a space |
| `load_channel_persona_map` | Build the space-to-persona map from persona frontmatter |
| `SpacePersonaEntry`, `RoutingRule`, `evaluate` | Routing entries and first-match rules |
| `detect_nickname` | Detect a mention by persona nickname |
| `extract_triage_section` | `(markdown, *, pack=None) -> str \| None`. Body of `## Light`, falling back to `## Triage`, then to legacy headings that `pack.persona_section_aliases` maps to `Light` or `Triage` (`pack` defaults to `get_language_pack()`) |
| `extract_json_object`, `prepare_profile_for_triage`, `TRIAGE_PROFILE_MAX_CHARS` (`6000`) | Other triage helpers |
| `ChannelPersonaEntry` | Deprecated. See [Deprecated API](#deprecated-api) |

### `mltgnt.conversation`

| Names | Purpose |
|-------|---------|
| `configure`, `get_config`, `ConversationConfig` | Set and read the active conversation paths. `get_config` raises `RuntimeError` before `configure` is called |
| `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage` | Re-exported from `mltgnt.interfaces.turn` |
| `thread_queue`, `session_store`, `session_compact`, `thread_index`, `thread_persona_store`, `fake_media` | Submodules: per-conversation queue (`drained_attachments` collects image metadata from drained inbox entries for `MediaBridge`), session ledger, log compaction, thread index, persona bindings, and a `TurnInput` builder that needs no real medium |

### `mltgnt.scheduler`

| Names | Purpose |
|-------|---------|
| `PersonaScheduler` | Runs YAML-defined jobs. See [Scheduler jobs](#scheduler-jobs) |
| `ScheduleJob` | One job definition |
| `SchedulePaths` | `SchedulePaths(state_dir)`. Per-day state files (`done`, `planned`, `missed`, `failed`, `skipped`) |
| `load_schedule_jobs` | `(yaml_path, *, default_timezone="Asia/Tokyo")` |
| `atomic_write_text` | Atomic file write used for state files |
| `_hash_offset` | Internal compatibility export; underscore-prefixed and unsupported |

`SchedulePaths.prune(today, keep_days=30) -> int` deletes state files named `<job_id>_<YYYY-MM-DD>.<ext>` whose date is older than `today - keep_days`, and returns how many it deleted. Files whose date does not parse, and the interval directory, are left alone. `keep_days < 1` raises `ValueError`. `PersonaScheduler` calls it once per calendar day with `state_keep_days`. A prune failure is logged and does not stop the tick.

### `mltgnt.daemon` and `mltgnt.config`

| Package | Names |
|---------|-------|
| `mltgnt.daemon` | `DaemonComponent`, `DaemonRunner(*, pid_file, components, logger=None)`, `PidLock(pid_file)`, `SkillWatcherComponent(registry, interval=5.0)` |
| `mltgnt.config` | `PersonaConfig`, `MemoryConfig`, `SchedulerConfig`, `ConversationConfig`, `DEFAULT_WEIGHT_MAP` |
| `mltgnt.config.language` | `LanguagePack`, `EN`, `get_language_pack`, `set_language_pack`. The last three can also be imported from `mltgnt.config` |

`mltgnt.daemon.__all__` is exactly `DaemonComponent`, `DaemonRunner`, `PidLock`, and `SkillWatcherComponent`.

### `mltgnt.interfaces`

Dependency-free DTOs and Protocols that every layer shares.

| Module | Names |
|--------|-------|
| `mltgnt.interfaces` | `PersonaProtocol`, `PersonaFMBase`, `Message`, `ChatInput`, `ChatOutput`, `ChatInputBase`, `ChatOutputBase`, `TurnHandler`, `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, `HubAddress`, `HubDisplay`, `HubMessage`, `HubTransport`, `SlackClientProtocol` (deprecated) |
| `mltgnt.interfaces.media` | `Status`, `MediaClient`, `adapt_client` |

| Name | Contract |
|------|----------|
| `Status` | `str` Enum: `RECEIVED`, `WORKING`, `DONE`, `FAILED`, `CANCELLED`. The values are the lowercase names |
| `MediaClient` | `post(text, space, thread=None, *, extra=None) -> str \| None`, `update(message_id, text) -> bool`, `set_status(message_id, status) -> bool`, `upload(path, space, thread=None) -> bool`, `react(message_id, name) -> bool`. Failures return `None` or `False` instead of raising |
| `adapt_client` | `(obj) -> MediaClient`. An object with `post` passes through unchanged. An object with only `post_message` is wrapped, with one `DeprecationWarning`. Anything else raises `TypeError` |
| `HubDisplay` | `name`, `icon_url` (optional). Self-declared sender presentation on hub envelopes; not used for routing or auth. `from_mapping` / `to_mapping` round-trip JSON |
| `HubMessage` | Envelope with `display: HubDisplay \| None` on outbound rows (since `v0.131.0`) |

### `mltgnt.media`

`mltgnt.media` re-exports 27 names from `mltgnt.media._core` (see `mltgnt.media.__all__`; does not import `media.slack` / `media.webchat` at package import time). Slack and WebChat adapters are still imported from their subpackages below.

`mltgnt.media._core` (independent of any medium):

| Module | Main names | Purpose |
|--------|------------|---------|
| `types` | `MediaEvent`, `OutboundMessage` | `MediaEvent(space_id, conversation_id, message_id, author, text, attachments=(), raw=None)`. `OutboundMessage(text, space_id, thread_id=None)` |
| `client` | `MediaClient`, `Status`, `adapt_client` | Re-exports `mltgnt.interfaces.media` |
| `config` | `MediaConfig` | See [Media settings](#media-settings) |
| `bridge` | `MediaBridge` | `MediaBridge(client, handler, config, hooks=None, *, pending_prefix="pending-")`. `handle_event(event)` runs one turn and either posts a reply or records a pending task. `deliver_result(uid, body, *, post_options=None)` posts the result of a delegated task |
| `hooks` | `HookRegistry`, `OnInbound`, `BeforeDispatch`, `AfterPost`, `OnResult` | Turn hooks, run in registration order. A hook that raises is logged and skipped |
| `component` | `MediaBridgeComponent` | `(start, *, mode, backoff_sec, on_exit=None)`. Runs `start` in a process or a thread and restarts it with backoff |
| `watchers` | `ExecDoneHandler`, `ProgressWatcher`, `DeliveryReconciler`, `catchup_pending_on_startup`, `iter_pending_with_done`, `CATCHUP_STATES` | Job completion, catch-up at startup, delivery reconciliation, progress polling |
| `progress` | `ProgressState`, `finalize_progress`, `status_for_done_marker`, `status_label`, `summarize_tool_use_block`, `summarize_work_loop_step` | Progress message text |
| `plan_gate` | `PlanGate`, `PlanState`, `is_approval`, `expire_pending`, `AWAITING_STATE`, `PENDING_KEY` | Plan approval state machine |
| `cancel` | `CancelOutcome`, `handle_cancel`, `is_cancel_request`, `find_pending_uids` | Cancel requests posted in a thread |
| `enqueue_guard` | `enqueue_or_report` | Posts `LanguagePack.enqueue_failed_text` when a submission fails |
| `pending` | `PendingStore` | One JSON record per pending request |
| `id_map` | `to_conversation_id`, `resolve`, `storage_key`, `storage_key_from_conversation_id` | Converts between a conversation id and `(space, thread)` |
| `thread_reactions` | `admit`, `status_for_admission`, `acknowledge_drained` | Maps queue admission to a `Status` |
| `sanitizer` | `sanitize_result_body`, `strip_status_lines`, `strip_leading_paragraphs`, `dedupe_trailing_repeated_block`, `extract_final_assistant_text` | Cleans a result before it is posted |

`MediaBridge` passes `TurnResult.post_options` to `MediaClient.post(extra=...)`, both for immediate replies and in `deliver_result`.

`mltgnt.media.slack` (requires the `slack` extra):

| Module | Main names | Purpose |
|--------|------------|---------|
| `config` | `SlackMediaConfig`, `DEFAULT_STATUS_REACTIONS` | See [Media settings](#media-settings) |
| `client` | `SlackClient`, `split_text`, `POST_EXTRA_KEYS`, `UPDATE_EXTRA_KEYS` | `SlackClient(web_client, config, *, default_channel="")`. Statuses become reactions. Long text is split at `chunk_max_chars`. `POST_EXTRA_KEYS = {"username", "icon_emoji", "icon_url", "blocks", "reply_broadcast"}` and `UPDATE_EXTRA_KEYS = {"blocks"}`. Any other `extra` key raises `ValueError` before the API call. Also provides `upload(..., *, title=None)`, `react`, and `unreact` |
| `app` | `build_app`, `start_socket_mode` | `build_app(config)` creates a Bolt app. `start_socket_mode(app, config)` blocks in Socket Mode |
| `inbound` | `to_media_event`, `strip_mentions`, `thread_ts_from_event`, `files_to_attachments` | Converts a Slack event to a `MediaEvent` |
| `history` | `fetch_thread_messages`, `extract_text_from_message`, `extract_text_from_blocks`, `table_block_to_markdown` | Fetches threads and extracts text |
| `media_store` | `save_images`, `SavedMedia`, `safe_filename`, `ext_from_mimetype` | Downloads image attachments |
| `mrkdwn` | `markdown_to_mrkdwn`, `normalize_markdown_residual` | Converts Markdown to Slack mrkdwn |

`mltgnt.media.webchat` (the app requires the `webchat` extra):

| Module | Main names | Purpose |
|--------|------------|---------|
| `config` | `WebChatMediaConfig`, `WebPushConfig` | See [Media settings](#media-settings) |
| `pwa` | `manifest`, `service_worker_source`, `resolve_pwa_icon` | PWA manifest, service worker source, and icon paths |
| `push` | `PushSubscriptionStore`, `PushWorker`, `push_enabled` | Web Push subscription JSONL store and tail worker |
| `client` | `WebChatClient` | `WebChatClient(config, *, store=None, files=None, author="assistant")`. `upload(path, space, thread)` posts a PNG / JPEG / GIF / WebP image as a message |
| `files` | `WebChatFiles`, `CONTENT_TYPES`, `content_type_for` | Message images under `<files_dir>/<YYYY-MM-DD>/<hex>.<ext>`; the relative path is the file id |
| `store` | `WebChatStore`, `KINDS`, `MESSAGE_KINDS` | `WebChatStore` subclasses `JsonlMessageStore` (`mltgnt.media._core.jsonl_store`). Daily JSONL of message snapshots. `KINDS = ("message", "update", "status", "bookmark", "reaction")` |
| `app` | `create_app`, `serve`, `EventHandler`, `parse_event_id` | `create_app(config, bridge, *, store=None, files=None, poll_interval_sec=0.5, stream_timeout_sec=None)` serves `GET /` (redirects to `/c/<channel_slug>`), slug routes `GET /c/{slug}`, `GET /c/{slug}/t/{message_id}` (thread permalink), `GET /c/{slug}/p/{message_id}` (message permalink), `GET /c/{slug}/bookmarks`, `GET /config` (includes `push_enabled` and `channel_slug`), PWA routes (`/manifest.webmanifest`, `/sw.js`, `/pwa/icon-192.png`, `/pwa/icon-512.png`), optional push routes when enabled (`/push/vapid-public-key`, `POST /push/subscriptions`, `DELETE /push/subscriptions`), `GET /assets/{name}`, `GET /messages` (`?day=`, or `?limit=` with optional `&before=<ISO ts>` to page back), `POST /messages` (`text` and/or `files: [{"id", "name"}]`; files become `MediaEvent.attachments` with the local path as `uri`), `POST /files?name=` (raw image body), `GET /files/{day}/{name}`, `GET /threads/{thread_id}`, `POST /messages/{message_id}/bookmark`, `GET /bookmarks`, and `GET /stream` (SSE). `PushWorker` tails the store and sends agent-reply notifications when push is enabled. `serve(app, config)` listens on `config.host:config.port` |
| `inbound` | `to_media_event`, `file_entries`, `DEFAULT_AUTHOR`, `MAX_FILES` | Converts a `POST /messages` body to a `MediaEvent` |
| `ui` | `INDEX_HTML` | Single-page UI served at `GET /`: dark, Slack-like single channel with a thread pane, image attach (button / drag and drop / paste) and a lightbox |


`mltgnt.media.hub` (requires the `hub` extra):

| Module | Main names | Purpose |
|--------|------------|---------|
| `config` | `HubMediaConfig` | Remote hub client settings: `hub_url`, `host_id`, `token_env`, `rooms`, `cursor_dir`, `poll_interval_sec`, `space_prefix`, `guard`, optional `outbound_filter` |
| `guard` | `GuardConfig`, `GuardVerdict`, `check` | Pure hop / rate / agent-run limits before posting (`hop_limit`, `max_agent_run`, `min_interval_sec`, `room_daily_limit`) |
| `client` | `HubClient`, `UrllibTransport` | `HubClient(config, *, persona, transport=None, display=None)` is a `MediaClient` over the hub HTTP API; optional `display` sets `HubMessage.display` on outbound posts. `UrllibTransport` implements `HubTransport` |
| `inbound` | `to_media_event` | Hub poll payload → `MediaEvent` (uses `space_prefix` + room id) |
| `subscriber` | `HubSubscriber` | `DaemonComponent` that polls rooms, persists cursors under `cursor_dir`, and calls `handle_event` |
| `server` | `HubServerConfig`, `load_server_config`, `create_app`, `serve`, `parse_cursor`, `purge_expired_daily_files` | FastAPI relay with per-room `JsonlMessageStore`, bearer auth from `token_env`, guard checks, SSE poll |

Hub envelopes live in `mltgnt.interfaces.hub` (`HubAddress`, `HubDisplay`, `HubMessage`, `HubTransport`) so core layers never import `mltgnt.media.hub`.

## Protocols / Extension Points

| Extension point | Module | Contract | Consumed by |
|-----------------|--------|----------|-------------|
| `MediaClient` | `mltgnt.interfaces.media` | `post` / `update` / `set_status` / `upload` / `react` | `PersonaScheduler`, `MediaBridge`. Implemented by `SlackClient` and `WebChatClient` |
| `TurnHandler` | `mltgnt.interfaces.turn` | `handle(turn: TurnInput) -> TurnResult`. `TurnResult.kind` is `"reply"` or `"task"`; it also carries `text`, `task_ref`, `reaction`, and `post_options` | `MediaBridge` |
| `HookRegistry` hooks | `mltgnt.media._core.hooks` | `OnInbound(event) -> bool` (`True` stops the turn), `BeforeDispatch(turn) -> TurnInput`, `AfterPost(result, message_id)`, `OnResult(uid, body)` | `MediaBridge` |
| `EventHandler` | `mltgnt.media.webchat.app` | `handle_event(event: MediaEvent)` | `create_app`. `MediaBridge` satisfies it |
| `DaemonComponent` | `mltgnt.daemon` | `name` property, non-blocking `start()`, `stop()` | `DaemonRunner`, `mltgnt run` |
| `HubTransport` | `mltgnt.interfaces.hub` | `send`, `poll`, `revise`, `guard_context` for hub HTTP relay | `HubClient`, `UrllibTransport`, `HubSubscriber` |
| `EventHandler` | `mltgnt.media.hub.subscriber` | `handle_event(event: MediaEvent)` | `HubSubscriber` polls rooms and delivers inbound events |
| `LLMCaller` | `mltgnt.agent._runner` | `(prompt, *, tool_result=None) -> str \| None` | `AgentRunner(llm_call=...)`, `run_work_loop(llm_call=...)` |
| `ToolExecutor` | `mltgnt.agent._runner` | `(tool_name, tool_args) -> str`. `MemoryToolExecutor` is a built-in implementation | `AgentRunner(tool_executor=...)`, `run_work_loop(tools=...)` |
| `ReflexionEvaluator` | `mltgnt.agent._runner` | `(prompt, tool_name, tool_args, tool_result, tool_trace) -> ReflexionVerdict` | `AgentRunner(evaluator=...)` |
| `SkillRunner` | `mltgnt.agent.work_loop` | `(skill_name, arguments, *, parent_correlation_id) -> tuple[bool, str]`. `GhdagSkillRunner` is the built-in implementation | `run_work_loop(skill_runner=...)`, `make_skill_tool` |
| `ActionFn` | `mltgnt.scheduler.models` | `(job: ScheduleJob) -> tuple[bool, str]` | `PersonaScheduler(actions={name: fn})` |
| `PersonaProtocol`, `PersonaFMBase`, `ChatInputBase`, `ChatOutputBase` | `mltgnt.interfaces` | Structural types for personas and chat I/O | Skill runner, routing |
| `LanguagePack` | `mltgnt.config.language` | Frozen dataclass of locale vocabulary | See [LanguagePack](#languagepack) |

## Architecture

### Packages

All packages live under `src/mltgnt/`.

| Package | Role |
|---------|------|
| `agent` | Tool-calling loop, work loop with `run_skill`, dispatch decision, Reflexion, plan tracking |
| `bridges` | ghdag integration: LLM calls, DAG submission and cancel requests, file I/O, VCS commits, audit hooks |
| `cli` | `mltgnt` console entry point (`run`, `memory dream`, `hub serve`) |
| `config` | Frozen config dataclasses and `LanguagePack` |
| `conversation` | Per-conversation queue, session ledger, thread index, persona bindings |
| `daemon` | `DaemonRunner`, PID lock, `SkillWatcherComponent` |
| `interfaces` | Dependency-free DTOs and Protocols (`media`, `turn`, `hub` envelopes) |
| `media` | `MediaBridge` core (`_core`, shared `jsonl_store`) plus `slack`, `webchat`, and `hub` adapters |
| `memory` | Episodic log, compaction, semantic store, dream summaries (`dream`), search |
| `persona` | Markdown persona loading, validation, compression, prompt execution |
| `routing` | Space-to-persona resolution, triage helpers, agentic skill discovery |
| `scheduler` | YAML-driven jobs with the built-in `noop`, `skill`, and `memory_dream` actions (`actions`) and state pruning |
| `skill` | Skill discovery, matching, variable substitution, execution |

`exceptions.py` defines `MltgntError`, `ConfigError`, and `DependencyError`. `__main__.py` makes `python -m mltgnt` work.

### Modules

Every Python module under `src/mltgnt/` (145 modules at `v0.134.0`). "Depends toward" follows `.importlinter` layers: higher layers import inward toward `interfaces` and ghdag via `bridges`.

| Module | Responsibility | Depends toward |
|--------|----------------|----------------|
| `mltgnt.__init__` | mltgnt — type contracts and chat I/O for persona × secretary (OSS core). | entry |
| `mltgnt.__main__` | Implementation module. | entry |
| `mltgnt.agent.__init__` | mltgnt.agent — generic agent loop + decision-layer skeleton (#3318). | L2 → L3–L5 |
| `mltgnt.agent._parse` | mltgnt.agent._parse — parse JSON from an LLM response. | L2 → L3–L5 |
| `mltgnt.agent._runner` | mltgnt.agent._runner — generic agent loop. | L2 → L3–L5 |
| `mltgnt.agent.action_classifier` | mltgnt.agent.action_classifier — auto-classify tool side-effect levels. | L2 → L3–L5 |
| `mltgnt.agent.deterministic_gate` | Deterministic gate (no side effects, medium-agnostic) (#3318). | L2 → L3–L5 |
| `mltgnt.agent.dispatch_decision` | Dispatch decision skeleton (#3318). | L2 → L3–L5 |
| `mltgnt.agent.dispatch_preflight` | Preflight skeleton before dispatch (#3318). | L2 → L3–L5 |
| `mltgnt.agent.plan` | mltgnt.agent.plan — work plan data structure and JSON contract (#3855). | L2 → L3–L5 |
| `mltgnt.agent.reflexion` | mltgnt.agent.reflexion — deterministic default Reflexion evaluator (#3855). | L2 → L3–L5 |
| `mltgnt.agent.work_loop` | mltgnt.agent.work_loop — generic work loop with optional skill delegation. | L2 → L3–L5 |
| `mltgnt.bridges.__init__` | Implementation module. | L4 → L5 (+ ghdag) |
| `mltgnt.bridges.audit_adapter` | Implementation module. | L4 → L5 (+ ghdag) |
| `mltgnt.bridges.files_adapter` | mltgnt.bridges.files_adapter | L4 → L5 (+ ghdag) |
| `mltgnt.bridges.ghdag_bridge` | mltgnt.bridges.ghdag_bridge — wrapper around LLMPipelineAPI + wait_for_result. | L4 → L5 (+ ghdag) |
| `mltgnt.bridges.hooks_adapter` | mltgnt.bridges.hooks_adapter — audit_writer callback factory for AgentRunner + DagHooks adapter. | L4 → L5 (+ ghdag) |
| `mltgnt.bridges.llm_adapter` | mltgnt.bridges.llm_adapter | L4 → L5 (+ ghdag) |
| `mltgnt.cli.__init__` | Implementation module. | entry (CLI) |
| `mltgnt.cli.hub` | mltgnt.cli.hub — hub serve subcommand. | entry (CLI) |
| `mltgnt.cli.main` | Implementation module. | entry (CLI) |
| `mltgnt.cli.memory` | mltgnt.cli.memory — memory dream show/forget subcommands. | entry (CLI) |
| `mltgnt.cli.run` | Implementation module. | entry (CLI) |
| `mltgnt.config.__init__` | mltgnt.config — diary-independent configuration dataclasses. | cross-cutting (any layer) |
| `mltgnt.config.language` | mltgnt.config.language — language-pack dataclass for locale-specific vocabulary. | cross-cutting (any layer) |
| `mltgnt.conversation.__init__` | mltgnt.conversation — media-agnostic conversation layer (#3317). | L3 → L4–L5 |
| `mltgnt.conversation.fake_media` | Fake media layer — build TurnInput without Slack (#3317). | L3 → L4–L5 |
| `mltgnt.conversation.session_compact` | Automatic compaction of conversation logs (#3317). | L3 → L4–L5 |
| `mltgnt.conversation.session_store` | Conversation-layer session ledger and turn log (#3317). | L3 → L4–L5 |
| `mltgnt.conversation.thread_index` | Conversation-layer thread index (#3317). | L3 → L4–L5 |
| `mltgnt.conversation.thread_persona_store` | JSONL persistence for conversation-bound personas (#3317). | L3 → L4–L5 |
| `mltgnt.conversation.thread_queue` | Conversation-layer wait queue (#3317); persists attachment metadata on enqueue and exposes `drained_attachments` when a turn is admitted. | L3 → L4–L5 |
| `mltgnt.conversation.types` | Conversation-layer boundary types — re-export mltgnt.interfaces.turn (#3317). | L3 → L4–L5 |
| `mltgnt.daemon.__init__` | Implementation module. | L1 → L2–L5 |
| `mltgnt.daemon._pidlock` | Implementation module. | L1 → L2–L5 |
| `mltgnt.daemon._runner` | Implementation module. | L1 → L2–L5 |
| `mltgnt.daemon._skill_watcher` | mltgnt.daemon._skill_watcher — DaemonComponent that watches skills/ and hot-reloads. | L1 → L2–L5 |
| `mltgnt.exceptions` | mltgnt.exceptions — shared exception type hierarchy. | shared |
| `mltgnt.interfaces.__init__` | Implementation module. | L5 (no inward deps) |
| `mltgnt.interfaces.hub` | Hub envelope types and transport boundary (no media-layer imports). | L5 (no inward deps) |
| `mltgnt.interfaces.media` | Media-agnostic client contract (MediaClient / Status) and the legacy adapter. | L5 (no inward deps) |
| `mltgnt.interfaces.persona` | Implementation module. | L5 (no inward deps) |
| `mltgnt.interfaces.slack` | Implementation module. | L5 (no inward deps) |
| `mltgnt.interfaces.turn` | Conversation-layer boundary types (neck: TurnInput / TurnResult) and TurnHandler Protocol. | L5 (no inward deps) |
| `mltgnt.interfaces.types` | L1 DTO — type definitions for the interfaces layer. | L5 (no inward deps) |
| `mltgnt.media.__init__` | Media layer (Slack / WebChat implementations live in submodules). | L1 → L2–L5 |
| `mltgnt.media._core.__init__` | Media-agnostic core shared by media implementations. | L1 → L2–L5 |
| `mltgnt.media._core.bridge` | Media-agnostic flow of one turn: MediaEvent -> admit -> TurnHandler -> post -> finalize. | L1 → L2–L5 |
| `mltgnt.media._core.cancel` | Cancel a conversation's jobs from the thread. | L1 → L2–L5 |
| `mltgnt.media._core.client` | Re-export of the MediaClient contract defined in ``mltgnt.interfaces.media``. | L1 → L2–L5 |
| `mltgnt.media._core.component` | Supervise a media bridge: run it, report each exit, restart with backoff. | L1 → L2–L5 |
| `mltgnt.media._core.config` | Settings shared by media implementations. Implementations subclass MediaConfig. | L1 → L2–L5 |
| `mltgnt.media._core.enqueue_guard` | Guard a job enqueue: a failure is reported in the thread instead of being dropped. | L1 → L2–L5 |
| `mltgnt.media._core.hooks` | Host hook points around one media turn. | L1 → L2–L5 |
| `mltgnt.media._core.id_map` | Opaque conversation id <-> (space, thread) mapping. | L1 → L2–L5 |
| `mltgnt.media._core.jsonl_store` | Append-only JSONL message store (one file per day). Shared by webchat and hub. | L1 → L2–L5 |
| `mltgnt.media._core.pending` | Pending-request metadata store (one JSON file per request id). | L1 → L2–L5 |
| `mltgnt.media._core.plan_gate` | Plan approval state machine: ``awaiting -> approved \| rejected \| expired``. | L1 → L2–L5 |
| `mltgnt.media._core.progress` | Summarize a job's events JSONL into a progress message and rewrite it in place. | L1 → L2–L5 |
| `mltgnt.media._core.sanitizer` | Pure helpers that strip internal markers from an agent result before posting. | L1 → L2–L5 |
| `mltgnt.media._core.thread_reactions` | Reaction layer of the conversation wait queue: admit result -> message status. | L1 → L2–L5 |
| `mltgnt.media._core.types` | Media-agnostic inbound / outbound message types. | L1 → L2–L5 |
| `mltgnt.media._core.watchers` | Job completion detection, startup catch-up, delivery reconciliation and progress polling. | L1 → L2–L5 |
| `mltgnt.media.hub.__init__` | Hub medium: multi-host JSONL message relay (no slack/webchat imports). | L1 → L2–L5 |
| `mltgnt.media.hub.client` | HubClient (MediaClient) and urllib-based HubTransport. | L1 → L2–L5 |
| `mltgnt.media.hub.config` | Hub media client / subscriber settings (stdlib only; no FastAPI import). | L1 → L2–L5 |
| `mltgnt.media.hub.guard` | Pure guard checks for hub message posting (no I/O). | L1 → L2–L5 |
| `mltgnt.media.hub.inbound` | Hub envelope rows -> MediaEvent (no FastAPI). | L1 → L2–L5 |
| `mltgnt.media.hub.server` | FastAPI hub server: per-room JSONL stores and bearer auth. | L1 → L2–L5 |
| `mltgnt.media.hub.subscriber` | Poll hub rooms and deliver inbound messages to a host handler. | L1 → L2–L5 |
| `mltgnt.media.slack.__init__` | Slack media implementation. Import from the submodules (requires the ``slack`` extra at runtime). | L1 → L2–L5 |
| `mltgnt.media.slack.app` | Slack Bolt App factory and Socket Mode start-up. Bolt is imported lazily. | L1 → L2–L5 |
| `mltgnt.media.slack.client` | SlackClient: MediaClient implementation over a Slack WebClient. | L1 → L2–L5 |
| `mltgnt.media.slack.config` | Slack-specific media settings. | L1 → L2–L5 |
| `mltgnt.media.slack.history` | Text extraction from Slack messages (blocks / attachments / tables) and thread fetch. | L1 → L2–L5 |
| `mltgnt.media.slack.inbound` | Slack event dict -> MediaEvent. | L1 → L2–L5 |
| `mltgnt.media.slack.media_store` | Download image files attached to a Slack event and store them locally. | L1 → L2–L5 |
| `mltgnt.media.slack.mrkdwn` | Markdown -> Slack mrkdwn conversion. | L1 → L2–L5 |
| `mltgnt.media.webchat.__init__` | WebChat media implementation. Import from the submodules (the app requires the ``webchat`` extra). | L1 → L2–L5 |
| `mltgnt.media.webchat.app` | FastAPI app of the WebChat medium. FastAPI / uvicorn are imported lazily. | L1 → L2–L5 |
| `mltgnt.media.webchat.client` | WebChatClient: MediaClient implementation over the WebChat store. | L1 → L2–L5 |
| `mltgnt.media.webchat.config` | WebChat-specific media settings. | L1 → L2–L5 |
| `mltgnt.media.webchat.files` | Image files attached to WebChat messages. | L1 → L2–L5 |
| `mltgnt.media.webchat.inbound` | WebChat ``POST /messages`` body -> MediaEvent. | L1 → L2–L5 |
| `mltgnt.media.webchat.push` | Web Push subscription JSONL store, validation, notification payload builder, and ``PushWorker`` tail for agent replies. | L1 → L2–L5 |
| `mltgnt.media.webchat.pwa` | PWA manifest, service worker source, and safe resolution of custom PWA icon paths. | L1 → L2–L5 |
| `mltgnt.media.webchat.store` | Append-only message store: one ``YYYY-MM-DD.jsonl`` file per day, one JSON row per line. | L1 → L2–L5 |
| `mltgnt.media.webchat.ui` | Static single-page UI of the WebChat medium (served by ``GET /``). | L1 → L2–L5 |
| `mltgnt.memory.__init__` | mltgnt.memory — re-export hub. | L3 → L4–L5 |
| `mltgnt.memory._chroma` | Semantic search backend using a Chroma vector DB. | L3 → L4–L5 |
| `mltgnt.memory._commit` | mltgnt.memory._commit — debounced git commits of memory files (nexus #3833). | L3 → L4–L5 |
| `mltgnt.memory._format` | mltgnt.memory._format — parse and format memory files. | L3 → L4–L5 |
| `mltgnt.memory._iterative` | mltgnt.memory._iterative — iterative retrieval using judge_sufficiency. | L3 → L4–L5 |
| `mltgnt.memory._scoring` | Scoring via cosine similarity. | L3 → L4–L5 |
| `mltgnt.memory._sufficiency` | mltgnt.memory._sufficiency — LLM-based sufficiency judgment. | L3 → L4–L5 |
| `mltgnt.memory._tfidf` | TF-IDF-based text vectorization. | L3 → L4–L5 |
| `mltgnt.memory.api` | mltgnt.memory.api — path resolve, lock, append, read (CRUD). | L3 → L4–L5 |
| `mltgnt.memory.archive` | mltgnt.memory.archive — move old episodes to monthly archive files. | L3 → L4–L5 |
| `mltgnt.memory.compaction` | mltgnt.memory.compaction — memory compaction (per-section cap). | L3 → L4–L5 |
| `mltgnt.memory.core_render` | mltgnt.memory.core_render — render the always-loaded memory core. | L3 → L4–L5 |
| `mltgnt.memory.dream.__init__` | mltgnt.memory.dream — dream.json synthesis API. | L3 → L4–L5 |
| `mltgnt.memory.dream._format` | mltgnt.memory.dream._format — DreamSection / DreamSummary and JSON conversion. | L3 → L4–L5 |
| `mltgnt.memory.dream.api` | mltgnt.memory.dream.api — read/write dream.json. | L3 → L4–L5 |
| `mltgnt.memory.dream.selector` | mltgnt.memory.dream.selector — select personas for synthesis. | L3 → L4–L5 |
| `mltgnt.memory.dream.synthesizer` | mltgnt.memory.dream.synthesizer — synthesize DreamSummary via LLM. | L3 → L4–L5 |
| `mltgnt.memory.reflection` | mltgnt.memory.reflection — reflection prompt, parser and applier. | L3 → L4–L5 |
| `mltgnt.memory.search` | mltgnt.memory.search — relevance search, sufficiency, iterative retrieval. | L3 → L4–L5 |
| `mltgnt.memory.semantic` | mltgnt.memory.semantic — semantic memory store (``semantic.jsonl``). | L3 → L4–L5 |
| `mltgnt.memory.tools` | mltgnt.memory.tools — ``remember`` / ``recall`` / ``forget`` tool executor. | L3 → L4–L5 |
| `mltgnt.persona.__init__` | mltgnt.persona — persona management module | L3 → L4–L5 |
| `mltgnt.persona.compress` | mltgnt.persona.compress | L3 → L4–L5 |
| `mltgnt.persona.extractor` | mltgnt.persona.extractor — H2 section parse and light/heavy text extract (#3318). | L3 → L4–L5 |
| `mltgnt.persona.formatter` | mltgnt.persona.formatter — tone formatting and persona prefixes (#3318). | L3 → L4–L5 |
| `mltgnt.persona.frontmatter` | mltgnt.persona.frontmatter | L3 → L4–L5 |
| `mltgnt.persona.loader` | mltgnt.persona.loader | L3 → L4–L5 |
| `mltgnt.persona.memory` | mltgnt.persona.memory — media-agnostic helpers for persona memory (#3318). | L3 → L4–L5 |
| `mltgnt.persona.phrases` | mltgnt.persona.phrases — read ops.phrases (#3318). | L3 → L4–L5 |
| `mltgnt.persona.registry` | mltgnt.persona.registry | L3 → L4–L5 |
| `mltgnt.persona.resolve` | mltgnt.persona.resolve — choose responder and PersonaContext (#3318). | L3 → L4–L5 |
| `mltgnt.persona.result_format` | mltgnt.persona.result_format — LLM call skeleton for persona formatting (#3318). | L3 → L4–L5 |
| `mltgnt.persona.runner` | mltgnt.persona.runner | L3 → L4–L5 |
| `mltgnt.persona.schema` | mltgnt.persona.schema | L3 → L4–L5 |
| `mltgnt.persona.types` | mltgnt.persona.types — persona-layer boundary data (#3318). | L3 → L4–L5 |
| `mltgnt.routing.__init__` | mltgnt.routing — space → persona routing (media-agnostic). | L2 → L3–L5 |
| `mltgnt.routing.agentic_discover` | mltgnt.routing.agentic_discover — skill discovery via AgenticSkillDiscoverer. | L2 → L3–L5 |
| `mltgnt.routing.channel_router` | src/mltgnt/routing/channel_router.py | L2 → L3–L5 |
| `mltgnt.routing.triage` | mltgnt.routing.triage | L2 → L3–L5 |
| `mltgnt.scheduler.__init__` | mltgnt.scheduler — job dispatch core + YAML parser. | L2 → L3–L5 |
| `mltgnt.scheduler.actions.__init__` | Implementation module. | L2 → L3–L5 |
| `mltgnt.scheduler.actions.dream` | mltgnt.scheduler.actions.dream — memory_dream schedule action. | L2 → L3–L5 |
| `mltgnt.scheduler.actions.skill` | Implementation module. | L2 → L3–L5 |
| `mltgnt.scheduler.base_runner` | mltgnt.scheduler.base_runner — shared ABC for the tick loop. | L2 → L3–L5 |
| `mltgnt.scheduler.fanout` | Implementation module. | L2 → L3–L5 |
| `mltgnt.scheduler.loader` | Implementation module. | L2 → L3–L5 |
| `mltgnt.scheduler.models` | Implementation module. | L2 → L3–L5 |
| `mltgnt.scheduler.runner` | Implementation module. | L2 → L3–L5 |
| `mltgnt.scheduler.state` | Implementation module. | L2 → L3–L5 |
| `mltgnt.skill.__init__` | mltgnt.skill — load and run Markdown-based skill files. | L3 → L4–L5 |
| `mltgnt.skill._registry` | mltgnt.skill._registry — skill registry. | L3 → L4–L5 |
| `mltgnt.skill.context` | mltgnt.skill.context — build extra_context from knowledge + memory. | L3 → L4–L5 |
| `mltgnt.skill.lint` | mltgnt.skill.lint — structural validation of SKILL.md frontmatter (V1–V14). | L3 → L4–L5 |
| `mltgnt.skill.loader` | mltgnt.skill.loader — glob SKILL.md and parse frontmatter. | L3 → L4–L5 |
| `mltgnt.skill.matcher` | mltgnt.skill.matcher — skill matching (slash / literal / triggers / LLM). | L3 → L4–L5 |
| `mltgnt.skill.models` | mltgnt.skill.models — SkillMeta / SkillFile dataclass definitions. | L3 → L4–L5 |
| `mltgnt.skill.runner` | mltgnt.skill.runner — variable substitution and prompt composition. | L3 → L4–L5 |


## Configuration

Inventory source: configuration dataclasses in `mltgnt.config`, media configs, and direct environment reads in `src/mltgnt/**/*.py` at `v0.134.0`.

mltgnt hardcodes no host paths. Every directory comes from a config object or a function argument.

### Environment variables

| Variable | Read by | Default | Effect |
|----------|---------|---------|--------|
| names in `get_skill_config().passthrough_env` | `mltgnt.skill.runner` | empty | Replaces `$KEY` in skill bodies. Declare at startup with `from mltgnt.config import SkillConfig, set_skill_config` and `set_skill_config(SkillConfig(passthrough_env=("NOTES_ROOT", ...)))`. Undeclared `$KEY` stay as is (since `v0.126.0`, `$NIKKI_ROOT` is not implicit). Built-in keys win over passthrough |
| `REPO_ROOT` | `mltgnt.skill.runner` | empty | Replaces `$REPO_ROOT` in skill bodies |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge.enqueue_dag` | enabled | `0` turns off the skill I/O type check between DAG steps |
| `THREAD_PERSONA_TTL_DAYS` | `mltgnt.conversation.thread_persona_store` | `30` | Lifetime of conversation-to-persona bindings. Invalid values fall back to `30`. Ignored once a `ConversationConfig` is configured |
| `MLTGNT_DEFAULT_ENGINE` | `mltgnt.persona.schema.system_default_engine` | `claude` | Engine used when neither the persona nor the caller names one. Read on every call. Must be `claude`, `codex`, `cursor`, or `gemini`; any other value raises `ValueError` |
| `SLACK_BOT_TOKEN` | `mltgnt.media.slack.app.build_app` | - | Slack bot token. The variable name comes from `SlackMediaConfig.bot_token_env`. If unset, `RuntimeError` is raised |
| `SLACK_APP_TOKEN` | `mltgnt.media.slack.app.start_socket_mode` | - | Slack app-level token. The variable name comes from `SlackMediaConfig.app_token_env`. If unset, `RuntimeError` is raised |

These variables are read by ghdag but change how mltgnt behaves:

| Variable | Effect |
|----------|--------|
| `ENABLE_GIT` | `1`, `true`, or `yes` turns on git commits of memory files. Otherwise the ghdag sink does nothing |
| `GHDAG_VCS_CONFIG` | Path to the ghdag VCS sink YAML. It must define the `memory` sink for memory commits to happen |

Memory appends, the final compaction write, and `dream.json` / `global.json` writes schedule a commit on the ghdag `memory` sink. Commits are debounced per path by `MemoryConfig.commit_debounce_sec` (`<= 0` commits immediately). Commit failures are logged and never raised. `mltgnt.memory.flush_memory_commits()` commits pending paths right away, and it also runs at process exit.

### Config dataclasses (`mltgnt.config`)

All of them are frozen dataclasses.

| Class | Required fields | Optional fields (default) |
|-------|-----------------|---------------------------|
| `PersonaConfig` | - | `weight_map` (copy of `DEFAULT_WEIGHT_MAP`), `section_aliases` (copy of `get_language_pack().persona_section_aliases` when the config is built; empty with `EN`), `exclude_stems` (`frozenset()`) |
| `MemoryConfig` | `chat_dir` | `chat_memory_dir` (`None`), `inject_max_bytes` (`10240`), `inject_max_entries` (`12`), `preferences_max_bytes` (`5120`), `lock_timeout_sec` (`30.0`), `lock_stale_threshold_sec` (`300.0`), `raw_days` (`7`), `mid_weeks` (`3`), `compact_threshold_bytes` (`40960`), `compact_target_bytes` (`25600`), `preferences_section_name`, `protected_layers` (`("caveat",)`), `timezone` (`"Asia/Tokyo"`), `dream_model` (`""`), `dream_engine` (`"claude"`), `use_dream_summary` (`False`), `dream_dir_name` (`"memory"`), `commit_debounce_sec` (`300.0`), `global_dream_exclude_personas` (`()`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir` | `timezone` (`"Asia/Tokyo"`), `salt` (`""`) |
| `SkillConfig` | - | `passthrough_env` (`()`). Configure with `set_skill_config(SkillConfig(...))` at process entry; read with `get_skill_config()`. Not in `mltgnt.config.__all__` but importable from `mltgnt.config` |
| `ConversationConfig` | `queue_dir`, `sessions_dir`, `ledger_dir`, `thread_index_dir`, `thread_persona_path` | `posts_dir` (`None`), `audit_path` (`None`), `stale_after_sec` (`3600`), `max_queued` (`20`), `cleanup_ttl_days` (`14`), `thread_persona_ttl_days` (`30`) |

`DEFAULT_WEIGHT_MAP` maps `Background`, `Values`, `Reaction patterns`, and `Tone` to `heavy`, `Output format` to `reference`, and `Light` to `light`. Its `get` also resolves alias keys through the current `LanguagePack.persona_section_aliases`.

`dream_engine` and `dream_model` choose the LLM for the `memory_dream` action. An empty model means the engine default; for `claude` that is `claude-haiku-4-5-20251001`. The `memory_dream` action is registered only when `use_dream_summary` is true.

### Media settings

`MediaConfig` (`mltgnt.media._core.config`) is a frozen dataclass that each medium subclasses.

| Field | Default | Meaning |
|-------|---------|---------|
| `state_dir` | required | Medium state directory |
| `pending_dir` | required | Pending records for delegated tasks |
| `events_dir` | required | Job events JSONL that the progress watcher reads |
| `language` | `get_language_pack()` when the config is built | `LanguagePack` for words, labels, and messages |
| `progress_min_interval_sec` | `5.0` | Minimum time between progress updates |
| `approval_ttl_sec` | `600.0` | Deadline for plan approval |

| Subclass | Extra fields (default) |
|----------|------------------------|
| `SlackMediaConfig` | `bot_token_env` (`"SLACK_BOT_TOKEN"`), `app_token_env` (`"SLACK_APP_TOKEN"`), `status_reactions` (`DEFAULT_STATUS_REACTIONS`), `chunk_max_chars` (`3000`, must be positive) |
| `WebChatMediaConfig` | `store_dir` (required, keyword-only), `host` (`"127.0.0.1"`), `port` (`8765`), `space_id` (`"webchat"`), `assets_dir` (`None`), `avatars` (`{}`; author -> file under `assets_dir` or an `http(s)://` URL), `display_names` (`{}`; author -> name. The browser user posts as `"user"`), `channel_name` (`"general"`), `channel_slug` (`"general"`; URL segment under `/c/<slug>/`), `files_dir` (`None` = `<store_dir>/../files`), `max_upload_bytes` (20 MiB), `push` (`WebPushConfig`; VAPID keys and `enabled` flag), `pwa_icon_192` / `pwa_icon_512` (optional basename under `assets_dir` for PWA icons) |
| `WebPushConfig` | `enabled` (`False`), `vapid_public_key` / `vapid_private_key` (`None`), `vapid_subject` (`mailto:admin@example.invalid`), `body_text` (fixed notification body; message text is never sent). Requires `pip install mltgnt[webchat-push]` and a valid subject (`mailto:` or `https://`) for `push_enabled` |
| `HubMediaConfig` | `hub_url`, `host_id`, `token_env`, `rooms` (required, keyword-only), `cursor_dir` (`None`), `poll_interval_sec` (`2.0`), `space_prefix` (`"hub."`), `guard` (`GuardConfig()`), `outbound_filter` (`None`) |

### LanguagePack

`mltgnt.config.language.LanguagePack` is a frozen dataclass that holds locale vocabulary. `EN` is the only pack that ships with mltgnt, and it is the default. Functions that take `pack=None`, and configs that leave out `language`, use the current pack from `get_language_pack()`.

To use another locale, define your own `LanguagePack` and call `set_language_pack(pack)` once at startup, before you build any `MediaConfig` or `PersonaConfig`. Passing anything other than a `LanguagePack` raises `TypeError`.

| Group | Fields | Used by |
|-------|--------|---------|
| Request gates | `work_request_markers`, `create_request_markers`, `deferred_patterns` | `mltgnt.agent.deterministic_gate` |
| Persona text | `compress_prompt_template`, `v21_required_sections`, `v21_example_section`, `meta_header_needles`, `dedupe_opener_re`, `persona_cut_re`, `persona_end_re`, `exclude_stems` | `mltgnt.persona`, persona listing |
| Persona headings | `persona_section_aliases` (default `{}`) | Maps localized legacy headings to canonical English names. Used by the persona loader, section validation, light/heavy extraction, `regenerate_light_block`, `extract_triage_section`, `DEFAULT_WEIGHT_MAP`, and the `PersonaConfig.section_aliases` default |
| Compaction | `phase1_meta_prefixes` (default `()`), `phase1_meta_markers` (default `()`) | Extra line prefixes and substrings that memory compaction drops from Phase 1 LLM output, on top of the English ones |
| Conversation | `cancel_words`, `composite_header`, `composite_cancel_suffix` | `mltgnt.conversation.thread_queue`, `mltgnt.media._core.cancel` |
| Media | `approval_words`, `status_labels`, `enqueue_failed_text`, `progress_line_pattern` | `mltgnt.media._core` |
| Memory tools | `remember_trigger_words`, `forget_trigger_words` | `mltgnt.memory.tools` |

The first ten fields (request gates, persona text up to `persona_cut_re`, and `exclude_stems`) are required. All other fields have English defaults or empty defaults.

### Scheduler jobs

`PersonaScheduler` loads jobs from `yaml_path` or `SchedulerConfig.schedule_yaml`. A YAML error raises `ConfigError`. `ScheduleJob.from_dict` validates each entry and raises `ValueError` on invalid input.

If an action raises an exception, the job is marked failed with `<ExceptionType>: <message>`, a failure notice is posted, and the result is recorded to memory. Once per calendar day the scheduler prunes state files older than `state_keep_days` (default `30`) with `SchedulePaths.prune`.

| Field | Default | Notes |
|-------|---------|-------|
| `id`, `mode`, `action` | required | `mode` is `scheduled` (needs `every_day_at`), `interval` (needs `interval_minutes > 0`), `fuzzy_window` (needs `window_start` and `window_end`; windows that cross midnight are rejected), or `chained` (runs once every `depends_on` job is done) |
| `notify` | `silent` | `silent`, `slack_secretary`, or `slack_custom` (needs `slack_channel`) |
| `timezone` | `Asia/Tokyo` | |
| `enabled` | `true` | |
| `action_args` | `{}` | Must be a mapping. See below |
| `every_day_at`, `window_start`, `window_end` | `null` | `HH:MM` |
| `every_week_on` | `null` | `monday` to `sunday` |
| `interval_minutes` | `null` | |
| `fuzzy_method` | `hash` | `hash` or `random` |
| `on_window_missed` | `notify` | `notify`, `silent`, or `mark_done` |
| `slack_channel`, `persona` | `null` | |
| `timeout_seconds` | `600` | Wait timeout. `action_args.timeout_seconds` takes precedence |
| `memory` | `false` | |
| `depends_on` | `[]` | Upstream job ids |
| `on_chain_failure` | `abort_notify` | |
| `on_exit` | `null` | Mapping with the required key `nonzero`: `fail` or `skip` |
| `chain_every_run` | `false` | Needs `mode: chained` and `depends_on`. Fires after every successful upstream run and passes the upstream output on as `upstream_output`. If the job is still running, the new upstream output is dropped with a warning |

`action_args` of the built-in `skill` action:

| Key | Default | Effect |
|-----|---------|--------|
| `skill`, `persona` | required | Skill name and persona name |
| `argv` | `[]` | Arguments joined into `$ARGUMENTS`. `upstream_output` is appended |
| `engine`, `model` | persona frontmatter | LLM engine and model |
| `permission` | `null` | Passed to ghdag |
| `task_timeout_sec` | `null` | Positive number of seconds, sent to ghdag as the task execution limit. Invalid values are ignored with a warning |
| `knowledge_count`, `memory_max_bytes`, `memory_exclude_source_tags` | `0`, `0`, `null` | Opt-in context from skill knowledge files and persona memory |
| `enable_pipeline` | `false` | Match a skill pipeline, compose it, type-check it, and run it with `enqueue_dag` |
| `enable_fanout` | `false` | Create dynamic child tasks from the skill result |
| `enforce_status_markers` | `false` | Fail the job when the `PIPELINE_STATUS` markers in the result do not match the skill contract |
| `result_frontmatter` | `false` | Write skill I/O frontmatter at the top of the result |

```yaml
- id: health-check
  mode: interval
  interval_minutes: 30
  action: health_check        # custom action registered with PersonaScheduler(actions=...)
- id: health-followup
  mode: chained
  depends_on: [health-check]
  chain_every_run: true
  action: skill
  action_args:
    skill: triage-health
    persona: guide
    task_timeout_sec: 900
```

## Error Reference

Inventory source: `mltgnt.exceptions`, package `__all__`, and raised types documented against CLI exit codes at `v0.134.0`.

| Exception | Module | Base | Raised when |
|-----------|--------|------|-------------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base class. `mltgnt run` maps any subclass not listed below to exit code `1` |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | `mltgnt run --components` is invalid; the scheduler YAML fails to load; a space has more than one `primary` persona in `load_channel_persona_map`. Exit code `2` |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | Another process holds the PID lock (`DaemonRunner.run`); the persona loader passed to `load_channel_persona_map` fails. Exit code `3` |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | `load_persona` finds frontmatter that does not parse or has no `persona` key |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | The ghdag tools listing times out, fails, or returns invalid JSON, or a skill names an unknown tool |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | `enqueue_dag` or `compose_pipeline` finds a skill I/O type mismatch between steps (turn the check off with `SKILL_IO_TYPECHECK=0`) |
| `LlmCallError` | `mltgnt.memory.compaction` (exported by `mltgnt.memory`) | `RuntimeError` | For hosts to signal that an injected `llm_call` failed during compaction. mltgnt never raises it itself |
| `ReflectionParseError` | `mltgnt.memory.reflection` | `ValueError` | `parse_reflection` gets no JSON object, invalid JSON, or fields of the wrong type |
| `WorkLoopDeadline` | `mltgnt.agent.work_loop` (exported by `mltgnt.agent`) | `Exception` | For the host's `llm_call` to raise when the time budget is spent. `run_work_loop` catches it and returns `BLOCKED` (`deadline_before_plan` or `deadline`) |

Built-in exceptions raised by the public API:

| Exception | Raised by |
|-----------|-----------|
| `ValueError` | `ScheduleJob.from_dict`, `SchedulePaths.prune` (`keep_days < 1`), `parse_plan`, `system_default_engine`, `regenerate_light_block` (no `## Heavy` block), `SlackMediaConfig` (`chunk_max_chars <= 0`), `SlackClient` (unknown `extra` keys) |
| `TypeError` | `adapt_client`, `set_language_pack` |
| `RuntimeError` | `mltgnt.conversation.get_config` before `configure`; `build_app` / `start_socket_mode` when the token variable is unset |
| `ImportError` | The Slack / WebChat app functions when the extra is not installed |

## Public API Stability

- Versions follow `0.Y.Z`. A minor release (`0.Y.0`) may change the API. A patch release does not. Changes are recorded in `CHANGELOG.md`.
- The supported surface is `mltgnt.__all__`, `mltgnt.interfaces` (including `mltgnt.interfaces.media` and `mltgnt.interfaces.hub`), `mltgnt.agent.__all__`, `mltgnt.media.__all__`, the CLI, and the configuration schema described here. Other modules under those packages may change in any minor release.
- A renamed or removed API usually keeps a deprecated alias for at least one minor release. Exceptions are marked `BREAKING` in `CHANGELOG.md`. For example, `0.125.0` added the keyword-only `pack` parameter to `extract_triage_section` and moved the legacy heading aliases into `LanguagePack.persona_section_aliases`.
- Pin an exact tag in production, for example `@v0.134.0`.

## Deprecated API

Inventory source: `CHANGELOG.md` entries from `v0.126.0` through `v0.134.0` (removed aliases are not listed as current API).

| Deprecated | Module | Replacement | Warning |
|------------|--------|-------------|---------|
| `SlackClientProtocol` (`post_message`) | `mltgnt.interfaces.slack`, `mltgnt.interfaces` | `MediaClient` | One `DeprecationWarning` when such a client goes through `adapt_client` (including `PersonaScheduler(slack=...)`). It is wrapped so that `post` calls `post_message`. Removal is planned for a future minor release |
| `ChannelPersonaEntry` | `mltgnt.routing` | `SpacePersonaEntry` | None (plain alias) |

## License

MIT (SPDX: `MIT`). See [`LICENSE`](LICENSE).
