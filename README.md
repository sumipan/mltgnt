# mltgnt

**Typed contracts for multi-agent hosts: personas, memory, skills, agent loops, scheduling, and conversation I/O.**
mltgnt is the middle layer of a three-layer stack: **L0 [ghdag](https://github.com/sumipan/ghdag)** runs DAGs and LLM engines, **L1 mltgnt** defines the domain contracts on top of it, and **L2 your host** owns processes, credentials, and channel I/O.
Unlike agent frameworks that bundle model clients and orchestration, mltgnt contains no model client and no DAG engine — every LLM call is delegated to ghdag through one bridge package.

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.72.0)-orange)

## Not (what this is not)

| mltgnt is not | Instead |
|---------------|---------|
| An LLM SDK | LLM calls go through ghdag via `mltgnt.bridges`. mltgnt never talks to a model provider directly. |
| A DAG engine | Queueing, dependency resolution, and DAG state belong to ghdag. mltgnt builds steps (`enqueue_dag`) and waits for their results. |
| A host runtime | Slack / CLI processes, deployment, and secrets belong to the L2 host. `mltgnt run` only starts the components a host factory returns. |

## Installation

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.72.0"
```

With development tools:

```bash
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.72.0"
```

| Item | Value |
|------|-------|
| Distribution | `mltgnt` `0.72.0` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.83.0` |
| `dev` extra | `pytest>=7.0`, `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `freezegun>=1.2`, `import-linter>=2.0`, `mypy>=1.10`, `ruff>=0.4` |
| Console script | `mltgnt` → `mltgnt.cli.main:main` (`python -m mltgnt` is equivalent) |
| Typed | Yes (`py.typed`) |

## Quick Start

The example below uses only names from `mltgnt.__all__` and runs offline — no LLM, no network. It writes a persona file, lists, loads, and validates it, stores and reads back a dream (memory summary), and drives one `AgentRunner` loop with a scripted LLM.

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

print(list_personas(agents))                    # ['guide']
persona = load_persona("guide", persona_dir=agents)
print(persona.name)                             # guide
print(validate_persona(persona))                # []

persona_chat_dir = root / "chat" / "guide"
write_dream(
    persona_chat_dir,
    DreamSummary(
        persona="guide",
        sections=[DreamSection(category="style", content="Prefers bullet points.", source_entries=2)],
        updated_at="2026-01-01T00:00:00+00:00",
    ),
)
summary = read_dream(persona_chat_dir)          # reads chat/guide/memory/dream.json
print([(s.category, s.source_entries) for s in summary.sections])  # [('style', 2)]

replies = iter([
    json.dumps({"thought": "look it up", "tool": "lookup", "args": {"key": "tz"}}),
    json.dumps({"thought": "done", "tool": "answer", "args": {"text": "Asia/Tokyo"}}),
])
runner = AgentRunner(
    llm_call=lambda prompt, *, tool_result=None: next(replies),
    tool_executor=lambda name, args: "Asia/Tokyo",
    terminal_tools=frozenset({"answer"}),
)
result = runner.run("Which timezone?")
print(result.tool, result.args)                 # answer {'text': 'Asia/Tokyo'}
```

`write_dream` schedules a debounced git commit of `dream.json`; it is a no-op unless `ENABLE_GIT` is set (see [Configuration](#configuration)). The stored summary can be printed with `mltgnt memory dream show guide --chat-dir <root>/chat`.

## CLI Reference

Defined with argparse in `src/mltgnt/cli/main.py` and `src/mltgnt/cli/memory.py`. With no subcommand, help is printed and the exit code is `0`.

| Command | Arguments | Behavior |
|---------|-----------|----------|
| `mltgnt run` | `--components MODULE:FUNCTION` (required); `--pid-file PATH` (default `/tmp/mltgnt_daemon.pid`) | Imports `MODULE`, calls `FUNCTION()` to get a list of `DaemonComponent`s, and runs them with `DaemonRunner` under a PID lock until SIGINT / SIGTERM. |
| `mltgnt memory dream show` | `persona` (positional); `--chat-dir PATH` (required) | Prints every section of `<chat-dir>/<persona>/memory/dream.json`. If no summary exists, prints a notice and exits `0`. |
| `mltgnt memory dream forget` | `persona` (positional); `--category NAME` (required); `--chat-dir PATH` (required) | Removes one category from the dream summary and rewrites the file. Exits `1` if the summary or the category does not exist. |

Exit codes of `mltgnt run` (`src/mltgnt/cli/main.py`, errors raised in `src/mltgnt/cli/run.py` and `mltgnt.daemon`):

| Code | Meaning |
|------|---------|
| `0` | Normal termination |
| `1` | Any other `MltgntError` |
| `2` | `ConfigError` — `--components` is not `module:function`, the module is not found, or the function is missing or not callable |
| `3` | `DependencyError` — another instance already holds the PID lock |

## Public API

### Top level (`mltgnt.__all__`)

`src/mltgnt/__init__.py` exports exactly these 23 names. Everything else is imported from its subpackage.

| Group | Name | Kind | Signature / fields | Defined in |
|-------|------|------|--------------------|------------|
| memory | `read_memory_iterative` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call, skill_paths=None, max_iterations=3) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_by_relevance` | function | `(config, persona_stem, query, *, max_bytes, max_entries, layers=None) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_with_sufficiency_check` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call=None) -> str` | `mltgnt.memory.search` |
| memory | `DreamSection` | frozen dataclass | `category`, `content`, `source_entries` | `mltgnt.memory.dream._format` |
| memory | `DreamSummary` | frozen dataclass | `persona`, `sections: list[DreamSection]`, `updated_at` | `mltgnt.memory.dream._format` |
| memory | `read_dream` | function | `(persona_dir, *, memory_dir_name="memory") -> DreamSummary \| None` | `mltgnt.memory.dream.api` |
| memory | `write_dream` | function | `(persona_dir, summary, *, memory_dir_name="memory") -> None` | `mltgnt.memory.dream.api` |
| persona | `Persona` | dataclass | `name`, `fm`, `sections`, `body`, `path`, `weight_map` | `mltgnt.persona.loader` |
| persona | `load_persona` | function | `(name, *, persona_dir=None, config=None) -> Persona` | `mltgnt.persona` |
| persona | `list_personas` | function | `(persona_dir=None) -> list[str]` | `mltgnt.persona` |
| persona | `validate_persona` | function | `(persona, *, available_skills=None) -> list[str]` (warnings; empty means valid) | `mltgnt.persona` |
| persona | `run_persona_prompt` | function | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str`; engine is the persona's `engine` or `system_default_engine()` | `mltgnt.persona.runner` |
| interfaces | `ChatInput` | dataclass | `source`, `session_key`, `messages`, `persona_name=""`, `model=None`, `context_files=[]`, `context_memory_excerpt=None`, `context_memory_preferences=None` | `mltgnt.interfaces.types` |
| interfaces | `ChatOutput` | dataclass | `content`, `persona_name`, `timestamp`, `session_key` | `mltgnt.interfaces.types` |
| interfaces | `Message` | TypedDict | `role`, `content` | `mltgnt.interfaces.types` |
| interfaces | `PersonaProtocol` | Protocol | Structural type satisfied by `Persona` | `mltgnt.interfaces.persona` |
| agent | `AgentResult` | dataclass | `tool`, `args`, `raw_response`, `tool_trace=None`, `reflexion_count=0`, `plan=None` | `mltgnt.agent._runner` |
| agent | `AgentRunner` | class | See [Agent loop](#agent-loop-mltgntagent) | `mltgnt.agent._runner` |
| bridges | `enqueue_dag` | function | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, skills=None, permission=None, order_builder=None) -> list[tuple[bool, str]]` | `mltgnt.bridges.ghdag_bridge` |
| bridges | `enqueue_and_wait` | function | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_name=None, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, permission=None, order_builder=None, run_result=None) -> tuple[bool, str]` | `mltgnt.bridges.ghdag_bridge` |
| scheduler | `PersonaScheduler` | class | `(slack, *, config=None, state_dir=None, yaml_path=None, salt="", jobs=None, notify_channel_resolver=None, default_slack_post_kwargs=None, persona_post_kwargs_resolver=None, repo_root=None, persona_dir=None, append_memory_fn=None, actions=None, memory_config=None)` | `mltgnt.scheduler.runner` |
| scheduler | `ScheduleJob` | dataclass | See [Scheduler jobs](#scheduler-jobs-schedulejob); build with `ScheduleJob.from_dict(raw)` | `mltgnt.scheduler.models` |
| version | `__version__` | `str` | Installed distribution version; `"0.0.0"` if metadata is unavailable | `mltgnt` |

### Agent loop (`mltgnt.agent`)

`AgentRunner(*, llm_call, tool_executor, terminal_tools, max_iterations=3, max_iterations_fn=None, evaluator=None, retry_config=None, logger=None, audit_writer=None, classifier=None, history_mode="last_result", history_max_chars=24_000, plan=None, max_reflexions=None, step_hook=None)` and `runner.run(prompt) -> AgentResult | None` (`None` when the LLM call or response parsing fails).

With the defaults the loop feeds back only the last tool result. The work-mode options are opt-in:

| Option | Effect |
|--------|--------|
| `history_mode="full_trace"` | Sends the whole numbered tool trace (truncated to `history_max_chars`) instead of only the last result. |
| `plan=Plan(...)` | Updated in place from the `plan_update` key of LLM responses (`Plan.apply`); returned as `AgentResult.plan`. Use `build_plan_prompt` / `parse_plan` to create it. |
| `max_reflexions=N` | Caps Reflexion retries; when exceeded, `AgentResult.tool == "__reflexion_exhausted__"`. |
| `step_hook(entry)` | Called after each tool-trace entry is appended. |

Other public names in `mltgnt.agent.__all__`:

| Name | Kind | Summary |
|------|------|---------|
| `Plan` / `PlanItem` | dataclass | `Plan(items)` with `apply(updates)` and `progress() -> (done, total)`; `PlanItem(id, title, depends, status="pending", note="")` |
| `parse_plan` | function | `(raw) -> Plan`; parses `{"items": [{"id", "title", "depends"?}]}` (omitted `depends` = previous item); raises `ValueError` |
| `build_plan_prompt` | function | `(prompt) -> str`; asks the LLM for the JSON that `parse_plan` accepts |
| `DefaultReflexionEvaluator` | class | `(failure_markers=(), repeat_window=3)`; retries on `[ERROR]` results, failure markers, or a repeated `(tool, args)` within `repeat_window` |
| `DispatchDecision`, `make_dispatch_decision`, `MODE_REPLY`, `MODE_DELEGATE`, `should_force_delegate`, `should_preempt_delegate`, `has_work_request`, `is_create_request`, `match_deferred_promise`, `extract_artifact_references` | functions / constants | Reply-vs-delegate decision and deterministic request gates |
| `PreflightContext`, `run_preflight`, `DirectAgentResult`, `SkillWorkerResult`, `MemoryWorkerResult` | class / function | Dispatch preflight and worker result types |

### Other subpackages

Each subpackage defines its own `__all__`; the main entry points are:

| Package | Main names |
|---------|------------|
| `mltgnt.bridges` | `enqueue_dag`, `enqueue_and_wait`, `DagStep`, `call_llm`, `MltgntHooks`, `create_audit_writer`, `md_read`, `md_write`, `files_adapter` (`files_adapter.commit(paths, message, *, sink="memory", trailers=None)` commits through a ghdag VCS sink) |
| `mltgnt.memory` | `append_memory_entry`, `read_memory_preferences`, `read_memory_tail_text`, `memory_file_path`, `persona_memory_lock`, `compact`, `needs_compaction`, `CompactionResult`, `LlmCallError`, `MemoryEntry`, `parse_jsonl`, `serialize_entry`, `flush_memory_commits() -> int` (commits every pending debounced memory path now; also registered with `atexit`) |
| `mltgnt.persona` | `Persona`, `PersonaContext`, `PersonaValidationError`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt`, `format_persona_body`, `format_result_for_persona`, `compress_heavy_to_light`, `regenerate_light_block`; `mltgnt.persona.schema.system_default_engine()` |
| `mltgnt.skill` | `discover`, `discover_bodies`, `load`, `match`, `resolve_skill`, `run`, `build_extra_context`, `lint_skill_meta`, `SkillMeta`, `SkillFile`, `SkillRegistry`, `SkillRunResult`, `SkillMatchResult`, `ArtifactSpec`, `ProducesSpec`, `ConsumesSpec` |
| `mltgnt.routing` | `resolve_persona`, `find_observers_in_space`, `load_channel_persona_map`, `SpacePersonaEntry`, `RoutingRule`, `evaluate`, `detect_nickname`, triage helpers |
| `mltgnt.conversation` | `configure`, `get_config`, `ConversationConfig`, `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, modules `thread_queue`, `session_store`, `session_compact`, `thread_index`, `thread_persona_store`, `fake_media` |
| `mltgnt.scheduler` | `PersonaScheduler`, `ScheduleJob`, `SchedulePaths`, `load_schedule_jobs`, `atomic_write_text` |
| `mltgnt.daemon` | `DaemonComponent`, `DaemonRunner`, `PidLock`, `SkillWatcherComponent` |
| `mltgnt.config` | `PersonaConfig`, `MemoryConfig`, `SchedulerConfig`, `ConversationConfig`, `DEFAULT_WEIGHT_MAP` |

Skill matching takes an engine: `match(user_input, skills, persona_skills=None, model=None, *, engine="claude")`, `mltgnt.skill.matcher.match_pipeline(..., *, engine="claude")`, and `resolve_skill(user_input, skill_paths, persona_skills=None, entry_file="SKILL.md", matcher_model=None, *, matcher_engine="claude")`. An empty engine means `claude`; with `claude` and no model the matcher uses `claude-haiku-4-5-20251001`, other engines use their own default model.

## Protocols / Extension Points

| Extension point | Module | Contract |
|-----------------|--------|----------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Structural type satisfied by `Persona`. |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, thread_ts=None, blocks=None, reply_broadcast=False) -> bool`; return `False` on failure. Passed as `PersonaScheduler(slack=...)`. |
| `TurnHandler` | `mltgnt.interfaces.turn` | `handle(turn: TurnInput) -> TurnResult`; media-independent boundary between the host's channel layer and the conversation layer. |
| `TurnInput` / `TurnResult` / `Attachment` / `HistoryMessage` | `mltgnt.interfaces.turn` | Frozen dataclasses. `TurnInput`: `conversation_id`, `text`, `attachments=()`, `history=()`, `persona_id=None`. `TurnResult`: `kind` (`"reply"` or `"task"`), `text=""`, `task_ref=None`. |
| `PersonaFMBase` / `ChatInputBase` / `ChatOutputBase` | `mltgnt.interfaces.types` | Protocols for persona frontmatter and chat input / output shapes. |
| `DaemonComponent` | `mltgnt.daemon` | `name` property, non-blocking `start()`, `stop()`. Returned by the factory given to `mltgnt run --components`. |
| `LLMCaller` | `mltgnt.agent._runner` | `(prompt, *, tool_result=None) -> str \| None`; `AgentRunner(llm_call=...)`. |
| `ToolExecutor` | `mltgnt.agent._runner` | `(tool_name, tool_args) -> str`; `AgentRunner(tool_executor=...)`. |
| `ReflexionEvaluator` | `mltgnt.agent._runner` | `(prompt, tool_name, tool_args, tool_result, tool_trace) -> ReflexionVerdict(should_retry, feedback)`; `AgentRunner(evaluator=...)`. `DefaultReflexionEvaluator` is the built-in implementation. |
| `ActionFn` | `mltgnt.scheduler.models` | `(job) -> tuple[bool, str]`; register with `PersonaScheduler(actions={name: fn})`. Built-in actions: `noop`, `skill`, and `memory_dream` (only when `memory_config.use_dream_summary` is true). |
| `LanguagePack` / `JA` | `mltgnt.config.language` | Frozen dataclass of locale-specific vocabulary; functions taking `pack=None` use `JA`. |

## Architecture

Contents of `src/mltgnt/`:

| Path | Responsibility |
|------|----------------|
| `agent/` | Tool-calling agent loop (`AgentRunner`) with retry, Reflexion, plans, and full-trace history; action classification, deterministic gate, dispatch preflight and dispatch decision. |
| `bridges/` | The only gateway to ghdag: DAG submission, LLM calls, DAG hooks, audit writer, file and VCS commit adapters, skill I/O type check. |
| `cli/` | argparse CLI: `mltgnt run`, `mltgnt memory dream show`, `mltgnt memory dream forget`. |
| `config/` | Configuration dataclasses and `LanguagePack`. |
| `conversation/` | Media-independent conversation layer: thread queue, session store and compaction, thread index, thread-to-persona binding. |
| `daemon/` | `DaemonComponent`, `DaemonRunner`, `PidLock`, `SkillWatcherComponent`. |
| `interfaces/` | Dependency-free DTOs and Protocols shared by every layer. |
| `memory/` | Persona memory append, read, and search (TF-IDF relevance, sufficiency check, iterative retrieval), compaction, and debounced git commits of memory files; `memory/dream/` synthesizes and stores `dream.json` / `global.json`. |
| `persona/` | Persona file loading, frontmatter schema, validation, registry, prompt formatting, compression, and `run_persona_prompt`. |
| `routing/` | Space-to-persona routing, observers, LLM triage, agentic skill discovery. |
| `scheduler/` | `ScheduleJob` model and YAML loader, `PersonaScheduler`, run state, fan-out; built-in `skill` and `memory_dream` actions in `scheduler/actions/`. |
| `skill/` | Markdown skill discovery, loading, lint, matching, context building, execution. |
| `exceptions.py` | Base exception hierarchy: `MltgntError`, `ConfigError`, `DependencyError`. |
| `__main__.py` | Enables `python -m mltgnt`. |
| `__init__.py` | Defines `mltgnt.__all__`. |
| `py.typed` | PEP 561 marker. |

### Layer contracts (`.importlinter`)

A layer may import only layers below it:

| Layer (top → bottom) | Packages |
|----------------------|----------|
| 1 | `daemon` |
| 2 | `scheduler`, `agent`, `routing` |
| 3 | `persona`, `skill`, `memory`, `conversation` |
| 4 | `bridges` |
| 5 | `interfaces` |

Allowed exceptions (`ignore_imports`): any module may import `mltgnt.config`; `mltgnt.bridges.*` may import `mltgnt.skill.*`; `mltgnt.skill.matcher` may import `mltgnt.routing.agentic_discover`. A second (`forbidden`) contract prevents `persona`, `skill`, `memory`, and `conversation` from importing `ghdag` directly; they go through `bridges`.

## Configuration

### Environment variables

All environment variables read under `src/`, plus `ENABLE_GIT`, which ghdag reads and which changes mltgnt behavior:

| Variable | Read by | Effect |
|----------|---------|--------|
| `MLTGNT_DEFAULT_ENGINE` | `mltgnt.persona.schema.system_default_engine()` | Engine used when neither the persona nor the caller names one (`run_persona_prompt`, dispatch decision, result formatting). Read on every call. Unset or blank → `claude`. Must be one of `claude`, `codex`, `cursor`, `gemini`, otherwise `ValueError`. |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge.enqueue_dag` | `0` skips the compose-time skill I/O type check. Any other value (or unset) keeps it on. |
| `THREAD_PERSONA_TTL_DAYS` | `mltgnt.conversation.thread_persona_store` | TTL in days for thread-to-persona bindings (default `30`; invalid values fall back to `30`). Ignored once a `ConversationConfig` is configured — its `thread_persona_ttl_days` wins. |
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Substituted for `$NIKKI_ROOT` in skill bodies (empty string when unset). |
| `REPO_ROOT` | `mltgnt.skill.runner` | Substituted for `$REPO_ROOT` in skill bodies (empty string when unset). |
| `ENABLE_GIT` | ghdag (`ghdag.vcs`) | Truthy (`1` / `true` / `yes`) enables git commits of memory files. Unset → the ghdag sink is a `NullSink` and memory commits do nothing. |

Memory commits: every memory append, compaction, `dream.json`, and `global.json` write schedules a commit through `mltgnt.bridges.files_adapter.commit` on the ghdag `memory` sink. Commits are debounced per path (the last write wins); append and compaction use `MemoryConfig.commit_debounce_sec`, dream files use 300 s, and `0` commits immediately. Commit failures are logged, never raised. Call `mltgnt.memory.flush_memory_commits()` to commit pending paths now; it also runs at process exit.

### Config dataclasses (`mltgnt.config`)

All are frozen dataclasses. Paths are injected by the host; mltgnt hardcodes none.

| Class | Required fields | Optional fields (default) |
|-------|-----------------|---------------------------|
| `PersonaConfig` | — | `weight_map` (`DEFAULT_WEIGHT_MAP`), `section_aliases` (`PERSONA_SECTION_ALIASES`), `exclude_stems` (`frozenset()`) |
| `MemoryConfig` | `chat_dir` | `chat_memory_dir` (`None`), `inject_max_bytes` (`10240`), `inject_max_entries` (`12`), `preferences_max_bytes` (`5120`), `lock_timeout_sec` (`30.0`), `lock_stale_threshold_sec` (`300.0`), `raw_days` (`7`), `mid_weeks` (`3`), `compact_threshold_bytes` (`40960`), `compact_target_bytes` (`25600`), `preferences_section_name`, `protected_layers` (`("caveat",)`), `timezone` (`"Asia/Tokyo"`), `dream_model` (`""`), `dream_engine` (`"claude"`), `use_dream_summary` (`False`), `dream_dir_name` (`"memory"`), `commit_debounce_sec` (`300.0`), `global_dream_exclude_personas` (`()`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir` | `timezone` (`"Asia/Tokyo"`), `salt` (`""`) |
| `ConversationConfig` | `queue_dir`, `sessions_dir`, `ledger_dir`, `thread_index_dir`, `thread_persona_path` | `posts_dir` (`None`), `audit_path` (`None`), `stale_after_sec` (`3600`), `max_queued` (`20`), `cleanup_ttl_days` (`14`), `thread_persona_ttl_days` (`30`) |

`dream_engine` and `dream_model` select the LLM for the `memory_dream` scheduler action. A blank engine means `claude`; a blank model means the engine's default, except that `claude` falls back to `claude-haiku-4-5-20251001`.

### Scheduler jobs (`ScheduleJob`)

`PersonaScheduler` loads jobs from the YAML file given as `yaml_path` or `SchedulerConfig.schedule_yaml` (a load failure raises `ConfigError`). Each entry is parsed by `ScheduleJob.from_dict`, which raises `ValueError` on invalid input.

| Field | Default | Notes |
|-------|---------|-------|
| `id`, `mode`, `action` | required | `mode`: `scheduled` (needs `every_day_at`), `interval` (needs `interval_minutes > 0`), `fuzzy_window` (needs `window_start` / `window_end`, no overnight windows), `chained` (runs when all `depends_on` jobs are done) |
| `notify` | `silent` | `silent`, `slack_secretary`, or `slack_custom` (needs `slack_channel`) |
| `timezone` | `Asia/Tokyo` | |
| `enabled` | `true` | |
| `action_args` | `{}` | Must be a mapping |
| `every_day_at`, `window_start`, `window_end` | `null` | `HH:MM` |
| `every_week_on` | `null` | `monday` … `sunday` |
| `interval_minutes` | `null` | |
| `fuzzy_method` | `hash` | `hash` or `random` |
| `on_window_missed` | `notify` | `notify`, `silent`, or `mark_done` |
| `slack_channel`, `persona` | `null` | |
| `timeout_seconds` | `600` | |
| `memory` | `false` | |
| `depends_on` | `[]` | Upstream job ids |
| `on_chain_failure` | `abort_notify` | `abort_notify` or `silent` |
| `on_exit` | `null` | Mapping with required `nonzero`: `fail` or `skip` |
| `chain_every_run` | `false` | See below |
| `upstream_output` | — | Set at runtime only; never read from YAML |

**`chain_every_run`** — a `mode: chained` job with `chain_every_run: true` fires after *every* successful run of a job in its `depends_on`, instead of once per day. It requires `mode: chained` and a non-empty `depends_on`. The upstream output text is passed as `ScheduleJob.upstream_output`, and the built-in `skill` action appends it to the skill input. Such jobs are never triggered by time and write no done / skipped / failed marks.

In this example `health_check` is a custom action registered with `PersonaScheduler(actions=...)`; the built-in `skill` action needs `action_args.skill` and `action_args.persona`.

```yaml
- id: health-check
  mode: interval
  interval_minutes: 30
  action: health_check
  notify: silent
- id: health-followup
  mode: chained
  depends_on: [health-check]
  chain_every_run: true
  action: skill
  action_args:
    skill: triage-health
    persona: guide
  notify: silent
```

## Error Reference

| Type | Module | Base | Raised when |
|------|--------|------|-------------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base of `ConfigError` and `DependencyError`; not raised directly. `mltgnt run` maps other subclasses to exit code `1`. |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Invalid `mltgnt run --components` (`mltgnt.cli.run`); scheduler YAML fails to load (`PersonaScheduler`); more than one `primary` persona in one space (`mltgnt.routing.load_channel_persona_map`). CLI exit code `2`. |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | The PID lock is held by another instance (`DaemonRunner.run`); the injected persona loader fails (`mltgnt.routing.load_channel_persona_map`). CLI exit code `3`. |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | `load_persona` finds YAML frontmatter that does not parse or lacks the required `persona` key. |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | Skill loading fails: the ghdag tools list times out, fails, or returns invalid JSON, or a skill references an unknown tool (`mltgnt.skill.loader`). |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | `enqueue_dag`'s compose-time skill I/O type check finds a pipe type mismatch between steps (disable with `SKILL_IO_TYPECHECK=0`). |
| `LlmCallError` | `mltgnt.memory.compaction` (re-exported by `mltgnt.memory`) | `RuntimeError` | Exported for hosts to wrap failures of an injected `llm_call` in memory compaction; mltgnt code does not raise it. |

`ScheduleJob.from_dict`, `parse_plan`, and `system_default_engine()` raise the built-in `ValueError` for invalid job definitions, invalid plan JSON, and invalid `MLTGNT_DEFAULT_ENGINE` values.

## Public API Stability

- mltgnt is pre-1.0 (`0.Y.Z`). A minor release (`0.Y.0`) may contain breaking changes; a patch release (`0.Y.Z`) does not. Every change is listed in `CHANGELOG.md`.
- The supported surface is `mltgnt.__all__`, `mltgnt.interfaces`, the CLI, and the configuration schema above. Other subpackage names may change in any minor release.
- Renamed or removed APIs keep a deprecated alias for at least one minor release before removal.
- Pin an exact tag (for example `@v0.72.0`) in production.

## Deprecated API

Still importable from `mltgnt.routing`:

| Deprecated | Use instead | Warning |
|------------|-------------|---------|
| `resolve_responding_persona` | `resolve_persona` | `DeprecationWarning` |
| `find_observers` | `find_observers_in_space` | `DeprecationWarning` |
| `ChannelPersonaEntry` | `SpacePersonaEntry` | none (plain alias) |

## License

MIT (SPDX: `MIT`). See `LICENSE`.
