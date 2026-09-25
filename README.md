# mltgnt

**Typed persona, memory, skill, scheduler, and conversation contracts for multi-agent hosts.**
mltgnt is the middle layer of a three-layer stack: **L0 [ghdag](https://github.com/sumipan/ghdag)** runs DAGs and LLM engines, **L1 mltgnt** defines the domain contracts, and **L2 your host** owns processes, credentials, and channel I/O.
Unlike agent frameworks that bundle model clients and orchestration, mltgnt ships no model client and no DAG engine — it is the typed layer you put between the two.

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.66.0)-orange)

## Not (what this is not)

| mltgnt is not | Instead |
|---------------|---------|
| An LLM SDK | Every LLM call goes through ghdag via `mltgnt.bridges`. mltgnt never talks to a model provider itself. |
| A DAG engine | Queueing, dependency resolution, and DAG state belong to ghdag. mltgnt builds steps (`enqueue_dag`) and waits for their results. |
| A host runtime | Slack / CLI processes, deployment, and secrets belong to the L2 host. `mltgnt run` only starts components that the host's factory returns. |

## Installation

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.66.0"
```

With development tools:

```bash
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.66.0"
```

| Item | Value |
|------|-------|
| Distribution | `mltgnt` `0.66.0` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.80.0` |
| `dev` extra | `pytest>=7.0`, `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `freezegun>=1.2`, `import-linter>=2.0`, `mypy>=1.10`, `ruff>=0.4` |
| Console script | `mltgnt` → `mltgnt.cli.main:main` |
| Typed | Yes (`py.typed`) |

## Quick Start

This example uses only names from `mltgnt.__all__` and runs offline — no LLM, no network. It writes a persona file, lists, loads, and validates it, then stores and reads back a dream (memory summary).

```python
import tempfile
from pathlib import Path

from mltgnt import (
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
    "---\n"
    "persona:\n"
    "  name: guide\n"
    "---\n"
    "## Background\n"
    "\n"
    "A calm, concise guide.\n",
    encoding="utf-8",
)

print(list_personas(agents))                    # ['guide']
persona = load_persona("guide", persona_dir=agents)
print(persona.name)                             # guide
print(validate_persona(persona))                # []

chat_dir = root / "chat"
write_dream(
    chat_dir / "guide",
    DreamSummary(
        persona="guide",
        sections=[DreamSection(category="style", content="Prefers bullet points.", source_entries=2)],
        updated_at="2026-01-01T00:00:00+00:00",
    ),
)
summary = read_dream(chat_dir / "guide")        # reads chat/guide/memory/dream.json
print([(s.category, s.source_entries) for s in summary.sections])  # [('style', 2)]
```

The stored summary can then be printed with `mltgnt memory dream show guide --chat-dir <root>/chat`.

## CLI Reference

Defined with argparse in `src/mltgnt/cli/main.py` and `src/mltgnt/cli/memory.py`. `python -m mltgnt` is equivalent to the `mltgnt` console script. With no subcommand, help is printed and the exit code is `0`.

| Command | Arguments | Behavior |
|---------|-----------|----------|
| `mltgnt run` | `--components MODULE:FUNCTION` (required); `--pid-file PATH` (default `/tmp/mltgnt_daemon.pid`) | Imports `MODULE`, calls `FUNCTION()` to get a list of `DaemonComponent`s, and runs them in `DaemonRunner` under a PID lock. |
| `mltgnt memory dream show` | `persona` (positional); `--chat-dir PATH` (required) | Prints every section of `<chat-dir>/<persona>/memory/dream.json`. If no summary exists, prints a notice and exits `0`. |
| `mltgnt memory dream forget` | `persona` (positional); `--category NAME` (required); `--chat-dir PATH` (required) | Removes one category from the dream summary and rewrites it. Exits `1` if the summary or the category does not exist. |

Exit codes of `mltgnt run`:

| Code | Meaning |
|------|---------|
| `0` | Normal termination |
| `1` | Any other `MltgntError` |
| `2` | `ConfigError` — `--components` not in `module:function` form, module not found, function missing or not callable |
| `3` | `DependencyError` — for example, another instance already holds the PID lock |

## Public API

`mltgnt.__all__` (`src/mltgnt/__init__.py`) contains exactly these 23 names. Anything else must be imported from its subpackage.

| Group | Name | Kind | Signature / fields | Defined in |
|-------|------|------|--------------------|------------|
| memory | `read_memory_iterative` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call, skill_paths=None, max_iterations=3) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_by_relevance` | function | `(config, persona_stem, query, *, max_bytes, max_entries, layers=None) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_with_sufficiency_check` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call=None) -> str` | `mltgnt.memory.search` |
| memory | `DreamSection` | frozen dataclass | `category: str`, `content: str`, `source_entries: int` | `mltgnt.memory.dream._format` |
| memory | `DreamSummary` | frozen dataclass | `persona: str`, `sections: list[DreamSection]`, `updated_at: str` | `mltgnt.memory.dream._format` |
| memory | `read_dream` | function | `(persona_dir, *, memory_dir_name="memory") -> DreamSummary \| None` | `mltgnt.memory.dream.api` |
| memory | `write_dream` | function | `(persona_dir, summary, *, memory_dir_name="memory") -> None` | `mltgnt.memory.dream.api` |
| persona | `Persona` | dataclass | `name`, `fm`, `sections`, `body`, `path`, `weight_map` | `mltgnt.persona.loader` |
| persona | `load_persona` | function | `(name, *, persona_dir=None, config=None) -> Persona` (`persona_dir` defaults to `./agents`) | `mltgnt.persona` |
| persona | `list_personas` | function | `(persona_dir=None) -> list[str]` | `mltgnt.persona` |
| persona | `validate_persona` | function | `(persona, *, available_skills=None) -> list[str]` (warning messages; empty list means valid) | `mltgnt.persona` |
| persona | `run_persona_prompt` | function | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str` (calls an LLM through ghdag) | `mltgnt.persona.runner` |
| interfaces | `ChatInput` | dataclass | `source`, `session_key`, `messages`, `persona_name`, `model=None`, `context_files=[]`, `context_memory_excerpt=None`, `context_memory_preferences=None` | `mltgnt.interfaces.types` |
| interfaces | `ChatOutput` | dataclass | `content`, `persona_name`, `timestamp`, `session_key` | `mltgnt.interfaces.types` |
| interfaces | `Message` | TypedDict | `role: str`, `content: str` | `mltgnt.interfaces.types` |
| interfaces | `PersonaProtocol` | Protocol | structural type for persona objects | `mltgnt.interfaces.persona` |
| agent | `AgentResult` | dataclass | `tool`, `args`, `raw_response`, `tool_trace=None`, `reflexion_count=0` | `mltgnt.agent._runner` |
| agent | `AgentRunner` | class | `(*, llm_call, tool_executor, terminal_tools, max_iterations=3, max_iterations_fn=None, evaluator=None, retry_config=None, logger=None, audit_writer=None, classifier=None)` | `mltgnt.agent._runner` |
| bridges | `enqueue_dag` | function | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, skills=None, permission=None, order_builder=None) -> list[tuple[bool, str]]` | `mltgnt.bridges.ghdag_bridge` |
| bridges | `enqueue_and_wait` | function | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_name=None, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, permission=None, order_builder=None, run_result=None) -> tuple[bool, str]` | `mltgnt.bridges.ghdag_bridge` |
| scheduler | `PersonaScheduler` | class | `(slack, *, config=None, state_dir=None, yaml_path=None, salt="", jobs=None, notify_channel_resolver=None, default_slack_post_kwargs=None, persona_post_kwargs_resolver=None, repo_root=None, persona_dir=None, append_memory_fn=None, actions=None, memory_config=None)` | `mltgnt.scheduler.runner` |
| scheduler | `ScheduleJob` | dataclass | see [Scheduler jobs](#scheduler-jobs-schedulejob); build from YAML with `ScheduleJob.from_dict(raw)` | `mltgnt.scheduler.models` |
| version | `__version__` | `str` | installed distribution version; `"0.0.0"` if package metadata is unavailable | `mltgnt` |

## Protocols / Extension Points

The host plugs into mltgnt by implementing these Protocols or passing callables. All types in `mltgnt.interfaces` are re-exported from that package.

| Extension point | Module | Contract |
|-----------------|--------|----------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Structural type satisfied by `Persona`. |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, thread_ts=None, blocks=None, reply_broadcast=False) -> bool`. Return `False` on failure; do not raise. Passed as `PersonaScheduler(slack=...)`. |
| `TurnHandler` | `mltgnt.interfaces.turn` | `handle(turn: TurnInput) -> TurnResult`. Media-independent boundary between the host's media layer and the conversation layer. |
| `TurnInput` / `TurnResult` / `Attachment` / `HistoryMessage` | `mltgnt.interfaces.turn` | Frozen dataclasses. `TurnInput`: `conversation_id`, `text`, `attachments`, `history`, `persona_id`. `TurnResult`: `kind`, `text`, `task_ref`. |
| `PersonaFMBase` / `ChatInputBase` / `ChatOutputBase` | `mltgnt.interfaces.types` | Protocols for persona frontmatter and chat input / output shapes. |
| `DaemonComponent` | `mltgnt.daemon` | `name` property, non-blocking `start()`, `stop()`. The factory given to `mltgnt run --components` returns these. |
| `LLMCaller` | `mltgnt.agent._runner` | `(prompt, *, tool_result=None) -> str \| None`; passed to `AgentRunner` as `llm_call`. |
| `ToolExecutor` | `mltgnt.agent._runner` | `(tool_name, tool_args) -> str`; passed to `AgentRunner` as `tool_executor`. |
| `ReflexionEvaluator` | `mltgnt.agent._runner` | `(prompt, tool_name, tool_args, tool_result, tool_trace) -> ReflexionVerdict`; optional `AgentRunner` `evaluator`. |
| `actions` | `mltgnt.scheduler.runner` | `PersonaScheduler(actions={name: ActionFn})` registers custom scheduler actions (`ActionFn`: `(job) -> tuple[bool, str]`). Built-in actions: `noop`, `skill`, and `memory_dream` (only when `memory_config.use_dream_summary` is true). |
| `LanguagePack` / `JA` | `mltgnt.config.language` | Frozen dataclass of locale-specific vocabulary. Functions taking `pack=None` use `JA`; pass another pack to change locale. |

## Architecture

Top-level contents of `src/mltgnt/`:

| Path | Responsibility |
|------|----------------|
| `agent/` | Tool-calling agent loop (`AgentRunner`) with retry and Reflexion hooks; action classification, deterministic gate, dispatch preflight and dispatch decision. |
| `bridges/` | The only gateway to ghdag: DAG submission (`enqueue_dag`, `enqueue_and_wait`), LLM calls, DAG hooks, audit and file adapters, skill I/O type check. |
| `cli/` | argparse CLI: `mltgnt run`, `mltgnt memory dream show`, `mltgnt memory dream forget`. |
| `config/` | Configuration dataclasses and `LanguagePack`. |
| `conversation/` | Media-independent conversation layer: thread queue, session store, thread index, thread-to-persona binding, session compaction. |
| `daemon/` | `DaemonComponent`, `DaemonRunner`, `PidLock`, `SkillWatcherComponent`. |
| `interfaces/` | Dependency-free DTOs and Protocols shared by every layer. |
| `memory/` | Persona memory read and search (TF-IDF relevance, sufficiency check, iterative retrieval) and compaction; `memory/dream/` synthesizes and stores `dream.json`. |
| `persona/` | Persona file loading, frontmatter schema, validation, registry, prompt formatting, compression, and `run_persona_prompt`. |
| `routing/` | Space-to-persona routing, observers, LLM triage, agentic skill discovery. |
| `scheduler/` | `ScheduleJob` model and YAML loader, `PersonaScheduler`, run state, fan-out; built-in actions in `scheduler/actions/`. |
| `skill/` | Markdown skill discovery, loading, lint, matching, context building, execution. |
| `exceptions.py` | Base exception hierarchy (`MltgntError`, `ConfigError`, `DependencyError`). |
| `__main__.py` | Makes `python -m mltgnt` work. |
| `__init__.py` | Defines `mltgnt.__all__`. |
| `py.typed` | PEP 561 marker. |

### Layer contracts (`.importlinter`)

A layer may import only the layers below it. Exceptions (`ignore_imports`): any module may import `mltgnt.config`; `mltgnt.bridges.*` may import `mltgnt.skill.*`; `mltgnt.skill.matcher` may import `mltgnt.routing.agentic_discover`.

| Layer (top → bottom) | Packages |
|----------------------|----------|
| 1 | `daemon` |
| 2 | `scheduler`, `agent`, `routing` |
| 3 | `persona`, `skill`, `memory`, `conversation` |
| 4 | `bridges` |
| 5 | `interfaces` |

A second contract (`forbidden`) forbids `persona`, `skill`, `memory`, and `conversation` from importing `ghdag` directly; they must go through `bridges`.

## Configuration

### Environment variables

These are all the environment variables read under `src/`:

| Variable | Read by | Effect |
|----------|---------|--------|
| `MLTGNT_DEFAULT_ENGINE` | `mltgnt.persona.schema.system_default_engine()` | Engine used when neither the persona (`ops.engine`) nor the caller specifies one. Read on every call. Unset or blank → `"claude"`. Must be one of `claude`, `codex`, `cursor`, `gemini`, otherwise `ValueError`. |
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Replaces `$NIKKI_ROOT` in skill bodies (empty string when unset). |
| `REPO_ROOT` | `mltgnt.skill.runner` | Replaces `$REPO_ROOT` in skill bodies (empty string when unset). |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge` | Set to `0` to skip the skill I/O type check in `enqueue_dag`. Any other value (or unset) keeps it on. |
| `THREAD_PERSONA_TTL_DAYS` | `mltgnt.conversation.thread_persona_store` | TTL in days for thread-to-persona bindings (default `30`; invalid values fall back to `30`). Ignored once a `ConversationConfig` is configured — its `thread_persona_ttl_days` wins. |

### Config dataclasses (`mltgnt.config`)

All are frozen dataclasses. Paths are injected by the host; mltgnt hardcodes none.

| Class | Required fields | Optional fields (default) |
|-------|-----------------|---------------------------|
| `PersonaConfig` | — | `weight_map` (`DEFAULT_WEIGHT_MAP`), `section_aliases` (`PERSONA_SECTION_ALIASES`), `exclude_stems` (`frozenset()`) |
| `MemoryConfig` | `chat_dir` | `chat_memory_dir` (`None`), `inject_max_bytes` (`10240`), `inject_max_entries` (`12`), `preferences_max_bytes` (`5120`), `lock_timeout_sec` (`30.0`), `lock_stale_threshold_sec` (`300.0`), `raw_days` (`7`), `mid_weeks` (`3`), `compact_threshold_bytes` (`40960`), `compact_target_bytes` (`25600`), `preferences_section_name`, `protected_layers` (`("caveat",)`), `timezone` (`"Asia/Tokyo"`), `dream_model` (`"claude-haiku-4-5-20251001"`), `use_dream_summary` (`False`), `dream_dir_name` (`"memory"`), `global_dream_exclude_personas` (`()`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir` | `timezone` (`"Asia/Tokyo"`), `salt` (`""`) |
| `ConversationConfig` | `queue_dir`, `sessions_dir`, `ledger_dir`, `thread_index_dir`, `thread_persona_path` | `posts_dir` (`None`), `audit_path` (`None`), `stale_after_sec` (`3600`), `max_queued` (`20`), `cleanup_ttl_days` (`14`), `thread_persona_ttl_days` (`30`) |

### Scheduler jobs (`ScheduleJob`)

`PersonaScheduler` loads jobs from the YAML file given as `yaml_path` / `SchedulerConfig.schedule_yaml`; each entry is parsed by `ScheduleJob.from_dict`, which raises `ValueError` on invalid input.

| Field | Default | Notes |
|-------|---------|-------|
| `id`, `mode`, `action` | required | `mode`: `scheduled` (needs `every_day_at`), `interval` (needs `interval_minutes > 0`), `fuzzy_window` (needs `window_start` / `window_end`), `chained` (needs `depends_on`) |
| `notify` | `silent` | `silent`, `slack_secretary`, or `slack_custom` (needs `slack_channel`) |
| `timezone` | `Asia/Tokyo` | |
| `enabled` | `true` | |
| `action_args` | `{}` | Must be a mapping |
| `every_day_at`, `every_week_on`, `interval_minutes`, `window_start`, `window_end` | `null` | Timing per mode |
| `fuzzy_method` | `hash` | `hash` or `random` |
| `on_window_missed` | `notify` | `notify`, `silent`, or `mark_done` |
| `slack_channel`, `persona` | `null` | |
| `timeout_seconds` | `600` | |
| `memory` | `false` | |
| `depends_on` | `[]` | Upstream job ids |
| `on_chain_failure` | `abort_notify` | `abort_notify` or `silent` |
| `on_exit` | `null` | Mapping; `nonzero` is required when present |
| `chain_every_run` | `false` | See below |
| `upstream_output` | — | Runtime only, never read from YAML |

**`chain_every_run`** — a `mode: chained` job with `chain_every_run: true` fires right after *every* successful run of a job listed in its `depends_on`, instead of once per day. It requires `mode: chained` and a non-empty `depends_on`; otherwise `from_dict` raises `ValueError`. The upstream job's output text is handed to the run as `ScheduleJob.upstream_output`, and the built-in `skill` action appends it to the skill input. Such jobs are never triggered by time and never write done / skipped / failed state marks.

In the example below, `health_check` is a custom action registered through `PersonaScheduler(actions=...)`; `skill` is built in and needs `action_args.skill` and `action_args.persona`.

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
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base class of `ConfigError` and `DependencyError`; catch it to handle both. |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Invalid configuration or arguments: bad `mltgnt run --components`, scheduler YAML that fails to load, more than one `primary` persona in one space in the routing map. CLI exit code `2`. |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | A dependency failed: the daemon PID lock is held by another instance, or an injected persona loader raised during routing. CLI exit code `3`. |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | `load_persona` found YAML frontmatter that does not parse or lacks the required `persona` key. |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | Loading skills failed: the ghdag tools list timed out, failed, or returned invalid JSON, or a skill references an unknown tool. |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | The compose-time skill I/O type check found a pipe type mismatch between DAG steps (disable with `SKILL_IO_TYPECHECK=0`). |
| `LlmCallError` | `mltgnt.memory.compaction` (re-exported by `mltgnt.memory`) | `RuntimeError` | Type for wrapping errors from an injected `llm_call` during memory compaction. mltgnt itself does not raise it in v0.66.0. |

`ScheduleJob.from_dict` and `system_default_engine()` raise the built-in `ValueError` for invalid job definitions and invalid `MLTGNT_DEFAULT_ENGINE` values.

## Public API Stability

- mltgnt is pre-1.0. A minor (`0.Y.0`) release may contain breaking changes; a patch (`0.Y.Z`) release does not. Every change is listed in `CHANGELOG.md`.
- The stable surface is `mltgnt.__all__` and `mltgnt.interfaces`. Other subpackage names may change in any minor release.
- Pin an exact tag (for example `@v0.66.0`) in production.

## Deprecated API

Still importable from `mltgnt.routing`:

| Deprecated | Use instead | Warning |
|------------|-------------|---------|
| `resolve_responding_persona` | `resolve_persona` | `DeprecationWarning` |
| `find_observers` | `find_observers_in_space` | `DeprecationWarning` |
| `ChannelPersonaEntry` | `SpacePersonaEntry` | none (plain alias) |

## License

MIT (SPDX: `MIT`). See `LICENSE`.
