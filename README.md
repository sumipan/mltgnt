# mltgnt

**Type contracts and chat I/O for multi-agent personas.** mltgnt is the L1 layer of the **L0 [ghdag](https://github.com/sumipan/ghdag) / L1 mltgnt / L2 host** stack: it owns persona, memory, skill, scheduling, and conversation contracts, while ghdag runs DAGs and LLM adapters and the host owns processes and channel I/O. Unlike agent frameworks that bundle model clients and orchestration, mltgnt is a typed domain layer you plug between a DAG runner and your own host.

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.62.0)-orange)

## Not (what this is not)

| Item | Why |
|------|-----|
| Not an LLM SDK | mltgnt never calls model providers directly; LLM calls go through ghdag (`mltgnt.bridges`). |
| Not a DAG engine | Queueing, dependency resolution, and DAG state are owned by ghdag. mltgnt only builds steps and waits for results. |
| Not a host runtime | Slack / CLI processes, deployment, and credentials are L2 host responsibilities. `mltgnt run` only hosts components that the host supplies. |

## Installation

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.62.0"
```

| Item | Value |
|------|-------|
| Package | `mltgnt` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.78.0` |
| Console script | `mltgnt` (entry point `mltgnt.cli.main:main`) |
| License | MIT |

Development extras (`pytest`, `pytest-asyncio`, `pytest-cov`, `freezegun`, `import-linter`, `mypy`, `ruff`):

```bash
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.62.0"
```

## Quick Start

The example below uses only `mltgnt.__all__` symbols and runs offline (no LLM, no network). It creates a persona file, loads and validates it, then writes and reads a dream summary.

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
(agents / "helper.md").write_text(
    "---\n"
    "persona:\n"
    "  name: helper\n"
    "---\n"
    "## Basic information\n"
    "\n"
    "A concise, friendly assistant.\n",
    encoding="utf-8",
)

print(list_personas(agents))                  # ['helper']
persona = load_persona("helper", persona_dir=agents)
print(persona.name)                           # helper
print(validate_persona(persona))              # [] (no warnings)

persona_dir = root / "chat" / "helper"
write_dream(
    persona_dir,
    DreamSummary(
        persona="helper",
        sections=[DreamSection(category="preferences", content="Prefers short answers.", source_entries=3)],
        updated_at="2026-01-01T00:00:00+00:00",
    ),
)
summary = read_dream(persona_dir)             # reads chat/helper/memory/dream.json
assert summary is not None
print([s.category for s in summary.sections])  # ['preferences']
```

The same dream summary can be inspected from the CLI with `mltgnt memory dream show <persona> --chat-dir <chat-dir>` (here `<persona>` is `helper` and `<chat-dir>` is `root / "chat"`).

## CLI Reference

The CLI is defined with argparse in `src/mltgnt/cli/main.py` and `src/mltgnt/cli/memory.py`. `python -m mltgnt` is equivalent to the `mltgnt` console script. Running without a subcommand prints help and exits with `0`.

| Command | Arguments | Behavior |
|---------|-----------|----------|
| `mltgnt run` | `--components MODULE:FUNCTION` (required), `--pid-file PATH` (default `/tmp/mltgnt_daemon.pid`) | Imports `MODULE`, calls `FUNCTION()` to obtain the daemon components, and runs them under `DaemonRunner` with a PID lock. |
| `mltgnt memory dream show` | `persona` (positional), `--chat-dir PATH` (required) | Prints each section of `<chat-dir>/<persona>/memory/dream.json`. Prints a notice and exits `0` when no summary exists. |
| `mltgnt memory dream forget` | `persona` (positional), `--category NAME` (required), `--chat-dir PATH` (required) | Removes one category from the dream summary. Exits `1` when the summary or category is missing. |

Exit codes for `mltgnt run`:

| Code | Cause |
|------|-------|
| `0` | Normal exit |
| `1` | Any other `MltgntError` |
| `2` | `ConfigError` (for example a malformed `--components` value, a missing module, or a missing / non-callable function) |
| `3` | `DependencyError` |

## Public API

`mltgnt.__all__` (`src/mltgnt/__init__.py`) exports exactly these 23 names. Everything else is reachable only through its subpackage.

| Group | Symbol | Kind | Signature / fields | Defined in |
|-------|--------|------|--------------------|------------|
| memory | `read_memory_iterative` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call, skill_paths=None, max_iterations=3) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_by_relevance` | function | `(config, persona_stem, query, *, max_bytes, max_entries, layers=None) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_with_sufficiency_check` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call=None) -> str` | `mltgnt.memory.search` |
| memory | `DreamSection` | frozen dataclass | `category: str`, `content: str`, `source_entries: int` | `mltgnt.memory.dream._format` |
| memory | `DreamSummary` | frozen dataclass | `persona: str`, `sections: list[DreamSection]`, `updated_at: str` | `mltgnt.memory.dream._format` |
| memory | `read_dream` | function | `(persona_dir, *, memory_dir_name="memory") -> DreamSummary \| None` | `mltgnt.memory.dream.api` |
| memory | `write_dream` | function | `(persona_dir, summary, *, memory_dir_name="memory") -> None` | `mltgnt.memory.dream.api` |
| persona | `Persona` | dataclass | `name`, `fm`, `sections`, `body`, `path`, `weight_map`; methods `format_prompt(instruction, *, weight="heavy")`, `register_prompt_filter(name, fn)` | `mltgnt.persona.loader` |
| persona | `load_persona` | function | `(name, *, persona_dir=None, config=None) -> Persona` (default directory `./agents`) | `mltgnt.persona` |
| persona | `list_personas` | function | `(persona_dir=None) -> list[str]` | `mltgnt.persona` |
| persona | `validate_persona` | function | `(persona, *, available_skills=None) -> list[str]` (warnings; empty means OK) | `mltgnt.persona` |
| persona | `run_persona_prompt` | function | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str` (calls an LLM through ghdag) | `mltgnt.persona.runner` |
| interfaces | `ChatInput` | dataclass | `source`, `session_key`, `messages`, `persona_name=""`, `model=None`, `context_files=[]`, `context_memory_excerpt=None`, `context_memory_preferences=None` | `mltgnt.interfaces.types` |
| interfaces | `ChatOutput` | dataclass | `content`, `persona_name`, `timestamp`, `session_key` | `mltgnt.interfaces.types` |
| interfaces | `Message` | TypedDict | `role: str`, `content: str` | `mltgnt.interfaces.types` |
| interfaces | `PersonaProtocol` | Protocol | `name`, `fm`, `format_prompt(instruction)`, `register_prompt_filter(name, fn)` | `mltgnt.interfaces.persona` |
| agent | `AgentResult` | dataclass | `tool`, `args`, `raw_response`, `tool_trace=None`, `reflexion_count=0` | `mltgnt.agent._runner` |
| agent | `AgentRunner` | class | `(*, llm_call, tool_executor, terminal_tools, max_iterations=3, max_iterations_fn=None, evaluator=None, retry_config=None, logger=None, audit_writer=None, classifier=None)` | `mltgnt.agent._runner` |
| bridges | `enqueue_dag` | function | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, skills=None, permission=None, order_builder=None) -> list[tuple[bool, str]]` | `mltgnt.bridges.ghdag_bridge` |
| bridges | `enqueue_and_wait` | function | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_name=None, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, permission=None, order_builder=None, run_result=None) -> tuple[bool, str]` | `mltgnt.bridges.ghdag_bridge` |
| scheduler | `PersonaScheduler` | class | `(slack, *, config=None, state_dir=None, yaml_path=None, salt="", jobs=None, ..., actions=None, memory_config=None)` | `mltgnt.scheduler.runner` |
| scheduler | `ScheduleJob` | dataclass | `id`, `mode` (`scheduled` / `fuzzy_window` / `interval` / `chained`), `action`, `notify`, ...; `ScheduleJob.from_dict(raw)` | `mltgnt.scheduler.models` |
| version | `__version__` | str | Installed distribution version (`"0.0.0"` when metadata is unavailable) | `mltgnt` |

## Protocols / Extension Points

Hosts extend mltgnt by implementing these Protocols or by injecting callables. The `mltgnt.interfaces` package re-exports all interface types.

| Extension point | Module | Contract |
|-----------------|--------|----------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Structural type for persona objects (`name`, `fm`, `format_prompt`, `register_prompt_filter`). |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, thread_ts=None, blocks=None, reply_broadcast=False) -> bool`; returns `False` on failure instead of raising. |
| `TurnHandler` | `mltgnt.interfaces.turn` | `handle(turn: TurnInput) -> TurnResult`. Media-agnostic boundary between the host media layer and the conversation layer. |
| `TurnInput` / `TurnResult` / `Attachment` / `HistoryMessage` | `mltgnt.interfaces.turn` | Frozen dataclasses carried across `TurnHandler`. `TurnResult.kind` is `"reply"` or `"task"`. |
| `PersonaFMBase` / `ChatInputBase` / `ChatOutputBase` | `mltgnt.interfaces.types` | Runtime-checkable Protocols matching persona frontmatter and chat input / output shapes. |
| `DaemonComponent` | `mltgnt.daemon` | `name` property, non-blocking `start()`, `stop()`. The `--components` factory of `mltgnt run` returns these. |
| `LLMCaller` | `mltgnt.agent._runner` | `(prompt, *, tool_result=None) -> str \| None`; injected into `AgentRunner` as `llm_call`. |
| `ToolExecutor` | `mltgnt.agent._runner` | `(tool_name, tool_args) -> str`; injected into `AgentRunner` as `tool_executor`. |
| `ReflexionEvaluator` | `mltgnt.agent._runner` | `(prompt, tool_name, tool_args, tool_result, tool_trace) -> ReflexionVerdict`; optional `AgentRunner` `evaluator`. |
| `LanguagePack` / `JA` | `mltgnt.config.language` | Frozen dataclass holding locale-specific vocabulary. Functions that take `pack=None` fall back to the default `JA` pack; pass your own pack to switch locale. |

## Architecture

| Path | Responsibility |
|------|----------------|
| `agent/` | Generic tool-calling agent loop (`AgentRunner`) with retry and Reflexion hooks, plus action classification and dispatch-decision gates. |
| `bridges/` | Adapters to ghdag: DAG submission (`enqueue_dag`, `enqueue_and_wait`), LLM calls, DAG hooks / audit writer, Markdown file I/O. |
| `cli/` | argparse CLI (`mltgnt run`, `mltgnt memory dream show`, `mltgnt memory dream forget`). |
| `config/` | Configuration dataclasses (`PersonaConfig`, `MemoryConfig`, `SchedulerConfig`, `ConversationConfig`) and `LanguagePack`. |
| `conversation/` | Media-agnostic conversation layer: queue, session ledger, thread index, thread-to-persona binding, session compaction. |
| `daemon/` | `DaemonComponent` Protocol, `DaemonRunner`, PID lock, skill registry watcher component. |
| `interfaces/` | Dependency-free DTOs and Protocols shared by all layers. |
| `memory/` | Persona memory JSONL read / search (TF-IDF relevance, sufficiency check, iterative retrieval), compaction; `memory/dream/` synthesizes and stores `dream.json` summaries. |
| `persona/` | Persona file loading, frontmatter schema and validation, registry, prompt formatting, light-block compression, persona prompt runner. |
| `routing/` | Space-to-persona routing, observer lookup, nickname detection, LLM triage, agentic skill discovery. |
| `scheduler/` | Job model and YAML loader, `PersonaScheduler` tick loop, run state, fan-out; `scheduler/actions/` holds built-in dream and skill actions. |
| `skill/` | Markdown skill discovery, loading, lint, matching, context building, and execution. |
| `exceptions.py` | Shared exception hierarchy (`MltgntError`, `ConfigError`, `DependencyError`). |
| `__main__.py` | Enables `python -m mltgnt`. |

### Layer contracts (`.importlinter`)

Layers, top to bottom. A layer may import only layers below it. `mltgnt.config` is importable from anywhere; the other allowed exceptions are listed under `ignore_imports` in `.importlinter`.

| Layer | Packages |
|-------|----------|
| 1 | `daemon` |
| 2 | `scheduler` \| `agent` \| `routing` |
| 3 | `persona` \| `skill` \| `memory` \| `conversation` |
| 4 | `bridges` |
| 5 | `interfaces` |

| Contract | Type | Rule |
|----------|------|------|
| mltgnt layered architecture | layers | Enforces the table above. |
| L3 domain must not import ghdag directly | forbidden | `persona`, `skill`, `memory`, `conversation` must not import `ghdag`; they go through `bridges`. |

## Configuration

Most settings are passed as config dataclasses from `mltgnt.config`. The only environment variables read under `src/` are:

| Variable | Used in | Meaning |
|----------|---------|---------|
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Value substituted for `$NIKKI_ROOT` in skill bodies (empty string when unset). |
| `REPO_ROOT` | `mltgnt.skill.runner` | Value substituted for `$REPO_ROOT` in skill bodies (empty string when unset). |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge` | Skill I/O type checking in `enqueue_dag` runs unless this is `"0"`. |
| `THREAD_PERSONA_TTL_DAYS` | `mltgnt.conversation.thread_persona_store` | TTL in days for thread-pinned personas (default `30`). Ignored when a `ConversationConfig` has been configured. |

## Error Reference

| Type | Module | Bases | Raised when |
|------|--------|-------|-------------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Common base; catch it to handle all mltgnt errors below. |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Configuration or argument error (for example invalid `--components`). CLI exit code `2`. |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | External dependency (callable, subprocess, API) failure. CLI exit code `3`. |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | Persona frontmatter cannot be parsed or lacks the required `persona` key (`load_persona`). |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | Skill load or tool validation failure. |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | Pipe type mismatch detected by the compose-time skill I/O type check. |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | Wraps an error raised by the injected `llm_call` during memory compaction. |

## Public API Stability

- mltgnt is pre-1.0 and follows SemVer in its pre-1.0 form: a minor (`Y`) bump may contain breaking changes; patch (`Z`) bumps do not.
- The stable surface is `mltgnt.__all__` and `mltgnt.interfaces`. Other subpackage symbols may change in any minor release; see `CHANGELOG.md`.
- Pin an exact tag (for example `@v0.62.0`) in production.

## Deprecated API

These remain importable from `mltgnt.routing` and emit `DeprecationWarning`.

| Deprecated | Replacement |
|------------|-------------|
| `resolve_responding_persona` | `resolve_persona` |
| `find_observers` | `find_observers_in_space` |
| `ChannelPersonaEntry` | `SpacePersonaEntry` (alias; no warning) |

## License

MIT (SPDX: `MIT`).
