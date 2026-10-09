# mltgnt

**Typed building blocks for multi-agent chat hosts: personas, memory, skills, an agent loop, routing, scheduling, conversation state, and a media I/O contract.**

mltgnt is a Python library that sits between an execution engine and a host application:

| Layer | Owner | Responsibility |
|-------|-------|----------------|
| Engine | [ghdag](https://github.com/sumipan/ghdag) | DAG execution, LLM engines, VCS sinks |
| Library | **mltgnt** | Domain types and behavior: personas, memory, skills, agent loop, routing, scheduler, conversation state, media contract |
| Host | Your application | Processes, credentials, filesystem paths, deployment |

Every LLM call and every git commit leaves mltgnt through `mltgnt.bridges`, the only package that imports ghdag. Chat media (Slack, WebChat, or your own) plug in through one Protocol, `MediaClient`.

## Status

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.123.0)-orange)

Pre-1.0. The current release is `0.123.0`. Minor releases may change the API; see [Public API Stability](#public-api-stability).

## Not

| mltgnt is not | Instead |
|---------------|---------|
| An LLM SDK | It ships no model client. `call_llm`, `enqueue_and_wait`, and `enqueue_dag` in `mltgnt.bridges` hand work to ghdag. |
| A DAG engine | Queueing, dependency resolution, and job state belong to ghdag. mltgnt builds steps and waits for results. |
| A host runtime | The host supplies paths, secrets, and process layout. `mltgnt run` only starts the components a host factory returns. |
| A Slack bot | Slack and WebChat are optional adapters behind `MediaClient`. Core packages never import `mltgnt.media`. |

## Installation

mltgnt is distributed as git tags, not on PyPI:

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.123.0"

# with the Slack medium (mltgnt.media.slack)
pip install "mltgnt[slack] @ git+https://github.com/sumipan/mltgnt.git@v0.123.0"

# with the WebChat medium (mltgnt.media.webchat)
pip install "mltgnt[webchat] @ git+https://github.com/sumipan/mltgnt.git@v0.123.0"

# with development tools (pytest, ruff, mypy, import-linter)
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.123.0"
```

| Item | Value |
|------|-------|
| Distribution | `mltgnt` `0.123.0` |
| Python | `>=3.10` |
| Dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.100.0` |
| Extra `slack` | `slack_sdk>=3.0`, `slack_bolt>=1.18` |
| Extra `webchat` | `fastapi>=0.100`, `uvicorn>=0.20` |
| Extra `dev` | `pytest>=7.0`, `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `freezegun>=1.2`, `import-linter>=2.0`, `mypy>=1.10`, `ruff>=0.4` |
| Console script | `mltgnt` -> `mltgnt.cli.main:main` (`python -m mltgnt` is equivalent) |
| Typing | Ships `py.typed` |
| Optional, not declared | `chromadb`: when importable, memory search also queries a Chroma collection; otherwise TF-IDF only |

The Slack and WebChat apps import their third-party packages lazily, so the rest of `mltgnt.media` imports without the extras.

## Quick Start

The example below needs no LLM and no network. It loads a persona, stores a dream summary, and drives the agent loop with a scripted LLM.

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

# Personas are Markdown files with YAML frontmatter.
print(list_personas(agents))                     # ['guide']
persona = load_persona("guide", persona_dir=agents)
print(persona.name, validate_persona(persona))   # guide []

# Dream summaries live in <chat_dir>/<persona>/memory/dream.json.
write_dream(
    root / "chat" / "guide",
    DreamSummary(
        persona="guide",
        sections=[DreamSection(category="style", content="Prefers bullet points.", source_entries=2)],
        updated_at="2026-01-01T00:00:00+00:00",
    ),
)
print(read_dream(root / "chat" / "guide").sections[0].category)  # style

# The agent loop takes the LLM and the tools as plain callables.
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

Read the stored summary back from the shell:

```bash
mltgnt memory dream show guide --chat-dir /path/to/root/chat
```

A custom chat medium only has to satisfy the `MediaClient` Protocol:

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

Defined in `src/mltgnt/cli/main.py` and `src/mltgnt/cli/memory.py`. `mltgnt` with no subcommand prints help and exits `0`.

### `mltgnt run`

Imports `MODULE`, calls `FUNCTION()` to get a list of `DaemonComponent` objects, and runs them with `DaemonRunner` under a PID lock until SIGINT or SIGTERM.

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--components MODULE:FUNCTION` | yes | - | Component factory, e.g. `myhost.daemon:components` |
| `--pid-file PATH` | no | `/tmp/mltgnt_daemon.pid` | PID lock file |

| Exit code | Meaning |
|-----------|---------|
| `0` | Normal shutdown |
| `1` | Any other `MltgntError` |
| `2` | `ConfigError`: `--components` is not `module:function`, the module cannot be imported, or the attribute is missing or not callable |
| `3` | `DependencyError`: another process holds the PID lock |

### `mltgnt memory dream show`

Prints each section of `<chat-dir>/<persona>/memory/dream.json` as `=== <category> (source_entries: N) ===` followed by its content.

| Argument / option | Required | Description |
|-------------------|----------|-------------|
| `persona` | yes | Persona name (directory name under `--chat-dir`) |
| `--chat-dir PATH` | yes | Parent directory of the persona directories |

Exits `0`, including when no summary exists (a notice is printed instead).

### `mltgnt memory dream forget`

Removes one category from the dream summary and rewrites `dream.json`.

| Argument / option | Required | Description |
|-------------------|----------|-------------|
| `persona` | yes | Persona name (directory name under `--chat-dir`) |
| `--category NAME` | yes | Category to remove |
| `--chat-dir PATH` | yes | Parent directory of the persona directories |

Exits `0` on success and `1` when the summary or the category does not exist.

## Public API

### Top level: `mltgnt.__all__`

These 23 names are the top-level surface. Everything else is imported from its subpackage.

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
| `validate_persona` | function | `(persona, *, available_skills=None) -> list[str]` (warnings; empty means valid) | `mltgnt.persona` |
| `run_persona_prompt` | function | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str` | `mltgnt.persona` |
| `ChatInput` | dataclass | `source`, `session_key`, `messages`, `persona_name=""`, `model=None`, `context_files=[]`, `context_memory_excerpt=None`, `context_memory_preferences=None` | `mltgnt.interfaces.types` |
| `ChatOutput` | dataclass | `content`, `persona_name`, `timestamp`, `session_key` | `mltgnt.interfaces.types` |
| `Message` | TypedDict | `role`, `content` | `mltgnt.interfaces.types` |
| `PersonaProtocol` | Protocol | Structural persona type; `Persona` satisfies it | `mltgnt.interfaces.persona` |
| `AgentRunner` | class | See [`mltgnt.agent`](#mltgntagent) | `mltgnt.agent._runner` |
| `AgentResult` | dataclass | `tool`, `args`, `raw_response`, `tool_trace=None`, `reflexion_count=0`, `plan=None` | `mltgnt.agent._runner` |
| `enqueue_and_wait` | function | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_name=None, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, permission=None, order_builder=None, run_result=None, task_timeout_sec=None) -> tuple[bool, str]` | `mltgnt.bridges.ghdag_bridge` |
| `enqueue_dag` | function | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, skills=None, permission=None, order_builder=None, task_timeout_sec=None) -> list[tuple[bool, str]]` | `mltgnt.bridges.ghdag_bridge` |
| `PersonaScheduler` | class | `(slack, *, config=None, state_dir=None, yaml_path=None, salt="", jobs=None, notify_channel_resolver=None, default_slack_post_kwargs=None, persona_post_kwargs_resolver=None, repo_root=None, persona_dir=None, append_memory_fn=None, actions=None, memory_config=None)`; `slack` is a `MediaClient` | `mltgnt.scheduler.runner` |
| `ScheduleJob` | dataclass | See [Scheduler jobs](#scheduler-jobs); build with `ScheduleJob.from_dict(raw)` | `mltgnt.scheduler.models` |
| `__version__` | `str` | Installed distribution version; `"0.0.0"` when metadata is unavailable | `mltgnt` |

`run_persona_prompt` uses the persona's `engine`, falling back to `mltgnt.persona.schema.system_default_engine()`. `task_timeout_sec` on the enqueue functions is passed to ghdag as the per-task execution limit (`None` keeps the host default).

### `mltgnt.agent`

`AgentRunner(*, llm_call, tool_executor, terminal_tools, max_iterations=3, max_iterations_fn=None, evaluator=None, retry_config=None, logger=None, audit_writer=None, classifier=None, history_mode="last_result", history_max_chars=24000, plan=None, max_reflexions=None, step_hook=None)`.
`runner.run(prompt)` returns an `AgentResult` once the LLM picks a terminal tool, or `None` when the LLM call or response parsing fails.

| Option | Effect |
|--------|--------|
| `history_mode="full_trace"` | Sends the numbered tool trace to the LLM instead of only the last result; older bodies are truncated beyond `history_max_chars` |
| `evaluator` | A `ReflexionEvaluator`. `DefaultReflexionEvaluator(failure_markers=(), repeat_window=3)` retries on `[ERROR]` results, host failure markers, or a repeated `(tool, args)` |
| `max_reflexions=N` | Caps Reflexion retries; when exceeded, `AgentResult.tool` is `"__reflexion_exhausted__"` |
| `plan=Plan(...)` | Updated from the `plan_update` key of LLM responses and returned as `AgentResult.plan` |
| `step_hook(entry)` | Called after each tool-trace entry; exceptions are logged and ignored |

| Names | Purpose |
|-------|---------|
| `AgentRunner`, `AgentResult`, `DefaultReflexionEvaluator` | Tool-calling loop, result, default evaluator |
| `run_work_loop`, `WorkLoopConfig`, `WorkLoopOutcome`, `GhdagSkillRunner` | Multi-step work loop with plan + Reflexion; optional `run_skill` via DAG (`mltgnt.agent.work_loop`) |

```python
from mltgnt.agent import WorkLoopConfig, run_work_loop

outcome = run_work_loop(
    order_text,
    llm_call=my_llm_call,
    tools=my_tools,
    cfg=WorkLoopConfig(max_iterations=40, tool_skills=("my-skill",)),
    skill_runner=my_skill_runner,  # optional; use GhdagSkillRunner for ghdag enqueue
    parent_correlation_id="loop-uuid",
)
# outcome.status: IMPL_DONE | IMPL_FAILED | BLOCKED; outcome.reason is a stable code
```
| `Plan`, `PlanItem`, `parse_plan`, `build_plan_prompt` | Plan tracking. `PlanItem(id, title, depends=[], status="pending", note="")`; `Plan.apply(updates)`, `Plan.progress() -> (done, total)`; `parse_plan` raises `ValueError` |
| `DispatchDecision`, `make_dispatch_decision`, `MODE_REPLY`, `MODE_DELEGATE` | Reply-or-delegate decision (`"reply"` / `"delegate"`) |
| `should_force_delegate`, `should_preempt_delegate`, `has_work_request`, `is_create_request`, `match_deferred_promise`, `extract_artifact_references` | Deterministic request gates driven by the current `LanguagePack` |
| `PreflightContext`, `run_preflight`, `DirectAgentResult`, `SkillWorkerResult`, `MemoryWorkerResult` | Preflight before dispatch and worker result types |

### `mltgnt.bridges`

| Names | Purpose |
|-------|---------|
| `enqueue_dag`, `enqueue_and_wait`, `DagStep` | Build ghdag steps, enqueue them, wait for results |
| `call_llm` | `(prompt, *, engine="", model="", timeout=120)`; a single text call through ghdag |
| `MltgntHooks`, `create_audit_writer` | ghdag DAG hooks and the audit writer for `AgentRunner(audit_writer=...)` |
| `md_read`, `md_write` | Wrappers of `ghdag.files` |
| `ghdag_bridge`, `llm_adapter`, `hooks_adapter`, `files_adapter` | Submodules. `files_adapter.commit(paths, message, *, sink="memory", trailers=None)` commits through a ghdag VCS sink |

### `mltgnt.persona`

| Names | Purpose |
|-------|---------|
| `Persona`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt` | Same as the top level |
| `PersonaContext` | Resolved persona context for one turn |
| `PersonaValidationError` | See [Error Reference](#error-reference) |
| `format_persona_body`, `format_result_for_persona` | Tone formatting and persona-voiced result text |
| `compress_heavy_to_light`, `regenerate_light_block` | LLM compression of the heavy persona block into the light block |

`mltgnt.persona.schema.system_default_engine()` returns the host-wide default engine; see `MLTGNT_DEFAULT_ENGINE`.

### `mltgnt.memory`

| Group | Names |
|-------|-------|
| Episodic log | `append_memory_entry`, `read_memory_preferences`, `read_memory_tail_text`, `memory_file_path`, `persona_memory_lock`, `tail_utf8_bytes`, `assemble_entries_text` |
| Search | `read_memory_by_relevance`, `read_memory_with_sufficiency_check`, `read_memory_iterative` |
| Entries | `MemoryEntry`, `parse_jsonl`, `serialize_entry` |
| Compaction | `compact`, `needs_compaction`, `CompactionResult`, `LlmCall`, `LlmCallError` |
| Commits | `flush_memory_commits` |
| Chroma (optional) | `get_collection`, `query_similar`, `upsert_entry`; `get_collection` returns `None` without `chromadb` |
| Constants | `MEMORY_CORRUPT_THRESHOLD_BYTES` (`10`), `MEMORY_DEDUPE_SCAN_BYTES` (`32768`), `MEMORY_DEDUPE_SCAN_LINES` (`200`) |

`mltgnt.memory.__all__` also contains underscore-prefixed helpers. They are internal and not supported.

Memory submodules with their own `__all__`:

| Module | Names | Purpose |
|--------|-------|---------|
| `mltgnt.memory.dream` | `DreamSection`, `DreamSummary`, `read_dream`, `write_dream`, `read_global`, `write_global`, `read_global_summary`, `DreamSelector`, `Synthesizer` | Per-persona `dream.json` and cross-persona `global.json` summaries |
| `mltgnt.memory.semantic` | `SemanticStore`, `SemanticEntry`, `KINDS`, `STATUSES`, `validate_entry`, `normalize_content` | `SemanticStore(path, *, persona_stem, commit_debounce_sec=None)` over `semantic.jsonl`. Kinds: `fact`, `preference`, `commitment`, `caveat`, `self`, `reflection`. Statuses: `active`, `superseded` |
| `mltgnt.memory.tools` | `MemoryToolExecutor`, `MemoryGate`, `MEMORY_TOOL_SPECS`, `MEMORY_TOOL_NAMES`, `query_terms` | `remember` / `recall` / `forget` tools shaped as a `ToolExecutor`; `MemoryGate(max_per_turn=3, max_chars=300, secret_check=None, allowed_subject_prefixes=None)` |
| `mltgnt.memory.reflection` | `build_reflection_prompt`, `parse_reflection`, `apply_reflection`, `ReflectionResult`, `ReflectionAdd`, `ApplyReport`, `ReflectionParseError` | Reflection prompt, parser, and applier for a `SemanticStore` |
| `mltgnt.memory.core_render` | `render_core`, `MANDATORY_KINDS`, `OPTIONAL_KINDS` | `render_core(entries, *, max_bytes=4096, heading="## Memory")`; `caveat` and `commitment` are always kept |
| `mltgnt.memory.archive` | `archive_episodes` | `archive_episodes(persona_dir, *, now, keep_days=90) -> int`; moves old episodes into monthly files |

### `mltgnt.skill`

| Names | Purpose |
|-------|---------|
| `discover`, `discover_bodies`, `load` | Find and parse skill files |
| `match` | `(user_input, skills, persona_skills=None, model=None, *, engine="claude") -> SkillMatchResult` |
| `resolve_skill` | `(user_input, skill_paths, persona_skills=None, entry_file="SKILL.md", matcher_model=None, *, matcher_engine="claude")` |
| `run` | `(skill, persona, arguments, chat_input, extra_context=None) -> SkillRunResult`; substitutes `$ARGUMENTS`, `$PERSONA`, `$SKILL_DIR`, `$NIKKI_ROOT`, `$REPO_ROOT`, `$0`, `$1`, ... |
| `build_extra_context` | Extra prompt context from skill knowledge files and persona memory |
| `lint_skill_meta` | Skill frontmatter validation |
| `SkillMeta`, `SkillFile`, `SkillRegistry`, `SkillRunResult`, `SkillMatchResult`, `ArtifactSpec`, `ProducesSpec`, `ConsumesSpec` | Data types |

With `engine="claude"` and no model, the LLM matcher uses `claude-haiku-4-5-20251001`; other engines use their ghdag default model.

### `mltgnt.routing`

| Names | Purpose |
|-------|---------|
| `resolve_persona` | Pick the responding persona for a message in a space / conversation |
| `find_observers_in_space` | Other personas present in a space |
| `load_channel_persona_map` | Build the space-to-persona map from persona frontmatter |
| `SpacePersonaEntry`, `RoutingRule`, `evaluate` | Routing entries and rules |
| `detect_nickname` | Mention detection by persona nickname |
| `extract_json_object`, `extract_triage_section`, `prepare_profile_for_triage`, `TRIAGE_PROFILE_MAX_CHARS` (`6000`) | Triage helpers |
| `resolve_responding_persona`, `find_observers`, `ChannelPersonaEntry` | Deprecated; see [Deprecated API](#deprecated-api) |

### `mltgnt.conversation`

| Names | Purpose |
|-------|---------|
| `configure`, `get_config`, `ConversationConfig` | Set and read the active conversation paths |
| `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage` | Re-exported from `mltgnt.interfaces.turn` |
| `thread_queue`, `session_store`, `session_compact`, `thread_index`, `thread_persona_store`, `fake_media` | Submodules: per-conversation queue, session ledger, log compaction, thread index, persona bindings, `TurnInput` builder without a real medium |

### `mltgnt.scheduler`, `mltgnt.daemon`, `mltgnt.config`

| Package | Names |
|---------|-------|
| `mltgnt.scheduler` | `PersonaScheduler`, `ScheduleJob`, `SchedulePaths`, `load_schedule_jobs(yaml_path, *, default_timezone="Asia/Tokyo")`, `atomic_write_text` |
| `mltgnt.daemon` | `DaemonComponent`, `DaemonRunner(*, pid_file, components, logger=None)`, `PidLock(pid_file)`, `SkillWatcherComponent(registry, interval=5.0)` |
| `mltgnt.config` | `PersonaConfig`, `MemoryConfig`, `SchedulerConfig`, `ConversationConfig`, `DEFAULT_WEIGHT_MAP` |
| `mltgnt.config.language` | `LanguagePack`, `EN`, `get_language_pack`, `set_language_pack` (the last three are also importable from `mltgnt.config`) |

### `mltgnt.interfaces`

Dependency-free DTOs and Protocols shared by every layer.

| Module | Names |
|--------|-------|
| `mltgnt.interfaces` | `PersonaProtocol`, `PersonaFMBase`, `Message`, `ChatInput`, `ChatOutput`, `ChatInputBase`, `ChatOutputBase`, `TurnHandler`, `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, `SlackClientProtocol` (deprecated) |
| `mltgnt.interfaces.media` | `Status`, `MediaClient`, `adapt_client` |

| Name | Contract |
|------|----------|
| `Status` | `str` Enum: `RECEIVED`, `WORKING`, `DONE`, `FAILED`, `CANCELLED` (values are the lowercase names) |
| `MediaClient` | `post(text, space, thread=None, *, extra=None) -> str \| None`, `update(message_id, text) -> bool`, `set_status(message_id, status) -> bool`, `upload(path, space, thread=None) -> bool`, `react(message_id, name) -> bool`. Failures return `None` / `False` instead of raising |
| `adapt_client` | `(obj) -> MediaClient`. Objects with `post` pass through; objects with only `post_message` are wrapped with one `DeprecationWarning`; anything else raises `TypeError` |

### `mltgnt.media`

`mltgnt.media` and its subpackages export nothing at package level (`__all__ = []`). Import from the modules below.

`mltgnt.media._core` (medium-independent):

| Module | Main names | Purpose |
|--------|------------|---------|
| `types` | `MediaEvent`, `OutboundMessage` | `MediaEvent(space_id, conversation_id, message_id, author, text, attachments=(), raw=None)`; `OutboundMessage(text, space_id, thread_id=None)` |
| `client` | `MediaClient`, `Status`, `adapt_client` | Re-export of `mltgnt.interfaces.media` |
| `config` | `MediaConfig` | See [Media settings](#media-settings) |
| `bridge` | `MediaBridge` | `MediaBridge(client, handler, config, hooks=None, *, pending_prefix="pending-")`. `handle_event(event)` runs one turn and posts a reply or records a pending task; `deliver_result(uid, body, *, post_options=None)` posts a delegated result |
| `hooks` | `HookRegistry`, `OnInbound`, `BeforeDispatch`, `AfterPost`, `OnResult` | Turn hooks, run in registration order; a hook that raises is logged and skipped |
| `component` | `MediaBridgeComponent` | `(start, *, mode, backoff_sec, on_exit=None)`; runs `start` in a process or thread and restarts it with backoff |
| `watchers` | `ExecDoneHandler`, `ProgressWatcher`, `DeliveryReconciler`, `catchup_pending_on_startup`, `iter_pending_with_done`, `CATCHUP_STATES` | Job completion, startup catch-up, delivery reconciliation, progress polling |
| `progress` | `ProgressState`, `finalize_progress`, `status_for_done_marker`, `status_label`, `summarize_tool_use_block`, `summarize_work_loop_step` | Progress message text |
| `plan_gate` | `PlanGate`, `PlanState`, `is_approval`, `expire_pending`, `AWAITING_STATE`, `PENDING_KEY` | Plan approval state machine |
| `cancel` | `CancelOutcome`, `handle_cancel`, `is_cancel_request`, `find_pending_uids` | Cancel requests posted in a thread |
| `enqueue_guard` | `enqueue_or_report` | Posts `LanguagePack.enqueue_failed_text` when an enqueue fails |
| `pending` | `PendingStore` | One JSON record per pending request |
| `id_map` | `to_conversation_id`, `resolve`, `storage_key`, `storage_key_from_conversation_id` | Conversation id <-> `(space, thread)` |
| `thread_reactions` | `admit`, `status_for_admission`, `acknowledge_drained` | Queue admission mapped to `Status` |
| `sanitizer` | `sanitize_result_body`, `strip_status_lines`, `strip_leading_paragraphs`, `dedupe_trailing_repeated_block`, `extract_final_assistant_text` | Result cleanup before posting |

`MediaBridge` forwards `TurnResult.post_options` to `MediaClient.post(extra=...)` for immediate replies and for `deliver_result`.

`mltgnt.media.slack` (needs extra `slack`):

| Module | Main names | Purpose |
|--------|------------|---------|
| `config` | `SlackMediaConfig`, `DEFAULT_STATUS_REACTIONS` | See [Media settings](#media-settings) |
| `client` | `SlackClient`, `split_text`, `POST_EXTRA_KEYS`, `UPDATE_EXTRA_KEYS` | `SlackClient(web_client, config, *, default_channel="")`. Statuses become reactions; long text is split at `chunk_max_chars`. `POST_EXTRA_KEYS = {"username", "icon_emoji", "icon_url", "blocks", "reply_broadcast"}`, `UPDATE_EXTRA_KEYS = {"blocks"}`; other `extra` keys raise `ValueError` before the API call. Also provides `upload(..., *, title=None)`, `react`, and `unreact` |
| `app` | `build_app`, `start_socket_mode` | `build_app(config)` creates a Bolt app; `start_socket_mode(app, config)` blocks in Socket Mode |
| `inbound` | `to_media_event`, `strip_mentions`, `thread_ts_from_event`, `files_to_attachments` | Slack event to `MediaEvent` |
| `history` | `fetch_thread_messages`, `extract_text_from_message`, `extract_text_from_blocks`, `table_block_to_markdown` | Thread fetch and text extraction |
| `media_store` | `save_images`, `SavedMedia`, `safe_filename`, `ext_from_mimetype` | Image attachment download |
| `mrkdwn` | `markdown_to_mrkdwn`, `normalize_markdown_residual` | Markdown to Slack mrkdwn |

`mltgnt.media.webchat` (the app needs extra `webchat`):

| Module | Main names | Purpose |
|--------|------------|---------|
| `config` | `WebChatMediaConfig` | See [Media settings](#media-settings) |
| `client` | `WebChatClient` | `WebChatClient(config, *, store=None, author="assistant")` |
| `store` | `WebChatStore`, `KINDS`, `MESSAGE_KINDS` | Daily JSONL of message snapshots. `KINDS = ("message", "update", "status", "bookmark", "reaction")` |
| `app` | `create_app`, `serve`, `EventHandler`, `parse_event_id` | `create_app(config, bridge, *, store=None, poll_interval_sec=0.5, stream_timeout_sec=None)` serves `GET /`, `GET /config`, `GET /assets/{name}`, `GET /messages`, `POST /messages`, `GET /threads/{thread_id}`, `POST /messages/{message_id}/bookmark`, `GET /bookmarks`, and `GET /stream` (SSE); `serve(app, config)` listens on `config.host:config.port` |
| `inbound` | `to_media_event`, `DEFAULT_AUTHOR` | `POST /messages` body to `MediaEvent` |
| `ui` | `INDEX_HTML` | Single-page UI served at `GET /` |

## Protocols / Extension Points

| Extension point | Module | Contract | Consumed by |
|-----------------|--------|----------|-------------|
| `MediaClient` | `mltgnt.interfaces.media` | `post` / `update` / `set_status` / `upload` / `react` | `PersonaScheduler`, `MediaBridge`; implemented by `SlackClient`, `WebChatClient` |
| `TurnHandler` | `mltgnt.interfaces.turn` | `handle(turn: TurnInput) -> TurnResult`; `TurnResult.kind` is `"reply"` or `"task"`, plus `text`, `task_ref`, `reaction`, `post_options` | `MediaBridge` |
| `HookRegistry` hooks | `mltgnt.media._core.hooks` | `OnInbound(event) -> bool` (`True` stops the turn), `BeforeDispatch(turn) -> TurnInput`, `AfterPost(result, message_id)`, `OnResult(uid, body)` | `MediaBridge` |
| `EventHandler` | `mltgnt.media.webchat.app` | `handle_event(event: MediaEvent)` | `create_app`; `MediaBridge` satisfies it |
| `DaemonComponent` | `mltgnt.daemon` | `name` property, non-blocking `start()`, `stop()` | `DaemonRunner`, `mltgnt run` |
| `LLMCaller` | `mltgnt.agent._runner` | `(prompt, *, tool_result=None) -> str \| None` | `AgentRunner(llm_call=...)` |
| `ToolExecutor` | `mltgnt.agent._runner` | `(tool_name, tool_args) -> str`; `MemoryToolExecutor` is a built-in implementation | `AgentRunner(tool_executor=...)` |
| `ReflexionEvaluator` | `mltgnt.agent._runner` | `(prompt, tool_name, tool_args, tool_result, tool_trace) -> ReflexionVerdict` | `AgentRunner(evaluator=...)` |
| `ActionFn` | `mltgnt.scheduler.models` | `(job: ScheduleJob) -> tuple[bool, str]` | `PersonaScheduler(actions={name: fn})` |
| `PersonaProtocol`, `PersonaFMBase`, `ChatInputBase`, `ChatOutputBase` | `mltgnt.interfaces` | Structural types for personas and chat I/O | Skill runner, routing |
| `LanguagePack` | `mltgnt.config.language` | Frozen dataclass of locale vocabulary | See [LanguagePack](#languagepack) |

## Architecture

### Packages

| Package | Role |
|---------|------|
| `agent` | Tool-calling loop, dispatch decision, Reflexion, plan tracking |
| `bridges` | ghdag integration: LLM calls, DAG enqueue, file I/O, VCS commits, audit hooks |
| `cli` | `mltgnt` console entry point |
| `config` | Frozen config dataclasses and `LanguagePack` |
| `conversation` | Per-conversation queue, session ledger, thread index, persona bindings |
| `daemon` | `DaemonRunner`, PID lock, `SkillWatcherComponent` |
| `interfaces` | Dependency-free DTOs and Protocols |
| `media` | `MediaBridge` core plus the `slack` and `webchat` adapters |
| `memory` | Episodic log, compaction, semantic store, dream summaries, search |
| `persona` | Markdown persona loading, validation, compression, prompt execution |
| `routing` | Space-to-persona resolution, triage helpers, agentic skill discovery |
| `scheduler` | YAML-driven jobs with built-in `noop`, `skill`, and `memory_dream` actions |
| `skill` | Skill discovery, matching, variable substitution, execution |

`exceptions.py` holds `MltgntError`, `ConfigError`, and `DependencyError`; `__main__.py` enables `python -m mltgnt`.

### Layers

Enforced by `.importlinter`. A layer imports only layers below it; packages on the same row are independent of each other.

| Layer (top to bottom) | Packages |
|-----------------------|----------|
| 1 | `daemon`, `media` |
| 2 | `scheduler`, `agent`, `routing` |
| 3 | `persona`, `skill`, `memory`, `conversation` |
| 4 | `bridges` |
| 5 | `interfaces` |

| Contract | Rule |
|----------|------|
| Allowed exceptions | Any module may import `mltgnt.config`; `mltgnt.bridges.*` may import `mltgnt.skill.*`; `mltgnt.skill.matcher` may import `mltgnt.routing.agentic_discover` |
| `l3_no_ghdag` | `persona`, `skill`, `memory`, `conversation` do not import `ghdag` directly |
| `core_no_media` | Every package except `daemon` and `media` is forbidden from importing `mltgnt.media` |
| `media_slack_webchat` | `mltgnt.media.slack` and `mltgnt.media.webchat` do not import each other |

### Turn flow

1. A medium adapter turns an incoming message into a `MediaEvent`.
2. `MediaBridge.handle_event` runs `on_inbound` hooks, admits the event into the conversation queue, builds a `TurnInput` with session history, runs `before_dispatch` hooks, and calls the host's `TurnHandler`.
3. A `reply` result is posted to the thread; a `task` result is stored as a pending record and the message is marked `WORKING`.
4. When ghdag finishes the job, the watchers call `MediaBridge.deliver_result`, which posts the body and runs `on_result` hooks.

## Configuration

mltgnt hardcodes no host paths. Every directory comes from a config object or a function argument.

### Environment variables

| Variable | Read by | Default | Effect |
|----------|---------|---------|--------|
| `MLTGNT_DEFAULT_ENGINE` | `mltgnt.persona.schema.system_default_engine` | `claude` | Engine used when neither the persona nor the caller names one. Read on every call. Must be `claude`, `codex`, `cursor`, or `gemini`; otherwise `ValueError` |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge.enqueue_dag` | enabled | `0` disables the skill I/O type check across DAG steps |
| `THREAD_PERSONA_TTL_DAYS` | `mltgnt.conversation.thread_persona_store` | `30` | TTL of conversation-to-persona bindings; invalid values fall back to `30`. Ignored when a `ConversationConfig` is configured |
| `NIKKI_ROOT` | `mltgnt.skill.runner` | empty | Substituted for `$NIKKI_ROOT` in skill bodies |
| `REPO_ROOT` | `mltgnt.skill.runner` | empty | Substituted for `$REPO_ROOT` in skill bodies |
| `SLACK_BOT_TOKEN` | `mltgnt.media.slack.app.build_app` | - | Slack bot token. The variable name is `SlackMediaConfig.bot_token_env`; unset raises `RuntimeError` |
| `SLACK_APP_TOKEN` | `mltgnt.media.slack.app.start_socket_mode` | - | Slack app-level token. The variable name is `SlackMediaConfig.app_token_env`; unset raises `RuntimeError` |

Read by ghdag, but they change mltgnt behavior:

| Variable | Effect |
|----------|--------|
| `ENABLE_GIT` | `1` / `true` / `yes` enables git commits of memory files. Otherwise the ghdag sink is a no-op |
| `GHDAG_VCS_CONFIG` | Path to the ghdag VCS sink YAML; it must define the `memory` sink for memory commits to happen |

Memory appends, the final compaction write, and `dream.json` / `global.json` writes schedule a commit on the ghdag `memory` sink, debounced per path by `MemoryConfig.commit_debounce_sec` (`<= 0` commits immediately). Commit failures are logged, never raised. `mltgnt.memory.flush_memory_commits()` commits pending paths immediately and also runs at process exit.

### Config dataclasses (`mltgnt.config`)

All are frozen dataclasses.

| Class | Required fields | Optional fields (default) |
|-------|-----------------|---------------------------|
| `PersonaConfig` | - | `weight_map` (`DEFAULT_WEIGHT_MAP`), `section_aliases` (built-in alias map), `exclude_stems` (`frozenset()`) |
| `MemoryConfig` | `chat_dir` | `chat_memory_dir` (`None`), `inject_max_bytes` (`10240`), `inject_max_entries` (`12`), `preferences_max_bytes` (`5120`), `lock_timeout_sec` (`30.0`), `lock_stale_threshold_sec` (`300.0`), `raw_days` (`7`), `mid_weeks` (`3`), `compact_threshold_bytes` (`40960`), `compact_target_bytes` (`25600`), `preferences_section_name`, `protected_layers` (`("caveat",)`), `timezone` (`"Asia/Tokyo"`), `dream_model` (`""`), `dream_engine` (`"claude"`), `use_dream_summary` (`False`), `dream_dir_name` (`"memory"`), `commit_debounce_sec` (`300.0`), `global_dream_exclude_personas` (`()`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir` | `timezone` (`"Asia/Tokyo"`), `salt` (`""`) |
| `ConversationConfig` | `queue_dir`, `sessions_dir`, `ledger_dir`, `thread_index_dir`, `thread_persona_path` | `posts_dir` (`None`), `audit_path` (`None`), `stale_after_sec` (`3600`), `max_queued` (`20`), `cleanup_ttl_days` (`14`), `thread_persona_ttl_days` (`30`) |

`dream_engine` and `dream_model` select the LLM for the `memory_dream` action. An empty model means the engine default; for `claude` that is `claude-haiku-4-5-20251001`. The `memory_dream` action is registered only when `use_dream_summary` is true.

### Media settings

`MediaConfig` (`mltgnt.media._core.config`) is a frozen dataclass that each medium subclasses.

| Field | Default | Meaning |
|-------|---------|---------|
| `state_dir` | required | Medium state directory |
| `pending_dir` | required | Pending records of delegated tasks |
| `events_dir` | required | Job events JSONL read by the progress watcher |
| `language` | `get_language_pack()` at construction time | `LanguagePack` for words, labels, and messages |
| `progress_min_interval_sec` | `5.0` | Minimum interval between progress updates |
| `approval_ttl_sec` | `600.0` | Plan approval deadline |

| Subclass | Extra fields (default) |
|----------|------------------------|
| `SlackMediaConfig` | `bot_token_env` (`"SLACK_BOT_TOKEN"`), `app_token_env` (`"SLACK_APP_TOKEN"`), `status_reactions` (`DEFAULT_STATUS_REACTIONS`), `chunk_max_chars` (`3000`, must be positive) |
| `WebChatMediaConfig` | `store_dir` (required, keyword-only), `host` (`"127.0.0.1"`), `port` (`8765`), `space_id` (`"webchat"`), `assets_dir` (`None`), `avatars` (`{}`), `display_names` (`{}`) |

### LanguagePack

`mltgnt.config.language.LanguagePack` is a frozen dataclass holding locale vocabulary. `EN` is the only pack shipped with mltgnt and the default. Functions that take `pack=None`, and configs that omit `language`, use the current pack from `get_language_pack()`.

To use another locale, define your own `LanguagePack` and call `set_language_pack(pack)` once at startup, before building any `MediaConfig`. Passing anything other than a `LanguagePack` raises `TypeError`.

| Group | Fields | Used by |
|-------|--------|---------|
| Request gates | `work_request_markers`, `create_request_markers`, `deferred_patterns` | `mltgnt.agent.deterministic_gate` |
| Persona text | `compress_prompt_template`, `v21_required_sections`, `v21_example_section`, `meta_header_needles`, `dedupe_opener_re`, `persona_cut_re`, `persona_end_re`, `exclude_stems` | `mltgnt.persona`, persona listing |
| Conversation | `cancel_words`, `composite_header`, `composite_cancel_suffix` | `mltgnt.conversation.thread_queue`, `mltgnt.media._core.cancel` |
| Media | `approval_words`, `status_labels`, `enqueue_failed_text`, `progress_line_pattern` | `mltgnt.media._core` |
| Memory tools | `remember_trigger_words`, `forget_trigger_words` | `mltgnt.memory.tools` |

The first ten fields (request gates, persona text up to `persona_cut_re`, and `exclude_stems`) are required; the rest have English defaults.

### Scheduler jobs

`PersonaScheduler` loads jobs from `yaml_path` or `SchedulerConfig.schedule_yaml`; a YAML error raises `ConfigError`. `ScheduleJob.from_dict` validates each entry and raises `ValueError` on invalid input.

| Field | Default | Notes |
|-------|---------|-------|
| `id`, `mode`, `action` | required | `mode` is `scheduled` (needs `every_day_at`), `interval` (needs `interval_minutes > 0`), `fuzzy_window` (needs `window_start` and `window_end`; overnight windows are rejected), or `chained` (runs once every `depends_on` job is done) |
| `notify` | `silent` | `silent`, `slack_secretary`, or `slack_custom` (needs `slack_channel`) |
| `timezone` | `Asia/Tokyo` | |
| `enabled` | `true` | |
| `action_args` | `{}` | Must be a mapping; see below |
| `every_day_at`, `window_start`, `window_end` | `null` | `HH:MM` |
| `every_week_on` | `null` | `monday` to `sunday` |
| `interval_minutes` | `null` | |
| `fuzzy_method` | `hash` | `hash` or `random` |
| `on_window_missed` | `notify` | `notify`, `silent`, or `mark_done` |
| `slack_channel`, `persona` | `null` | |
| `timeout_seconds` | `600` | Wait timeout; `action_args.timeout_seconds` takes precedence |
| `memory` | `false` | |
| `depends_on` | `[]` | Upstream job ids |
| `on_chain_failure` | `abort_notify` | |
| `on_exit` | `null` | Mapping with required `nonzero`: `fail` or `skip` |
| `chain_every_run` | `false` | Needs `mode: chained` and `depends_on`; fires after every successful upstream run and passes its output on as `upstream_output` |

`action_args` of the built-in `skill` action:

| Key | Default | Effect |
|-----|---------|--------|
| `skill`, `persona` | required | Skill name and persona name |
| `argv` | `[]` | Arguments joined into `$ARGUMENTS`; `upstream_output` is appended |
| `engine`, `model` | persona frontmatter | LLM engine and model |
| `permission` | `null` | Passed to ghdag |
| `task_timeout_sec` | `null` | Positive number of seconds forwarded to ghdag as the task execution limit; invalid values are ignored with a warning |
| `knowledge_count`, `memory_max_bytes`, `memory_exclude_source_tags` | `0`, `0`, `null` | Opt-in context injection from skill knowledge files and persona memory |
| `enable_pipeline` | `false` | Match a skill pipeline, compose it, type-check it, and run it with `enqueue_dag` |
| `enable_fanout` | `false` | Spawn dynamic child tasks from the skill result |
| `enforce_status_markers` | `false` | Fail the job when the result's `PIPELINE_STATUS` markers do not match the skill contract |
| `result_frontmatter` | `false` | Write skill I/O frontmatter at the top of the result |

```yaml
- id: health-check
  mode: interval
  interval_minutes: 30
  action: health_check        # custom action registered via PersonaScheduler(actions=...)
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

| Exception | Module | Base | Raised when |
|-----------|--------|------|-------------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base class; `mltgnt run` maps otherwise unhandled subclasses to exit code `1` |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Invalid `mltgnt run --components`; the scheduler YAML fails to load; a space has more than one `primary` persona in `load_channel_persona_map`. Exit code `2` |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | Another process holds the PID lock (`DaemonRunner.run`); the persona loader passed to `load_channel_persona_map` fails. Exit code `3` |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | `load_persona` finds frontmatter that does not parse or lacks the `persona` key |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | The ghdag tools listing times out, fails, or returns invalid JSON, or a skill names an unknown tool |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | `enqueue_dag` / `compose_pipeline` find a skill I/O type mismatch between steps (disable with `SKILL_IO_TYPECHECK=0`) |
| `LlmCallError` | `mltgnt.memory.compaction` (exported by `mltgnt.memory`) | `RuntimeError` | For hosts to signal a failed injected `llm_call` during compaction; mltgnt itself does not raise it |
| `ReflectionParseError` | `mltgnt.memory.reflection` | `ValueError` | `parse_reflection` receives no JSON object, invalid JSON, or fields of the wrong type |

Built-in exceptions raised by the public API: `ValueError` from `ScheduleJob.from_dict`, `parse_plan`, `system_default_engine`, `SlackMediaConfig` (`chunk_max_chars <= 0`), and `SlackClient` (unknown `extra` keys); `TypeError` from `adapt_client` and `set_language_pack`; `RuntimeError` from `build_app` / `start_socket_mode` when the token variable is unset; `ImportError` from the Slack / WebChat app functions when the extra is not installed.

## Public API Stability

- Versions follow `0.Y.Z`. A minor release (`0.Y.0`) may change the API; a patch release does not. Changes are recorded in `CHANGELOG.md`.
- The supported surface is `mltgnt.__all__`, `mltgnt.interfaces` (including `mltgnt.interfaces.media`), the CLI, and the configuration schema documented here. Other subpackage names, including the `mltgnt.media` modules, may change in any minor release.
- A renamed or removed API normally keeps a deprecated alias for at least one minor release. Exceptions are called out as `BREAKING` in `CHANGELOG.md`.
- Pin an exact tag in production, for example `@v0.123.0`.

## Deprecated API

| Deprecated | Module | Replacement | Warning |
|------------|--------|-------------|---------|
| `SlackClientProtocol` (`post_message`) | `mltgnt.interfaces.slack`, `mltgnt.interfaces` | `MediaClient` | One `DeprecationWarning` when such a client passes through `adapt_client` (including `PersonaScheduler(slack=...)`); it is wrapped so `post` calls `post_message`. Removal is planned for a future minor release |
| `resolve_responding_persona` | `mltgnt.routing` | `resolve_persona` | `DeprecationWarning` |
| `find_observers` | `mltgnt.routing` | `find_observers_in_space` | `DeprecationWarning` |
| `ChannelPersonaEntry` | `mltgnt.routing` | `SpacePersonaEntry` | None (plain alias) |

## License

MIT (SPDX: `MIT`). See [`LICENSE`](LICENSE).
