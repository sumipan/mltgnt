# mltgnt

**Typed building blocks for multi-agent chat hosts: personas, memory, skills, agent loops, routing, scheduling, conversation state, and media I/O.**

mltgnt sits in the middle of a three-layer stack. It owns domain types and behavior; it delegates execution down to ghdag and leaves deployment to the host above it.

| Layer | Owner | Responsibility |
|-------|-------|----------------|
| L0 | [ghdag](https://github.com/sumipan/ghdag) | DAG execution, LLM engines, VCS sinks |
| L1 | **mltgnt** | Personas, memory, skills, agent loop, routing, scheduler, conversation state, media contract |
| L2 | Your host | Processes, credentials, paths, deployment |

All LLM calls and git commits leave mltgnt through `mltgnt.bridges`, which is the only package that imports ghdag. Chat media (Slack, WebChat, or your own) connect through the single `MediaClient` Protocol.

## Status

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.85.0)-orange)

Pre-1.0, current release `0.85.0`. Minor releases may change the API; see [Public API Stability](#public-api-stability).

## Not

| mltgnt is not | Instead |
|---------------|---------|
| An LLM SDK | There is no model client. `mltgnt.bridges.call_llm`, `enqueue_and_wait`, and `enqueue_dag` hand work to ghdag. |
| A DAG engine | Queueing, dependency resolution, and job state belong to ghdag. mltgnt builds steps and waits for their results. |
| A host runtime | The host injects paths, secrets, and process layout. `mltgnt run` only starts the components a host factory returns. |
| A Slack bot | Slack and WebChat are optional adapters behind `MediaClient`; core packages never import `mltgnt.media`. |

## Installation

mltgnt is installed from a git tag:

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.85.0"

# Slack medium (mltgnt.media.slack)
pip install "mltgnt[slack] @ git+https://github.com/sumipan/mltgnt.git@v0.85.0"

# WebChat medium (mltgnt.media.webchat)
pip install "mltgnt[webchat] @ git+https://github.com/sumipan/mltgnt.git@v0.85.0"

# Development tools (tests, lint, type check, import-linter)
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.85.0"
```

| Item | Value |
|------|-------|
| Distribution | `mltgnt` `0.85.0` |
| Python | `>=3.10` |
| Dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.86.0` |
| Extra `slack` | `slack_sdk>=3.0`, `slack_bolt>=1.18` |
| Extra `webchat` | `fastapi>=0.100`, `uvicorn>=0.20` |
| Extra `dev` | `pytest>=7.0`, `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `freezegun>=1.2`, `import-linter>=2.0`, `mypy>=1.10`, `ruff>=0.4` |
| Console script | `mltgnt` = `mltgnt.cli.main:main` (`python -m mltgnt` is equivalent) |
| Type information | Ships `py.typed` |
| Optional, undeclared | `chromadb`: if importable, memory search also queries a Chroma collection; otherwise it uses TF-IDF only |

The Slack and WebChat apps import their third-party packages lazily, so the other media modules import without the extras installed.

## Quick Start

This example needs no LLM and no network. It creates a persona, stores a dream summary, and runs the agent loop with a scripted LLM.

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

# 1. Personas: Markdown files with YAML frontmatter
print(list_personas(agents))                     # ['guide']
persona = load_persona("guide", persona_dir=agents)
print(persona.name, validate_persona(persona))   # guide []

# 2. Dream memory: <chat_dir>/<persona>/memory/dream.json
write_dream(
    root / "chat" / "guide",
    DreamSummary(
        persona="guide",
        sections=[DreamSection(category="style", content="Prefers bullet points.", source_entries=2)],
        updated_at="2026-01-01T00:00:00+00:00",
    ),
)
print(read_dream(root / "chat" / "guide").sections[0].category)  # style

# 3. Agent loop: the LLM and the tools are plain callables
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

To inspect the stored dream summary from the shell:

```bash
mltgnt memory dream show guide --chat-dir /path/to/root/chat
```

A custom medium implements the four `MediaClient` methods:

```python
from mltgnt.interfaces.media import MediaClient, Status


class ConsoleMedia:
    def post(self, text, space, thread=None):
        print(f"[{space}] {text}")
        return "m-1"

    def update(self, message_id, text):
        return True

    def set_status(self, message_id, status):
        return True

    def upload(self, path, space, thread=None):
        return False


media = ConsoleMedia()
assert isinstance(media, MediaClient)            # runtime-checkable Protocol
media.set_status(media.post("hello", "general"), Status.DONE)
```

## CLI Reference

Source: `src/mltgnt/cli/main.py`, `src/mltgnt/cli/memory.py`. Running `mltgnt` without a subcommand prints help and exits `0`.

### `mltgnt run`

Imports `MODULE`, calls `FUNCTION()` to obtain a list of `DaemonComponent`s, and runs them with `DaemonRunner` under a PID lock until SIGINT or SIGTERM.

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--components MODULE:FUNCTION` | yes | — | Component factory, for example `myhost.daemon:components` |
| `--pid-file PATH` | no | `/tmp/mltgnt_daemon.pid` | PID lock file |

| Exit code | Meaning |
|-----------|---------|
| `0` | Normal termination |
| `1` | Any other `MltgntError` |
| `2` | `ConfigError`: `--components` is not `module:function`, the module is not found, or the attribute is missing or not callable |
| `3` | `DependencyError`: another instance holds the PID lock |

### `mltgnt memory dream show`

Prints every section of `<chat-dir>/<persona>/memory/dream.json` as `=== <category> (source_entries: N) ===` followed by its content.

| Argument / option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `persona` | yes | — | Persona name (directory name under `--chat-dir`) |
| `--chat-dir PATH` | yes | — | Parent directory of the persona directories |

Exit code `0`, also when no summary exists (a notice is printed).

### `mltgnt memory dream forget`

Removes one category from the dream summary and rewrites `dream.json`.

| Argument / option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `persona` | yes | — | Persona name (directory name under `--chat-dir`) |
| `--category NAME` | yes | — | Category to remove |
| `--chat-dir PATH` | yes | — | Parent directory of the persona directories |

Exit code `0` on success, `1` when the summary or the category does not exist.

## Public API

### `mltgnt` (top level)

`mltgnt.__all__` has exactly these 23 names. Everything else is imported from its subpackage.

| Name | Kind | Signature / fields | Defined in |
|------|------|--------------------|------------|
| `read_memory_iterative` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call, skill_paths=None, max_iterations=3) -> str` | `mltgnt.memory.search` |
| `read_memory_by_relevance` | function | `(config, persona_stem, query, *, max_bytes, max_entries, layers=None) -> str` | `mltgnt.memory.search` |
| `read_memory_with_sufficiency_check` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call=None) -> str` | `mltgnt.memory.search` |
| `DreamSection` | frozen dataclass | `category`, `content`, `source_entries` | `mltgnt.memory.dream` |
| `DreamSummary` | frozen dataclass | `persona`, `sections`, `updated_at` | `mltgnt.memory.dream` |
| `read_dream` | function | `(persona_dir, *, memory_dir_name="memory") -> DreamSummary \| None` | `mltgnt.memory.dream` |
| `write_dream` | function | `(persona_dir, summary, *, memory_dir_name="memory") -> None` | `mltgnt.memory.dream` |
| `Persona` | dataclass | `name`, `fm`, `sections`, `body`, `path`, `weight_map` | `mltgnt.persona.loader` |
| `load_persona` | function | `(name, *, persona_dir=None, config=None) -> Persona` | `mltgnt.persona` |
| `list_personas` | function | `(persona_dir=None) -> list[str]` | `mltgnt.persona` |
| `validate_persona` | function | `(persona, *, available_skills=None) -> list[str]`; returns warnings, empty means valid | `mltgnt.persona` |
| `run_persona_prompt` | function | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str`; engine is the persona's `engine`, else `system_default_engine()` | `mltgnt.persona` |
| `ChatInput` | dataclass | `source`, `session_key`, `messages`, `persona_name=""`, `model=None`, `context_files=[]`, `context_memory_excerpt=None`, `context_memory_preferences=None` | `mltgnt.interfaces.types` |
| `ChatOutput` | dataclass | `content`, `persona_name`, `timestamp`, `session_key` | `mltgnt.interfaces.types` |
| `Message` | TypedDict | `role`, `content` | `mltgnt.interfaces.types` |
| `PersonaProtocol` | Protocol | Structural type satisfied by `Persona` | `mltgnt.interfaces.persona` |
| `AgentResult` | dataclass | `tool`, `args`, `raw_response`, `tool_trace=None`, `reflexion_count=0`, `plan=None` | `mltgnt.agent._runner` |
| `AgentRunner` | class | See [mltgnt.agent](#mltgntagent) | `mltgnt.agent._runner` |
| `enqueue_dag` | function | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, skills=None, permission=None, order_builder=None) -> list[tuple[bool, str]]` | `mltgnt.bridges.ghdag_bridge` |
| `enqueue_and_wait` | function | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_name=None, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, permission=None, order_builder=None, run_result=None) -> tuple[bool, str]` | `mltgnt.bridges.ghdag_bridge` |
| `PersonaScheduler` | class | `(slack, *, config=None, state_dir=None, yaml_path=None, salt="", jobs=None, notify_channel_resolver=None, default_slack_post_kwargs=None, persona_post_kwargs_resolver=None, repo_root=None, persona_dir=None, append_memory_fn=None, actions=None, memory_config=None)`; `slack` is a `MediaClient` | `mltgnt.scheduler.runner` |
| `ScheduleJob` | dataclass | See [Scheduler jobs](#scheduler-jobs-schedulejob); build with `ScheduleJob.from_dict(raw)` | `mltgnt.scheduler.models` |
| `__version__` | `str` | Installed distribution version, `"0.0.0"` when metadata is unavailable | `mltgnt` |

### `mltgnt.agent`

`AgentRunner(*, llm_call, tool_executor, terminal_tools, max_iterations=3, max_iterations_fn=None, evaluator=None, retry_config=None, logger=None, audit_writer=None, classifier=None, history_mode="last_result", history_max_chars=24000, plan=None, max_reflexions=None, step_hook=None)`.
`runner.run(prompt)` returns an `AgentResult` when a terminal tool is chosen, or `None` when the LLM call or parsing fails.

| Option | Effect |
|--------|--------|
| `history_mode="full_trace"` | Feed the numbered tool trace back to the LLM instead of only the last result; old bodies are truncated past `history_max_chars` |
| `evaluator` | A `ReflexionEvaluator`; `DefaultReflexionEvaluator` retries on `[ERROR]` results, failure markers, or a repeated `(tool, args)` |
| `max_reflexions=N` | Cap Reflexion retries; when exceeded, `AgentResult.tool` is `"__reflexion_exhausted__"` |
| `plan=Plan(...)` | Updated from the `plan_update` key of LLM responses and returned in `AgentResult.plan` |
| `step_hook(entry)` | Called after each tool-trace entry; exceptions are logged and ignored |

| Name | Kind | Summary |
|------|------|---------|
| `AgentRunner`, `AgentResult` | class / dataclass | Tool-calling loop and its result |
| `Plan`, `PlanItem` | dataclass | `Plan(items)` with `apply(updates)` and `progress() -> (done, total)`; `PlanItem(id, title, depends=[], status="pending", note="")` |
| `parse_plan` | function | `(raw) -> Plan`; parses `{"items": [...]}`; raises `ValueError` |
| `build_plan_prompt` | function | `(prompt) -> str`; asks the LLM for JSON accepted by `parse_plan` |
| `DefaultReflexionEvaluator` | class | `(failure_markers=(), repeat_window=3)` |
| `DispatchDecision`, `make_dispatch_decision`, `MODE_REPLY` (`"reply"`), `MODE_DELEGATE` (`"delegate"`) | class / function / constants | Reply-vs-delegate decision |
| `should_force_delegate`, `should_preempt_delegate`, `has_work_request`, `is_create_request`, `match_deferred_promise`, `extract_artifact_references` | functions | Deterministic request gates driven by `LanguagePack` |
| `PreflightContext`, `run_preflight`, `DirectAgentResult`, `SkillWorkerResult`, `MemoryWorkerResult` | class / function | Preflight before dispatch and worker result types |

### `mltgnt.bridges`

The only package that talks to ghdag.

| Name | Summary |
|------|---------|
| `enqueue_dag`, `enqueue_and_wait`, `DagStep` | Build ghdag DAG steps, enqueue them, and wait for results |
| `call_llm` | `(prompt, *, engine="", model="", timeout=120)`; one text call through ghdag |
| `MltgntHooks` | ghdag DAG hooks |
| `create_audit_writer` | Audit writer factory for `AgentRunner(audit_writer=...)` |
| `md_read`, `md_write` | Wrappers of `ghdag.files` |
| `ghdag_bridge`, `llm_adapter`, `hooks_adapter`, `files_adapter` | Submodules; `files_adapter.commit(paths, message, *, sink="memory", trailers=None)` commits through a ghdag VCS sink |

### `mltgnt.persona`

| Name | Summary |
|------|---------|
| `Persona`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt` | Same as the top level |
| `PersonaContext` | Resolved persona context for a turn |
| `PersonaValidationError` | See [Error Reference](#error-reference) |
| `format_persona_body`, `format_result_for_persona` | Tone formatting and persona-voiced result text |
| `compress_heavy_to_light`, `regenerate_light_block` | LLM compression of the heavy persona block into the light block |

`mltgnt.persona.schema.system_default_engine()` returns the host-wide default engine (see `MLTGNT_DEFAULT_ENGINE`).

### `mltgnt.memory`

| Group | Names |
|-------|-------|
| Episodic log | `append_memory_entry`, `read_memory_preferences`, `read_memory_tail_text`, `memory_file_path`, `persona_memory_lock`, `tail_utf8_bytes`, `assemble_entries_text` |
| Search | `read_memory_by_relevance`, `read_memory_with_sufficiency_check`, `read_memory_iterative` |
| Entries | `MemoryEntry`, `parse_jsonl`, `serialize_entry` |
| Compaction | `compact`, `needs_compaction`, `CompactionResult`, `LlmCall`, `LlmCallError` |
| Commits | `flush_memory_commits` |
| Chroma (optional) | `get_collection`, `query_similar`, `upsert_entry`; `get_collection` returns `None` when `chromadb` is unavailable |
| Constants | `MEMORY_CORRUPT_THRESHOLD_BYTES` (`10`), `MEMORY_DEDUPE_SCAN_BYTES` (`32768`), `MEMORY_DEDUPE_SCAN_LINES` (`200`) |

`mltgnt.memory.__all__` also lists underscore-prefixed helpers; they are internal and not part of the supported surface.

Submodules with their own `__all__` (none of them calls an LLM):

| Module | Names | Summary |
|--------|-------|---------|
| `mltgnt.memory.dream` | `DreamSection`, `DreamSummary`, `read_dream`, `write_dream`, `read_global`, `write_global`, `read_global_summary`, `DreamSelector`, `Synthesizer` | Per-persona `dream.json` and cross-persona `global.json` summaries |
| `mltgnt.memory.semantic` | `SemanticStore`, `SemanticEntry`, `KINDS`, `STATUSES`, `validate_entry`, `normalize_content` | `SemanticStore(path, *, persona_stem, commit_debounce_sec=None)` over `semantic.jsonl`; kinds `fact`, `preference`, `commitment`, `caveat`, `self`, `reflection`; statuses `active`, `superseded` |
| `mltgnt.memory.tools` | `MemoryToolExecutor`, `MemoryGate`, `MEMORY_TOOL_SPECS`, `MEMORY_TOOL_NAMES`, `query_terms` | `remember` / `recall` / `forget` tools in the `ToolExecutor` shape; `MemoryGate(max_per_turn=3, max_chars=300, secret_check=None, allowed_subject_prefixes=None)` |
| `mltgnt.memory.reflection` | `build_reflection_prompt`, `parse_reflection`, `apply_reflection`, `ReflectionResult`, `ReflectionAdd`, `ApplyReport`, `ReflectionParseError` | Reflection prompt, parser, and applier for a `SemanticStore` |
| `mltgnt.memory.core_render` | `render_core`, `MANDATORY_KINDS`, `OPTIONAL_KINDS` | `render_core(entries, *, max_bytes=4096, heading="## Memory")`; `caveat` and `commitment` are always included |
| `mltgnt.memory.archive` | `archive_episodes` | `archive_episodes(persona_dir, *, now, keep_days=90) -> int`; moves old episodes to monthly files |

### `mltgnt.skill`

| Name | Summary |
|------|---------|
| `discover`, `discover_bodies`, `load` | Find and parse skill files |
| `match` | `(user_input, skills, persona_skills=None, model=None, *, engine="claude") -> SkillMatchResult` |
| `resolve_skill` | `(user_input, skill_paths, persona_skills=None, entry_file="SKILL.md", matcher_model=None, *, matcher_engine="claude")` |
| `run` | `(skill, persona, arguments, chat_input, extra_context=None) -> SkillRunResult`; substitutes `$ARGUMENTS`, `$PERSONA`, `$SKILL_DIR`, `$NIKKI_ROOT`, `$REPO_ROOT`, `$0`, `$1`, ... |
| `build_extra_context` | Extra prompt context from knowledge files and memory |
| `lint_skill_meta` | Frontmatter validation |
| `SkillMeta`, `SkillFile`, `SkillRegistry`, `SkillRunResult`, `SkillMatchResult`, `ArtifactSpec`, `ProducesSpec`, `ConsumesSpec` | Data types |

With `engine="claude"` and no model, the LLM matcher uses `claude-haiku-4-5-20251001`; other engines use their ghdag default model.

### `mltgnt.routing`

| Name | Summary |
|------|---------|
| `resolve_persona` | Choose the responding persona for a message in a space / conversation |
| `find_observers_in_space` | Personas in a space other than the responder |
| `load_channel_persona_map` | Build the space-to-persona map from persona frontmatter |
| `SpacePersonaEntry`, `RoutingRule`, `evaluate` | Routing entries and rules |
| `detect_nickname` | Nickname mention detection |
| `extract_json_object`, `extract_triage_section`, `prepare_profile_for_triage`, `TRIAGE_PROFILE_MAX_CHARS` (`6000`) | Triage helpers |
| `resolve_responding_persona`, `find_observers`, `ChannelPersonaEntry` | Deprecated; see [Deprecated API](#deprecated-api) |

### `mltgnt.conversation`

| Name | Summary |
|------|---------|
| `configure`, `get_config`, `ConversationConfig` | Set and read the active conversation paths |
| `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage` | Re-exports of `mltgnt.interfaces.turn` |
| `thread_queue`, `session_store`, `session_compact`, `thread_index`, `thread_persona_store`, `fake_media` | Submodules; see [Architecture](#architecture) |

### `mltgnt.scheduler`, `mltgnt.daemon`, `mltgnt.config`

| Package | Names |
|---------|-------|
| `mltgnt.scheduler` | `PersonaScheduler`, `ScheduleJob`, `SchedulePaths`, `load_schedule_jobs(yaml_path, *, default_timezone="Asia/Tokyo")`, `atomic_write_text` |
| `mltgnt.daemon` | `DaemonComponent`, `DaemonRunner(*, pid_file, components, logger=None)`, `PidLock(pid_file)`, `SkillWatcherComponent(registry, interval=5.0)` |
| `mltgnt.config` | `PersonaConfig`, `MemoryConfig`, `SchedulerConfig`, `ConversationConfig`, `DEFAULT_WEIGHT_MAP` |
| `mltgnt.config.language` | `LanguagePack`, `JA` |

### `mltgnt.interfaces`

Dependency-free DTOs and Protocols shared by every layer.

| Module | Names |
|--------|-------|
| `mltgnt.interfaces` | `PersonaProtocol`, `PersonaFMBase`, `Message`, `ChatInput`, `ChatOutput`, `ChatInputBase`, `ChatOutputBase`, `TurnHandler`, `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, `SlackClientProtocol` (deprecated) |
| `mltgnt.interfaces.media` | `Status`, `MediaClient`, `adapt_client` |

| Name | Contract |
|------|----------|
| `Status` | `str` Enum: `RECEIVED` (`"received"`), `WORKING` (`"working"`), `DONE` (`"done"`), `FAILED` (`"failed"`), `CANCELLED` (`"cancelled"`) |
| `MediaClient` | `post(text, space, thread=None) -> str \| None`, `update(message_id, text) -> bool`, `set_status(message_id, status) -> bool`, `upload(path, space, thread=None) -> bool`. Failures return `None` / `False` instead of raising. |
| `adapt_client` | `(obj) -> MediaClient`. Objects with `post` are returned unchanged; objects with only `post_message` are wrapped with one `DeprecationWarning`; anything else raises `TypeError`. |

### `mltgnt.media`

`mltgnt.media` and its subpackages export nothing at package level (`__all__ = []`); import from the modules below.

`mltgnt.media._core` (medium-independent):

| Module | Names | Summary |
|--------|-------|---------|
| `types` | `MediaEvent`, `OutboundMessage` | `MediaEvent(space_id, conversation_id, message_id, author, text, attachments=(), raw=None)`; `OutboundMessage(text, space_id, thread_id=None)` |
| `client` | `MediaClient`, `Status`, `adapt_client` | Re-export of `mltgnt.interfaces.media` |
| `config` | `MediaConfig` | See [Media settings](#media-settings) |
| `bridge` | `MediaBridge` | `MediaBridge(client, handler, config, hooks=None, *, pending_prefix="pending-")`; `handle_event(event)` runs one turn and posts a reply or records a pending task; `deliver_result(uid, body)` posts a task result |
| `hooks` | `HookRegistry`, `OnInbound`, `BeforeDispatch`, `AfterPost`, `OnResult` | Turn hooks, run in registration order; a hook that raises is logged and skipped |
| `component` | `MediaBridgeComponent` | `(start, *, mode, backoff_sec, on_exit=None)`; runs `start` in a process or thread and restarts it with backoff |
| `watchers` | `ExecDoneHandler`, `ProgressWatcher`, `DeliveryReconciler`, `catchup_pending_on_startup`, `iter_pending_with_done`, `CATCHUP_STATES` | Job completion, startup catch-up, delivery reconciliation, progress polling |
| `progress` | `ProgressState`, `finalize_progress`, `status_for_done_marker`, `status_label`, `summarize_tool_use_block`, `summarize_work_loop_step` | Progress message summarization |
| `plan_gate` | `PlanGate`, `PlanState`, `is_approval`, `expire_pending`, `AWAITING_STATE`, `PENDING_KEY` | Plan approval state machine |
| `cancel` | `CancelOutcome`, `handle_cancel`, `is_cancel_request`, `find_pending_uids` | Cancel requests posted in a thread |
| `enqueue_guard` | `enqueue_or_report` | Posts `LanguagePack.enqueue_failed_text` when an enqueue fails |
| `pending` | `PendingStore` | One JSON record per pending request |
| `id_map` | `to_conversation_id`, `resolve`, `storage_key`, `storage_key_from_conversation_id` | Conversation id to `(space, thread)` |
| `thread_reactions` | `admit`, `status_for_admission`, `acknowledge_drained` | Queue admission to message `Status` |
| `sanitizer` | `sanitize_result_body`, `strip_status_lines`, `strip_leading_paragraphs`, `dedupe_trailing_repeated_block`, `extract_final_assistant_text` | Cleans a result before posting |

`mltgnt.media.slack` (extra `slack`):

| Module | Names | Summary |
|--------|-------|---------|
| `config` | `SlackMediaConfig`, `DEFAULT_STATUS_REACTIONS` | See [Media settings](#media-settings) |
| `client` | `SlackClient`, `split_text` | `SlackClient(web_client, config, *, default_channel="")`; statuses become reactions, long text is split at `chunk_max_chars` |
| `app` | `build_app`, `start_socket_mode` | `build_app(config)` creates a Bolt app; `start_socket_mode(app, config)` blocks in Socket Mode |
| `inbound` | `to_media_event`, `strip_mentions`, `thread_ts_from_event`, `files_to_attachments` | Slack event to `MediaEvent` |
| `history` | `fetch_thread_messages`, `extract_text_from_message`, `extract_text_from_blocks`, `table_block_to_markdown` | Thread fetch and text extraction |
| `media_store` | `save_images`, `SavedMedia`, `safe_filename`, `ext_from_mimetype` | Image attachment download |
| `mrkdwn` | `markdown_to_mrkdwn`, `normalize_markdown_residual` | Markdown to Slack mrkdwn |

`mltgnt.media.webchat` (extra `webchat` for the app):

| Module | Names | Summary |
|--------|-------|---------|
| `config` | `WebChatMediaConfig` | See [Media settings](#media-settings) |
| `client` | `WebChatClient` | `WebChatClient(config, *, store=None, author="assistant")` |
| `store` | `WebChatStore`, `KINDS` | Daily JSONL of message snapshots of kind `message`, `update`, `status` |
| `app` | `create_app`, `serve`, `EventHandler`, `parse_event_id` | `create_app(config, bridge, *, store=None, poll_interval_sec=0.5, stream_timeout_sec=None)` serves `GET /`, `GET /messages`, `POST /messages`, `GET /threads/{thread_id}`, `GET /stream` (SSE); `serve(app, config)` runs it on `config.host:config.port` |
| `inbound` | `to_media_event`, `DEFAULT_AUTHOR` | `POST /messages` body to `MediaEvent` |
| `ui` | `INDEX_HTML` | Single-page UI served at `GET /` |

## Protocols / Extension Points

| Extension point | Module | Contract | Consumed by |
|-----------------|--------|----------|-------------|
| `MediaClient` | `mltgnt.interfaces.media` | `post` / `update` / `set_status` / `upload` | `PersonaScheduler`, `MediaBridge`; implemented by `SlackClient`, `WebChatClient` |
| `TurnHandler` | `mltgnt.interfaces.turn` | `handle(turn: TurnInput) -> TurnResult`; `TurnResult.kind` is `"reply"` or `"task"` | `MediaBridge` |
| `HookRegistry` hooks | `mltgnt.media._core.hooks` | `OnInbound(event) -> bool` (return `True` to stop the turn), `BeforeDispatch(turn) -> TurnInput`, `AfterPost(result, message_id) -> None`, `OnResult(uid, body) -> None` | `MediaBridge` |
| `EventHandler` | `mltgnt.media.webchat.app` | `handle_event(event: MediaEvent)` | `create_app`; satisfied by `MediaBridge` |
| `DaemonComponent` | `mltgnt.daemon` | `name` property, non-blocking `start()`, `stop()` | `DaemonRunner`, `mltgnt run` |
| `LLMCaller` | `mltgnt.agent._runner` | `(prompt, *, tool_result=None) -> str \| None` | `AgentRunner(llm_call=...)` |
| `ToolExecutor` | `mltgnt.agent._runner` | `(tool_name, tool_args) -> str`; `MemoryToolExecutor` is a built-in implementation | `AgentRunner(tool_executor=...)` |
| `ReflexionEvaluator` | `mltgnt.agent._runner` | `(prompt, tool_name, tool_args, tool_result, tool_trace) -> ReflexionVerdict` | `AgentRunner(evaluator=...)` |
| `ActionFn` | `mltgnt.scheduler.models` | `(job: ScheduleJob) -> tuple[bool, str]` | `PersonaScheduler(actions={name: fn})`; built-in actions are `noop`, `skill`, and `memory_dream` (only when `memory_config.use_dream_summary` is true) |
| `PersonaProtocol`, `PersonaFMBase`, `ChatInputBase`, `ChatOutputBase` | `mltgnt.interfaces` | Structural types for personas and chat I/O | Skill runner, routing |
| `LanguagePack` | `mltgnt.config.language` | Frozen dataclass of locale vocabulary | See [LanguagePack](#languagepack) |

## Architecture

### Layers (`.importlinter`)

A layer imports only the layers below it. Packages on the same row are independent.

| Layer (top to bottom) | Packages |
|-----------------------|----------|
| 1 | `daemon`, `media` |
| 2 | `scheduler`, `agent`, `routing` |
| 3 | `persona`, `skill`, `memory`, `conversation` |
| 4 | `bridges` |
| 5 | `interfaces` |

| Contract | Rule |
|----------|------|
| Ignored imports | Any module may import `mltgnt.config`; `mltgnt.bridges.*` may import `mltgnt.skill.*`; `mltgnt.skill.matcher` may import `mltgnt.routing.agentic_discover` |
| `l3_no_ghdag` | `persona`, `skill`, `memory`, `conversation` must not import `ghdag` directly |
| `core_no_media` | `scheduler`, `agent`, `routing`, `persona`, `skill`, `memory`, `conversation`, `bridges`, `interfaces` must not import `mltgnt.media` |
| `media_slack_webchat` | `mltgnt.media.slack` and `mltgnt.media.webchat` must not import each other |

### Modules

Every module under `src/mltgnt/` (package `__init__.py` files re-export the public names listed in [Public API](#public-api)):

| Module | Role |
|--------|------|
| `__main__.py` | `python -m mltgnt` entry point |
| `exceptions.py` | `MltgntError`, `ConfigError`, `DependencyError` |
| `agent/_parse.py` | Extracts the JSON object from an LLM response |
| `agent/_runner.py` | `AgentRunner`, `AgentResult`, and the loop Protocols |
| `agent/action_classifier.py` | Side-effect level classification of tools |
| `agent/deterministic_gate.py` | Work / create request and deferred-promise gates |
| `agent/dispatch_decision.py` | `make_dispatch_decision` (reply or delegate) |
| `agent/dispatch_preflight.py` | `run_preflight` and worker result types |
| `agent/plan.py` | `Plan`, `PlanItem`, `parse_plan`, `build_plan_prompt` |
| `agent/reflexion.py` | `DefaultReflexionEvaluator` |
| `bridges/audit_adapter.py` | Orchestration audit events |
| `bridges/files_adapter.py` | `md_read`, `md_write`, `commit` through ghdag |
| `bridges/ghdag_bridge.py` | `enqueue_dag`, `enqueue_and_wait`, `DagStep`, skill I/O type check |
| `bridges/hooks_adapter.py` | `MltgntHooks` and `create_audit_writer` |
| `bridges/llm_adapter.py` | `call_llm` |
| `cli/main.py` | argparse entry point and exit-code mapping |
| `cli/memory.py` | `mltgnt memory dream show` / `forget` |
| `cli/run.py` | `mltgnt run` component factory loading |
| `config/language.py` | `LanguagePack`, `JA` |
| `conversation/fake_media.py` | Builds `TurnInput` without a real medium |
| `conversation/session_compact.py` | Automatic compaction of conversation logs |
| `conversation/session_store.py` | Session ledger and turn log |
| `conversation/thread_index.py` | Thread index |
| `conversation/thread_persona_store.py` | Conversation-to-persona bindings with a TTL |
| `conversation/thread_queue.py` | Per-conversation wait queue and cancel detection |
| `conversation/types.py` | Re-export of `mltgnt.interfaces.turn` |
| `daemon/_pidlock.py` | `PidLock` |
| `daemon/_runner.py` | `DaemonRunner` with signal handling |
| `daemon/_skill_watcher.py` | `SkillWatcherComponent` (reloads a `SkillRegistry`) |
| `interfaces/media.py` | `Status`, `MediaClient`, `adapt_client` |
| `interfaces/persona.py` | `PersonaProtocol` |
| `interfaces/slack.py` | Deprecated `SlackClientProtocol` |
| `interfaces/turn.py` | `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, `TurnHandler` |
| `interfaces/types.py` | `ChatInput`, `ChatOutput`, `Message`, and base Protocols |
| `media/_core/bridge.py` | `MediaBridge` |
| `media/_core/cancel.py` | Thread cancel handling |
| `media/_core/client.py` | Re-export of the media contract |
| `media/_core/component.py` | `MediaBridgeComponent` supervisor |
| `media/_core/config.py` | `MediaConfig` |
| `media/_core/enqueue_guard.py` | `enqueue_or_report` |
| `media/_core/hooks.py` | `HookRegistry` and hook types |
| `media/_core/id_map.py` | Conversation id mapping |
| `media/_core/pending.py` | `PendingStore` |
| `media/_core/plan_gate.py` | Plan approval state machine |
| `media/_core/progress.py` | Progress summarization and status labels |
| `media/_core/sanitizer.py` | Result body cleanup |
| `media/_core/thread_reactions.py` | Queue admission to `Status` |
| `media/_core/types.py` | `MediaEvent`, `OutboundMessage` |
| `media/_core/watchers.py` | Completion, catch-up, reconciliation, progress watchers |
| `media/slack/app.py` | Bolt app and Socket Mode |
| `media/slack/client.py` | `SlackClient`, `split_text` |
| `media/slack/config.py` | `SlackMediaConfig` |
| `media/slack/history.py` | Thread fetch and text extraction |
| `media/slack/inbound.py` | Slack event to `MediaEvent` |
| `media/slack/media_store.py` | Image download |
| `media/slack/mrkdwn.py` | Markdown to mrkdwn |
| `media/webchat/app.py` | FastAPI app with SSE stream |
| `media/webchat/client.py` | `WebChatClient` |
| `media/webchat/config.py` | `WebChatMediaConfig` |
| `media/webchat/inbound.py` | Request body to `MediaEvent` |
| `media/webchat/store.py` | `WebChatStore` |
| `media/webchat/ui.py` | `INDEX_HTML` |
| `memory/_chroma.py` | Optional Chroma vector search |
| `memory/_commit.py` | Debounced memory commits, `flush_memory_commits` |
| `memory/_format.py` | `MemoryEntry`, `parse_jsonl`, `serialize_entry` |
| `memory/_iterative.py` | Iterative retrieval loop |
| `memory/_scoring.py` | Cosine-similarity scoring |
| `memory/_sufficiency.py` | LLM sufficiency judgment |
| `memory/_tfidf.py` | TF-IDF vectorization |
| `memory/api.py` | Memory paths, lock, append, and read |
| `memory/archive.py` | `archive_episodes` |
| `memory/compaction.py` | `compact`, `needs_compaction`, `LlmCallError` |
| `memory/core_render.py` | `render_core` |
| `memory/dream/_format.py` | `DreamSection`, `DreamSummary` and JSON conversion |
| `memory/dream/api.py` | `dream.json` / `global.json` read and write |
| `memory/dream/selector.py` | `DreamSelector` |
| `memory/dream/synthesizer.py` | `Synthesizer` |
| `memory/reflection.py` | Reflection prompt, parser, applier |
| `memory/search.py` | Relevance, sufficiency, and iterative search |
| `memory/semantic.py` | `SemanticStore` |
| `memory/tools.py` | `MemoryToolExecutor`, `MemoryGate` |
| `persona/compress.py` | Heavy-to-light compression |
| `persona/extractor.py` | H2 section parsing and light / heavy extraction |
| `persona/formatter.py` | `format_persona_body` |
| `persona/frontmatter.py` | YAML frontmatter parsing |
| `persona/loader.py` | `Persona` and file loading |
| `persona/memory.py` | Persona memory helpers (dedupe keys, tool-trace blocks) |
| `persona/phrases.py` | Persona phrase loading |
| `persona/registry.py` | Persona listing and alias resolution |
| `persona/resolve.py` | Responder selection and `PersonaContext` building |
| `persona/result_format.py` | `format_result_for_persona` |
| `persona/runner.py` | `run_persona_prompt` |
| `persona/schema.py` | Frontmatter schema, validation, `system_default_engine` |
| `persona/types.py` | `PersonaContext` |
| `routing/agentic_discover.py` | Agentic skill discovery |
| `routing/channel_router.py` | Space-to-persona routing and observers |
| `routing/triage.py` | Triage preprocessing |
| `scheduler/actions/dream.py` | Built-in `memory_dream` action |
| `scheduler/actions/skill.py` | Built-in `skill` action |
| `scheduler/base_runner.py` | `BaseRunner` tick loop |
| `scheduler/fanout.py` | Dynamic child tasks for skill jobs (`action_args.enable_fanout`) |
| `scheduler/loader.py` | `load_schedule_jobs` |
| `scheduler/models.py` | `ScheduleJob`, `OnExitPolicy`, `ActionFn` |
| `scheduler/runner.py` | `PersonaScheduler` |
| `scheduler/state.py` | `SchedulePaths`, `atomic_write_text` |
| `skill/_registry.py` | `SkillRegistry` |
| `skill/context.py` | `build_extra_context` |
| `skill/lint.py` | `lint_skill_meta` |
| `skill/loader.py` | Skill discovery and frontmatter parsing |
| `skill/matcher.py` | Slash, literal, trigger, and LLM matching |
| `skill/models.py` | `SkillMeta`, `SkillFile`, `SkillLoadError` |
| `skill/runner.py` | Variable substitution and prompt composition |

## Configuration

mltgnt hardcodes no host paths: every directory comes from a config object or an argument.

### Environment variables

| Variable | Read by | Default | Effect |
|----------|---------|---------|--------|
| `MLTGNT_DEFAULT_ENGINE` | `mltgnt.persona.schema.system_default_engine` | `claude` | Engine used when neither the persona nor the caller names one. Read on every call. Must be `claude`, `codex`, `cursor`, or `gemini`, else `ValueError`. |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge.enqueue_dag` | on | `0` disables the compose-time skill I/O type check |
| `THREAD_PERSONA_TTL_DAYS` | `mltgnt.conversation.thread_persona_store` | `30` | TTL of conversation-to-persona bindings; invalid values fall back to `30`. Ignored when a `ConversationConfig` is configured. |
| `NIKKI_ROOT` | `mltgnt.skill.runner` | empty | Value substituted for `$NIKKI_ROOT` in skill bodies |
| `REPO_ROOT` | `mltgnt.skill.runner` | empty | Value substituted for `$REPO_ROOT` in skill bodies |
| `SLACK_BOT_TOKEN` | `mltgnt.media.slack.app.build_app` | — | Bot token; name set by `SlackMediaConfig.bot_token_env`. Unset raises `RuntimeError`. |
| `SLACK_APP_TOKEN` | `mltgnt.media.slack.app.start_socket_mode` | — | App-level token; name set by `SlackMediaConfig.app_token_env`. Unset raises `RuntimeError`. |

Read by ghdag, not by mltgnt, but they change mltgnt behavior:

| Variable | Effect |
|----------|--------|
| `ENABLE_GIT` | `1` / `true` / `yes` enables git commits of memory files. Otherwise the ghdag sink is a `NullSink` and commits do nothing. |
| `GHDAG_VCS_CONFIG` | Path to the ghdag VCS sink YAML; the `memory` sink must be defined there for memory commits to happen. |

Memory appends, compaction, and `dream.json` / `global.json` writes schedule a commit on the ghdag `memory` sink, debounced per path by `MemoryConfig.commit_debounce_sec` (`<= 0` commits immediately). Commit failures are logged, not raised. `mltgnt.memory.flush_memory_commits()` commits pending paths now and also runs at process exit.

### Config dataclasses (`mltgnt.config`)

All are frozen dataclasses.

| Class | Required fields | Optional fields (default) |
|-------|-----------------|---------------------------|
| `PersonaConfig` | — | `weight_map` (`DEFAULT_WEIGHT_MAP`), `section_aliases` (built-in alias map), `exclude_stems` (`frozenset()`) |
| `MemoryConfig` | `chat_dir` | `chat_memory_dir` (`None`), `inject_max_bytes` (`10240`), `inject_max_entries` (`12`), `preferences_max_bytes` (`5120`), `lock_timeout_sec` (`30.0`), `lock_stale_threshold_sec` (`300.0`), `raw_days` (`7`), `mid_weeks` (`3`), `compact_threshold_bytes` (`40960`), `compact_target_bytes` (`25600`), `preferences_section_name` (`"User’s preferences and tendencies"`), `protected_layers` (`("caveat",)`), `timezone` (`"Asia/Tokyo"`), `dream_model` (`""`), `dream_engine` (`"claude"`), `use_dream_summary` (`False`), `dream_dir_name` (`"memory"`), `commit_debounce_sec` (`300.0`), `global_dream_exclude_personas` (`()`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir` | `timezone` (`"Asia/Tokyo"`), `salt` (`""`) |
| `ConversationConfig` | `queue_dir`, `sessions_dir`, `ledger_dir`, `thread_index_dir`, `thread_persona_path` | `posts_dir` (`None`), `audit_path` (`None`), `stale_after_sec` (`3600`), `max_queued` (`20`), `cleanup_ttl_days` (`14`), `thread_persona_ttl_days` (`30`) |

`dream_engine` / `dream_model` select the LLM of the `memory_dream` action. A blank model means the engine default; for `claude` that is `claude-haiku-4-5-20251001`.

### Media settings

`MediaConfig` (`mltgnt.media._core.config`) is a frozen dataclass that each medium subclasses.

| Field | Default | Meaning |
|-------|---------|---------|
| `state_dir` | required | State directory of the medium |
| `pending_dir` | required | Pending records of delegated tasks |
| `events_dir` | required | Job events JSONL read by the progress watcher |
| `language` | `JA` | `LanguagePack` used for words, labels, and messages |
| `progress_min_interval_sec` | `5.0` | Minimum interval between progress updates |
| `approval_ttl_sec` | `600.0` | Plan approval deadline |

| Subclass | Extra fields (default) |
|----------|------------------------|
| `SlackMediaConfig` | `bot_token_env` (`"SLACK_BOT_TOKEN"`), `app_token_env` (`"SLACK_APP_TOKEN"`), `status_reactions` (`DEFAULT_STATUS_REACTIONS`), `chunk_max_chars` (`3000`, must be positive) |
| `WebChatMediaConfig` | `store_dir` (required, keyword-only), `host` (`"127.0.0.1"`), `port` (`8765`), `space_id` (`"webchat"`) |

### LanguagePack

`mltgnt.config.language.LanguagePack` is a frozen dataclass of locale vocabulary. `JA` is the built-in instance and the default wherever a function takes `pack=None` or a config takes `language`.

| Group | Fields | Used by |
|-------|--------|---------|
| Request gates | `work_request_markers`, `create_request_markers`, `deferred_patterns` | `mltgnt.agent.deterministic_gate` |
| Persona text | `compress_prompt_template`, `v21_required_sections`, `v21_example_section`, `meta_header_needles`, `dedupe_opener_re`, `persona_cut_re`, `persona_end_re`, `exclude_stems` | `mltgnt.persona` |
| Conversation | `cancel_words`, `composite_header`, `composite_cancel_suffix` | `mltgnt.conversation.thread_queue`, `mltgnt.media._core.cancel` |
| Media | `approval_words`, `status_labels`, `enqueue_failed_text`, `progress_line_pattern` | `mltgnt.media._core` |
| Memory tools | `remember_trigger_words`, `forget_trigger_words` | `mltgnt.memory.tools` |

The first ten fields are required; the rest have defaults (for example `approval_words` is `{"OK", "ok", "yes", "approve"}` and `enqueue_failed_text` is `"Failed to enqueue the request. Please try again later."`).

### Scheduler jobs (`ScheduleJob`)

`PersonaScheduler` loads jobs from `yaml_path` or `SchedulerConfig.schedule_yaml` (a YAML error raises `ConfigError`). `ScheduleJob.from_dict` validates each entry and raises `ValueError` on invalid input.

| Field | Default | Notes |
|-------|---------|-------|
| `id`, `mode`, `action` | required | `mode` is `scheduled` (needs `every_day_at`), `interval` (needs `interval_minutes > 0`), `fuzzy_window` (needs `window_start` and `window_end`; overnight windows are rejected), or `chained` (runs when every `depends_on` job is done) |
| `notify` | `silent` | `silent`, `slack_secretary`, or `slack_custom` (needs `slack_channel`) |
| `timezone` | `Asia/Tokyo` | |
| `enabled` | `true` | |
| `action_args` | `{}` | Must be a mapping |
| `every_day_at`, `window_start`, `window_end` | `null` | `HH:MM` |
| `every_week_on` | `null` | `monday` to `sunday` |
| `interval_minutes` | `null` | |
| `fuzzy_method` | `hash` | `hash` or `random` |
| `on_window_missed` | `notify` | `notify`, `silent`, or `mark_done` |
| `slack_channel`, `persona` | `null` | |
| `timeout_seconds` | `600` | `action_args.timeout_seconds` takes precedence |
| `memory` | `false` | |
| `depends_on` | `[]` | Upstream job ids |
| `on_chain_failure` | `abort_notify` | |
| `on_exit` | `null` | Mapping with required `nonzero`: `fail` or `skip` |
| `chain_every_run` | `false` | Requires `mode: chained` and `depends_on`; fires after every successful upstream run and passes its output as `upstream_output` |

```yaml
- id: health-check
  mode: interval
  interval_minutes: 30
  action: health_check        # custom action registered via PersonaScheduler(actions=...)
- id: health-followup
  mode: chained
  depends_on: [health-check]
  chain_every_run: true
  action: skill               # built-in; needs action_args.skill and action_args.persona
  action_args:
    skill: triage-health
    persona: guide
```

## Error Reference

| Exception | Module | Base | Raised when |
|-----------|--------|------|-------------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base class; `mltgnt run` maps uncaught subclasses to exit code `1` |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Invalid `mltgnt run --components`; scheduler YAML fails to load; a space has more than one `primary` persona in `load_channel_persona_map`. Exit code `2`. |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | The PID lock is held by another instance (`DaemonRunner.run`); the persona loader passed to `load_channel_persona_map` fails. Exit code `3`. |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | `load_persona` finds frontmatter that does not parse or lacks the `persona` key |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | The ghdag tools list times out, fails, or returns invalid JSON, or a skill names an unknown tool |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | `enqueue_dag` finds a skill I/O type mismatch between steps (disable with `SKILL_IO_TYPECHECK=0`) |
| `LlmCallError` | `mltgnt.memory.compaction` (exported by `mltgnt.memory`) | `RuntimeError` | Provided for hosts to signal a failed injected `llm_call` during compaction; mltgnt itself does not raise it |
| `ReflectionParseError` | `mltgnt.memory.reflection` | `ValueError` | `parse_reflection` gets no JSON object, invalid JSON, or fields of the wrong type |

Built-in exceptions from the public API: `ValueError` from `ScheduleJob.from_dict`, `parse_plan`, `system_default_engine`, and `SlackMediaConfig` (`chunk_max_chars <= 0`); `TypeError` from `adapt_client`; `RuntimeError` from `build_app` / `start_socket_mode` when the token variable is unset; `ImportError` from the Slack / WebChat app functions when the extra is missing.

## Public API Stability

- Versions follow `0.Y.Z`. A minor release (`0.Y.0`) may change the API; a patch release does not. Every change is recorded in `CHANGELOG.md`.
- The supported surface is `mltgnt.__all__`, `mltgnt.interfaces` (including `mltgnt.interfaces.media`), the CLI, and the configuration schema documented here. Other subpackage names, including the `mltgnt.media` modules, may change in any minor release.
- A renamed or removed API keeps a deprecated alias for at least one minor release.
- Pin an exact tag in production, for example `@v0.85.0`.

## Deprecated API

| Deprecated | Module | Replacement | Warning |
|------------|--------|-------------|---------|
| `SlackClientProtocol` (`post_message`) | `mltgnt.interfaces.slack`, `mltgnt.interfaces` | `MediaClient` | One `DeprecationWarning` when such a client goes through `adapt_client` (including `PersonaScheduler(slack=...)`); it is wrapped so `post` calls `post_message` |
| `resolve_responding_persona` | `mltgnt.routing` | `resolve_persona` | `DeprecationWarning` |
| `find_observers` | `mltgnt.routing` | `find_observers_in_space` | `DeprecationWarning` |
| `ChannelPersonaEntry` | `mltgnt.routing` | `SpacePersonaEntry` | None (plain alias) |

## License

MIT (SPDX: `MIT`). See `LICENSE`.
