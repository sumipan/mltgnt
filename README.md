# mltgnt

**Typed contracts for multi-agent hosts: personas, memory, skills, agent loops, scheduling, conversation, and media I/O.**
mltgnt sits between **[ghdag](https://github.com/sumipan/ghdag)** (L0: DAG execution and LLM engines) and **your host** (L2: processes, credentials, deployment). It defines the domain types and behavior that a host wires together; every LLM call is delegated to ghdag through the `mltgnt.bridges` package, and chat media (Slack, WebChat) plug in through one `MediaClient` contract.

## Status

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.82.0)-orange)

Pre-1.0: minor releases may break the API (see [Public API Stability](#public-api-stability)).

## Not (what this is not)

| mltgnt is not | Instead |
|---------------|---------|
| An LLM SDK | No model client lives here. LLM calls go through ghdag (`mltgnt.bridges.call_llm`, `enqueue_and_wait`, `enqueue_dag`). |
| A DAG engine | Queueing, dependency resolution, and DAG state belong to ghdag. mltgnt builds steps and waits for their results. |
| A host runtime | Process layout, secrets, and deployment belong to the host. `mltgnt run` only starts the components a host factory returns, and the media layer reads tokens only from environment variables the host names. |

## Installation

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.82.0"
```

Optional extras:

```bash
# Slack medium (mltgnt.media.slack): slack_sdk, slack_bolt
pip install "mltgnt[slack] @ git+https://github.com/sumipan/mltgnt.git@v0.82.0"

# WebChat medium (mltgnt.media.webchat): fastapi, uvicorn
pip install "mltgnt[webchat] @ git+https://github.com/sumipan/mltgnt.git@v0.82.0"

# Development tools
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.82.0"
```

| Item | Value |
|------|-------|
| Distribution | `mltgnt` `0.82.0` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.85.0` |
| `slack` extra | `slack_sdk>=3.0`, `slack_bolt>=1.18` |
| `webchat` extra | `fastapi>=0.100`, `uvicorn>=0.20` |
| `dev` extra | `pytest>=7.0`, `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `freezegun>=1.2`, `import-linter>=2.0`, `mypy>=1.10`, `ruff>=0.4` |
| Console script | `mltgnt` → `mltgnt.cli.main:main` (`python -m mltgnt` is equivalent) |
| Typed | Yes (`py.typed`) |

The Slack and WebChat modules import their third-party packages lazily; without the extra, `build_app` / `start_socket_mode` / `create_app` / `serve` raise `ImportError` with an install hint.

## Quick Start

Both examples run offline — no LLM, no network.

**Personas, dream memory, and one agent loop** (names from `mltgnt.__all__` only):

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
    json.dumps({"thought": "done", "tool": "answer", "args": {"text": "UTC"}}),
])
runner = AgentRunner(
    llm_call=lambda prompt, *, tool_result=None: next(replies),
    tool_executor=lambda name, args: "UTC",
    terminal_tools=frozenset({"answer"}),
)
result = runner.run("Which timezone?")
print(result.tool, result.args)                 # answer {'text': 'UTC'}
```

`write_dream` schedules a debounced git commit of `dream.json`; it does nothing unless `ENABLE_GIT` is set (see [Environment variables](#environment-variables)). The stored summary can be printed with `mltgnt memory dream show guide --chat-dir <root>/chat`.

**The media client contract** (`mltgnt.interfaces.media`):

```python
from mltgnt.interfaces.media import MediaClient, Status, adapt_client


class InMemoryMedia:
    def __init__(self) -> None:
        self.posts = []

    def post(self, text, space, thread=None):
        self.posts.append((text, space, thread))
        return "M1"

    def update(self, message_id, text):
        return True

    def set_status(self, message_id, status):
        return True

    def upload(self, path, space, thread=None):
        return False


client = InMemoryMedia()
print(isinstance(client, MediaClient))          # True (runtime-checkable Protocol)
media = adapt_client(client)                    # returned unchanged, no warning
message_id = media.post("hello", "space-1")
print(message_id, media.set_status(message_id, Status.DONE))  # M1 True
```

## CLI Reference

Defined with argparse in `src/mltgnt/cli/main.py` and `src/mltgnt/cli/memory.py`. `python -m mltgnt` runs the same entry point. With no subcommand, help is printed and the exit code is `0`.

| Command | Argument / option | Required | Default | Description |
|---------|-------------------|----------|---------|-------------|
| `mltgnt run` | `--components MODULE:FUNCTION` | yes | — | Imports `MODULE`, calls `FUNCTION()` to get a list of `DaemonComponent`s, and runs them with `DaemonRunner` under a PID lock until SIGINT / SIGTERM. |
| | `--pid-file PATH` | no | `/tmp/mltgnt_daemon.pid` | PID lock file. |
| `mltgnt memory dream show` | `persona` (positional) | yes | — | Persona name / stem. Prints every section of `<chat-dir>/<persona>/memory/dream.json`; if none exists, prints a notice and exits `0`. |
| | `--chat-dir PATH` | yes | — | Parent directory of the persona directories. |
| `mltgnt memory dream forget` | `persona` (positional) | yes | — | Persona name / stem. Removes one category from the dream summary and rewrites the file; exits `1` if the summary or the category does not exist. |
| | `--category NAME` | yes | — | Category to remove. |
| | `--chat-dir PATH` | yes | — | Parent directory of the persona directories. |

Exit codes of `mltgnt run`:

| Code | Meaning |
|------|---------|
| `0` | Normal termination |
| `1` | Any other `MltgntError` |
| `2` | `ConfigError`: `--components` is not `module:function`, the module is not found, or the function is missing or not callable |
| `3` | `DependencyError`: another instance already holds the PID lock |

## Public API

### Top level (`mltgnt.__all__`)

`src/mltgnt/__init__.py` exports exactly these 23 names. Everything else is imported from its subpackage.

| Group | Name | Kind | Signature / fields | Defined in |
|-------|------|------|--------------------|------------|
| memory | `read_memory_iterative` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call, skill_paths=None, max_iterations=3) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_by_relevance` | function | `(config, persona_stem, query, *, max_bytes, max_entries, layers=None) -> str` | `mltgnt.memory.search` |
| memory | `read_memory_with_sufficiency_check` | function | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call=None) -> str` | `mltgnt.memory.search` |
| memory | `DreamSection` | frozen dataclass | `category`, `content`, `source_entries` | `mltgnt.memory.dream` |
| memory | `DreamSummary` | frozen dataclass | `persona`, `sections: list[DreamSection]`, `updated_at` | `mltgnt.memory.dream` |
| memory | `read_dream` | function | `(persona_dir, *, memory_dir_name="memory") -> DreamSummary \| None` | `mltgnt.memory.dream` |
| memory | `write_dream` | function | `(persona_dir, summary, *, memory_dir_name="memory") -> None` | `mltgnt.memory.dream` |
| persona | `Persona` | dataclass | `name`, `fm`, `sections`, `body`, `path`, `weight_map` | `mltgnt.persona` |
| persona | `load_persona` | function | `(name, *, persona_dir=None, config=None) -> Persona` | `mltgnt.persona` |
| persona | `list_personas` | function | `(persona_dir=None) -> list[str]` | `mltgnt.persona` |
| persona | `validate_persona` | function | `(persona, *, available_skills=None) -> list[str]` (warnings; empty means valid) | `mltgnt.persona` |
| persona | `run_persona_prompt` | function | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str`; the engine is the persona's `engine`, else `system_default_engine()` | `mltgnt.persona` |
| interfaces | `ChatInput` | dataclass | `source`, `session_key`, `messages`, `persona_name=""`, `model=None`, `context_files=[]`, `context_memory_excerpt=None`, `context_memory_preferences=None` | `mltgnt.interfaces.types` |
| interfaces | `ChatOutput` | dataclass | `content`, `persona_name`, `timestamp`, `session_key` | `mltgnt.interfaces.types` |
| interfaces | `Message` | TypedDict | `role`, `content` | `mltgnt.interfaces.types` |
| interfaces | `PersonaProtocol` | Protocol | Structural type satisfied by `Persona` | `mltgnt.interfaces.persona` |
| agent | `AgentResult` | dataclass | `tool`, `args`, `raw_response`, `tool_trace=None`, `reflexion_count=0`, `plan=None` | `mltgnt.agent` |
| agent | `AgentRunner` | class | See [Agent loop](#agent-loop-mltgntagent) | `mltgnt.agent` |
| bridges | `enqueue_dag` | function | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, skills=None, permission=None, order_builder=None) -> list[tuple[bool, str]]` | `mltgnt.bridges.ghdag_bridge` |
| bridges | `enqueue_and_wait` | function | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, persona_name=None, persona_dir=None, correlation_id=None, parent_correlation_id=None, request_id=None, permission=None, order_builder=None, run_result=None) -> tuple[bool, str]` | `mltgnt.bridges.ghdag_bridge` |
| scheduler | `PersonaScheduler` | class | `(slack, *, config=None, state_dir=None, yaml_path=None, salt="", jobs=None, notify_channel_resolver=None, default_slack_post_kwargs=None, persona_post_kwargs_resolver=None, repo_root=None, persona_dir=None, append_memory_fn=None, actions=None, memory_config=None)`; `slack` is a `MediaClient` (or a deprecated `SlackClientProtocol` client, wrapped by `adapt_client`) | `mltgnt.scheduler` |
| scheduler | `ScheduleJob` | dataclass | See [Scheduler jobs](#scheduler-jobs-schedulejob); build with `ScheduleJob.from_dict(raw)` | `mltgnt.scheduler` |
| version | `__version__` | `str` | Installed distribution version; `"0.0.0"` if metadata is unavailable | `mltgnt` |

### Agent loop (`mltgnt.agent`)

`AgentRunner(*, llm_call, tool_executor, terminal_tools, max_iterations=3, max_iterations_fn=None, evaluator=None, retry_config=None, logger=None, audit_writer=None, classifier=None, history_mode="last_result", history_max_chars=24000, plan=None, max_reflexions=None, step_hook=None)`; `runner.run(prompt) -> AgentResult | None` (`None` when the LLM call or response parsing fails).

By default only the last tool result is fed back to the LLM. Opt-in options:

| Option | Effect |
|--------|--------|
| `history_mode="full_trace"` | Sends the whole numbered tool trace (older result bodies truncated past `history_max_chars`) instead of the last result only. |
| `plan=Plan(...)` | Updated in place from the `plan_update` key of LLM responses and returned as `AgentResult.plan`. Create it with `build_plan_prompt` / `parse_plan`. |
| `max_reflexions=N` | Caps Reflexion retries; when exceeded, `AgentResult.tool == "__reflexion_exhausted__"`. |
| `step_hook(entry)` | Called after each tool-trace entry is appended; exceptions are logged and ignored. |

Other names in `mltgnt.agent.__all__`:

| Name | Kind | Summary |
|------|------|---------|
| `Plan` / `PlanItem` | dataclass | `Plan(items)` with `apply(updates)` and `progress() -> (done, total)`; `PlanItem(id, title, depends, status="pending", note="")` |
| `parse_plan` | function | `(raw) -> Plan`; parses `{"items": [{"id", "title", "depends"?}]}`; raises `ValueError` |
| `build_plan_prompt` | function | `(prompt) -> str`; asks the LLM for the JSON that `parse_plan` accepts |
| `DefaultReflexionEvaluator` | class | `(failure_markers=(), repeat_window=3)`; retries on `[ERROR]` results, failure markers, or a repeated `(tool, args)` |
| `DispatchDecision`, `make_dispatch_decision`, `MODE_REPLY`, `MODE_DELEGATE`, `should_force_delegate`, `should_preempt_delegate`, `has_work_request`, `is_create_request`, `match_deferred_promise`, `extract_artifact_references` | class / functions / constants | Reply-vs-delegate decision and deterministic request gates |
| `PreflightContext`, `run_preflight`, `DirectAgentResult`, `SkillWorkerResult`, `MemoryWorkerResult` | class / function | Dispatch preflight and worker result types |

### Media contract (`mltgnt.interfaces.media`)

`__all__ = ["Status", "MediaClient", "adapt_client"]`. It lives in the lowest layer, so core packages can use it without importing `mltgnt.media`.

| Name | Kind | Contract |
|------|------|----------|
| `Status` | `str` Enum | `RECEIVED` (`"received"`), `WORKING` (`"working"`), `DONE` (`"done"`), `FAILED` (`"failed"`), `CANCELLED` (`"cancelled"`) |
| `MediaClient` | runtime-checkable Protocol | `post(text, space, thread=None) -> str \| None` (message id, `None` on failure); `update(message_id, text) -> bool`; `set_status(message_id, status) -> bool`; `upload(path, space, thread=None) -> bool` (default `False`). Failures return `None` / `False` and never raise. |
| `adapt_client` | function | `(obj) -> MediaClient`. An object with `post` is returned unchanged. An object with only `post_message` (deprecated `SlackClientProtocol`) is wrapped with one `DeprecationWarning`. Anything else raises `TypeError`. |

### Media layer (`mltgnt.media`)

`mltgnt.media`, `mltgnt.media._core`, `mltgnt.media.slack`, and `mltgnt.media.webchat` all have `__all__ = []`: import names from the submodules listed below.

`mltgnt.media._core` (media-independent):

| Module | Main names | Summary |
|--------|------------|---------|
| `types` | `MediaEvent`, `OutboundMessage` | Frozen dataclasses. `MediaEvent(space_id, conversation_id, message_id, author, text, attachments=(), raw=None)`; `OutboundMessage(text, space_id, thread_id)` |
| `client` | `MediaClient`, `Status`, `adapt_client` | Re-export of `mltgnt.interfaces.media` |
| `config` | `MediaConfig` | Shared settings; see [Media settings](#media-settings-mediaconfig) |
| `bridge` | `MediaBridge` | `MediaBridge(client, handler, config, hooks=None, *, pending_prefix="pending-")`. `handle_event(event) -> TurnResult \| None` runs one turn: `on_inbound` hooks → queue admission → `TurnInput` with session history → `before_dispatch` hooks → `TurnHandler.handle`. A reply is posted to the event's thread; a task result is saved as a pending record and marked `Status.WORKING` until `deliver_result(uid, body) -> str \| None` posts it. |
| `hooks` | `HookRegistry`, `OnInbound`, `BeforeDispatch`, `AfterPost`, `OnResult` | Register with `on_inbound` / `before_dispatch` / `after_post` / `on_result` (usable as decorators). `on_inbound(event) -> bool` stops the turn on `True`; `before_dispatch(turn) -> TurnInput` replaces the handler input. Hooks run in registration order; a hook that raises is logged and skipped. |
| `component` | `MediaBridgeComponent` | `(start, *, mode, backoff_sec, on_exit=None)`: runs `start` in a child process (`mode="process"`) or thread (`mode="thread"`), reports each exit code to `on_exit`, restarts after each `backoff_sec` delay, then gives up |
| `watchers` | `ExecDoneHandler`, `ProgressWatcher`, `DeliveryReconciler`, `catchup_pending_on_startup`, `iter_pending_with_done`, `CATCHUP_STATES` | Job completion detection, startup catch-up, delivery reconciliation, and progress polling over `pending_dir` / `events_dir`; the host passes `is_done` and `deliver` |
| `progress` | `ProgressState`, `finalize_progress`, `status_for_done_marker`, `status_label`, `summarize_tool_use_block`, `summarize_work_loop_step` | Summarizes a job's events JSONL into one progress message, rate-limited by `progress_min_interval_sec` |
| `plan_gate` | `PlanGate`, `PlanState`, `is_approval`, `expire_pending`, `AWAITING_STATE`, `PENDING_KEY` | Plan approval state machine `awaiting -> approved \| rejected \| expired` (approval words from `LanguagePack.approval_words`, deadline from `approval_ttl_sec`) |
| `cancel` | `CancelOutcome`, `handle_cancel`, `is_cancel_request`, `find_pending_uids` | Cancels a conversation's running or queued jobs from the thread; the host supplies `cancel_job` |
| `enqueue_guard` | `enqueue_or_report` | Runs an enqueue; on failure posts `LanguagePack.enqueue_failed_text` to the thread instead of dropping it |
| `pending` | `PendingStore` | Pending-request metadata, one `<store_dir>/<prefix><uid>.json` per request |
| `id_map` | `to_conversation_id`, `resolve`, `storage_key`, `storage_key_from_conversation_id` | Opaque conversation id ↔ `(space, thread)` |
| `thread_reactions` | `admit`, `status_for_admission`, `acknowledge_drained` | Maps queue admission (`accepted` / `running` / `queued` / `rejected`) to a message `Status` |
| `sanitizer` | `sanitize_result_body`, `strip_status_lines`, `strip_leading_paragraphs`, `dedupe_trailing_repeated_block`, `extract_final_assistant_text` | Pure helpers that strip internal markers from a result before posting |

`mltgnt.media.slack` (needs the `slack` extra):

| Module | Main names | Summary |
|--------|------------|---------|
| `config` | `SlackMediaConfig`, `DEFAULT_STATUS_REACTIONS` | See [Media settings](#media-settings-mediaconfig) |
| `client` | `SlackClient`, `split_text` | `SlackClient(web_client, config, *, default_channel="")`: `MediaClient` over a Slack `WebClient`; statuses become reactions, long text is split at `chunk_max_chars` |
| `app` | `build_app`, `start_socket_mode` | `build_app(config)` creates a Bolt `App` with the token from `config.bot_token_env`; `start_socket_mode(app, config)` runs Socket Mode with `config.app_token_env` (blocking). A missing token raises `RuntimeError`. |
| `inbound` | `to_media_event`, `strip_mentions`, `thread_ts_from_event`, `files_to_attachments` | Slack event dict → `MediaEvent` |
| `history` | `fetch_thread_messages`, `extract_text_from_message`, `extract_text_from_blocks`, `table_block_to_markdown` | Text extraction and thread fetch |
| `media_store` | `save_images`, `SavedMedia`, `safe_filename`, `ext_from_mimetype` | Downloads image attachments to local files |
| `mrkdwn` | `markdown_to_mrkdwn`, `normalize_markdown_residual` | Markdown → Slack mrkdwn |

`mltgnt.media.webchat` (the app needs the `webchat` extra):

| Module | Main names | Summary |
|--------|------------|---------|
| `config` | `WebChatMediaConfig` | See [Media settings](#media-settings-mediaconfig) |
| `client` | `WebChatClient` | `WebChatClient(config, *, store=None, author="assistant")`: single-channel `MediaClient` over the store |
| `store` | `WebChatStore`, `KINDS` | Append-only `YYYY-MM-DD.jsonl` per day; each row is a full message snapshot of kind `message`, `update`, or `status` |
| `app` | `create_app`, `serve`, `EventHandler`, `parse_event_id` | `create_app(config, bridge, *, store=None, poll_interval_sec=0.5, stream_timeout_sec=None)` builds a FastAPI app with `GET /`, `GET /messages?day=YYYY-MM-DD`, `POST /messages`, `GET /threads/{thread_id}`, and `GET /stream` (SSE, resumable with `Last-Event-ID`); `serve(app, config)` runs it on `config.host:config.port` |
| `inbound` | `to_media_event`, `DEFAULT_AUTHOR` | `POST /messages` body → `MediaEvent` |
| `ui` | `INDEX_HTML` | Static single-page UI served by `GET /` |

### Semantic memory (`mltgnt.memory` submodules)

Importable from `mltgnt.memory` but kept out of its `__all__`; each submodule has its own `__all__`. None of them calls an LLM.

| Module | Main names | Summary |
|--------|------------|---------|
| `mltgnt.memory.semantic` | `SemanticStore`, `SemanticEntry`, `KINDS`, `STATUSES`, `validate_entry`, `normalize_content` | `SemanticStore(path, *, persona_stem, commit_debounce_sec=None)` over `semantic.jsonl`. Kinds: `fact`, `preference`, `commitment`, `caveat`, `self`, `reflection`; statuses: `active`, `superseded`. Entries are superseded, never deleted. |
| `mltgnt.memory.tools` | `MemoryToolExecutor`, `MemoryGate`, `MEMORY_TOOL_SPECS`, `MEMORY_TOOL_NAMES`, `query_terms` | `remember` / `recall` / `forget` tools with the `ToolExecutor` shape; never raises. `MemoryGate(max_per_turn=3, max_chars=300, secret_check=None, allowed_subject_prefixes=None)` limits writes. |
| `mltgnt.memory.reflection` | `build_reflection_prompt`, `parse_reflection`, `apply_reflection`, `ReflectionResult`, `ReflectionAdd`, `ApplyReport`, `ReflectionParseError` | Build a reflection prompt, run it with your own LLM call, parse the JSON reply, and apply it to a `SemanticStore` |
| `mltgnt.memory.core_render` | `render_core`, `MANDATORY_KINDS`, `OPTIONAL_KINDS` | `render_core(entries, *, max_bytes=4096, heading="## Memory") -> str`; `caveat` and `commitment` are always included |
| `mltgnt.memory.archive` | `archive_episodes` | `archive_episodes(persona_dir, *, now, keep_days=90) -> int`; moves old `episodes.jsonl` lines to `episodes/YYYY-MM.jsonl` |

### Other subpackages

| Package | Main names (`__all__`) |
|---------|------------------------|
| `mltgnt.bridges` | `enqueue_dag`, `enqueue_and_wait`, `DagStep`, `call_llm`, `MltgntHooks`, `create_audit_writer`, `md_read`, `md_write`, modules `ghdag_bridge`, `llm_adapter`, `hooks_adapter`, `files_adapter` (`files_adapter.commit(paths, message, *, sink="memory", trailers=None)` commits through a ghdag VCS sink) |
| `mltgnt.memory` | `append_memory_entry`, `read_memory_preferences`, `read_memory_tail_text`, `read_memory_by_relevance`, `read_memory_with_sufficiency_check`, `read_memory_iterative`, `memory_file_path`, `persona_memory_lock`, `compact`, `needs_compaction`, `CompactionResult`, `LlmCallError`, `LlmCall`, `MemoryEntry`, `parse_jsonl`, `serialize_entry`, `assemble_entries_text`, `tail_utf8_bytes`, `flush_memory_commits` |
| `mltgnt.memory.dream` | `DreamSection`, `DreamSummary`, `read_dream`, `write_dream`, `read_global`, `write_global`, `read_global_summary`, `DreamSelector`, `Synthesizer` |
| `mltgnt.persona` | `Persona`, `PersonaContext`, `PersonaValidationError`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt`, `format_persona_body`, `format_result_for_persona`, `compress_heavy_to_light`, `regenerate_light_block`; also `mltgnt.persona.schema.system_default_engine()` |
| `mltgnt.skill` | `discover`, `discover_bodies`, `load`, `match`, `resolve_skill`, `run`, `build_extra_context`, `lint_skill_meta`, `SkillMeta`, `SkillFile`, `SkillRegistry`, `SkillRunResult`, `SkillMatchResult`, `ArtifactSpec`, `ProducesSpec`, `ConsumesSpec` |
| `mltgnt.routing` | `resolve_persona`, `find_observers_in_space`, `load_channel_persona_map`, `SpacePersonaEntry`, `RoutingRule`, `evaluate`, `detect_nickname`, `extract_json_object`, `extract_triage_section`, `prepare_profile_for_triage`, `TRIAGE_PROFILE_MAX_CHARS` (plus the deprecated names below) |
| `mltgnt.conversation` | `configure`, `get_config`, `ConversationConfig`, `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, modules `thread_queue`, `session_store`, `session_compact`, `thread_index`, `thread_persona_store`, `fake_media` |
| `mltgnt.scheduler` | `PersonaScheduler`, `ScheduleJob`, `SchedulePaths`, `load_schedule_jobs`, `atomic_write_text` |
| `mltgnt.daemon` | `DaemonComponent`, `DaemonRunner`, `PidLock`, `SkillWatcherComponent` |
| `mltgnt.config` | `PersonaConfig`, `MemoryConfig`, `SchedulerConfig`, `ConversationConfig`, `DEFAULT_WEIGHT_MAP`; `mltgnt.config.language`: `LanguagePack`, `JA` |
| `mltgnt.interfaces` | Dependency-free DTOs and Protocols: `PersonaProtocol`, `PersonaFMBase`, `Message`, `ChatInput`, `ChatOutput`, `ChatInputBase`, `ChatOutputBase`, `TurnHandler`, `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, `SlackClientProtocol` (deprecated); the media contract is in `mltgnt.interfaces.media` |

Skill matching takes an engine: `match(user_input, skills, persona_skills=None, model=None, *, engine="claude")` and `resolve_skill(user_input, skill_paths, persona_skills=None, entry_file="SKILL.md", matcher_model=None, *, matcher_engine="claude")`. With `claude` and no model the matcher uses `claude-haiku-4-5-20251001`; other engines use their ghdag default model.

## Protocols / Extension Points

| Extension point | Module | Contract |
|-----------------|--------|----------|
| `MediaClient` | `mltgnt.interfaces.media` | `post` / `update` / `set_status` / `upload`; see [Media contract](#media-contract-mltgntinterfacesmedia). Implemented by `SlackClient` and `WebChatClient`; accepted by `PersonaScheduler(slack=...)` and `MediaBridge`. |
| `TurnHandler` | `mltgnt.interfaces.turn` | `handle(turn: TurnInput) -> TurnResult`; the media-independent boundary between a medium and the conversation layer. |
| `TurnInput` / `TurnResult` / `Attachment` / `HistoryMessage` | `mltgnt.interfaces.turn` | Frozen dataclasses. `TurnInput(conversation_id, text, attachments=(), history=(), persona_id=None)`; `TurnResult(kind, text="", task_ref=None)` with `kind` `"reply"` or `"task"`. |
| `HookRegistry` hooks | `mltgnt.media._core.hooks` | `OnInbound = (MediaEvent) -> bool`, `BeforeDispatch = (TurnInput) -> TurnInput`, `AfterPost = (TurnResult, str \| None) -> None`, `OnResult = (uid, body) -> None`. |
| `EventHandler` | `mltgnt.media.webchat.app` | `handle_event(event: MediaEvent)`; satisfied by `MediaBridge`. |
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Structural type satisfied by `Persona`. |
| `PersonaFMBase` / `ChatInputBase` / `ChatOutputBase` | `mltgnt.interfaces.types` | Protocols for persona frontmatter and chat input / output shapes. |
| `DaemonComponent` | `mltgnt.daemon` | `name` property, non-blocking `start()`, `stop()`. Returned by the factory given to `mltgnt run --components`. |
| `LLMCaller` | `mltgnt.agent._runner` | `(prompt, *, tool_result=None) -> str \| None`; `AgentRunner(llm_call=...)`. |
| `ToolExecutor` | `mltgnt.agent._runner` | `(tool_name, tool_args) -> str`; `AgentRunner(tool_executor=...)`. `MemoryToolExecutor` is a built-in implementation. |
| `ReflexionEvaluator` | `mltgnt.agent._runner` | `(prompt, tool_name, tool_args, tool_result, tool_trace) -> ReflexionVerdict(should_retry, feedback)`; `AgentRunner(evaluator=...)`. |
| `ActionFn` | `mltgnt.scheduler.models` | `(job) -> tuple[bool, str]`; register with `PersonaScheduler(actions={name: fn})`. Built-in actions: `noop`, `skill`, and `memory_dream` (only when `memory_config.use_dream_summary` is true). |
| `LanguagePack` | `mltgnt.config.language` | Frozen dataclass of locale-specific vocabulary; see [LanguagePack](#languagepack-mltgntconfiglanguage). |

## Architecture

Every module under `src/mltgnt/`:

| Module | Role |
|--------|------|
| `__init__.py` | Defines `mltgnt.__all__` and `__version__`. |
| `__main__.py` | Enables `python -m mltgnt`. |
| `exceptions.py` | Base exceptions: `MltgntError`, `ConfigError`, `DependencyError`. |
| `py.typed` | PEP 561 marker. |
| `agent/__init__.py` | Public agent loop and decision-layer names. |
| `agent/_runner.py` | `AgentRunner` tool-calling loop with retry, Reflexion, plans, and full-trace history. |
| `agent/_parse.py` | Parses the JSON object in an LLM response. |
| `agent/action_classifier.py` | Classifies tool side-effect levels. |
| `agent/deterministic_gate.py` | Deterministic request gates (work / create requests, deferred promises). |
| `agent/dispatch_decision.py` | Reply-vs-delegate decision (`make_dispatch_decision`). |
| `agent/dispatch_preflight.py` | Preflight before dispatch (`run_preflight`). |
| `agent/plan.py` | `Plan` / `PlanItem` and their JSON contract. |
| `agent/reflexion.py` | `DefaultReflexionEvaluator`. |
| `bridges/__init__.py` | Public bridge names; the only gateway to ghdag. |
| `bridges/ghdag_bridge.py` | `enqueue_dag`, `enqueue_and_wait`, `DagStep`, skill I/O type check. |
| `bridges/llm_adapter.py` | `call_llm`, a thin wrapper around the ghdag text call. |
| `bridges/hooks_adapter.py` | `MltgntHooks` (ghdag DAG hooks) and the `AgentRunner` audit writer factory. |
| `bridges/audit_adapter.py` | Orchestration audit events (`record_event`, `start_orchestration`, `end_orchestration`). |
| `bridges/files_adapter.py` | `md_read`, `md_write`, and `commit` through the ghdag VCS sink. |
| `cli/main.py` | argparse entry point and exit-code mapping. |
| `cli/run.py` | `mltgnt run`: loads the component factory and runs `DaemonRunner`. |
| `cli/memory.py` | `mltgnt memory dream show` / `forget`. |
| `config/__init__.py` | `PersonaConfig`, `MemoryConfig`, `SchedulerConfig`, `ConversationConfig`, `DEFAULT_WEIGHT_MAP`. |
| `config/language.py` | `LanguagePack` and `JA`. |
| `conversation/__init__.py` | `configure` / `get_config` and re-exports of the turn types. |
| `conversation/types.py` | Re-export of `mltgnt.interfaces.turn`. |
| `conversation/thread_queue.py` | Per-conversation wait queue (admission, queued-message composition, cancel words). |
| `conversation/session_store.py` | Session ledger and turn log. |
| `conversation/session_compact.py` | Automatic compaction of conversation logs. |
| `conversation/thread_index.py` | Thread index. |
| `conversation/thread_persona_store.py` | JSONL store of conversation-bound personas with a TTL. |
| `conversation/fake_media.py` | Builds `TurnInput` without a real medium (tests, local runs). |
| `daemon/__init__.py` | `DaemonComponent` Protocol and public names. |
| `daemon/_runner.py` | `DaemonRunner`: starts components under a PID lock, stops them on SIGINT / SIGTERM. |
| `daemon/_pidlock.py` | `PidLock`. |
| `daemon/_skill_watcher.py` | `SkillWatcherComponent`: watches skill files and reloads the `SkillRegistry`. |
| `interfaces/__init__.py` | Dependency-free DTOs and Protocols shared by every layer. |
| `interfaces/media.py` | `Status`, `MediaClient`, `adapt_client`. |
| `interfaces/slack.py` | Deprecated `SlackClientProtocol`. |
| `interfaces/turn.py` | `TurnInput`, `TurnResult`, `Attachment`, `HistoryMessage`, `TurnHandler`. |
| `interfaces/persona.py` | `PersonaProtocol`. |
| `interfaces/types.py` | `ChatInput`, `ChatOutput`, `Message`, `PersonaFMBase`, `ChatInputBase`, `ChatOutputBase`. |
| `media/__init__.py`, `media/_core/__init__.py`, `media/slack/__init__.py`, `media/webchat/__init__.py` | Package markers with `__all__ = []`. |
| `media/_core/types.py` | `MediaEvent`, `OutboundMessage`. |
| `media/_core/client.py` | Re-export of the `MediaClient` contract. |
| `media/_core/config.py` | `MediaConfig`. |
| `media/_core/bridge.py` | `MediaBridge`: one turn from `MediaEvent` to posted reply or pending task. |
| `media/_core/hooks.py` | `HookRegistry` and hook types. |
| `media/_core/component.py` | `MediaBridgeComponent` supervisor with restart backoff. |
| `media/_core/watchers.py` | Job completion, startup catch-up, delivery reconciliation, progress polling. |
| `media/_core/progress.py` | Progress message summarization and status labels. |
| `media/_core/plan_gate.py` | Plan approval state machine. |
| `media/_core/cancel.py` | Cancel requests from the thread. |
| `media/_core/enqueue_guard.py` | `enqueue_or_report`. |
| `media/_core/pending.py` | `PendingStore`. |
| `media/_core/id_map.py` | Conversation id to `(space, thread)` mapping. |
| `media/_core/thread_reactions.py` | Queue admission result to message `Status`. |
| `media/_core/sanitizer.py` | Strips internal markers from results before posting. |
| `media/slack/config.py` | `SlackMediaConfig`, `DEFAULT_STATUS_REACTIONS`. |
| `media/slack/client.py` | `SlackClient`, `split_text`. |
| `media/slack/app.py` | `build_app`, `start_socket_mode`. |
| `media/slack/inbound.py` | Slack event to `MediaEvent`. |
| `media/slack/history.py` | Slack message text extraction and thread fetch. |
| `media/slack/media_store.py` | Image attachment download. |
| `media/slack/mrkdwn.py` | Markdown to Slack mrkdwn. |
| `media/webchat/config.py` | `WebChatMediaConfig`. |
| `media/webchat/client.py` | `WebChatClient`. |
| `media/webchat/store.py` | `WebChatStore` (daily JSONL message snapshots). |
| `media/webchat/app.py` | FastAPI app (`create_app`, `serve`) with SSE stream. |
| `media/webchat/inbound.py` | `POST /messages` body to `MediaEvent`. |
| `media/webchat/ui.py` | Static single-page UI (`INDEX_HTML`). |
| `memory/__init__.py` | Re-export hub for memory APIs. |
| `memory/api.py` | Memory path resolution, lock, append, and read. |
| `memory/_format.py` | `MemoryEntry`, `parse_jsonl`, `serialize_entry`. |
| `memory/search.py` | Relevance search, sufficiency check, iterative retrieval. |
| `memory/_tfidf.py` | TF-IDF vectorization. |
| `memory/_scoring.py` | Cosine-similarity scoring. |
| `memory/_sufficiency.py` | LLM-based sufficiency judgment. |
| `memory/_iterative.py` | Iterative retrieval loop. |
| `memory/_chroma.py` | Optional Chroma vector search (`get_collection`, `query_similar`, `upsert_entry`); inactive when `chromadb` is not installed. |
| `memory/_commit.py` | Debounced git commits of memory files and `flush_memory_commits`. |
| `memory/compaction.py` | Memory compaction (`compact`, `needs_compaction`, `LlmCallError`). |
| `memory/semantic.py` | `SemanticStore` over `semantic.jsonl`. |
| `memory/tools.py` | `remember` / `recall` / `forget` tool executor. |
| `memory/reflection.py` | Reflection prompt, parser, and applier. |
| `memory/core_render.py` | Always-loaded memory core rendering. |
| `memory/archive.py` | Monthly archiving of old episodes. |
| `memory/dream/__init__.py` | Dream API names. |
| `memory/dream/_format.py` | `DreamSection`, `DreamSummary`, JSON conversion. |
| `memory/dream/api.py` | Read / write `dream.json` and `global.json`. |
| `memory/dream/selector.py` | `DreamSelector`: chooses personas for synthesis. |
| `memory/dream/synthesizer.py` | `Synthesizer`: builds a `DreamSummary` with an LLM. |
| `persona/__init__.py` | Public persona API and `PersonaValidationError`. |
| `persona/loader.py` | `Persona` and persona file loading. |
| `persona/frontmatter.py` | YAML frontmatter parsing. |
| `persona/schema.py` | Frontmatter schema, validation, `system_default_engine`. |
| `persona/registry.py` | Persona listing and name / alias resolution. |
| `persona/extractor.py` | H2 section parsing and light / heavy text extraction. |
| `persona/formatter.py` | Tone formatting and persona prefixes (`format_persona_body`). |
| `persona/compress.py` | LLM compression of the heavy block into the light block. |
| `persona/result_format.py` | `format_result_for_persona`. |
| `persona/runner.py` | `run_persona_prompt`. |
| `persona/resolve.py` | Responder selection and `PersonaContext` building. |
| `persona/memory.py` | Media-independent persona memory helpers (dedupe keys, tool-trace blocks). |
| `persona/phrases.py` | Persona phrase loading. |
| `persona/types.py` | `PersonaContext`. |
| `routing/__init__.py` | Public routing names. |
| `routing/channel_router.py` | Space-to-persona routing and observers. |
| `routing/triage.py` | Triage preprocessing helpers. |
| `routing/agentic_discover.py` | Agentic skill discovery. |
| `scheduler/__init__.py` | Public scheduler names. |
| `scheduler/models.py` | `ScheduleJob`, `OnExitPolicy`, `ActionFn`. |
| `scheduler/loader.py` | `load_schedule_jobs` (YAML). |
| `scheduler/runner.py` | `PersonaScheduler`. |
| `scheduler/base_runner.py` | `BaseRunner` tick-loop base class. |
| `scheduler/state.py` | `SchedulePaths`, `atomic_write_text`. |
| `scheduler/fanout.py` | Fan-out of dynamic child tasks for skill jobs with `action_args.enable_fanout`. |
| `scheduler/actions/skill.py` | Built-in `skill` action. |
| `scheduler/actions/dream.py` | Built-in `memory_dream` action. |
| `skill/__init__.py` | Public skill API. |
| `skill/models.py` | `SkillMeta`, `SkillFile`, `SkillLoadError`. |
| `skill/loader.py` | Skill file discovery and frontmatter parsing. |
| `skill/lint.py` | Frontmatter validation (`lint_skill_meta`). |
| `skill/matcher.py` | Skill matching (slash, literal, triggers, LLM). |
| `skill/context.py` | `build_extra_context` from knowledge and memory. |
| `skill/runner.py` | Variable substitution and prompt composition (`run`). |
| `skill/_registry.py` | `SkillRegistry`. |

### Layer contracts (`.importlinter`)

A layer may import only layers below it:

| Layer (top → bottom) | Packages |
|----------------------|----------|
| 1 | `daemon`, `media` |
| 2 | `scheduler`, `agent`, `routing` |
| 3 | `persona`, `skill`, `memory`, `conversation` |
| 4 | `bridges` |
| 5 | `interfaces` |

Additional contracts:

| Contract | Rule |
|----------|------|
| `ignore_imports` | Any module may import `mltgnt.config`; `mltgnt.bridges.*` may import `mltgnt.skill.*`; `mltgnt.skill.matcher` may import `mltgnt.routing.agentic_discover`. |
| `l3_no_ghdag` (forbidden) | `persona`, `skill`, `memory`, and `conversation` must not import `ghdag` directly; they go through `bridges`. |
| `core_no_media` (forbidden) | `scheduler`, `agent`, `routing`, `persona`, `skill`, `memory`, `conversation`, `bridges`, and `interfaces` must not import `mltgnt.media`. |
| `media_slack_webchat` (independence) | `mltgnt.media.slack` and `mltgnt.media.webchat` must not import each other. |

## Configuration

### Environment variables

All environment variables read under `src/`, plus `ENABLE_GIT`, which ghdag reads and which changes mltgnt behavior:

| Variable | Read by | Effect |
|----------|---------|--------|
| `MLTGNT_DEFAULT_ENGINE` | `mltgnt.persona.schema.system_default_engine()` | Engine used when neither the persona nor the caller names one (`run_persona_prompt`, dispatch decision, result formatting). Read on every call. Unset or blank → `claude`. Must be one of `claude`, `codex`, `cursor`, `gemini`, otherwise `ValueError`. |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge.enqueue_dag` | `0` skips the compose-time skill I/O type check. Any other value (or unset) keeps it on. |
| `THREAD_PERSONA_TTL_DAYS` | `mltgnt.conversation.thread_persona_store` | TTL in days for thread-to-persona bindings (default `30`; invalid values fall back to `30`). Ignored once a `ConversationConfig` is configured; its `thread_persona_ttl_days` wins. |
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Substituted for `$NIKKI_ROOT` in skill bodies (empty string when unset). |
| `REPO_ROOT` | `mltgnt.skill.runner` | Substituted for `$REPO_ROOT` in skill bodies (empty string when unset). |
| `SLACK_BOT_TOKEN` | `mltgnt.media.slack.app.build_app` | Slack bot token. The variable name is `SlackMediaConfig.bot_token_env`; unset → `RuntimeError`. |
| `SLACK_APP_TOKEN` | `mltgnt.media.slack.app.start_socket_mode` | Slack app-level token for Socket Mode. The variable name is `SlackMediaConfig.app_token_env`; unset → `RuntimeError`. |
| `ENABLE_GIT` | ghdag (`ghdag.vcs`) | Truthy enables git commits of memory files. Unset → the ghdag sink is a `NullSink` and memory commits do nothing. ghdag also needs `GHDAG_VCS_CONFIG` pointing to a YAML file with a `sinks.memory` entry; without it the sink stays disabled. |

Memory commits: memory appends, compaction, `dream.json`, and `global.json` writes schedule a commit through `mltgnt.bridges.files_adapter.commit` on the ghdag `memory` sink. Commits are debounced per path; append and compaction use `MemoryConfig.commit_debounce_sec` (`<= 0` commits immediately). Commit failures are logged, never raised. `mltgnt.memory.flush_memory_commits()` commits pending paths now and also runs at process exit.

### Config dataclasses (`mltgnt.config`)

All are frozen dataclasses. Paths are injected by the host; mltgnt hardcodes none.

| Class | Required fields | Optional fields (default) |
|-------|-----------------|---------------------------|
| `PersonaConfig` | — | `weight_map` (`DEFAULT_WEIGHT_MAP`), `section_aliases` (`PERSONA_SECTION_ALIASES`), `exclude_stems` (`frozenset()`) |
| `MemoryConfig` | `chat_dir` | `chat_memory_dir` (`None`), `inject_max_bytes` (`10240`), `inject_max_entries` (`12`), `preferences_max_bytes` (`5120`), `lock_timeout_sec` (`30.0`), `lock_stale_threshold_sec` (`300.0`), `raw_days` (`7`), `mid_weeks` (`3`), `compact_threshold_bytes` (`40960`), `compact_target_bytes` (`25600`), `preferences_section_name` (`"User’s preferences and tendencies"`), `protected_layers` (`("caveat",)`), `timezone` (`"Asia/Tokyo"`), `dream_model` (`""`), `dream_engine` (`"claude"`), `use_dream_summary` (`False`), `dream_dir_name` (`"memory"`), `commit_debounce_sec` (`300.0`), `global_dream_exclude_personas` (`()`) |
| `SchedulerConfig` | `schedule_yaml`, `state_dir` | `timezone` (`"Asia/Tokyo"`), `salt` (`""`) |
| `ConversationConfig` | `queue_dir`, `sessions_dir`, `ledger_dir`, `thread_index_dir`, `thread_persona_path` | `posts_dir` (`None`), `audit_path` (`None`), `stale_after_sec` (`3600`), `max_queued` (`20`), `cleanup_ttl_days` (`14`), `thread_persona_ttl_days` (`30`) |

`dream_engine` and `dream_model` select the LLM for the `memory_dream` scheduler action. A blank engine means `claude`; a blank model means the engine's default, except that `claude` falls back to `claude-haiku-4-5-20251001`.

### Media settings (`MediaConfig`)

`mltgnt.media._core.config.MediaConfig` is a frozen dataclass; media implementations subclass it.

| Field | Default | Meaning |
|-------|---------|---------|
| `state_dir` | required | State directory of the medium; its location is chosen by the host. |
| `pending_dir` | required | Pending-request records of delegated tasks. |
| `events_dir` | required | Job events JSONL read by the progress watcher. |
| `language` | `JA` | `LanguagePack` for cancel / approval words, status labels, and messages. |
| `progress_min_interval_sec` | `5.0` | Minimum interval between progress message updates. |
| `approval_ttl_sec` | `600.0` | Plan approval deadline. |

`SlackMediaConfig(MediaConfig)` (`mltgnt.media.slack.config`) adds `bot_token_env` (`"SLACK_BOT_TOKEN"`), `app_token_env` (`"SLACK_APP_TOKEN"`), `status_reactions` (`DEFAULT_STATUS_REACTIONS`: `Status` → reaction name), and `chunk_max_chars` (`3000`; must be positive, otherwise `ValueError`).

`WebChatMediaConfig(MediaConfig)` (`mltgnt.media.webchat.config`) adds `store_dir` (required, keyword-only), `host` (`"127.0.0.1"`), `port` (`8765`), and `space_id` (`"webchat"`).

### LanguagePack (`mltgnt.config.language`)

`LanguagePack` is a frozen dataclass of locale-specific vocabulary; `JA` is the built-in instance and the default wherever a function takes `pack=None` or a config takes `language`. Hosts that want other values pass their own instance.

| Field group | Fields | Used by |
|-------------|--------|---------|
| Request gates | `work_request_markers`, `create_request_markers`, `deferred_patterns` | `mltgnt.agent.deterministic_gate` |
| Persona text | `compress_prompt_template`, `v21_required_sections`, `v21_example_section`, `meta_header_needles`, `dedupe_opener_re`, `persona_cut_re`, `persona_end_re`, `exclude_stems` | `mltgnt.persona` compression, formatting, and listing |
| Conversation | `cancel_words`, `composite_header`, `composite_cancel_suffix` | `mltgnt.conversation.thread_queue` (queued-message composition, cancel detection) and `mltgnt.media._core.cancel` |
| Media | `approval_words` (default `{"OK", "ok", "yes", "approve"}`), `status_labels` (default `received` → `Received`, `working` → `Working`, `done` → `Done`, `failed` → `Failed`, `cancelled` → `Cancelled`), `enqueue_failed_text` (default `"Failed to enqueue the request. Please try again later."`), `progress_line_pattern` (default matches `[progress] <text>` lines) | `mltgnt.media._core` |
| Memory tools | `remember_trigger_words`, `forget_trigger_words` (default empty) | `mltgnt.memory.tools` |

### Scheduler jobs (`ScheduleJob`)

`PersonaScheduler` loads jobs from `yaml_path` or `SchedulerConfig.schedule_yaml` (a load failure raises `ConfigError`). Each entry is parsed by `ScheduleJob.from_dict`, which raises `ValueError` on invalid input.

| Field | Default | Notes |
|-------|---------|-------|
| `id`, `mode`, `action` | required | `mode`: `scheduled` (needs `every_day_at`), `interval` (needs `interval_minutes > 0`), `fuzzy_window` (needs `window_start` / `window_end`, no overnight windows), `chained` (runs when all `depends_on` jobs are done) |
| `notify` | `silent` | `silent`, `slack_secretary`, or `slack_custom` (needs `slack_channel`); notifications are posted through the scheduler's `MediaClient` |
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
| `chain_every_run` | `false` | Requires `mode: chained` and non-empty `depends_on`; fires after every successful upstream run, passes the upstream output as `ScheduleJob.upstream_output`, and writes no done / skipped / failed marks |
| `upstream_output` | — | Set at runtime only; never read from YAML |

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
| `MltgntError` | `mltgnt.exceptions` | `Exception` | Base of `ConfigError` and `DependencyError`. `mltgnt run` maps other subclasses to exit code `1`. |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | Invalid `mltgnt run --components`; scheduler YAML fails to load (`PersonaScheduler`); more than one `primary` persona in one space (`mltgnt.routing.load_channel_persona_map`). CLI exit code `2`. |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | The PID lock is held by another instance (`DaemonRunner.run`); the injected persona loader fails (`load_channel_persona_map`). CLI exit code `3`. |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | `load_persona` finds YAML frontmatter that does not parse or lacks the required `persona` key. |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | Skill loading fails: the ghdag tools list times out, fails, or returns invalid JSON, or a skill references an unknown tool. |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | `enqueue_dag`'s compose-time skill I/O type check finds a pipe type mismatch between steps (disable with `SKILL_IO_TYPECHECK=0`). |
| `LlmCallError` | `mltgnt.memory.compaction` (re-exported by `mltgnt.memory`) | `RuntimeError` | Exported for hosts to wrap failures of an injected `llm_call` in memory compaction; mltgnt code does not raise it. |
| `ReflectionParseError` | `mltgnt.memory.reflection` (importable from `mltgnt.memory`) | `ValueError` | `parse_reflection` gets text with no JSON object, invalid JSON, or wrong field types (`reflection` not a string, `add` not a list of objects, `supersede` not a list of strings). |

Built-in exceptions raised by the public API: `ValueError` from `ScheduleJob.from_dict` (invalid job), `parse_plan` (invalid plan JSON), `system_default_engine()` (invalid `MLTGNT_DEFAULT_ENGINE`), and `SlackMediaConfig` (`chunk_max_chars <= 0`); `TypeError` from `adapt_client` (no `post` or `post_message`); `RuntimeError` from `build_app` / `start_socket_mode` (token variable unset); `ImportError` from the Slack / WebChat app functions when the extra is missing.

## Public API Stability

- mltgnt is pre-1.0 (`0.Y.Z`). A minor release (`0.Y.0`) may contain breaking changes; a patch release (`0.Y.Z`) does not. Every change is listed in `CHANGELOG.md`.
- The supported surface is `mltgnt.__all__`, `mltgnt.interfaces` (including `mltgnt.interfaces.media`), the CLI, and the configuration schema above. Other subpackage names, including `mltgnt.media` submodules, may change in any minor release.
- Renamed or removed APIs keep a deprecated alias for at least one minor release before removal.
- Pin an exact tag (for example `@v0.82.0`) in production.

## Deprecated API

| Deprecated | Module | Use instead | Warning | Removal |
|------------|--------|-------------|---------|---------|
| `SlackClientProtocol` (`post_message`) | `mltgnt.interfaces.slack` (re-exported by `mltgnt.interfaces`) | `MediaClient` (`mltgnt.interfaces.media`) | One `DeprecationWarning` when such a client passes through `adapt_client` (including `PersonaScheduler(slack=...)`); it is wrapped so `post` calls `post_message(text, channel=space, thread_ts=thread)` | Next minor release |
| `resolve_responding_persona` | `mltgnt.routing` | `resolve_persona` | `DeprecationWarning` | — |
| `find_observers` | `mltgnt.routing` | `find_observers_in_space` | `DeprecationWarning` | — |
| `ChannelPersonaEntry` | `mltgnt.routing` | `SpacePersonaEntry` | none (plain alias) | — |

## License

MIT (SPDX: `MIT`). See `LICENSE`.
