# Changelog

## Unreleased

## v0.126.0

- **Added `mltgnt.config.SkillConfig`** (#5060): frozen dataclass with `passthrough_env: tuple[str, ...] = ()` plus process-wide `get_skill_config()` / `set_skill_config()` (importable from `mltgnt.config`, not in `__all__`). A `str` value raises `TypeError`; other iterables are normalized to a tuple.
- **BREAKING:** `mltgnt.skill.runner.run` no longer substitutes `$NIKKI_ROOT` implicitly. Only the names declared in `SkillConfig.passthrough_env` are replaced with their environment value (empty when unset); undeclared `$KEY` stay as is. Built-in keys (`$ARGUMENTS` / `$PERSONA` / `$SKILL_DIR` / `$REPO_ROOT` / positional) are unchanged and take precedence. Hosts that relied on the old behavior must declare the names at each process entry point before bumping, e.g.:

  ```python
  try:
      from mltgnt.config import SkillConfig, set_skill_config
  except ImportError:
      pass
  else:
      set_skill_config(SkillConfig(passthrough_env=("NOTES_ROOT", "NIKKI_ROOT")))
  ```

## v0.125.3

- **Public API exports** (#5037): `mltgnt.media.__all__` re-exports 27 names from `mltgnt.media._core.*` (without importing `media.slack` / `media.webchat`); `mltgnt.agent` exports `LLMCaller`, `RetryConfig`, `ToolExecutor` and `REFLEXION_EXHAUSTED_TOOL`; `bridges.hooks_adapter` imports `check_pipeline_status` and `default_check_rejected` from public `ghdag.dag`. (`eb16d03`)

## v0.125.2

- Rewrite the README for v0.125.1. (`14ff1ef`)

## v0.125.1

- Bump ghdag to v0.101.2. (`7331104`)

## v0.125.0

- **Localized legacy persona headings and Phase 1 meta lines come from the language pack** (#5040): `LanguagePack` gains `persona_section_aliases`, `phase1_meta_prefixes` and `phase1_meta_markers` (all empty in `EN`); `persona.extractor.extract`, `routing.triage.extract_triage_section` and `memory.compaction._sanitize_phase1_output` take a `pack` keyword argument. **BREAKING:** `mltgnt.config.PERSONA_SECTION_ALIASES` is removed; consumers read `get_language_pack().persona_section_aliases` at run time. (`d9528c4`)
- Use an ASCII dummy language pack for legacy persona heading tests. (`793c714`)

## v0.124.1

- **Scheduler robustness** (#5034): an exception raised by a job action is logged with a traceback, marks the job failed (`"<ExceptionType>: <message>"`) and posts a `Job raised` notification instead of killing the job thread silently. `ghdag_bridge.enqueue_and_wait` / `enqueue_dag` request a cancel of a timed-out task via the `jobs/cancel/<uuid>` marker. New `SchedulePaths.prune(today, keep_days=30)`, called once per day by `PersonaScheduler` (`state_keep_days=30`). A dropped `chain_every_run` run now logs a warning, and `stop()` no longer waits for the rest of the loop sleep. (`2bb3747`)

## v0.124.0

- **`mltgnt.agent.work_loop`** (#5002): `run_work_loop` wraps `AgentRunner` with planning, `RepeatGuard`, optional `run_skill` via `SkillRunner` / `GhdagSkillRunner` (`enqueue_and_wait` with `parent_correlation_id`), and `events_sink` callbacks (`work_loop_step` / `work_loop_plan_failed`). Exports include `WorkLoopConfig`, `WorkLoopOutcome`, `WorkLoopDeadline`, `FINISH_TOOL`, `TrackingCaller`, `is_plan_prompt`, `make_skill_tool` and `run_skill_contract`. (`55335ec`)
- CP2 review fixes for `work_loop`. (`ebc399f`)

## v0.123.10

- Bump ghdag to v0.101.1. (`a315ada`)

## v0.123.9

- Bump ghdag to v0.101.0. (`72b6e5a`)

## v0.123.8

- Bump ghdag to v0.100.8. (`9fbe543`)

## v0.123.7

- Bump ghdag to v0.100.7. (`125d810`)
- Bump ghdag to v0.100.7. (`9ea8b3b`)

## v0.123.6

- Bump ghdag to v0.100.5. (`bb79ca6`)

## v0.123.5

- Bump ghdag to v0.100.4. (`fc34b7f`)

## v0.123.4

- Bump ghdag to v0.100.3. (`9c1d4aa`)

## v0.123.3

- Bump ghdag to v0.100.2. (`af80f04`)

## v0.123.2

- Bump ghdag to v0.100.1. (`7fe5b91`)

## v0.123.1

- Rewrite the README for v0.123.0 (#4499). (`7c5475b`)

## v0.123.0

- Forward `action_args.task_timeout_sec` from the scheduler as the ghdag task timeout (#4539). (`d524caf`)

## v0.122.0

- Document the `JA` removal, the `EN` default and the `set_language_pack` migration (#4493). Host migration (keep this order): (1) define your own `LanguagePack` and replace every `JA` import with it; (2) at daemon startup, call `set_language_pack(your_pack)` once, before any `MediaConfig` is built; (3) only then raise the mltgnt version pin. (`4395a6f`)

## v0.121.0

- Bump ghdag to v0.100.0. (`007c9b5`)

## v0.120.0

- **BREAKING: `mltgnt.config.language.JA` is removed** (#4487). No compatibility alias is kept, so `from mltgnt.config.language import JA` raises `ImportError`; defaults now come from the current pack (`EN` unless replaced). The CJK exception for `src/mltgnt/config/language.py` is gone: all of `src/` is covered by the same zero-CJK check as `tests/`. (`480e73a`)

## v0.119.0

- Switch persona directory layout tests in memory from the `JA` to the `EN` pack (#4492). (`89f2191`)

## v0.118.0

- Switch `media/_core` tests from the `JA` to the `EN` language pack (#4491). (`5f5f188`)

## v0.117.0

- Switch media core types tests from the `JA` to the `EN` pack (#4490). (`9f29435`)

## v0.116.0

- Pin the call-time `thread_queue` language pack lookup in tests (#4489). (`a616d65`)

## v0.115.0

- Pin agent URL extraction around full-width parentheses and periods in tests (#4488). (`15396cc`)

## v0.114.0

- Translate the scheduler `enable_pipeline` docstring to ASCII English (#4486). (`53c5a3b`)

## v0.113.0

- `memory.dream.synthesizer` reads its default `exclude_stems` from the current language pack (#4484). (`0b58dfb`)

## v0.112.0

- `thread_queue` cancel words and composite message text are read from the current language pack at call time, not fixed at import time (#4480). (`860a9bd`)
- Replace CJK punctuation in the skill runner docstring with ASCII (#4482). (`722b3d8`)

## v0.111.0

- Resolve the `deterministic_gate` language pack at call time and drop CJK literals (#4479). (`fea9faa`)
- Build CJK punctuation via `chr()` and update the default-pack test for call-time resolution (#4479). (`af8939f`)

## v0.110.0

- Persona code resolves the default language pack at call time instead of `JA`, including the excluded persona stems in `persona.registry` (#4477). (`5cffa39`)
- Follow the `EN` default exclude stem in persona tests and cover call-time pack resolution (#4477). (`bf881ce`)

## v0.109.0

- Media defaults resolve the current language pack at call time instead of `JA`: `MediaConfig.language` defaults to the current pack at construction time. (`028202e`)
- Align the `MediaConfig` default language test with the current pack (#4475). (`3883c06`)

## v0.108.0

- Bump ghdag to v0.99.0. (`a74d1fc`)

## v0.107.0

- **English pack and current-pack API**: new `EN`, `get_language_pack()` and `set_language_pack(pack)` in `mltgnt.config.language`, also re-exported from `mltgnt.config`. `set_language_pack` raises `TypeError` when `pack` is not a `LanguagePack`. (`a385282`)

## v0.106.0

- Bump ghdag to v0.98.0. (`b67e646`)

## v0.105.0

- Bump ghdag to v0.97.0. (`429767e`)

## v0.104.0

- Bump ghdag to v0.96.0. (`d04c767`)

## v0.103.0

- Bump ghdag to v0.95.0. (`391a5ca`)

## v0.102.0

- Bump ghdag to v0.94.0. (`33d80e5`)

## v0.101.0

- Bump ghdag to v0.93.0. (`5b384f0`)

## v0.100.0

- Bump ghdag to v0.91.0. (`bf9ccce`)
- Bump ghdag to v0.92.0. (`7f78cb0`)

## v0.99.0

- Bump ghdag to v0.91.0. (`4d77240`)

## v0.98.0

- Bump ghdag to v0.89.0. (`50a14c8`)

## v0.97.0

- Rewrite the README for v0.96.0. (`bbca0c8`)

## v0.96.0

- Update README version references to v0.95.0. (`2164002`)

## v0.95.0

- Update README version references to v0.94.0. (`6263051`)

## v0.94.0

- Rewrite the README for v0.93.0. (`6e25978`)

## v0.93.0

- Bump ghdag to v0.88.0. (`8d28b0c`)

## v0.92.0

- Align README version references with v0.91.0. (`1e90269`)

## v0.91.0

- Rewrite the README for the v0.90.0 public API and media layer. (`a140541`)

## v0.90.0

- Bump ghdag to v0.87.0. (`505df1e`)

## v0.89.0

- Align README version references with the v0.88.0 release. (`477ac94`)
- Add `TurnResult.post_options` and pass it through the media bridge. (`115dfed`)

## v0.88.0

- Align the WebChat UI with Slack and enrich the SSE stream. (`749b667`)

## v0.87.0

- Extend `SlackClient` with extra kwargs, upload and unreact. (`f445a4d`)

## v0.86.0

- Rewrite the README for v0.85.0. (`0842091`)
- Add the WebChat 2-pane UI, bookmarks and the cross-media reaction contract. (`fe858a7`)
- Align the `TurnResult` field test with reactions (#4211). (`a159cda`)
- CP2 review fixes: WebChat reply counts use the latest snapshot per message, Markdown links only allow safe URL schemes, and the bookmark panel updates live. (`2b3af11`)

## v0.85.0

- Rewrite the README for v0.84.0. (`cbc2687`)

## v0.84.0

- Bump ghdag to v0.86.0. (`51a3e09`)

## v0.83.0

- Rewrite the README for v0.82.0. (`2d0ac25`)

## v0.82.0

- Bump ghdag to v0.85.0. (`f75b65b`)

## v0.81.0

- Bump ghdag to v0.84.0. (`283dae0`)

## v0.80.0

- Add the WebChat medium and cross-media contract tests (#4032). (`c7e70df`)

## v0.79.0

- **Media turn bridge and host hooks** (#4031): new `mltgnt.media._core.hooks.HookRegistry` with `on_inbound` / `before_dispatch` / `after_post` / `on_result` hooks (run in registration order; a raising hook is logged and skipped) and `mltgnt.media._core.bridge.MediaBridge(client, handler, config, hooks)`. `handle_event(event)` admits the event through `thread_reactions`, builds a `TurnInput` with `session_store` history, calls `TurnHandler.handle` and posts the reply to the same thread; a delegated task result is kept as a pending record until `deliver_result(uid, body)` posts it and fires `on_result`. (`30a60d0`)

## v0.78.0

- Add progress, watchers, component, guards and the plan gate to `media._core` (#4030). (`e724a6f`)

## v0.77.0

- Add the semantic memory store, core renderer, memory tools, reflection and archive (#4037). (`8ba4b85`)

## v0.76.0

- Add `media._core` helpers and the Slack media implementation (#4029). (`198bac0`)

## v0.75.0

- **Media layer contract** (#4027): new `mltgnt.interfaces.media` with `Status` (`str` Enum: `RECEIVED` / `WORKING` / `DONE` / `FAILED` / `CANCELLED`), the `MediaClient` Protocol (`post` / `update` / `set_status` / `upload`) and `adapt_client(obj)`, plus the `mltgnt.media` package skeleton (`media._core.types`, `media._core.client`, `media._core.config.MediaConfig`), the `mltgnt[slack]` / `mltgnt[webchat]` extras and `.importlinter` layering for `media`. `PersonaScheduler(slack=...)` accepts a `MediaClient`. `SlackClientProtocol` (`post_message`) is deprecated: `adapt_client` wraps such a client and emits one `DeprecationWarning`; removal is planned for the next minor (Y) bump. (`af69590`)
- Use ASCII defaults for the `LanguagePack` media vocabulary (`approval_words`, `status_labels`, `enqueue_failed_text`, `progress_line_pattern`) (#4027). (`e9b739c`)
- Describe the `LanguagePack` media vocabulary defaults accurately in the changelog (#4027). (`a6abf56`)

## v0.74.0

- Expose public names and injection points for host-used private conversation helpers (#4024). (`9aa2519`)

## v0.73.0

- Rewrite the README for v0.72.0. (`dae389b`)

## v0.72.0

- Bump ghdag to v0.83.0. (`aea3fb4`)

## v0.71.0

- **Debounced git commits of memory files** (nexus #3833): `mltgnt.bridges.files_adapter.commit(paths, message, *, sink="memory", trailers=None)` passes through to the ghdag `ghdag.vcs` sink. `append_memory_entry`, `compact` (final write only), `write_dream` and `write_global` schedule a per-path debounced commit after a successful write; new `MemoryConfig.commit_debounce_sec` (default `300.0`, `<= 0` commits synchronously) and `mltgnt.memory.flush_memory_commits()` (also registered with `atexit`). Commit failures are logged and never affect the write, and nothing is committed unless `ENABLE_GIT` is truthy. (`3470731`)
- Pin ghdag v0.82.0 (the first release with `ghdag.vcs`). (`0791364`)

## v0.70.0

- **Agent work mode** (#3855): `AgentRunner` gains keyword arguments that all default to the previous behavior: `history_mode="full_trace"` (numbered tool trace plus `[REFLEXION] <feedback>` lines, old results folded past `history_max_chars`), `plan` (`mltgnt.agent.plan.Plan`, updated from `plan_update` and returned as `AgentResult.plan`), `max_reflexions` (stops with `REFLEXION_EXHAUSTED_TOOL`) and `step_hook(entry)`. New exports: `Plan`, `PlanItem`, `parse_plan`, `build_plan_prompt` and `DefaultReflexionEvaluator`. (`d1750a4`)

## v0.69.0

- Bump ghdag to v0.81.0. (`5cba7d0`)
- **`MemoryConfig.dream_engine`**: selects the LLM engine (`"claude"` / `"cursor"` / `"codex"`, default `"claude"`; empty means `"claude"`) used by the `memory_dream` schedule action, so cursor- or codex-only hosts no longer fail the daily dream job. `MemoryConfig.dream_model` now defaults to `""`: `claude` then falls back to `claude-haiku-4-5-20251001` (unchanged), other engines use their ghdag default, and an explicit value is passed through. (`e6636c8`)

## v0.68.0

- **Selectable engine for skill matcher LLM stages** (#3806): `match` / `match_pipeline` accept a keyword-only `engine` (default `"claude"`) used by both the agentic discover stage and the LLM intent-classification stage. `_DEFAULT_MATCHER_MODEL` applies only to claude; other engines get `model=None` unless `model` is given. `resolve_skill` gains `matcher_engine`, and the scheduler `enable_pipeline` path passes the job / persona engine to `match_pipeline`. (`ef5fe37`)
- Keep added comments and docs ASCII (no CJK in the public repository). (`093dea0`)
- CP2 review fixes: drop a duplicate CHANGELOG entry with full-width parentheses and use an ASCII arrow in a test docstring. (`5a33c17`)

## v0.67.1

- `tests/scheduler/test_chain_every_run.py::test_dependent_fires_again_on_next_upstream_run` no longer races the runner thread: it waits until no job is in `_running` before the second `tick` (#352). No runtime change. (`60e667a`)

## v0.67.0

- Rewrite the README for v0.66.0. (`cf3132a`)
- Align the README version with pyproject 0.67.0 (CP2 review fix). (`73e16b4`)

## v0.66.0

- **`MLTGNT_DEFAULT_ENGINE` host-wide default engine** (#3809): new `mltgnt.persona.schema.system_default_engine()` reads `MLTGNT_DEFAULT_ENGINE` on every call (unset/blank -> `SYSTEM_DEFAULT_ENGINE` = `"claude"`; values outside `VALID_ENGINES` raise `ValueError`). `run_persona_prompt`, `format_result_for_persona(engine="")` and `dispatch_decision._normalize_primary_engine_model` use it when no engine is given; an explicit engine always wins. `SYSTEM_DEFAULT_ENGINE` is kept unchanged for compatibility. (`65da7ff`)

## v0.65.0

- Bump ghdag to v0.80.0. (`96804e9`)

## v0.64.0

- Bump ghdag to v0.79.0. (`33ca4b7`)

## v0.63.0

- **`chain_every_run` for chained scheduler jobs** (#347): a `mode: chained` job with `chain_every_run: true` fires right after *every* successful run of its `depends_on` job (previously chained jobs fired once per day via date-marked done files, so chaining after an `interval` job never fired). The upstream job's output text is passed as `ScheduleJob.upstream_output`; `action: skill` appends it to the persona's user message. Such jobs never write done / skipped / failed marks and are never time-triggered by `tick()`. (`4204940`)

## v0.62.0

- Rewrite the README for v0.61.0. (`767679a`)
- Align README version pins with pyproject 0.62.0 (CP2). (`1f9ae9e`)

## v0.61.0

- Bump ghdag to v0.78.0. (`21d31e7`)

## v0.60.0

- Add structural convention tests (#3573). (`26c3c68`)

## v0.59.0

- Bump ghdag to v0.77.0. (`3943c38`)

## v0.58.0

- Bump ghdag to v0.76.0. (`2962ed2`)

## v0.57.0

- Bump ghdag to v0.75.0. (`ee8f3dc`)

## v0.56.0

- Bump ghdag to v0.74.0. (`b005f98`)

## v0.55.0

- Bump ghdag to v0.73.0. (`75cd1f5`)

## v0.54.0

- Bump ghdag to v0.72.0. (`842af34`)

## v0.53.0

- Bump ghdag to v0.71.0. (`3fc247d`)

## v0.52.0

- Bump ghdag to v0.70.0. (`ca7be7b`)

## v0.51.0

- Bump ghdag to v0.69.0. (`997a6fb`)

## v0.50.0

- Bump ghdag to v0.68.0. (`5959381`)

## v0.49.0

- Bump ghdag to v0.66.0. (`6eff144`)

## v0.48.1

- Bump ghdag to v0.65.1. (`0c5a910`)

## v0.48.0

- Bump ghdag to v0.65.0. (`7fe5c6d`)

## v0.47.0

- Bump the ghdag pin from v0.62.0 to v0.63.0. (`014782a`)

## v0.46.0

- Enforce a recursive CJK gate over the source tree. (`448cb0e`)
- Encode the excluded sample persona stem. (`19f55cc`)

## v0.45.0

- Remove CJK from structural source keys. (`44d7d9e`)

## v0.44.0

- **Normalize test fixtures** (#3383, #3337): replace CJK test data with ASCII `LanguagePack` fixtures; the tone cut in `format_persona_body` uses a generic pattern instead of a name-specific literal, and name-specific test data is replaced with synthetic identifiers. (`fa692b8`)
- Enforce a repository-wide CJK fixture gate in tests. (`51cccad`)

## v0.43.0

- Remove `docs/` (the Japanese documents `MLTGNT.md` / `improvement_hub.md`); design documents live only in nexus `docs/MLTGNT.md`. (`3f5d4df`)
- Remove `tools/` (the `mltgnt_fugu` prototype package) and `tests/tools/`; `tools/lint-boundary.sh` moves to `scripts/lint-boundary.sh` (the CI boundary policy lint is kept). (`3e7bde2`)
- **`LanguagePack` dataclass for locale-specific vocabulary** (#3382): `mltgnt.config.language` gains the frozen `LanguagePack` dataclass and a default `JA` instance. Hard-coded localized vocabulary in `deterministic_gate` / `persona/compress` / `persona/formatter` moves into `LanguagePack`, and a `pack=None` argument (`JA` when None) lets callers swap the locale. `persona/registry` `EXCLUDE_STEMS` becomes `frozenset()`, and `list_personas` / `resolve_with_alias` gain an `exclude_stems` argument. The duplicate `_EXCLUDE_PERSONA_STEMS` in `memory/dream/synthesizer` is removed, and `PersonaConfig` gains an `exclude_stems` field. (`c13b2d4`)
- Resolve the ruff SIM300 yoda condition in `test_config.py`. (`59604c3`)

## v0.42.1

- Bump the ghdag pin from v0.61.0 to v0.62.0. (`46ac7b9`)

## v0.42.0

- Translate mltgnt `src` docstrings, log messages and errors to English. (`bb7fe18`)

## v0.41.0

- **Removed unused modules** (#3321 / #3301 sub 6): `mltgnt.loops` / `mltgnt.ooda` / `mltgnt.improvement` / `mltgnt.kpi` / `mltgnt.chat` / `mltgnt.execution` are removed, together with `mltgnt.interfaces.loops` / `ooda` / `chat` (and the ooda-only `dispatch`), `LoopsConfig` / `ChatConfig` and the top-level `run_pipeline`. `BaseRunner` moves to `mltgnt.scheduler.base_runner`. The loops-only `enqueue_step` / `poll_step` are also removed. (`25d94cf`)
- Scrub nexus persona identifiers from docstrings and `EXCLUDE_STEMS`. (`60f30da`)

## v0.40.0

- Translate the test suite to English for OSS quality (#3339). (`0af5700`)
- Strip trailing whitespace in tests for ruff W291 (#3339). (`3935222`)
- Escape the `EXCLUDE_STEMS` sample name in tests for the leak scan (#3339). (`6430617`)

## v0.39.0

- Scrub host persona names from the persona formatter and tests. (`3383204`)

## v0.38.0

- Add media-independent persona and decision layers (#3318). (`595f035`)
- Collapse a nested `if` in the agent layer for ruff SIM102 (#3318). (`8fea525`)

## v0.37.0

- Add a media-independent conversation layer (#3317). (`4fd0606`)
- Fix mypy findings (narrow `Optional` in `session_store`, `thread_index` return type). (`50fc574`)

## v0.36.0

- **Waist contract `TurnInput` / `TurnResult` / `TurnHandler`** (#3286): media-independent boundary data types and the `TurnHandler` Protocol in `mltgnt.interfaces.turn`. The package exports `TurnInput` / `TurnResult` / `TurnHandler` / `Attachment` / `HistoryMessage`. Queue and ledger implementations are not brought in. (`a905123`)

## v0.35.0

- **Media-agnostic routing API** (#3285): new `resolve_persona(text, *, space_id, conversation_id, persona_map, pinned_personas)` and `find_observers_in_space`. `SpacePersonaEntry` is the canonical name and `ChannelPersonaEntry` stays as a backward-compatible alias. The old `resolve_responding_persona` / `find_observers` are deprecated (they are tied to Slack vocabulary such as `channel` / `thread_ts`) and kept as compatibility wrappers that emit `DeprecationWarning`; migrate to `resolve_persona` / `find_observers_in_space`. (`d4dbe26`)

## v0.34.1

- Pin ghdag to v0.61.0. (`197c577`)

## v0.34.0

- Pin ghdag to v0.55.0. (`2f324d3`)
- Align the fanout integration test with the ghdag v0.55.0 API. (`e37db85`)
## v0.33.0

### Changed

- **Pillar 5 back to opt-in** (#3179): V7 only checks types (it does not check that output artifacts exist). Diagnostic files are opt-in via `discover(diagnostics_dir=...)` (when unset, nothing is written to `{base}/_unresolved/`). Result frontmatter is opt-in via `action_args.result_frontmatter: true` (default is `run_result=None`)

## v0.20.0

### Added

- **loops Phase 3 (execution)**: `kind: action` subtasks and host-facing `ActionRequest` / `ActionResult` / `ActionExecutor`. Deterministic actions that match the public `action_schemas` run synchronously with an idempotency key
- **Persona memory**: an optional `MemoryAppender` appends short summaries on plan approval, iteration completion and done/failed (a dedupe key prevents double appends)
- **LLM / watch / replan budgets**: `llm_call_budget_per_loop` (default 200) / `llm_call_budget_per_day` (default 1000, shared per JST day) / `max_watch_subtasks_per_loop` (default 50) / `max_replans_per_loop` (default 20). Exceeding a budget moves the loop to `paused`; an exact-match "resume" message grants a `budget_override`
- **Events**: `action_executed` / `memory_appended` / `memory_append_failed` / `budget_resumed`

### Compatibility

- `schema_version: 1` is kept. Old state restores new fields with defaults. `ActionExecutor` / `MemoryAppender` are optional and do not break existing host construction
- Existing `auto` / `human` / `watch`, approval, comment dialogue and deliverable paths are non-breaking

## v0.19.6

### Added

- **loops Phase 2 (dialogue)**: inbox `kind=comment` is handled with a deterministic status check and LLM classification (`status` / `instruction` / `question` / `chitchat`). Progress queries are answered immediately with `post_progress`, change instructions go to the existing `replanning`, questions get a persona answer, and chitchat is appended to `clarification_context`
- **Settings**: `comment_model` / `max_comments_per_tick` (default 10) / `comment_reply_budget_per_hour` (default 10) / `comment_reply_max_chars` (default 800)
- **Events**: `comment_classified` (`source`: deterministic / llm / budget_fallback) / `comment_replied` / `comment_warning`
- **`render_progress_summary`**: human-readable progress summary without an LLM

### Compatibility

- `HumanChannel` and the Phase 1 watch / replan / approval gate are non-breaking. `schema_version: 1` is kept
- The old bulk `comment_received` supplement path is replaced by dialogue handling (supplements are appended only for chitchat or LLM failure)

## v0.19.5

### Added

- **loops Phase 1 (reactivity)**: `kind: watch` subtasks, a `depends` DAG, local `path_exists` / `path_changed` evaluation (`PathConditionEvaluator`), and host-facing `ConditionEvaluator` / `WatchVerdict`
- **Immediate replan on watch failure**: `replanning` state and `max_replans_per_iteration` (default 3). running / success must be kept
- **Plan approval gate**: Objective `plan_approval` (when unset, `LoopsConfig.plan_approval_default`, default true) and `awaiting_plan_approval`. Approval words are exact full-text matches of `ok` / "approve" / "proceed" / `go`. Human revisions are allowed up to `max_plan_revisions` (default 3) and do not consume `replan_count`
- **Settings**: `watch_root` / `max_replans_per_iteration` / `max_plan_revisions` / `plan_approval_default`
- **Events**: `watch_polled` / `replan_triggered` / `plan_proposed` / `plan_approved` / `plan_revised`

### Compatibility

- `schema_version: 1` is kept. v0.19.4 state (without the added keys) loads with defaults. A missing depends key is normalized to sequential dependencies
- The existing `auto` / `human` submit -> poll -> evaluate path is non-breaking. GitHub Issue/PR/label evaluation lives on the host side (#2585)

## v0.19.4

### Added

- **loops single deliverable contract**: `state_dir/<loop_id>/deliverable.md` is the canonical deliverable, initialized from the Objective body by `start_loop`. auto subtasks edit the same file step by step, and evaluate uses `result_summary` and a deliverable excerpt as input
- **`HumanChannel.post_progress` / `post_deliverable`**: host notification contract for plan, progress and deliverable notices (`progress_notify` can suppress progress only)
- **Observation events**: `state_change` / `question_asked` / `subtask_submitted` / `subtask_done` / `deliverable_updated`
- **`Subtask.result_summary` / `result_filename`**: backward-compatible fields for evaluation and notification (old state restores them as empty strings)

### Compatibility

- Existing `result` / `submission` / HumanChannel methods, state names and schema_version=1 are kept. The nexus-side Slack/diary implementation is #2582

## v0.19.3

### Fixed

- **Missing `close_thread` on loops failed termination**: when a loop moves to `failed` (e.g. after consecutive errors), `HumanChannel.close_thread` is always called through the same finalize path as done / cancelled, so no pending thread is left on the host side
- **Ingest inbox `kind: "comment"`**: user messages outside a pending question are appended to `clarification_context` as `Supplement: <text>` and a `comment_received` event is recorded (the same message_id is never consumed twice)

## v0.19.0

### BREAKING

- **Auto-start by placing an Objective is removed**: placing a `.md` in `objectives_dir` no longer creates loop state. Loops start only by consuming requests in `state_dir/requests/*.json`.
- **Migration order**: install this release (the mltgnt consumer) first, and switch the nexus-side request producer / Slack wiring (#2560) **afterwards**.

### Added

- **`ensure_frontmatter`**: deterministically fills only missing `id` / `title` / `status` / `max_iterations` (`agent` is not filled)
- **`mltgnt.loops.requests`**: validation and listing of start-request JSON, with isolation into `consumed/` / `corrupt/`
- **`LoopsEngine.start_loop(..., thread=)`**: the request thread (`HumanThreadRef`) is carried into the initial state. The existing `start_loop(objective)` stays compatible
- **`store.archive_terminal_state`**: moves terminal state to `state_dir/archive/` so a new request can start the loop again

### Compatibility

- Restoring non-terminal state, cancellation by deleting the Objective / `status: cancelled`, and the content hash change warning are kept.
- The public Protocols (`interfaces/loops.py`) and `LoopsConfig` fields are unchanged.

## v0.18.0

### Added

- **`mltgnt.loops`**: Objective-driven loop execution (clarify -> decompose -> execute -> evaluate)
- **`LoopsComponent`**: Objective snapshot polling compliant with `DaemonComponent` (default 10 seconds)
- **`LoopsConfig`**: objectives/state/status/jobs paths, LLM/subtask engines, limits
- **`HumanChannel` / `SubtaskExecutor` Protocol**: Slack/ghdag implementations live on the host (nexus #2512) side
- **`enqueue_step` / `poll_step`**: non-blocking subtask enqueue and completion check in ghdag_bridge
- **status Markdown**: writes the human-readable current state to `<status_dir>/<loop_id>.md`

### Compatibility

- Backward compatible. No changes to the existing scheduler / chat / OODA APIs.
- nexus host wiring is implemented separately in #2512.

### Operational limits

- `max_iterations`: 1..10 (default 5)
- `max_clarify_rounds`: 1..3 (default 3)
- `max_subtasks_per_iteration`: 1..5 (default 5)
- `subtask_timeout_sec`: 1800 seconds (30 minutes)

## Phase Progress

### Phase D: exit_code routing ✓
- SkillRunResult.exit_code -> ExitStatus enum conversion implemented
- scheduler permission pass-through (v0.15.1)
- ⚠️ exit_code propagation to enqueue_dag() child tasks is not supported yet (#2235)

### Phase E: side_effects audit ⚠️ In Progress
- SkillMeta.side_effects declaration exists. The measuring audit wrapper is not implemented (#2234)
- BaseRunner ABC extraction and ActDispatcher Protocol unification (v0.16.0)
- Removal of the deprecated compact() / needs_compaction() public APIs (planned)

### Phase F: pipe composition runtime ⚠️ Not Started
- typecheck_dag() exists but skips everything when skill_io != "v1" (silent compatibility mode)
- Making skill_io: v1 explicit and enforcing type checks has not started

## v0.15.1

### Added

- scheduler permission pass-through: `action_args.permission` is passed through `enqueue_and_wait` to `StepConfig.permission`

## v0.10.0

### Removed (BREAKING): deprecated APIs

The following APIs, which emitted DeprecationWarning in v0.9.x, have been removed.

**chat module**
- `mltgnt.chat.models` -> import directly from `mltgnt.interfaces.types`
- `mltgnt.chat.run_chat()` -> use `run_pipeline()`

**memory module**
- `mltgnt.memory.read_memory_agentic()` -> use `read_memory_iterative()`
- `mltgnt.memory._compaction` -> import directly from `mltgnt.memory.compaction`
- `mltgnt.memory.api.normalize_source_prefix()` -> removed (inline it at the call site)

**persona / agent modules**
- `mltgnt.agent._parse` acceptance of JSON without an args key -> the `{"tool": str, "args": dict}` form is required
- flat keys (`chat_model`, `slack`) in `mltgnt.persona.schema` -> use the `ops:` namespace
- `ops.chat_model` in `mltgnt.persona.schema` -> use `ops.engine` / `ops.model`
- `Persona.WEIGHT_MAP` / `Persona.ops_config` / `Persona.slack_post_kwargs()` / `Persona.delegate_ack()` -> removed
- `legacy_keys` warning in `validate_persona()` / `validate_fm()` -> removed

**scheduler module**
- `mltgnt.scheduler.ghdag_bridge` -> import directly from `mltgnt.bridges.ghdag_bridge`
