# mltgnt

**L1 agent runtime for host-integrated operations.** In the L0 / L1 / L2 stack (**L0 [ghdag](https://github.com/sumipan/ghdag) / L1 mltgnt / L2 host**), mltgnt owns type contracts, persona and memory orchestration, and loop boundaries. ghdag owns DAG transport and LLM adapter wiring; the host owns process lifecycle and channel I/O.

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.30.0)-orange)

## Not (what this is not)

| Item | Why |
|------|-----|
| Not an LLM SDK | mltgnt does not call model providers directly; LLM calls go through ghdag adapters. |
| Not a DAG engine | Task scheduling, queueing, and DAG state management are owned by ghdag. |
| Not a host runtime | Slack / CLI host process management and deployment are L2 responsibilities. |

## Installation

Requires Python `>=3.10`.

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.30.0"
```

| Item | Value |
|------|-------|
| Package | `mltgnt` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.43.0` |
| Console script | `mltgnt = mltgnt.cli.main:main` |
| License | MIT |

Dev extras (`pytest`, `pytest-asyncio`, `pytest-cov`, `freezegun`, `import-linter`, `mypy`, `ruff`):

```bash
pip install "mltgnt[dev] @ git+https://github.com/sumipan/mltgnt.git@v0.30.0"
```

## Quick Start

Examples use symbols from `mltgnt.__all__` and match patterns covered by `tests/`.

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
  model: claude-sonnet-5
---

## Background
A curious multi-legged AI tank.
"""

with tempfile.TemporaryDirectory() as tmp:
    persona_dir = Path(tmp)
    (persona_dir / "Tachikoma.md").write_text(persona_md, encoding="utf-8")
    persona = load_persona("Tachikoma", persona_dir=persona_dir)
    print(persona.name)
```

### Write and read a dream summary

```python
from pathlib import Path
import tempfile

from mltgnt import DreamSection, DreamSummary, read_dream, write_dream

with tempfile.TemporaryDirectory() as tmp:
    persona_dir = Path(tmp) / "Tachikoma"
    persona_dir.mkdir()
    summary = DreamSummary(
        persona="Tachikoma",
        sections=[
            DreamSection(
                category="facts",
                content="Likes curiosity.",
                source_entries=1,
            )
        ],
        updated_at="2026-09-11T00:00:00+09:00",
    )
    write_dream(persona_dir, summary)
    loaded = read_dream(persona_dir)
    assert loaded is not None
    print(loaded.sections[0].category, loaded.sections[0].content)
```

### Run the chat pipeline (requires a live ghdag LLM path)

```python
from pathlib import Path
import tempfile

from mltgnt import load_persona, run_pipeline

persona_md = """\
---
persona:
  name: Tachikoma
ops:
  engine: claude
  model: claude-sonnet-5
---

## Style
Answer in one short paragraph.
"""

with tempfile.TemporaryDirectory() as tmp:
    persona_dir = Path(tmp)
    (persona_dir / "Tachikoma.md").write_text(persona_md, encoding="utf-8")
    persona = load_persona("Tachikoma", persona_dir=persona_dir)
    out = run_pipeline(
        "Say hello from mltgnt.",
        persona,
        engine="claude",
        model="claude-sonnet-5",
    )
    print(out.persona_name, out.content)
```

## CLI Reference

Entry point: `mltgnt` → `mltgnt.cli.main:main` (argparse).

### `mltgnt run`

Start the daemon with a host-provided component factory.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--components` | yes | — | Component factory as `MODULE:FUNCTION` |
| `--pid-file` | no | `/tmp/mltgnt_daemon.pid` | PID lock file path |

| Exit code | Meaning |
|-----------|---------|
| `0` | Daemon started |
| `1` | Generic `MltgntError` |
| `2` | `ConfigError` (invalid `--components`, missing module/function, non-callable) |
| `3` | `DependencyError` (PID lock or other blocked dependency) |

### `mltgnt memory dream show`

Print dream summary sections for one persona.

| Argument | Required | Description |
|----------|----------|-------------|
| `persona` | yes | Persona name / stem |
| `--chat-dir` | yes | Parent directory that contains persona subdirectories |

| Exit code | Meaning |
|-----------|---------|
| `0` | Sections printed, or no dream summary found (informational message on stdout) |

### `mltgnt memory dream forget`

Remove one dream summary category for a persona.

| Argument | Required | Description |
|----------|----------|-------------|
| `persona` | yes | Persona name / stem |
| `--category` | yes | Category name to remove |
| `--chat-dir` | yes | Parent directory that contains persona subdirectories |

| Exit code | Meaning |
|-----------|---------|
| `0` | Category removed |
| `1` | No dream summary, or category not found (message on stderr) |

## Public API

Stable public surface: `mltgnt.__all__` (**24** symbols, including `__version__`).

| Symbol | Signature / shape | Purpose |
|--------|-------------------|---------|
| `run_pipeline` | `(prompt, persona, *, engine='', model='', timeout=300, memory=None, orchestration_ctx=None, audit_path=None) -> ChatOutput` | Run one chat turn through the L1 pipeline |
| `read_memory_iterative` | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call, skill_paths=None, max_iterations=3) -> str` | Iterative memory retrieval with an LLM sufficiency loop |
| `read_memory_by_relevance` | `(config, persona_stem, query, *, max_bytes, max_entries, layers=None) -> str` | Retrieve memory entries ranked by relevance |
| `read_memory_with_sufficiency_check` | `(config, persona_stem, query, *, max_bytes, max_entries, llm_call=None) -> str` | Retrieve memory with an optional sufficiency check |
| `DreamSection` | `(category, content, source_entries)` | One category block inside a dream summary |
| `DreamSummary` | `(persona, sections, updated_at)` | Aggregated dream summary for a persona |
| `read_dream` | `(persona_dir, *, memory_dir_name='memory') -> DreamSummary \| None` | Load a persona dream summary from disk |
| `write_dream` | `(persona_dir, summary, *, memory_dir_name='memory') -> None` | Persist a persona dream summary to disk |
| `Persona` | dataclass (`name`, `fm`, `sections`, `body`, `path`, …) | Loaded persona document and frontmatter |
| `load_persona` | `(name, *, persona_dir=None, config=None) -> Persona` | Load a persona by name or alias |
| `list_personas` | `(persona_dir=None) -> list[str]` | List available persona stems |
| `validate_persona` | `(persona, *, available_skills=None) -> list[str]` | Validate persona frontmatter/body; returns warnings |
| `run_persona_prompt` | `(persona_name, prompt, persona_dir=None, timeout=120, memory=None) -> str` | Render and run a persona prompt |
| `ChatInput` | dataclass | Chat pipeline input DTO |
| `ChatOutput` | dataclass | Chat pipeline output DTO |
| `Message` | `TypedDict` with `role`, `content` | Single chat message record |
| `PersonaProtocol` | Protocol (`name`, `fm`, `format_prompt`, `register_prompt_filter`) | Minimal persona contract for runtime APIs |
| `AgentResult` | dataclass | Result of one agent action classification |
| `AgentRunner` | class | Agent orchestration runner |
| `enqueue_dag` | `(steps, timeout, idempotency_key, jobs_dir, exec_done_dir, ...) -> list[tuple[bool, str]]` | Enqueue a ghdag job without waiting |
| `enqueue_and_wait` | `(prompt, engine, model, timeout, idempotency_key, jobs_dir, exec_done_dir, ...) -> tuple[bool, str]` | Enqueue a ghdag job and wait for completion |
| `PersonaScheduler` | class | Schedule persona-driven actions |
| `ScheduleJob` | dataclass | One scheduled job record |
| `__version__` | `str` | Installed package version string |

## Protocols / Extension Points

Host-facing contracts you can implement without depending on L3 concretes:

| Contract | Module | Purpose |
|----------|--------|---------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Minimal persona identity + prompt formatting |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | Host chat pipeline (`ChatInputBase` → `ChatOutputBase`) |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | Slack post boundary (returns `bool`, does not raise) |
| `DaemonComponent` | `mltgnt.daemon` | Daemon lifecycle (`start` / `stop` / `name`) |
| `HumanChannel` | `mltgnt.interfaces.loops` | Thread open / ask / notify / progress / deliverable |
| `SubtaskExecutor` | `mltgnt.interfaces.loops` | Async submit / poll for auto subtasks |
| `ConditionEvaluator` | `mltgnt.interfaces.loops` | Watcher condition evaluation |
| `ActionExecutor` | `mltgnt.interfaces.loops` | Deterministic side-effect execution |
| `MemoryAppender` | `mltgnt.interfaces.loops` | Optional host sink for memory append events |
| `ObserveSource` | `mltgnt.interfaces.ooda` | OODA observe-phase event source |
| `ActDispatcher` | `mltgnt.interfaces.dispatch` | OODA act-phase dispatch boundary |

## Architecture

Top-level packages under `src/mltgnt/` (**17** directories):

| Path | Responsibility |
|------|----------------|
| `agent/` | `AgentRunner` orchestration and action classification |
| `bridges/` | ghdag integration (audit, files, hooks, LLM, DAG enqueue) |
| `chat/` | Chat pipeline (`run_pipeline`) |
| `cli/` | CLI entry points (`run`, `memory dream show\|forget`) |
| `config/` | Runtime configuration dataclasses |
| `daemon/` | Daemon lifecycle (`DaemonRunner`, `PidLock`, skill watcher) |
| `execution/` | Shared execution runner base interfaces |
| `improvement/` | Self-improvement loop (analyzer, hub, patch, rollback) |
| `interfaces/` | Type contracts and host boundary protocols |
| `kpi/` | KPI calculation and reporting |
| `loops/` | Objective loops (engine, budget, conditions, store) |
| `memory/` | Memory retrieval, compaction, and dream summaries |
| `ooda/` | OODA orchestration (audit source, exec dispatcher, runner) |
| `persona/` | Persona loading, validation, and helpers |
| `routing/` | Channel routing and agentic triage |
| `scheduler/` | `PersonaScheduler` and schedule actions |
| `skill/` | Skill loading, matching, linting, and execution |

Shared exception module (package root, not a directory): `exceptions.py` → `MltgntError`, `ConfigError`, `DependencyError`.

### Layer structure (import-linter)

From `.importlinter`:

```
daemon | loops
scheduler | agent | routing
persona | chat | skill | memory
bridges
interfaces
```

Contracts:

- **Layered architecture**: upper layers may depend on lower layers; reverse imports are forbidden.
- **L3 domain isolation**: `persona`, `chat`, `skill`, `memory`, and `loops` must not import `ghdag` directly (bridges mediate access).

## Configuration

### Environment variables

Verified via `os.environ.get` / `os.getenv` under `src/mltgnt/`:

| Variable | Used in | Meaning |
|----------|---------|---------|
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Diary root path for template substitutions |
| `REPO_ROOT` | `mltgnt.skill.runner` | Repository root path for template substitutions |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge` | Skill I/O type checking is enabled unless set to `"0"` |
| `MLTGNT_AS_OF_DATE` | `mltgnt.improvement.loop` | Optional `YYYY-MM-DD` override for improvement-cycle "today" |

## Error Reference

Public types from `mltgnt.exceptions.__all__`:

```text
MltgntError
├── ConfigError
└── DependencyError
```

| Type | Bases | Notes |
|------|-------|-------|
| `MltgntError` | `Exception` | Package base; catch with `except MltgntError` |
| `ConfigError` | `MltgntError` | YAML / configuration load or parse failure |
| `DependencyError` | `MltgntError` | External dependency (callable / subprocess / API) failure |

Related types outside `exceptions.py` (not in that `__all__`):

| Type | Module | Bases | Notes |
|------|--------|-------|-------|
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | Persona frontmatter validation failure |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | Memory compaction LLM call failure |

## Public API Stability

mltgnt is pre-1.0 (`0.Y.Z`):

- `Y` increments may include breaking API changes.
- `Z` increments are intended to be backward-compatible fixes or additions.
- The documented stable surface is `mltgnt.__all__`, plus host contracts under `mltgnt.interfaces`.

## License

MIT (SPDX: `MIT`, matching `license = "MIT"` in `pyproject.toml`).

- Source: https://github.com/sumipan/mltgnt
- Issues: https://github.com/sumipan/mltgnt/issues
- L0 runtime: https://github.com/sumipan/ghdag
