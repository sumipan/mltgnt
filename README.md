# mltgnt

**L1 agent runtime for host-integrated operations.** In the L0/L1/L2 stack (**L0 ghdag / L1 mltgnt / L2 host**), mltgnt defines type contracts, loop transitions, and orchestration boundaries while ghdag owns DAG transport and model execution wiring.

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.25.0)-orange)

## Not (what this is not)

| Item | Why |
|------|-----|
| Not an LLM SDK | mltgnt does not call model providers directly; LLM calls go through ghdag adapters. |
| Not a DAG engine | Task scheduling, queueing, and DAG state management are owned by ghdag. |
| Not a host runtime | Slack/CLI host process management and deployment concerns are L2 responsibilities. |

## Installation

| Item | Value |
|------|-------|
| Package | `pip install mltgnt` |
| Python | `>=3.10` |
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.39.1` |
| Script entry point | `mltgnt = mltgnt.cli.main:main` |
| License | MIT |

## Quick Start

### 1) Parse an objective file

```python
from pathlib import Path
import tempfile

from mltgnt.loops import Objective, ObjectiveError, parse_objective

objective_md = """\
```yaml
id: release-readme
title: Rewrite README for v0.25.0
agent: operator
max_iterations: 3
status: active
```

Draft and verify a new README.
"""

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "release-readme.md"
    path.write_text(objective_md, encoding="utf-8")
    result = parse_objective(path, default_persona="operator", default_max_iterations=5)
    if isinstance(result, Objective):
        print(result.loop_id, result.agent, result.max_iterations)
    else:
        assert isinstance(result, ObjectiveError)
        print(result.message)
```

### 2) Load a persona

```python
from pathlib import Path
import tempfile

from mltgnt import load_persona

persona_md = """\
```yaml
persona:
  name: Tachikoma
ops:
  engine: claude
  model: claude-sonnet-5
```

## Background
A curious multi-legged AI tank.
"""

with tempfile.TemporaryDirectory() as tmp:
    persona_dir = Path(tmp)
    (persona_dir / "Tachikoma.md").write_text(persona_md, encoding="utf-8")
    persona = load_persona("Tachikoma", persona_dir=persona_dir)
    print(persona.name)
```

### 3) Run one chat pipeline call

```python
from pathlib import Path
import tempfile

from mltgnt import load_persona, run_pipeline

persona_md = """\
```yaml
persona:
  name: Tachikoma
ops:
  engine: claude
  model: claude-sonnet-5
```

## Style
Answer in one short paragraph.
"""

with tempfile.TemporaryDirectory() as tmp:
    persona_dir = Path(tmp)
    (persona_dir / "Tachikoma.md").write_text(persona_md, encoding="utf-8")
    persona = load_persona("Tachikoma", persona_dir=persona_dir)
    out = run_pipeline("Say hello from mltgnt.", persona, engine="claude", model="claude-sonnet-5")
    print(out.persona_name, out.content)
```

## CLI Reference

### `mltgnt run`

Start the daemon runner with a host-provided component factory.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--components` | yes | — | Component factory as `module.path:function_name` |
| `--pid-file` | no | `/tmp/mltgnt_daemon.pid` | Path to the PID lock file |

| Exit code | Meaning |
|-----------|---------|
| `0` | Daemon started successfully |
| `1` | Generic `MltgntError` |
| `2` | `ConfigError` (invalid `--components`, missing module/function, non-callable factory) |
| `3` | `DependencyError` (PID lock failure or other blocked external dependency) |

### `mltgnt memory dream show`

Print dream summary sections for one persona.

| Argument | Required | Description |
|----------|----------|-------------|
| `persona` | yes | Persona name/stem |
| `--chat-dir` | yes | Parent directory containing persona subdirectories |

| Exit code | Meaning |
|-----------|---------|
| `0` | Sections printed, or no dream summary found (informational message on stdout) |

### `mltgnt memory dream forget`

Remove one dream summary category for a persona.

| Argument | Required | Description |
|----------|----------|-------------|
| `persona` | yes | Persona name/stem |
| `--category` | yes | Category name to remove |
| `--chat-dir` | yes | Parent directory containing persona subdirectories |

| Exit code | Meaning |
|-----------|---------|
| `0` | Category removed successfully |
| `1` | No dream summary found, or category not found (message on stderr) |

## Public API

The stable public surface is `mltgnt.__all__` (24 symbols):

| Symbol | Module | Description |
|--------|--------|-------------|
| `run_pipeline` | `mltgnt.chat.pipeline` | Run one chat turn through the L3 pipeline |
| `read_memory_iterative` | `mltgnt.memory` | Iteratively retrieve memory entries |
| `read_memory_by_relevance` | `mltgnt.memory` | Retrieve memory entries ranked by relevance |
| `read_memory_with_sufficiency_check` | `mltgnt.memory` | Retrieve memory with sufficiency gating |
| `DreamSection` | `mltgnt.memory.dream` | One category block inside a dream summary |
| `DreamSummary` | `mltgnt.memory.dream` | Aggregated dream summary for a persona |
| `read_dream` | `mltgnt.memory.dream` | Load a persona dream summary from disk |
| `write_dream` | `mltgnt.memory.dream` | Persist a persona dream summary to disk |
| `Persona` | `mltgnt.persona` | Loaded persona document and frontmatter |
| `load_persona` | `mltgnt.persona` | Load a persona by name from a directory |
| `list_personas` | `mltgnt.persona` | List available persona stems in a directory |
| `validate_persona` | `mltgnt.persona` | Validate persona frontmatter and body |
| `run_persona_prompt` | `mltgnt.persona` | Render and run a persona prompt template |
| `ChatInput` | `mltgnt.interfaces.types` | Chat input dataclass |
| `ChatOutput` | `mltgnt.interfaces.types` | Chat output dataclass |
| `Message` | `mltgnt.interfaces.types` | Single chat message record |
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Minimal persona contract for runtime APIs |
| `AgentResult` | `mltgnt.agent` | Result of one agent action classification |
| `AgentRunner` | `mltgnt.agent` | Agent orchestration runner |
| `enqueue_dag` | `mltgnt.bridges.ghdag_bridge` | Enqueue a ghdag job without waiting |
| `enqueue_and_wait` | `mltgnt.bridges.ghdag_bridge` | Enqueue a ghdag job and wait for completion |
| `PersonaScheduler` | `mltgnt.scheduler` | Schedule persona-driven actions |
| `ScheduleJob` | `mltgnt.scheduler` | One scheduled job record |
| `__version__` | `mltgnt` | Installed package version string |

## Protocols / Extension Points

| Contract | Module | Purpose |
|----------|--------|---------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Minimal contract for prompt formatting and persona identity |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | Host-implemented L1 chat pipeline (`ChatInputBase` → `ChatOutputBase`) |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | Slack message posting boundary (returns `bool`, does not raise) |
| `DaemonComponent` | `mltgnt.daemon` | Daemon lifecycle component (`start` / `stop` / `name`) |
| `HumanChannel` | `mltgnt.interfaces.loops` | Host callbacks for thread open/ask/notify/progress/deliverable |
| `SubtaskExecutor` | `mltgnt.interfaces.loops` | Async submit/poll boundary for auto subtasks |
| `ConditionEvaluator` | `mltgnt.interfaces.loops` | Host-defined watcher condition evaluation |
| `PathConditionEvaluator` | `mltgnt.loops.conditions` | Deterministic `path_exists` / `path_changed` evaluation under a root |
| `ActionExecutor` | `mltgnt.interfaces.loops` | Deterministic side-effect execution boundary |
| `MemoryAppender` | `mltgnt.interfaces.loops` | Optional host sink for memory append events |
| `ObserveSource` | `mltgnt.interfaces.ooda` | OODA observe-phase event source |

## Architecture

Top-level subpackages under `src/mltgnt/`:

| Path | Responsibility |
|------|----------------|
| `agent/` | `AgentRunner` orchestration and action classification |
| `bridges/` | ghdag integration (`audit`, `files`, `hooks`, `llm`, DAG bridge) |
| `chat/` | Chat pipeline (`run_pipeline`) |
| `cli/` | CLI entry points (`run`, `memory`) |
| `config/` | Runtime configuration models and defaults |
| `daemon/` | Daemon lifecycle (`PidLock`, `SkillWatcherComponent`) |
| `execution/` | Shared execution runner base interfaces |
| `improvement/` | Improvement loop (`analyzer`, `hub`, `patch`, `rollback`) |
| `interfaces/` | Type contracts and host boundary protocols |
| `kpi/` | KPI calculation and reporting |
| `loops/` | Objective loops (`engine`, `budget`, `conditions`, `requests`, `store`) |
| `memory/` | Memory retrieval, compaction, and dream summaries |
| `ooda/` | OODA orchestration (`audit_source`, exec dispatcher) |
| `persona/` | Persona loading, validation, and compression helpers |
| `routing/` | Channel routing and agentic triage |
| `scheduler/` | `PersonaScheduler` and dream/skill schedule actions |
| `skill/` | Skill loading, matching, linting, and execution |

### Layer structure (import-linter)

```
daemon | loops
scheduler | agent | routing
persona | chat | skill | memory
bridges
interfaces
```

Additional contracts enforced by import-linter:

- **Layered architecture**: upper layers may depend on lower layers; reverse imports are forbidden.
- **L3 domain isolation**: `persona`, `chat`, `skill`, `memory`, and `loops` must not import `ghdag` directly (bridges mediate all ghdag access).

### v0.25.0 highlights

- `loops/budget.py`: JST daily shared LLM budget with `BudgetReserveResult` and `BudgetExceeded`
- `loops/conditions.py`: `PathConditionEvaluator` for local path-based watcher conditions
- ghdag dependency pinned to `v0.39.1`

## Configuration

### Environment variables

| Variable | Used in | Meaning |
|----------|---------|---------|
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Diary root path for template substitutions |
| `REPO_ROOT` | `mltgnt.skill.runner` | Repository root path for template substitutions |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge` | Enables skill I/O type checking unless set to `"0"` |
| `MLTGNT_AS_OF_DATE` | `mltgnt.improvement.loop` | Optional `YYYY-MM-DD` override for improvement cycle "today" |

## Error Reference

### `mltgnt.exceptions` hierarchy

```text
MltgntError
|- ConfigError
`- DependencyError
```

### Publicly relevant error and exception types

| Type | Module | Notes |
|------|--------|-------|
| `MltgntError` | `mltgnt.exceptions` | Base package exception type |
| `ConfigError` | `mltgnt.exceptions` | Invalid configuration or component wiring |
| `DependencyError` | `mltgnt.exceptions` | Missing or blocked external dependency |
| `PersonaValidationError` | `mltgnt.persona` | Persona frontmatter validation failure |
| `LlmCallError` | `mltgnt.memory.compaction` | Memory compaction LLM call failure |
| `LlmCallError` | `mltgnt.loops.prompts` | Prompt rendering / JSON contract failure in loops |
| `SkillLoadError` | `mltgnt.skill.models` | Skill metadata/schema loading failure |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | Skill I/O contract mismatch during type checking |
| `BudgetExceeded` | `mltgnt.loops.engine` | Budget guard raised during loop execution |

## Public API Stability

mltgnt is pre-1.0 (`0.Y.Z`):

- `Y` increments may include breaking API changes.
- `Z` increments are intended to be backward-compatible fixes or additions.
- The documented stable surface is `mltgnt.__all__`, plus explicitly exported subpackage symbols and host contracts in `mltgnt.interfaces`.

## License

MIT (SPDX: `MIT`, matching `license = "MIT"` in `pyproject.toml`).

- Source: https://github.com/sumipan/mltgnt
- Issues: https://github.com/sumipan/mltgnt/issues
- L0 runtime: https://github.com/sumipan/ghdag
