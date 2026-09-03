# mltgnt

**L1 agent runtime for host-integrated operations.** In the L0/L1/L2 stack (**L0 ghdag / L1 mltgnt / L2 host**), mltgnt defines type contracts, loop transitions, and orchestration boundaries while ghdag owns DAG transport and model execution wiring.

![Status](https://img.shields.io/badge/status-Pre--1.0%20(v0.24.0)-orange)

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
| Runtime dependencies | `PyYAML>=6.0`, `scikit-learn>=1.0`, `numpy>=1.21`, `ghdag @ git+https://github.com/sumipan/ghdag.git@v0.34.5` |
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
title: Rewrite README for v0.24.0
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

| Command | Description | Required arguments |
|---------|-------------|--------------------|
| `mltgnt run` | Start the daemon runner with your component factory | `--components MODULE:FUNCTION` |
| `mltgnt memory dream show` | Print dream summary sections for one persona | `persona`, `--chat-dir` |
| `mltgnt memory dream forget` | Remove one dream summary category | `persona`, `--category`, `--chat-dir` |

## Public API

### `mltgnt.__all__` (24 symbols)

`run_pipeline`, `read_memory_iterative`, `read_memory_by_relevance`, `read_memory_with_sufficiency_check`, `DreamSection`, `DreamSummary`, `read_dream`, `write_dream`, `Persona`, `load_persona`, `list_personas`, `validate_persona`, `run_persona_prompt`, `ChatInput`, `ChatOutput`, `Message`, `PersonaProtocol`, `AgentResult`, `AgentRunner`, `enqueue_dag`, `enqueue_and_wait`, `PersonaScheduler`, `ScheduleJob`, `__version__`

## Protocols / Extension Points

| Contract | Module | Purpose |
|----------|--------|---------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | Minimal contract for prompt formatting and persona identity used by runtime APIs. |
| `HumanChannel` | `mltgnt.interfaces.loops` | Host-side callbacks for thread open/ask/notify/progress/deliverable. |
| `SubtaskExecutor` | `mltgnt.interfaces.loops` | Async submit/poll boundary for auto subtasks. |
| `ConditionEvaluator` | `mltgnt.interfaces.loops` | Host-defined watcher condition evaluation. |
| `ActionExecutor` | `mltgnt.interfaces.loops` | Deterministic side-effect execution boundary. |
| `MemoryAppender` | `mltgnt.interfaces.loops` | Optional host sink for memory append events. |

## Architecture

Top-level subpackages under `src/mltgnt/`:

| Path | Responsibility |
|------|----------------|
| `agent/` | `AgentRunner` orchestration and action classification |
| `bridges/` | ghdag integration (`audit`, `files`, `hooks`, `llm`, DAG bridge) |
| `chat/` | Chat pipeline (`run_pipeline`) |
| `cli/` | CLI entry points (`run`, `memory`) |
| `config/` | Runtime configuration models and defaults |
| `daemon/` | Daemon lifecycle (`PidLock`, skill watcher) |
| `execution/` | Shared execution runner base interfaces |
| `improvement/` | Improvement loop (`analyzer`, `hub`, `patch`, `rollback`) |
| `interfaces/` | Type contracts (`ChatInput`, `ChatOutput`, `PersonaProtocol`) |
| `kpi/` | KPI calculation and reporting |
| `loops/` | Objective loops (`engine`, budget, conditions, request/store) |
| `memory/` | Memory retrieval, compaction, and dream summaries |
| `ooda/` | OODA orchestration (`audit_source`, exec dispatcher) |
| `persona/` | Persona loading, validation, and compression helpers |
| `routing/` | Channel routing and agentic triage |
| `scheduler/` | `PersonaScheduler` and dream/skill schedule actions |
| `skill/` | Skill loading, matching, linting, and execution |

## Configuration

### Environment variables

| Variable | Used in | Meaning |
|----------|---------|---------|
| `NIKKI_ROOT` | `mltgnt.skill.runner` | Root fallback path for diary-style substitutions. |
| `REPO_ROOT` | `mltgnt.skill.runner` | Repository root fallback for substitutions. |
| `SKILL_IO_TYPECHECK` | `mltgnt.bridges.ghdag_bridge` | Enables skill I/O type checking unless set to `"0"`. |
| `MLTGNT_AS_OF_DATE` | `mltgnt.improvement.loop` | Optional `YYYY-MM-DD` override for improvement cycle "today". |

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
| `MltgntError` | `mltgnt.exceptions` | Base package exception type. |
| `ConfigError` | `mltgnt.exceptions` | Invalid configuration or component wiring. |
| `DependencyError` | `mltgnt.exceptions` | Missing or blocked external dependency. |
| `PersonaValidationError` | `mltgnt.persona` | Persona frontmatter validation failure. |
| `LlmCallError` | `mltgnt.memory.compaction` | Memory compaction LLM call failure. |
| `LlmCallError` | `mltgnt.loops.prompts` | Prompt rendering / JSON contract failure in loops. |
| `SkillLoadError` | `mltgnt.skill.models` | Skill metadata/schema loading failure. |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | Skill I/O contract mismatch during type checking. |
| `ObjectiveError` | `mltgnt.loops.objective` | Dataclass result for invalid objective files. |
| `RequestError` | `mltgnt.loops.requests` | Dataclass result for invalid start-request files. |
| `BudgetExceeded` | `mltgnt.loops.engine` | Budget guard raised during loop execution. |

## Public API Stability

mltgnt is pre-1.0 (`0.Y.Z`):

- `Y` increments may include breaking API changes.
- `Z` increments are intended to be backward-compatible fixes or additions.
- The documented stable surface is `mltgnt.__all__`, plus explicitly exported subpackage symbols and host contracts in `mltgnt.interfaces`.

## License

MIT (`license = "MIT"` in `pyproject.toml`).

- Source: https://github.com/sumipan/mltgnt
- Issues: https://github.com/sumipan/mltgnt/issues
- L0 runtime: https://github.com/sumipan/ghdag
