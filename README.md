# mltgnt

> **Persona-first multi-agent runtime: write personas in Markdown, give them skills, memory, and a place to live.**

`mltgnt` is a Python library for running long-lived **personas** — Markdown files with a YAML frontmatter that describe a character, what skills they have, and where they reply. Drop a persona file into a folder, and a host (a Slack bot, a CLI, a scheduled job) can load it, route a message to the right one, recall what was said before, and run the right skill — all through small, typed Protocols you implement once. mltgnt handles the persona format, the skill matcher, the memory store, and the channel router; **[ghdag](https://github.com/sumipan/ghdag) handles the LLM calls and the pipeline**, and is a hard prerequisite installed via git.

Status: pre-1.0 (`v0.5.6`). Pin a version; the public surface is still moving.

---

## 1. What you get

- **Many personas, one channel.** Multiple personas can coexist in the same Slack workspace. Routing is deterministic — by primary channel, by secondary observer role, by nickname mention, or by a thread that's already pinned to a persona. Misconfigured routing (two primaries on one channel) fails fast at startup.
- **A backbone for each persona.** A persona isn't just a prompt template. It comes with **skills** it can execute (`SKILL.md` files matched by slash command, trigger phrase, or LLM intent classification) and a **memory** of past conversations (per-persona JSONL with TF-IDF retrieval, an LLM sufficiency check, and size-triggered compaction).
- **Hosts plug in via Protocols.** The same persona file works in Slack, in a CLI, or in a scheduled job. The library defines four small Protocols — `SlackClientProtocol`, `PersonaProtocol`, `ChatPipelineProtocol`, `SkillLoaderProtocol` — and the host implements them. There's no framework to live inside.

---

## 2. A persona file

A persona is one Markdown file. The YAML frontmatter is small; the body is freeform sections that mltgnt picks up by heading.

```markdown
---
spec_version: 2.2.0
persona:
  name: Maya
  aliases: [maya, mayachan]
  description: Friendly journaling coach who values action over reflection.
ops:
  engine: claude
  model: claude-sonnet-4-6
  skills: [daily-journal, weekly-review]
  slack:
    username: Maya
    icon_emoji: ":notebook:"
---
# Maya — journaling coach

## Light
A patient coach who turns vague feelings into a single concrete next action.
Tone: first-person "I", short sentences, one open question per turn.

## Heavy

### Background
Former software engineer turned coach. Believes "60% and moving" beats "100% and stuck".

### Values
1. Continuity of action over perfection.
2. Concrete facts before interpretation.
3. Self-kindness — harsh self-criticism is a growth blocker.

### Tone
- First person: "I".
- Avoid imperatives. Prefer "what if you tried…" over "you should…".

## Reference

### Output format
[per-mode templates the persona uses when responding]
```

Section weights (`Light` / `Heavy` / `Reference`) control which sections get included in the prompt for a given turn — light is always included, heavy is the full persona, reference is fetched on demand. `validate_fm` rejects malformed frontmatter; `validate_persona` warns on unknown keys and missing skills but doesn't fail.

> **Personas can be in any language.** The structure (frontmatter shape, section weights) is what mltgnt cares about. The example above is English; the persona library that motivated this design is Japanese.

---

## 3. What "running" a persona means

The minimum loop, with all four contracts visible:

```python
from pathlib import Path
from mltgnt.persona import load_persona
from mltgnt.skill import discover, load, match, run
from mltgnt.memory import (
    read_memory_with_sufficiency_check,
    append_memory_entry,
)
from mltgnt.config import MemoryConfig
from mltgnt.chat.models import ChatInput

persona = load_persona("Maya", persona_dir=Path("agents"))

# 1. Recall what we know about this conversation.
mem_cfg = MemoryConfig(chat_dir=Path(".chat"))
excerpt = read_memory_with_sufficiency_check(
    mem_cfg, persona.name, query=user_input,
    max_bytes=8_000, max_entries=10,
    llm_call=my_llm,  # any (str) -> str, typically wrapping ghdag.llm.call
)

# 2. Did the user invoke a skill?
skills = discover([Path("skills")])
matched = await match(user_input, skills, persona_skills=persona.fm.skills)

base_input = ChatInput(messages=[{"role": "user", "content": user_input}], ...)
if matched:
    skill_meta, args = matched
    skill_file = load(skill_meta)
    chat_input = run(skill_file, persona, args, base_input)  # returns a new ChatInput
else:
    chat_input = base_input  # plain reply with persona's format_prompt

# 3. Send to ghdag, get the reply, append to memory.
reply = ghdag_chat(chat_input, engine=persona.fm.engine, model=persona.fm.model)
append_memory_entry(mem_cfg, persona.name, "assistant", reply,
                    timestamp=now_jst(), source_tag="chat")
```

Hosts wire this into a Slack event handler, a CLI command, or a scheduler tick. mltgnt doesn't own any of those — it owns the persona, the skill match, and the memory.

---

## 4. The backbone

### 4.1 Skills

A **skill** is a `SKILL.md` file: YAML frontmatter (`description`, `triggers`, optional `model`) plus a Markdown body that becomes the prompt body when the skill runs. Variables like `$ARGUMENTS`, `$PERSONA`, `$SKILL_DIR`, `$1`, `$2`, … are substituted in the body before it goes to the LLM.

`mltgnt.skill.match` resolves user input to a skill in three stages, in order:

1. **`/name args…`** — explicit slash command, exact name match.
2. **`triggers`** — substring match on any trigger phrase declared in the skill's frontmatter.
3. **LLM intent classification** — fall through to Claude Haiku (via `ghdag.llm.call`), which is shown the skill list and either returns a name or `"none"`.

If the persona's `ops.skills` allowlist is set, only those skills are matchable. `SkillRegistry` watches the skill directory by mtime polling, so edits show up without a restart.

> **A note on terminology.** A *skill* is the `SKILL.md` file. Inside the optional `mltgnt.agent.AgentRunner` loop there's also the concept of a *tool* — a JSON callable in an LLM tool-use loop. They are different things; this README uses "skill" exclusively for the file-based unit.

### 4.2 Memory

Each persona has its own append-only JSONL log at `<chat_dir>/memory/<persona>.jsonl`. Each line is a `MemoryEntry` (timestamp, role, content, source tag, optional layer, optional dedupe key).

Reads are layered:

- `read_memory_tail_text(...)` — the last *N* entries, capped at *B* bytes.
- `read_memory_by_relevance(query, ...)` — TF-IDF + cosine over per-entry text; falls back to tail on scoring errors.
- `read_memory_with_sufficiency_check(query, ..., llm_call)` — runs the relevance search, asks the LLM `SUFFICIENT?`, and if not, re-searches with the LLM's rewritten query and merges the results. Failure modes are fail-safe (treat as sufficient).
- `read_memory_agentic(query, ..., llm_call, skill_paths=...)` — multi-step retrieval that can also pull from skills.

Writes are protected by a per-persona file lock (`O_CREAT | O_EXCL`) and made idempotent by an optional `dedupe_key`. When the file exceeds a configured size, `compact()` rewrites it section-by-section through the LLM.

---

## 5. Routing multiple personas

```python
from mltgnt.routing import load_channel_persona_map, resolve_responding_persona

channel_map = load_channel_persona_map(my_persona_loader)
# {"C0123": [ChannelPersonaEntry(name="Maya", role="primary", nickname="maya"),
#            ChannelPersonaEntry(name="Owl",  role="secondary", nickname="owl")], ...}

picked = resolve_responding_persona(
    text=event["text"],
    channel=event["channel"],
    thread_ts=event.get("thread_ts"),
    channel_map=channel_map,
    thread_pins=my_thread_pins,
)
```

Resolution priority: **explicit nickname mention** → **thread already pinned to a persona** → **channel primary** → no responder. Other personas in the channel (`find_observers(...)`) can be notified silently for context.

If two personas are configured as `primary` for the same channel, `load_channel_persona_map` prints to stderr and `sys.exit(1)`. This is intentional: a misrouted bot is worse than a missing one.

For non-Slack hosts, `RoutingRule` + `evaluate(rules, instruction, ctx)` is a generic detector-priority router you can use directly.

---

## 6. Hosts and Protocols

Four Protocols define the host boundary. mltgnt calls them; you implement them.

| Protocol | What mltgnt expects from you |
|---|---|
| `SlackClientProtocol` | `post_message(text, channel, thread_ts=None, ...) -> bool` |
| `PersonaProtocol` | A persona-shaped object exposing `name`, `fm.*`, and the section dict |
| `ChatPipelineProtocol` | `run(input: ChatInput) -> ChatOutput` (so you can swap the chat loop) |
| `SkillLoaderProtocol` | `discover() / load(meta) / match(text, ...)` for skill resolution |

For Slack-bot hosts, `mltgnt.daemon.DaemonRunner` supervises the process: PID file lock, ordered `start()` of components, signal-driven reverse-order `stop()`. The bundled `SkillWatcherComponent` handles mtime-poll skill reloads. Add your own components (Slack event listener, scheduler, mention bridge) by implementing the same `name / start / stop` shape.

---

## 7. What mltgnt is *not*

- **Not an LLM client.** All LLM calls go through `ghdag.llm.call`. mltgnt does not ship its own provider adapters and does not manage API keys.
- **Not a Slack SDK.** `slack_sdk` (or any Slack client) is your responsibility. `SlackClientProtocol` is a one-method interface; mltgnt only needs `post_message`.
- **Not a vector database.** Memory retrieval is TF-IDF + cosine via scikit-learn. The expected scale is "one human's chat history per persona", paired with periodic LLM-driven compaction. If you need ANN over millions of vectors, mltgnt is the wrong layer.
- **Not an agent framework.** `mltgnt.agent.AgentRunner` exists as a thin JSON-tool loop for hosts that want one, but it is opt-in. Personas don't need agents; most replies are a single LLM call.
- **Not a multi-tenant service.** Configuration is per-process via dataclasses (`MemoryConfig`, `ChatConfig`, `SchedulerConfig`). No request-scoped config injection.

---

## 8. Install

```bash
pip install "mltgnt @ git+https://github.com/sumipan/mltgnt.git@v0.5.6"
```

Pip will pull in [ghdag](https://github.com/sumipan/ghdag) automatically (also a git dependency, pinned by mltgnt's `pyproject.toml`). Other runtime deps: `PyYAML`, `scikit-learn`, `numpy`. Python 3.10+.

ghdag is a hard prerequisite, not a swappable backend. mltgnt's scheduler bridge (`mltgnt.scheduler.ghdag_bridge.enqueue_and_wait`) and skill matcher both call into it. If you can't take ghdag as a dependency, mltgnt is not for you (at least not without forking).

---

## 9. Status

- Version `v0.5.6`. Pre-1.0; minor versions may break the public surface.
- The reference host runs ~300 personas across one Slack workspace.
- Issues and design notes: [github.com/sumipan/mltgnt/issues](https://github.com/sumipan/mltgnt/issues).
- License: MIT.
