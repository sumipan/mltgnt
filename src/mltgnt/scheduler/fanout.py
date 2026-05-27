from __future__ import annotations

import yaml

# fanout 指示: ペルソナが動的子タスクを生成できるよう result 末尾に YAML を出力させる。
# action_args.enable_fanout=true の skill ジョブでプロンプトに付加する。
_FANOUT_PROMPT_SUFFIX = """
---
If you determine that the task should be split into independent parallel subtasks,
append the following YAML block at the very end of your response (after a `---` separator).
Each child must have a unique `id` and a `command` that can be executed independently.

```
---
ghdag_fanout:
  children:
    - id: "subtask-1"
      command: "agent -p --force < order-subtask-1.md"
    - id: "subtask-2"
      command: "agent -p --force < order-subtask-2.md"
```

If no parallel subtasks are needed, omit the `ghdag_fanout` block entirely.
"""


def _parse_fanout_steps(
    content: str,
    engine: str,
    model: "str | None",
) -> "list | None":
    """ghdag_fanout YAML ブロックをパースして DagStep リストに変換する。

    content: enqueue_and_wait の応答文字列
    Returns DagStep のリスト。ブロックが存在しない場合は None。
    """
    lines = content.splitlines()
    last_sep_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "---":
            last_sep_idx = i
            break
    if last_sep_idx is None:
        return None

    yaml_text = "\n".join(lines[last_sep_idx + 1:])
    try:
        data = yaml.safe_load(yaml_text)
    except Exception:
        return None

    if not isinstance(data, dict) or "ghdag_fanout" not in data:
        return None

    fanout_data = data.get("ghdag_fanout") or {}
    children = fanout_data.get("children") or []
    if not children:
        return None

    from mltgnt.bridges.ghdag_bridge import DagStep
    steps = []
    for c in children:
        try:
            steps.append(DagStep(id=c["id"], prompt=c["command"], engine=engine, model=model))
        except (KeyError, TypeError):
            return None
    return steps or None
