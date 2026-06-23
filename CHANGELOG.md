# Changelog

## Phase Progress

### Phase D: exit_code routing ✓
- SkillRunResult.exit_code → ExitStatus enum 変換 実装済み
- scheduler permission pass-through (v0.15.1)
- ⚠️ enqueue_dag() 子タスクへの exit_code 伝播は未対応（#2235）

### Phase E: side_effects audit ⚠️ In Progress
- SkillMeta.side_effects 宣言は存在。実測 audit ラッパは未実装（#2234）
- BaseRunner ABC 抽出・ActDispatcher Protocol 統一 (v0.16.0)
- deprecated compact() / needs_compaction() 公開 API 削除（予定）

### Phase F: pipe composition runtime ⚠️ Not Started
- typecheck_dag() は存在するが skill_io != "v1" で全スキップ（silent compatibility mode）
- skill_io: v1 明示化・型検証の強制化は未着手

## v0.15.1

### Added

- scheduler permission pass-through: `action_args.permission` を `enqueue_and_wait` → `StepConfig.permission` へ透過

## v0.10.0

### BREAKING: 非推奨 API の削除

v0.9.x で DeprecationWarning を発行していた以下の API を削除しました。

**chat モジュール**
- `mltgnt.chat.models` → `mltgnt.interfaces.types` から直接 import してください
- `mltgnt.chat.run_chat()` → `run_pipeline()` を使用してください

**memory モジュール**
- `mltgnt.memory.read_memory_agentic()` → `read_memory_iterative()` を使用してください
- `mltgnt.memory._compaction` → `mltgnt.memory.compaction` から直接 import してください
- `mltgnt.memory.api.normalize_source_prefix()` → 削除（呼び出し元でインライン化してください）

**persona / agent モジュール**
- `mltgnt.agent._parse` の args キーなし JSON 受理 → `{"tool": str, "args": dict}` 形式を必須化
- `mltgnt.persona.schema` の flat キー (`chat_model`, `slack`) → `ops:` namespace を使用
- `mltgnt.persona.schema` の `ops.chat_model` → `ops.engine` / `ops.model` を使用
- `Persona.WEIGHT_MAP` / `Persona.ops_config` / `Persona.slack_post_kwargs()` / `Persona.delegate_ack()` → 削除
- `validate_persona()` / `validate_fm()` の `legacy_keys` 警告 → 削除

**scheduler モジュール**
- `mltgnt.scheduler.ghdag_bridge` → `mltgnt.bridges.ghdag_bridge` から直接 import してください
