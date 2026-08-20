# Changelog

## v0.18.0

### Added

- **`mltgnt.loops`**: Objective 駆動ループ実行（clarify → decompose → execute → evaluate）
- **`LoopsComponent`**: `DaemonComponent` 準拠の Objective snapshot ポーリング（既定 10 秒）
- **`LoopsConfig`**: objectives/state/status/jobs パス、LLM/subtask エンジン、上限値
- **`HumanChannel` / `SubtaskExecutor` Protocol**: Slack/ghdag 実装はホスト（nexus #2512）側
- **`enqueue_step` / `poll_step`**: ghdag_bridge の非ブロックサブタスク投入・完了確認
- **status Markdown**: `<status_dir>/<loop_id>.md` に人間向け現在状態を出力

### Compatibility

- 後方互換。既存 scheduler / chat / OODA API に変更なし。
- nexus ホスト配線は #2512 で別途実装。

### Operational limits

- `max_iterations`: 1..10（既定 5）
- `max_clarify_rounds`: 1..3（既定 3）
- `max_subtasks_per_iteration`: 1..5（既定 5）
- `subtask_timeout_sec`: 1800 秒（30 分）

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
