# Changelog

## v0.20.0

### Added

- **loops Phase 3（実行力）**: `kind: action` サブタスクとホスト向け `ActionRequest` / `ActionResult` / `ActionExecutor`。公開 `action_schemas` に適合する決定論的 action を冪等キー付きで同期実行
- **ペルソナメモリ**: 任意 `MemoryAppender` による計画承認・iteration 完了・done/failed の短文要約追記（dedupe key で二重防止）
- **LLM / watch / replan 予算**: `llm_call_budget_per_loop`（既定 200）/ `llm_call_budget_per_day`（既定 1000, JST 共有）/ `max_watch_subtasks_per_loop`（既定 50）/ `max_replans_per_loop`（既定 20）。超過時は `paused`、完全一致の「再開」で `budget_override`
- **イベント**: `action_executed` / `memory_appended` / `memory_append_failed` / `budget_resumed`

### Compatibility

- `schema_version: 1` 維持。旧 state は新フィールドを既定値で復元可能。`ActionExecutor` / `MemoryAppender` は省略可能で既存ホスト構築を壊さない
- 既存 `auto` / `human` / `watch`、承認、コメント対話、deliverable 経路は非破壊

## v0.19.6

### Added

- **loops Phase 2（対話）**: inbox `kind=comment` を決定論的 status 判定・LLM 分類（`status` / `instruction` / `question` / `chitchat`）で処理。進捗照会は `post_progress` で即答、修正指示は既存 `replanning` へ、質問はペルソナ回答、雑談は `clarification_context` に補足
- **設定**: `comment_model` / `max_comments_per_tick`（既定 10）/ `comment_reply_budget_per_hour`（既定 10）/ `comment_reply_max_chars`（既定 800）
- **イベント**: `comment_classified`（`source`: deterministic / llm / budget_fallback）/ `comment_replied` / `comment_warning`
- **`render_progress_summary`**: LLM なしの人間向け進捗サマリ

### Compatibility

- `HumanChannel` / Phase 1 の watch・replan・承認ゲートは非破壊。`schema_version: 1` 維持
- 旧 `comment_received` 一括補足経路は対話処理に置換（chitchat / LLM 失敗時のみ補足追記）

## v0.19.5

### Added

- **loops Phase 1（反応性）**: `kind: watch` サブタスク、`depends` DAG、ローカル `path_exists` / `path_changed` 評価（`PathConditionEvaluator`）、ホスト向け `ConditionEvaluator` / `WatchVerdict`
- **watch 失敗時の即時 replan**: `replanning` 状態と `max_replans_per_iteration`（既定 3）。running / success は keep 必須
- **計画承認ゲート**: Objective `plan_approval`（未指定時は `LoopsConfig.plan_approval_default`、既定 true）と `awaiting_plan_approval`。承認語は `ok` / `承認` / `進めて` / `go` の全文一致。人間修正は `max_plan_revisions`（既定 3）まで、`replan_count` は消費しない
- **設定**: `watch_root` / `max_replans_per_iteration` / `max_plan_revisions` / `plan_approval_default`
- **イベント**: `watch_polled` / `replan_triggered` / `plan_proposed` / `plan_approved` / `plan_revised`

### Compatibility

- `schema_version: 1` を維持。v0.19.4 state（追加キーなし）は既定値でロード可能。depends キー欠落時は逐次依存へ正規化
- 既存 `auto` / `human` の submit→poll→evaluate 経路は非破壊。GitHub Issue/PR/label 評価はホスト（#2585）側

## v0.19.4

### Added

- **loops 単一 deliverable 契約**: `state_dir/<loop_id>/deliverable.md` を正規成果物とし、`start_loop` で Objective 本文から初期化。auto サブタスクは同ファイルを段階編集し、evaluate は `result_summary` と deliverable 抜粋を入力に使う
- **`HumanChannel.post_progress` / `post_deliverable`**: 計画・進捗・成果物案内のホスト通知契約（`progress_notify` で進捗のみ抑止可）
- **観測イベント**: `state_change` / `question_asked` / `subtask_submitted` / `subtask_done` / `deliverable_updated`
- **`Subtask.result_summary` / `result_filename`**: 評価・通知用の後方互換フィールド（旧 state は空文字で復元）

### Compatibility

- 既存 `result` / `submission` / HumanChannel メソッド・状態名・ schema_version=1 は維持。nexus 側 Slack/diary 実装は #2582

## v0.19.3

### Fixed

- **loops failed 終端で `close_thread` 漏れ**: 連続エラー等で `failed` に遷移したとき、done / cancelled と同じ finalize 経路で `HumanChannel.close_thread` を必ず呼び、ホスト側 pending スレッドが残らないようにした
- **inbox `kind: "comment"` の取り込み**: 質問待ち以外のユーザー発言を `clarification_context` に `補足: <text>` として追記し、`comment_received` イベントを記録する（同一 message_id は二重消費しない）

## v0.19.0

### BREAKING

- **Objective 配置による自動起動を廃止**: `objectives_dir` へ `.md` を置いただけでは loop state を作成しない。起動は `state_dir/requests/*.json` の依頼消費のみ。
- **移行順序**: 本リリース（mltgnt consumer）を先に入れ、nexus 側の request producer / Slack 配線（#2560）は **後から** 切り替えること。

### Added

- **`ensure_frontmatter`**: 欠落した `id` / `title` / `status` / `max_iterations` だけを決定論的に補完（`agent` は補完しない）
- **`mltgnt.loops.requests`**: 起動依頼 JSON の検証・列挙・`consumed/` / `corrupt/` 隔離
- **`LoopsEngine.start_loop(..., thread=)`**: 依頼スレッド（`HumanThreadRef`）を初回 state に継承。既存の `start_loop(objective)` は互換維持
- **`store.archive_terminal_state`**: 終端 state を `state_dir/archive/` へ退避し、再依頼で新規起動可能にする

### Compatibility

- 非終端 state の復元、Objective 削除 / `status: cancelled` による取消、content hash 変更警告は維持。
- 公開 Protocol（`interfaces/loops.py`）と `LoopsConfig` のフィールドは変更なし。

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
