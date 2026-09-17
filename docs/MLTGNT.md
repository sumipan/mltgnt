# MLTGNT — package layout notes（#3321）

nexus 側の総合設計書（`docs/MLTGNT.md`）のうち、本リポジトリで実装する公開パッケージの現状をここに追記する。

## 残すパッケージ（`src/mltgnt/`）

| パッケージ | 役割 |
|---|---|
| `agent/` | 汎用エージェントループ・委譲判定骨格 |
| `bridges/` | ghdag / LLM / files / hooks アダプタ |
| `cli/` | CLI エントリ |
| `config/` | ホスト注入用 dataclass（`ConversationConfig` / `MemoryConfig` 等） |
| `conversation/` | 媒体非依存の会話層（待機列・セッション・台帳） |
| `daemon/` | デーモンランナーとコンポーネント |
| `interfaces/` | Protocol / DTO（slack / persona / types / turn） |
| `memory/` | メモリ読み取り・compact・dream |
| `persona/` | ペルソナ loader / registry / 整形 |
| `routing/` | 媒体非依存ルーティング |
| `scheduler/` | ジョブスケジューラ（`BaseRunner` / `PersonaScheduler`） |
| `skill/` | スキル discover / match / runner / pipeline 合成 |

## 削除済み（#3321 / #3301 サブ6）

`loops` / `ooda` / `improvement` / `kpi` / `chat` / `execution` および `interfaces` の loops / ooda / chat Protocol は削除した。

## 7.4 柱 / Phase F（スキルパイプライン）

| 項目 | 状態 | 根拠 |
|---|---|---|
| `split_pipe_segments` / `match_pipeline` | ✅ | `skill/matcher.py` |
| compose-time typecheck | ✅ | `bridges/ghdag_bridge.typecheck_dag`（既定オン） |
| **合成器 `compose_pipeline`** | ✅（#3031） | `SkillMatchResult` → 直線 `DagStep`（`pipe_{i}_{name}` / `depends`） |
| scheduler 配線 | ✅（#3031） | `action_args.enable_pipeline: true` → match → compose → typecheck → `enqueue_dag` |
| `PIPELINE_STATUS` 下流注入 | ✅（#3031） | `{step_id}_pipeline_status`。`INVALID_STATE` は downstream 未投入 |
| `diagnostics` 下流同伴 | ❌ | 未実装（設計対象外） |

`enable_fanout` と同時指定時は `enable_pipeline` を優先する。

エンジン: `compose_pipeline(..., engine=)` が全 `DagStep.engine` に設定（claude / cursor / codex）。

## 7.6-2 `consumes.producer` の宣言空転（解消）

- 以前: `typecheck_dag` のみで検証し、パイプ自動合成は無かった。
- #3031 以降: `compose_pipeline` が v1 下流の `consumes.producer` と前段 `decisive.name` を照合し、不一致なら `SkillIOTypeError`（exec.jsonl 書込前）。legacy はスキップ。
