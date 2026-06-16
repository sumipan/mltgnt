# mltgnt

**L1: persona-driven multi-agent orchestration layer.** 3 層アーキテクチャ（**L0 [ghdag](https://github.com/sumipan/ghdag)** / **L1 mltgnt** / **L2 ホストアプリ**）の中間層として、ペルソナ定義・スキルマッチング・メモリ管理・チャネルルーティング・スケジューリングを担います。LLM 呼び出し・ファイル I/O・DAG 投入は `bridges` 経由で L0 ghdag に委譲します。

**Status:** Pre-1.0 (`v0.15.0`)

---

## Not（これは何でないか）

| 項目 | 説明 |
|------|------|
| LLM ライブラリではない | LLM 呼び出しは ghdag（L0）経由。mltgnt 自体はモデル推論を行わない |
| 汎用チャットボットフレームワークではない | ペルソナ駆動のマルチエージェントオーケストレーション層に特化 |
| DAG 実行エンジンではない | DAG 実行は ghdag の責務。mltgnt は `enqueue_dag` / `enqueue_and_wait` で投入する |
| ホストアプリではない | Slack 連携・ファイル保存・チャネル固有ロジックは L2 ホストが実装する |

---

## Installation

| 項目 | 内容 |
|------|------|
| インストール | `pip install mltgnt` |
| Python | 3.10 以上（`requires-python = ">=3.10"`） |
| コア依存 | [ghdag](https://github.com/sumipan/ghdag)（`pyproject.toml` で pin 済み）、`PyYAML`、`scikit-learn`、`numpy` |

ghdag は差し替え不可の前提依存です。LLM 呼び出し・DAG 投入はすべて ghdag 経由で行われます。

---

## Quick Start

`load_persona()` でペルソナを読み込み、`run_pipeline()` で 1 往復チャットを実行します。以下は `tests/chat/test_pipeline.py` と同型の最小例です。

```python
from pathlib import Path

from mltgnt import load_persona, run_pipeline

persona_dir = Path("agents")
persona = load_persona("タチコマ", persona_dir=persona_dir)

output = run_pipeline(
    "こんにちは",
    persona,
    engine=persona.fm.engine,
    model=persona.fm.model,
)

print(output.content)       # LLM 応答テキスト
print(output.persona_name)  # "タチコマ"
```

`run_pipeline` は例外を送出せず、失敗時は `ChatOutput.content` にエラー文字列を格納します。

---

## CLI Reference

### `mltgnt run`

デーモンプロセスを起動します。

| オプション | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--components MODULE:FUNCTION` | ✅ | — | コンポーネントファクトリ。`importlib` でモジュールを読み込み、callable を呼び出して `DaemonComponent` のリストを取得 |
| `--pid-file PATH` | — | `/tmp/mltgnt_daemon.pid` | PID ロックファイルのパス |

```bash
mltgnt run --components myhost.daemon:build_components --pid-file /tmp/mltgnt.pid
```

| 終了コード | 意味 | 対応例外 |
|-----------|------|---------|
| 0 | 正常終了（ヘルプ表示・シグナルシャットダウン含む） | — |
| 1 | 一般エラー | `MltgntError` その他 |
| 2 | 設定エラー | `ConfigError` |
| 3 | 依存エラー | `DependencyError` |

### `mltgnt memory dream show`

ペルソナの dream サマリーを表示します。

| 引数 / オプション | 必須 | 説明 |
|------------------|------|------|
| `persona` | ✅ | ペルソナ名（stem） |
| `--chat-dir PATH` | ✅ | ペルソナディレクトリの親パス |

```bash
mltgnt memory dream show タチコマ --chat-dir /path/to/chat
```

| 終了コード | 意味 |
|-----------|------|
| 0 | 正常終了（dream サマリー未存在時も 0） |

### `mltgnt memory dream forget`

dream サマリーから指定カテゴリを削除します。

| 引数 / オプション | 必須 | 説明 |
|------------------|------|------|
| `persona` | ✅ | ペルソナ名（stem） |
| `--category NAME` | ✅ | 削除するカテゴリ名 |
| `--chat-dir PATH` | ✅ | ペルソナディレクトリの親パス |

```bash
mltgnt memory dream forget タチコマ --category "会話の傾向" --chat-dir /path/to/chat
```

| 終了コード | 意味 |
|-----------|------|
| 0 | カテゴリ削除成功 |
| 1 | dream サマリー未存在、またはカテゴリ未発見 |

---

## Public API

`mltgnt.__init__.__all__` に列挙された公開シンボルです。トップレベルから import してください。

```python
from mltgnt import run_pipeline, Persona, load_persona
```

| カテゴリ | シンボル | 型 | 備考 |
|---------|---------|-----|------|
| Chat | `run_pipeline` | function | |
| Memory | `read_memory_iterative` | function | |
| Memory | `read_memory_by_relevance` | function | |
| Memory | `read_memory_with_sufficiency_check` | function | |
| Memory | `compact` | function | ⚠️ deprecated — 代替: `PersonaScheduler` dream アクション |
| Memory | `needs_compaction` | function | ⚠️ deprecated — 代替: `PersonaScheduler` dream アクション |
| Memory | `DreamSection` | dataclass | |
| Memory | `DreamSummary` | dataclass | |
| Memory | `read_dream` | function | |
| Memory | `write_dream` | function | |
| Persona | `Persona` | class | |
| Persona | `load_persona` | function | |
| Persona | `list_personas` | function | |
| Persona | `validate_persona` | function | |
| Persona | `run_persona_prompt` | function | |
| Types | `ChatInput` | dataclass | |
| Types | `ChatOutput` | dataclass | |
| Types | `Message` | TypedDict | |
| Persona | `PersonaProtocol` | Protocol | |
| Agent | `AgentResult` | dataclass | |
| Agent | `AgentRunner` | class | |
| Bridge | `enqueue_dag` | function | |
| Bridge | `enqueue_and_wait` | function | |
| Scheduler | `PersonaScheduler` | class | |
| Scheduler | `ScheduleJob` | dataclass | |
| Version | `__version__` | str | |

### Public API Stability

Pre-1.0（`0.Y.Z`）のため、マイナー・パッチリリース間で破壊的変更が入る場合があります。安定版 API は `__all__` に列挙されたシンボルに限定してください。SemVer 運用では `0.Y.Z` の Y バンプが破壊的変更、`Z` バンプが後方互換変更を示します。

---

## Deprecated API

以下のシンボルは `__all__` に残存しますが、新規コードでの使用は非推奨です。Public API 表では deprecated としてマーク済みです。

| シンボル | 廃止予定 | 代替手段 |
|---------|---------|---------|
| `compact` | v0.16.0 削除予定 | `PersonaScheduler` の dream アクション（スケジューラ経由のメモリ圧縮） |
| `needs_compaction` | v0.16.0 削除予定 | `PersonaScheduler` の dream アクション（スケジューラ経由の圧縮判定） |

---

## Architecture

`src/mltgnt/` 配下のパッケージ構成です。

| モジュール | 責務 |
|-----------|------|
| `agent/` | 汎用エージェントループ（LLM + ツール実行） |
| `bridges/` | アダプタ層（ファイル・audit・hooks・LLM・ghdag 連携） |
| `chat/` | 1 ラウンドトリップチャットパイプライン |
| `cli/` | CLI エントリ（`mltgnt run` / `mltgnt memory dream`） |
| `config/` | 設定 dataclass（`MemoryConfig` / `PersonaConfig` / `SchedulerConfig` / `ChatConfig`） |
| `daemon/` | デーモンランナー（PID ロック・skill ウォッチャー） |
| `exceptions.py` | `MltgntError` 階層の定義 |
| `improvement/` | 改善分析・提案・パッチ適用・ロールバック判定 |
| `interfaces/` | Protocol 定義（chat, persona, slack, types, ooda） |
| `kpi/` | audit.jsonl からの KPI 集計 |
| `memory/` | メモリ検索・圧縮・dream サマリー API |
| `ooda/` | OODA ループ実行基盤（audit_source, exec_dispatcher, runner） |
| `persona/` | ペルソナ読み込み・レジストリ・スキーマ検証 |
| `routing/` | チャネルルーティング・トリアージ |
| `scheduler/` | ペルソナスケジューラ（scheduled, interval, fuzzy_window, chained） |
| `skill/` | スキル読み込み・マッチング・実行・lint |

### Protocols / 拡張点

ホスト（L2）が実装する拡張点です。mltgnt はこれらの Protocol を呼び出し側として利用します。

| Protocol | モジュール | 責務 |
|----------|-----------|------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | `name`、`fm`（frontmatter）、`format_prompt(instruction) -> str` — ペルソナ契約 |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | `run(inp, repo_root) -> ChatOutputBase` — ホスト側チャットパイプライン差し替え |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, ...) -> bool` — Slack 投稿。失敗時は `False`（例外なし） |
| `ChatInputBase` | `mltgnt.interfaces.types` | チャットパイプライン入力の L1 Protocol |
| `ChatOutputBase` | `mltgnt.interfaces.types` | チャットパイプライン出力の L1 Protocol |
| `PersonaFMBase` | `mltgnt.interfaces.types` | ペルソナフロントマターの L1 Protocol（`name` 必須） |

---

## Configuration

### 環境変数

| 変数名 | 定義箇所 | 用途 |
|--------|---------|------|
| `SKILL_IO_TYPECHECK` | `bridges/ghdag_bridge.py` | `"0"` で無効化。未設定時は skill I/O 型検査を実行（opt-out） |
| `NIKKI_ROOT` | `skill/runner.py` | nikki（diary/memory）ルートパス。スキル本文の `$NIKKI_ROOT` 変数置換に使用 |
| `REPO_ROOT` | `skill/runner.py` | リポジトリルートのフォールバック。スキル本文の `$REPO_ROOT` 変数置換に使用 |
| `MLTGNT_AS_OF_DATE` | `improvement/loop.py` | 改善サイクルの基準日（`YYYY-MM-DD`）。未設定時は `date.today()` |

### 設定 dataclass（`mltgnt.config`）

| dataclass | 主なフィールド | 用途 |
|-----------|--------------|------|
| `MemoryConfig` | `chat_dir`, `inject_max_bytes`, `compact_threshold_bytes`, `timezone`, `dream_model` 等 | メモリ JSONL のパス・閾値・圧縮・dream 設定 |
| `PersonaConfig` | `weight_map` | ペルソナ Markdown セクションの重み付け（`light` / `heavy` / `reference`） |
| `SchedulerConfig` | `schedule_yaml`, `state_dir`, `timezone`, `salt` | スケジュール YAML と状態ディレクトリ |
| `ChatConfig` | `persona_dir`, `memory_dir`, `matcher_model` | チャットパイプラインのパス・マッチャモデル |

---

## Error Reference

### MltgntError 階層

```
MltgntError (mltgnt.exceptions) ← 基底
├── ConfigError (mltgnt.exceptions)
└── DependencyError (mltgnt.exceptions)
```

### 全公開例外型

| クラス | モジュール | 継承 | 用途 |
|--------|-----------|------|------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | 共通基底。`except MltgntError` で一括捕捉可能 |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | 設定ファイルの読み込み・パースエラー、`--components` 形式不正 |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | 外部依存（callable, subprocess, API, PID ロック）の呼び出し失敗 |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | ペルソナ frontmatter 不正（`MltgntError` 階層外） |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | メモリ圧縮 LLM 呼び出し失敗（`MltgntError` 階層外） |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | DAG ステップ間の skill I/O 型不整合（`MltgntError` 階層外） |
| `SkillLoadError` | `mltgnt.skill.models` | `Exception` | スキル読み込み失敗・未知 Tool 参照（`MltgntError` 階層外） |

---

## License

MIT — SPDX: `MIT`（`pyproject.toml` の `license = "MIT"` と一致）
