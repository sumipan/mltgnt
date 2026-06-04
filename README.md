# mltgnt

3 層アーキテクチャ（**L0 [ghdag](https://github.com/sumipan/ghdag)** / **L1 mltgnt** / **L2 ホスト**）の **L1 層** に位置する Python ライブラリです。ペルソナ定義・スキルマッチング・メモリ管理・チャンネルルーティングを担い、LLM 呼び出し・ファイル I/O・DAG 投入は `bridges` 経由で L0 ghdag に委譲します。ghdag が DAG 実行基盤であるのに対し、mltgnt はペルソナ駆動のマルチエージェント実行レイヤです。

**Status:** pre-1.0 (`v0.13.0`)

---

## Installation

| 項目 | 内容 |
|------|------|
| インストール | `pip install mltgnt` |
| Python | 3.10 以上（`requires-python = ">=3.10"`） |
| 必須依存 | [ghdag](https://github.com/sumipan/ghdag)（`pyproject.toml` で pin 済み）、`PyYAML`、`scikit-learn`、`numpy` |

ghdag は差し替え不可の前提依存です。LLM 呼び出し・DAG 投入はすべて ghdag 経由で行われます。

---

## Quick Start

`run_pipeline` でペルソナ 1 件に対する 1 往復チャットを実行します。以下は `tests/chat/test_pipeline.py` と同型の最小例です。

```python
from pathlib import Path

from mltgnt import load_persona, run_pipeline

persona_dir = Path("agents")
persona = load_persona("Maya", persona_dir=persona_dir)

output = run_pipeline(
    "Hello!",
    persona,
    engine=persona.fm.engine,
    model=persona.fm.model,
)

print(output.content)       # LLM 応答テキスト
print(output.persona_name)  # "Maya"
```

`run_pipeline` は例外を送出せず、失敗時は `ChatOutput.content` にエラー文字列を格納します。

---

## CLI Reference

| サブコマンド | 説明 |
|-------------|------|
| `mltgnt run` | デーモンプロセスを起動する（現時点で唯一のサブコマンド） |

### `mltgnt run`

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
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

---

## Public API

`mltgnt.__init__.__all__` に列挙された公開シンボルです。トップレベルから import してください。

```python
from mltgnt import run_pipeline, Persona, load_persona
```

| カテゴリ | シンボル | 型 |
|---------|---------|-----|
| chat | `run_pipeline` | function |
| memory | `read_memory_iterative` | function |
| memory | `read_memory_by_relevance` | function |
| memory | `read_memory_with_sufficiency_check` | function |
| memory | `compact` | function |
| memory | `needs_compaction` | function |
| persona | `Persona` | class |
| persona | `load_persona` | function |
| persona | `list_personas` | function |
| persona | `validate_persona` | function |
| persona | `run_persona_prompt` | function |
| interfaces | `ChatInput` | dataclass |
| interfaces | `ChatOutput` | dataclass |
| interfaces | `Message` | dataclass |
| interfaces | `PersonaProtocol` | Protocol |
| agent | `AgentResult` | dataclass |
| agent | `AgentRunner` | class |
| bridges | `enqueue_dag` | function |
| bridges | `enqueue_and_wait` | function |
| scheduler | `PersonaScheduler` | class |
| scheduler | `ScheduleJob` | dataclass |
| version | `__version__` | str |

---

## Protocols

ホスト（L2）が実装する拡張点です。mltgnt はこれらの Protocol を呼び出し側として利用します。

| Protocol | モジュール | 責務 |
|----------|-----------|------|
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, thread_ts=..., ...) -> bool` — Slack 投稿。失敗時は `False`（例外なし） |
| `PersonaProtocol` | `mltgnt.interfaces.persona` | `name`、`fm`（frontmatter）、`format_prompt(instruction) -> str` — ペルソナ契約 |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | `run(inp, repo_root) -> ChatOutputBase` — ホスト側チャットパイプライン差し替え |
| `DaemonComponent` | `mltgnt.daemon` | `name` プロパティ、`start()` / `stop()` — デーモンに登録するコンポーネント契約 |

---

## Architecture

`src/mltgnt/` 配下のパッケージ構成です。

| パッケージ | 責務 |
|-----------|------|
| `agent/` | エージェント実行（`AgentRunner`、JSON ツールループ） |
| `bridges/` | L0 ghdag への委譲アダプタ（audit, files, LLM, hooks, DAG 投入） |
| `chat/` | チャットパイプライン（`run_pipeline`） |
| `cli/` | CLI エントリポイント（`mltgnt run`） |
| `config/` | diary 非依存の設定 dataclass |
| `daemon/` | デーモン管理（`DaemonRunner`, `PidLock`, `SkillWatcherComponent`） |
| `improvement/` | 自己改善ループ（analyzer, proposal, reporter） |
| `interfaces/` | 型契約 Protocol と入出力 dataclass |
| `kpi/` | KPI メトリクス収集・パース |
| `memory/` | ペルソナメモリ（TF-IDF 検索、充足性チェック、圧縮） |
| `ooda/` | OODA ループ実行（`OODARunner`, audit source, exec dispatcher） |
| `persona/` | ペルソナ定義・ロード・バリデーション・レジストリ |
| `routing/` | チャンネル→ペルソナルーティング（primary/secondary, nickname 検出, triage） |
| `scheduler/` | スケジュールジョブ管理（fanout, state, actions） |
| `skill/` | スキル定義・マッチング・実行（`SKILL.md` ベース） |

---

## Configuration

### 環境変数

| 変数 | 値 | 説明 |
|------|-----|------|
| `SKILL_IO_TYPECHECK` | `"1"` で有効 | `enqueue_dag` 実行時にスキル I/O 型チェックを有効化。未設定時はスキップ |

### 設定 dataclass（`mltgnt.config`）

| dataclass | 主なフィールド | 用途 |
|-----------|--------------|------|
| `MemoryConfig` | `chat_dir`, `inject_max_bytes`, `compact_threshold_bytes`, `timezone` 等 | メモリ JSONL のパス・閾値・圧縮設定 |
| `PersonaConfig` | `weight_map` | ペルソナ Markdown セクションの重み付け（`light` / `heavy` / `reference`） |
| `SchedulerConfig` | `schedule_yaml`, `state_dir`, `timezone`, `salt` | スケジュール YAML と状態ディレクトリ |
| `ChatConfig` | `persona_dir`, `memory_dir`, `matcher_model` | チャットパイプラインのパス・マッチャモデル |

---

## Error Reference

| クラス | モジュール | 継承 | 用途 |
|--------|-----------|------|------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | 共通基底。`except MltgntError` で一括捕捉可能 |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | 設定ファイルの読み込み・パースエラー、`--components` 形式不正 |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | 外部依存（callable, subprocess, API, PID ロック）の呼び出し失敗 |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | ペルソナ frontmatter 不正（`MltgntError` 階層外） |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | メモリ圧縮 LLM 呼び出し失敗（`MltgntError` 階層外） |

```
Exception
├── MltgntError
│   ├── ConfigError
│   └── DependencyError
├── PersonaValidationError
RuntimeError
└── LlmCallError
```

---

## License

MIT — SPDX: `MIT`（`pyproject.toml` の `license = "MIT"` と一致）
