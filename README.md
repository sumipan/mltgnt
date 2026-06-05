# mltgnt

**L1: ペルソナ駆動マルチエージェントオーケストレーション層。** 3 層アーキテクチャ（**L0 [ghdag](https://github.com/sumipan/ghdag)** / **L1 mltgnt** / **L2 ホストアプリ**）の中間層として、ペルソナ定義・スキルマッチング・メモリ管理・チャンネルルーティング・スケジューリングを担います。LLM 呼び出し・ファイル I/O・DAG 投入は `bridges` 経由で L0 ghdag に委譲します。ghdag が DAG 実行基盤であるのに対し、mltgnt はペルソナ駆動のマルチエージェント実行レイヤです。

**Status:** Pre-1.0 (`v0.14.0`)

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

`run_pipeline` でペルソナ 1 件に対する 1 往復チャットを実行します。以下は `tests/chat/test_pipeline.py` と同型の最小例です。

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

| サブコマンド | 説明 |
|-------------|------|
| `mltgnt run` | デーモンプロセスを起動する（現時点で唯一のサブコマンド） |

### `mltgnt run`

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

---

## Public API

`mltgnt.__init__.__all__` に列挙された公開シンボルです。トップレベルから import してください。

```python
from mltgnt import run_pipeline, Persona, load_persona
```

| カテゴリ | シンボル | 型 |
|---------|---------|-----|
| Chat | `run_pipeline` | function |
| Memory | `read_memory_iterative` | function |
| Memory | `read_memory_by_relevance` | function |
| Memory | `read_memory_with_sufficiency_check` | function |
| Memory | `compact` | function |
| Memory | `needs_compaction` | function |
| Persona | `Persona` | class |
| Persona | `load_persona` | function |
| Persona | `list_personas` | function |
| Persona | `validate_persona` | function |
| Persona | `run_persona_prompt` | function |
| Interfaces | `ChatInput` | dataclass |
| Interfaces | `ChatOutput` | dataclass |
| Interfaces | `Message` | TypedDict |
| Interfaces | `PersonaProtocol` | Protocol |
| Agent | `AgentResult` | dataclass |
| Agent | `AgentRunner` | class |
| Bridges | `enqueue_dag` | function |
| Bridges | `enqueue_and_wait` | function |
| Scheduler | `PersonaScheduler` | class |
| Scheduler | `ScheduleJob` | dataclass |
| Version | `__version__` | str |

### Public API Stability

Pre-1.0（`0.Y.Z`）のため、マイナー・パッチリリース間で破壊的変更が入る場合があります。安定版 API は `__all__` に列挙されたシンボルに限定してください。

---

## Architecture

`src/mltgnt/` 配下のパッケージ構成です。

| モジュール | 責務 |
|-----------|------|
| `agent/` | 汎用エージェントループ（LLM + ツール実行） |
| `bridges/` | アダプタ層（ファイル・audit・hooks・LLM・ghdag 連携） |
| `chat/` | 1 ラウンドトリップチャットパイプライン |
| `cli/` | CLI エントリ（`mltgnt run`） |
| `config/` | 設定読み込み |
| `daemon/` | デーモンランナー（PID ロック・skill ウォッチャー） |
| `exceptions.py` | エラー階層 |
| `improvement/` | 改善分析・提案・レポート |
| `interfaces/` | Protocol 定義（chat, persona, slack, types） |
| `kpi/` | メトリクス・パーサー |
| `memory/` | メモリ検索・圧縮・API |
| `ooda/` | OODA ループ（audit_source, exec_dispatcher, runner） |
| `persona/` | ペルソナ読み込み・レジストリ・スキーマ |
| `routing/` | チャネルルーティング・トリアージ |
| `scheduler/` | ペルソナスケジューラ（scheduled, interval, fuzzy_window, chained） |
| `skill/` | スキル読み込み・マッチング・実行・lint |

### Not（スコープ外）

mltgnt は LLM 呼び出しライブラリではありません。ペルソナ・メモリ・スケジューリングを統合するオーケストレーション層であり、実際の LLM 実行・DAG 実行は ghdag（L0）に委譲します。

### Protocols / 拡張点

ホスト（L2）が実装する拡張点です。mltgnt はこれらの Protocol を呼び出し側として利用します。

| Protocol | モジュール | 責務 |
|----------|-----------|------|
| `PersonaProtocol` | `mltgnt.interfaces.persona` | `name`、`fm`（frontmatter）、`format_prompt(instruction) -> str` — ペルソナ契約 |
| `ChatPipelineProtocol` | `mltgnt.interfaces.chat` | `run(inp, repo_root) -> ChatOutputBase` — ホスト側チャットパイプライン差し替え |
| `PersonaFMBase` | `mltgnt.interfaces.types` | ペルソナフロントマターの L1 Protocol（`name` 必須） |
| `SlackClientProtocol` | `mltgnt.interfaces.slack` | `post_message(text, channel, ...) -> bool` — Slack 投稿。失敗時は `False`（例外なし） |

---

## Configuration

### 環境変数

| 変数名 | 定義箇所 | 用途 |
|--------|---------|------|
| `SKILL_IO_TYPECHECK` | `bridges/ghdag_bridge.py` | `"1"` で skill I/O 型検査を有効化。未設定時はスキップ |
| `NIKKI_ROOT` | `skill/runner.py` | nikki（diary/memory）ルートパス。スキル本文の `$NIKKI_ROOT` 変数置換に使用 |
| `REPO_ROOT` | `skill/runner.py` | リポジトリルートのフォールバック。スキル本文の `$REPO_ROOT` 変数置換に使用 |

---

## Error Reference

| クラス | モジュール | 継承 | 用途 |
|--------|-----------|------|------|
| `MltgntError` | `mltgnt.exceptions` | `Exception` | 共通基底。`except MltgntError` で一括捕捉可能 |
| `ConfigError` | `mltgnt.exceptions` | `MltgntError` | 設定ファイルの読み込み・パースエラー、`--components` 形式不正 |
| `DependencyError` | `mltgnt.exceptions` | `MltgntError` | 外部依存（callable, subprocess, API, PID ロック）の呼び出し失敗 |
| `PersonaValidationError` | `mltgnt.persona` | `Exception` | ペルソナ frontmatter 不正（`MltgntError` 階層外） |
| `SkillIOTypeError` | `mltgnt.bridges.ghdag_bridge` | `TypeError` | DAG ステップ間の skill I/O 型不整合（`MltgntError` 階層外） |
| `LlmCallError` | `mltgnt.memory.compaction` | `RuntimeError` | メモリ圧縮 LLM 呼び出し失敗（`MltgntError` 階層外） |

```
MltgntError (mltgnt.exceptions) ← 基底
├── ConfigError (mltgnt.exceptions)
└── DependencyError (mltgnt.exceptions)

PersonaValidationError (mltgnt.persona) ← Exception 直接継承（階層外）
SkillIOTypeError (mltgnt.bridges.ghdag_bridge) ← TypeError 直接継承（階層外）
LlmCallError (mltgnt.memory.compaction) ← RuntimeError 直接継承（階層外）
```

---

## License

MIT — SPDX: `MIT`（`pyproject.toml` の `license = "MIT"` と一致）
