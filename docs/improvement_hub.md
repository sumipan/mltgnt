# ImprovementHub

mltgnt を三系統（ghdag / nexus / mltgnt）の改善ループを集約するハブとして位置づけるための設計ドキュメント。

## 背景

mltgnt は自身の改善ループ（`run_improvement_cycle`）を持つだけでなく、関連リポジトリ（ghdag, nexus）の改善サイクルも統括するオーケストレータとなることを目指す。`ImprovementHub` はその中心的なインタフェースを提供する。

## 三系統の役割

| 系統 | リポジトリ | 責務 |
|------|-----------|------|
| mltgnt | sumipan/mltgnt | エージェント実行・ペルソナ管理・スキル実行の改善ループ |
| ghdag | sumipan/ghdag | GitHub DAG ワークフロー自動化の改善ループ |
| nexus | sumipan/nexus | LLM パイプライン・ジョブ実行系の改善ループ |

各系統は `ImprovementSource` プロトコルを実装し、`ImprovementHub` に登録することで統一的に管理される。

## コンポーネント

### ImprovementSource（Protocol）

```python
@runtime_checkable
class ImprovementSource(Protocol):
    @property
    def name(self) -> str: ...

    def run_cycle(self) -> CycleResult: ...
```

改善ループ 1 系統の抽象インタフェース。`@runtime_checkable` により `isinstance` で準拠チェックが可能。

### MltgntSource

mltgnt 系統の `ImprovementSource` 実装。既存の `run_improvement_cycle()` 関数をアダプタパターンで包む。

```python
src = MltgntSource(
    audit_path=Path("/path/to/audit.jsonl"),
    persona_dir=Path("/path/to/personas"),
    skills_dir=Path("/path/to/skills"),
    since_days=7,
)
result = src.run_cycle()  # CycleResult を返す
```

### ImprovementHub

登録された全 `ImprovementSource` を順次実行するオーケストレータ。

```python
hub = ImprovementHub()
hub.register(MltgntSource(...))
# 将来: hub.register(GhdagSource(...))
# 将来: hub.register(NexusSource(...))

results = hub.run_all_cycles()  # list[CycleResult]
```

- 同名ソースの重複登録は `ValueError`
- ソース未登録時は空リスト `[]` を返す
- 実行順は登録順（並列化は Phase 3 以降で検討）

## 拡張方針

ghdag / nexus 系統の `ImprovementSource` 実装は各リポジトリの別イシューで行う。本モジュールはプロトコル定義とオーケストレータ機能のみを担う。
