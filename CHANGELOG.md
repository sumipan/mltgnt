# Changelog

## v0.8.0

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
