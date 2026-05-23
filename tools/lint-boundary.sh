#!/usr/bin/env bash
# 越境ポリシー lint: mltgnt から order/result/.md への直接 I/O を検出する
set -euo pipefail

violations=0

# Rule 1: memory module 内の .md ファイルへの書き込みを検出
# JSONL ファイルへの既知の書き込み（__init__.py, _compaction.py）は対象外
non_jsonl_writes=$(grep -rn '\.write_text\|open(' src/mltgnt/memory/ --include="*.py" \
  | grep -v '__pycache__\|#.*lint-ok\|\.jsonl\|jsonl' \
  | grep '\.md\b' || true)
if [ -n "$non_jsonl_writes" ]; then
  echo "ERROR: memory module に .md ファイルへの書き込みが検出されました:"
  echo "$non_jsonl_writes"
  violations=$((violations + 1))
fi

# Rule 2: ghdag_bridge での result 直接読み取りを検出
direct_reads=$(grep -n 'read_text\|\.read()' src/mltgnt/scheduler/ghdag_bridge.py 2>/dev/null | grep -v 'ghdag\.files\|#.*lint-ok' || true)
if [ -n "$direct_reads" ]; then
  echo "ERROR: ghdag_bridge が ghdag.files 以外でファイルを読み取っています:"
  echo "$direct_reads"
  violations=$((violations + 1))
fi

# Rule 3: 新規ファイルでの order/result パスへの直接書き込みを検出
order_result_writes=$(grep -rn 'open.*order/\|open.*result/\|write_text.*order\|write_text.*result' src/mltgnt/ --include="*.py" | grep -v '__pycache__\|ghdag\.files\|#.*lint-ok' || true)
if [ -n "$order_result_writes" ]; then
  echo "ERROR: order/result への直接ファイル操作が検出されました:"
  echo "$order_result_writes"
  violations=$((violations + 1))
fi

if [ $violations -eq 0 ]; then
  echo "OK: 越境ポリシー lint passed"
fi
exit $violations
