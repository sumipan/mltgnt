#!/usr/bin/env bash
# Boundary policy lint: detect direct I/O from mltgnt to order / result / .md files
set -euo pipefail

violations=0

# Rule 1: detect writes to .md files inside the memory module
# Known JSONL writes (__init__.py, _compaction.py) are out of scope
non_jsonl_writes=$(grep -rn '\.write_text\|open(' src/mltgnt/memory/ --include="*.py" \
  | grep -v '__pycache__\|#.*lint-ok\|\.jsonl\|jsonl' \
  | grep '\.md\b' || true)
if [ -n "$non_jsonl_writes" ]; then
  echo "ERROR: the memory module writes to .md files:"
  echo "$non_jsonl_writes"
  violations=$((violations + 1))
fi

# Rule 2: detect direct result reads in ghdag_bridge
direct_reads=$(grep -n 'read_text\|\.read()' src/mltgnt/scheduler/ghdag_bridge.py 2>/dev/null | grep -v 'ghdag\.files\|#.*lint-ok' || true)
if [ -n "$direct_reads" ]; then
  echo "ERROR: ghdag_bridge reads files without ghdag.files:"
  echo "$direct_reads"
  violations=$((violations + 1))
fi

# Rule 3: detect direct writes to order / result paths in new files
order_result_writes=$(grep -rn 'open.*order/\|open.*result/\|write_text.*order\|write_text.*result' src/mltgnt/ --include="*.py" | grep -v '__pycache__\|ghdag\.files\|#.*lint-ok' || true)
if [ -n "$order_result_writes" ]; then
  echo "ERROR: direct file operations on order / result paths:"
  echo "$order_result_writes"
  violations=$((violations + 1))
fi

# Rule 4: detect read_text on .md files anywhere under src/mltgnt/
md_reads=$(grep -rn 'read_text' src/mltgnt/ --include="*.py" \
  | grep -v '__pycache__\|#.*lint-ok\|\.jsonl\|jsonl' \
  | grep '\.md\b' || true)
if [ -n "$md_reads" ]; then
  echo "ERROR: read_text on .md files under src/mltgnt/:"
  echo "$md_reads"
  violations=$((violations + 1))
fi

if [ $violations -eq 0 ]; then
  echo "OK: boundary policy lint passed"
fi
exit $violations
