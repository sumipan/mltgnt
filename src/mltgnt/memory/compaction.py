"""
mltgnt.memory.compaction — メモリコンパクション（per-section cap 方式）。

設計: Issue #123, #137, #823, #1135
Issue #1135: diary の高度な圧縮ロジックを upstream。
- per-section cap（preferences / long_term / mid_term 各 25%）
- Phase 1: recent → preferences 抽出・マージ
- ロールアップループ: recent → mid_term チャンク分割 + 1 行要約
- mid_term → long_term 玉突き昇格
- インクリメンタル保存（セクション・チャンクごと）
- ratio guard（[slack-observe] エントリ除外）
- date coverage post-check
- entry 再分類（_redistribute_entries）
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

from mltgnt.memory._format import MemoryEntry, parse_jsonl, serialize_entry

_log = logging.getLogger(__name__)

# エントリ見出しのタイムスタンプ抽出用
_ENTRY_HEADER_TS_RE = re.compile(
    r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) — (?:user|assistant)",
    re.MULTILINE,
)

LlmCall = Callable[[str], str]

# per-section cap 割合
PREFS_CAP_RATIO = 0.25
LONG_TERM_CAP_RATIO = 0.25
MID_TERM_CAP_RATIO = 0.25
# preferences / long_term の統合圧縮時の最大圧縮率（これ未満への圧縮は拒否）
PREFS_MAX_RATIO = 0.90
LONG_TERM_MAX_RATIO = 0.90
# mid_term → long_term 玉突き昇格の定数
LONG_TERM_PROMOTE_FLUSH_BYTES = 10 * 1024  # buffer 累積でこの値を超えたら LLM 圧縮を発火
LONG_TERM_PROMOTE_MAX_ITER = 10  # LLM 呼び出し回数の上限。超過分は次回持ち越し

ROLLUP_CHUNK = 50 * 1024  # 50KB: ロールアップ1回あたりの取り出し上限バイト数
ROLLUP_FINE_CHUNK = 5 * 1024  # 5KB: 小粒度モードの切り出し上限
ROLLUP_MIN_KEEP_BYTES = 10 * 1024  # 10KB: recent の最小保持量（この値以下でループ停止）
ROLLUP_MAX_ITER = 20  # 無限ループ防御の上限回数
ROLLUP_SUMMARY_TARGET_BYTES = 5 * 1024  # 5KB: LLM圧縮後の1行サマリ目標上限バイト数
ROLLUP_SUMMARY_MAX_RETRIES = 2  # リトライ上限（初回試行を除く）

__all__ = [
    "LlmCallError",
    "CompactionResult",
    "PromoteCandidate",
    "extract_promote_candidates",
    "needs_compaction",
    "compact",
    "_effective_bytes_for_ratio",
    "_build_section_prompt",
    "_strip_heading",
    "_compact_section",
    "_promote_with_compression",
    "_promote_mid_to_long",
    "_extract_and_merge_preferences",
    "_sanitize_phase1_output",
    "_strip_observe_entries",
    "_rollup_recent_chunk",
    "_extract_chunk_date_range",
    "_check_date_coverage",
    "_compress_rollup_chunk",
    "_redistribute_entries",
    "_entry_to_block",
    "_entries_to_body",
]


class LlmCallError(RuntimeError):
    """llm_call の実行中に発生したエラーをラップする例外。"""


@dataclass(frozen=True)
class PromoteCandidate:
    topic: str
    summary: str
    source_entries: int
    recurrence: int


@dataclass(frozen=True)
class CompactionResult:
    before_bytes: int
    after_bytes: int
    summary: str
    warnings: list[str] = field(default_factory=list)
    promote_candidates: list[PromoteCandidate] = field(default_factory=list)


def extract_promote_candidates(
    entries: list[MemoryEntry],
    *,
    min_recurrence: int = 3,
) -> list[PromoteCandidate]:
    """コンパクション対象のエントリから promote 候補を抽出する。

    同一 source_tag が min_recurrence 回以上出現するエントリを候補とする。
    promote の実行判定は呼び出し側に委譲する。

    Args:
        entries: コンパクション対象の MemoryEntry リスト
        min_recurrence: 同一トピックの最小出現回数

    Returns:
        PromoteCandidate のリスト。
    """
    from collections import defaultdict

    groups: dict[str, list[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.source_tag].append(entry)

    candidates: list[PromoteCandidate] = []
    for source_tag, group in groups.items():
        if len(group) < min_recurrence:
            continue
        summary = "\n\n".join(e.content for e in group if e.content.strip())
        candidates.append(PromoteCandidate(
            topic=source_tag,
            summary=summary,
            source_entries=len(group),
            recurrence=len(group),
        ))
    return candidates


def needs_compaction(config: "MemoryConfig", persona_stem: str) -> bool:
    """メモリファイルがコンパクション閾値を超えているか判定する。"""
    from mltgnt.memory import memory_file_path
    path = memory_file_path(config, persona_stem)
    if not path.exists():
        return False
    return path.stat().st_size >= config.compact_threshold_bytes


def _effective_bytes_for_ratio(text: str) -> int:
    """ratio 計算用の有効バイト数（[slack-observe] ブロックを除外）。

    ``[slack-observe]`` タグ付きエントリは情報密度が低く、LLM 圧縮後に
    極端に小さくなるため ratio チェックが誤検知しやすい。
    このメソッドではそれらを除外したサイズを返すことで、
    「非 observe コンテンツが 5% 未満に圧縮された」場合だけアボートするようにする。

    JSONL 形式（各行が JSON）と Markdown 形式（``\\n---\\n`` 区切り）の両方に対応する。
    """
    import json as _json
    lines = text.splitlines()
    # JSONL 形式の判定: 最初の非空行が { で始まる場合
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty and non_empty[0].strip().startswith('{'):
        # JSONL 形式: source_tag か content に [slack-observe] を含まない行のみ集計
        kept_lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                obj = _json.loads(line)
                tag = obj.get('source_tag', '') or ''
                content = obj.get('content', '') or ''
                if '[slack-observe]' not in tag and '[slack-observe]' not in content:
                    kept_lines.append(line)
            except _json.JSONDecodeError:
                kept_lines.append(line)
        cleaned = '\n'.join(kept_lines)
        result = len(cleaned.encode('utf-8'))
    else:
        # Markdown 形式: \n---\n で区切られる形式
        blocks = re.split(r'\n---\n', text)
        kept = [b for b in blocks if '[slack-observe]' not in b]
        cleaned = '\n---\n'.join(kept)
        result = len(cleaned.encode('utf-8'))
    return result if result > 0 else len(text.encode('utf-8'))


def _build_section_prompt(section_text: str, target_bytes: int) -> str:
    """個別セクション用のコンパクションプロンプトを生成する。"""
    return (
        "以下の文章を要約・圧縮してください。"
        "各エントリの日時見出し行（「## YYYY-MM-DD HH:MM — user/assistant」形式）は"
        "削除・変更せずそのまま保持してください。"
        "見出し以外の本文を圧縮対象としてください。"
        f"目標サイズ: {target_bytes}バイト以内。"
        "出力は要約された本文のみとしてください。"
        "バイト数・トークン数・サイズ情報・圧縮率などのメタ情報を出力に含めないでください。"
        "「指示通り圧縮しました」「目標サイズに収めました」等のプロンプト指示への自己言及も禁止です。"
        "\n\n"
        f"{section_text}"
    )


def _strip_heading(section_text: str) -> str:
    """セクションテキストから先頭の ``## ...`` 見出し行を除去して本文だけ返す。"""
    return re.sub(r"^##\s+[^\n]*\n*", "", section_text, count=1).strip()


def _compact_section(
    section_name: str,
    section_text: str,
    target_bytes: int,
    llm_call: LlmCall,
    *,
    skip_min_ratio: bool = False,
) -> tuple[str, str | None]:
    """1 セクションをコンパクションする。

    Returns:
        (compacted_body, warning_or_none)
        失敗時は元の本文をそのまま返し、warning に理由を入れる。
    """
    body = _strip_heading(section_text)
    if not body:
        return body, None

    MIN_RATIO = 0.05
    original_size = len(body.encode("utf-8"))

    try:
        prompt = _build_section_prompt(body, target_bytes)
        result = llm_call(prompt)
    except Exception as e:
        warning = f"{section_name}: LLM call failed ({e}), using original text"
        _log.warning(warning)
        return body, warning

    result_size = len(result.encode("utf-8"))
    if result_size < original_size * MIN_RATIO:
        if skip_min_ratio:
            warning = (
                f"{section_name}: result very small "
                f"({result_size}B < {original_size}B * {MIN_RATIO}), "
                f"accepted (--no-min-ratio-guard)"
            )
            _log.warning(warning)
            return result.strip(), warning
        warning = (
            f"{section_name}: result too small "
            f"({result_size}B < {original_size}B * {MIN_RATIO}), "
            f"using original text"
        )
        _log.warning(warning)
        return body, warning

    return result.strip(), None


def _promote_with_compression(
    section_name: str,
    existing_body: str,
    incoming_body: str,
    cap_bytes: int,
    llm_call: LlmCall,
    *,
    max_ratio: float = 0.90,
    skip_min_ratio: bool = False,
) -> tuple[str, str | None]:
    """昇格統合圧縮: existing_body + incoming_body を一括で LLM 圧縮する。

    preferences / long_term 用。cap 超過時のみ呼ばれる。
    max_ratio ガード: 圧縮結果が ``existing_body の max_ratio 未満`` なら過剰圧縮として拒否し
    結合テキストをそのまま返す。

    Args:
        section_name: セクション名（ログ用）
        existing_body: 既存セクション本文（max_ratio の基準サイズ）
        incoming_body: 昇格で追加されるテキスト（空文字列可）
        cap_bytes: 目標バイト数上限
        llm_call: LLM 呼び出し callable
        max_ratio: 圧縮率下限（result < existing * max_ratio で拒否）

    Returns:
        (result_body, warning_or_none)
    """
    _SEP = "\n\n---\n\n"
    if existing_body and incoming_body:
        combined = existing_body + _SEP + incoming_body
    else:
        combined = existing_body or incoming_body

    if not combined:
        return "", None

    # max_ratio ガードの基準は existing_body のサイズ（存在する場合）
    check_size = len(existing_body.encode("utf-8")) if existing_body else len(combined.encode("utf-8"))

    try:
        prompt = _build_section_prompt(combined, cap_bytes)
        result = llm_call(prompt)
    except Exception as e:
        warning = f"{section_name}: LLM call failed ({e}), using original text"
        _log.warning(warning)
        return combined, warning

    result_size = len(result.encode("utf-8"))

    # max_ratio ガード: existing の max_ratio 未満への過剰圧縮を拒否
    if result_size < check_size * max_ratio:
        if skip_min_ratio:
            warning = (
                f"{section_name}: result over-compressed "
                f"({result_size}B < {check_size}B * {max_ratio}), "
                f"accepted (--no-min-ratio-guard)"
            )
            _log.warning(warning)
            return result.strip(), warning
        warning = (
            f"{section_name}: result over-compressed "
            f"({result_size}B < {check_size}B * {max_ratio}), "
            f"using original text"
        )
        _log.warning(warning)
        return combined, warning

    return result.strip(), None


def _promote_mid_to_long(
    compacted: dict[str, str],
    mid_term_cap: int,
    long_term_cap: int,
    llm_call: LlmCall,
    *,
    flush_threshold: int = LONG_TERM_PROMOTE_FLUSH_BYTES,
    max_iter: int = LONG_TERM_PROMOTE_MAX_ITER,
    skip_min_ratio: bool = False,
) -> list[str]:
    """mid_term の古いエントリを long_term に玉突き昇格する。

    compacted を in-place で変更し、warnings のみ返す。

    アルゴリズム:
    1. mid_term をブロック分割（"\\n\\n---\\n\\n" 区切り）
    2. 古い順（先頭）に buffer に積む。buffer 累積 > flush_threshold で flush:
       - long_term + buffer → _promote_with_compression(long_term_cap, max_ratio=0.90)
       - LLM 入力上限 = long_term_cap + flush_threshold（構造的に固定）
    3. mid_term_size <= mid_term_cap になったら停止
    4. max_iter 回の LLM 呼び出しで打ち切り、残りは次回持ち越し
    5. ループ終了後、buffer に余りがあれば最終 flush（この flush も iter_count にカウントする）
    """
    SEP = "\n\n---\n\n"
    blocks = compacted["mid_term"].split(SEP)
    blocks = [b for b in blocks if b.strip()]
    buffer: list[str] = []
    warnings: list[str] = []
    iter_count = 0

    while blocks and iter_count < max_iter:
        remaining = SEP.join(blocks)
        if len(remaining.encode("utf-8")) <= mid_term_cap:
            break

        block = blocks.pop(0)
        buffer.append(block)
        buffer_size = sum(len(b.encode("utf-8")) for b in buffer)

        if buffer_size >= flush_threshold:
            incoming = SEP.join(buffer)
            new_long_term, warning = _promote_with_compression(
                "long_term",
                compacted["long_term"],
                incoming,
                long_term_cap,
                llm_call,
                max_ratio=LONG_TERM_MAX_RATIO,
                skip_min_ratio=skip_min_ratio,
            )
            if warning:
                warnings.append(warning)
            compacted["long_term"] = new_long_term
            buffer = []
            iter_count += 1

    # buffer に余りがあり、かつ iter_count < max_iter なら最終 flush
    if buffer and iter_count < max_iter:
        incoming = SEP.join(buffer)
        new_long_term, warning = _promote_with_compression(
            "long_term",
            compacted["long_term"],
            incoming,
            long_term_cap,
            llm_call,
            max_ratio=LONG_TERM_MAX_RATIO,
            skip_min_ratio=skip_min_ratio,
        )
        if warning:
            warnings.append(warning)
        compacted["long_term"] = new_long_term
        buffer = []
        iter_count += 1

    # mid_term を残りブロック + 未消化 buffer で再構成
    remaining_blocks = buffer + blocks
    compacted["mid_term"] = SEP.join(remaining_blocks) if remaining_blocks else ""

    if iter_count >= max_iter and (blocks or buffer):
        _log.warning(
            "mid→long promotion hit max_iter=%d, %d blocks remain in mid_term",
            max_iter,
            len(blocks) + len(buffer),
        )

    return warnings


_PHASE1_PROMPT_TEMPLATE = """\
以下の2つのテキストを処理してください。

【タスク】
1. 「最近の記録」からユーザーの好み・傾向・習慣・パターンを抽出してください。
   - 一時的な状態（「今日は疲れた」等）は除外し、繰り返し現れる傾向のみを対象としてください。
2. 抽出した内容を「既存の好み・傾向」とマージしてください。
   - 重複する項目は統合してください。
   - 矛盾する項目は新しい方（最近の記録側）を優先してください。
3. 出力は好み・傾向の箇条書きのみとしてください。
   目標サイズ: {target_bytes}バイト以内。

出力規則（厳守）:
- 箇条書き本文のみ出力。前置き・後書き・見出し行・サイズ情報を含めるな
- 「承知しました」「分析します」等の自己言及文を含めるな
- 対話的な応答（質問・確認・提案）を含めるな
- 入力が空または「(なし)」の場合は空文字列を返せ（説明不要）

【既存の好み・傾向】
{existing_prefs}

【最近の記録】
{recent_text}"""


def _sanitize_phase1_output(text: str) -> str:
    """Phase 1 LLM の生出力からメタ発話・見出し・メタ行を除去する。

    除去対象:
    - 先頭が「承知」「分析」「了解」「以下」で始まる行
    - ``## `` で始まる見出し行
    - ``**サイズ**``、``**統計**``、``**分析**`` を含む行
    """
    if not text:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("承知", "分析", "了解", "以下")):
            continue
        if stripped.startswith("## "):
            continue
        if any(marker in stripped for marker in ("**サイズ**", "**統計**", "**分析**")):
            continue
        kept.append(line)
    return "\n".join(kept)


def _extract_and_merge_preferences(
    existing_prefs: str,
    recent_text: str,
    target_bytes: int,
    llm_call: LlmCall,
    *,
    skip_min_ratio: bool = False,
) -> tuple[str, str | None]:
    """recent テキストから好み・傾向を抽出し、既存 preferences とマージした結果を返す。

    Args:
        existing_prefs: 現在の preferences セクション本文（heading 除去済み）
        recent_text: recent セクション本文（heading 除去済み）
        target_bytes: preferences の cap バイト数
        llm_call: LLM 呼び出し callable

    Returns:
        (merged_prefs, warning)
        - merged_prefs: マージ後の preferences 本文
        - warning: 異常時の警告メッセージ（正常時は None）
    """
    if not recent_text:
        return existing_prefs, None

    prompt = _PHASE1_PROMPT_TEMPLATE.format(
        target_bytes=target_bytes,
        existing_prefs=existing_prefs if existing_prefs else "（既存の好み・傾向は未登録。recent からの抽出のみで初期化してよい）",
        recent_text=recent_text,
    )

    try:
        result = llm_call(prompt)
    except Exception as e:
        warning = f"Phase 1 extract-merge failed: {e}"
        _log.warning(warning)
        return existing_prefs, warning

    if not result or not result.strip():
        warning = "Phase 1 extract-merge: empty output from LLM"
        _log.warning(warning)
        return existing_prefs, warning

    result = _sanitize_phase1_output(result).strip()

    # 過剰圧縮ガード: 既存 preferences が存在する場合、
    # LLM 出力が既存の PREFS_MAX_RATIO 未満なら拒否
    existing_bytes = len(existing_prefs.encode("utf-8"))
    if existing_bytes > 0:
        result_bytes = len(result.encode("utf-8"))
        if result_bytes < existing_bytes * PREFS_MAX_RATIO:
            if skip_min_ratio:
                warning = (
                    f"Phase 1 extract-merge: over-compressed "
                    f"({result_bytes}B < {existing_bytes}B * {PREFS_MAX_RATIO}), "
                    f"accepted (--no-min-ratio-guard)"
                )
                _log.warning(warning)
                return result.strip(), warning
            warning = (
                f"Phase 1 extract-merge: over-compressed "
                f"({result_bytes}B < {existing_bytes}B * {PREFS_MAX_RATIO}), "
                f"keeping original preferences"
            )
            _log.warning(warning)
            return existing_prefs, warning

    return result, None


def _strip_observe_entries(body: str) -> str:
    """recent セクション本文から ``[slack-observe]`` を含むエントリブロックを除去する。

    ``\\n---\\n`` で split し、``[slack-observe]`` を含まないブロックのみ再結合する。
    """
    if not body:
        return body
    blocks = re.split(r'\n---\n', body)
    kept = [b for b in blocks if '[slack-observe]' not in b]
    return '\n---\n'.join(kept)


def _rollup_recent_chunk(recent_body: str, rollup_chunk: int) -> tuple[str, str]:
    """recent 本文から古い順にエントリ単位で rollup_chunk バイト分を取り出す。

    Args:
        recent_body: recent セクション本文
        rollup_chunk: 取り出し上限バイト数

    Returns:
        (remaining_body, promoted_body)
        - remaining_body: recent に残すテキスト
        - promoted_body: mid_term に昇格するテキスト（生エントリ）

    境界条件:
    - エントリが0個: (recent_body, "") を返す
    - エントリが1個で rollup_chunk 超: その1個を丸ごと昇格
    """
    if not recent_body:
        return recent_body, ""

    blocks = re.split(r'\n---\n', recent_body)
    blocks = [b for b in blocks if b]  # 空ブロック除去
    if not blocks:
        return recent_body, ""

    accumulated: list[str] = []
    acc_bytes = 0

    for i, block in enumerate(blocks):
        block_bytes = len(block.encode("utf-8"))
        if acc_bytes + block_bytes > rollup_chunk and accumulated:
            # 既に rollup_chunk を超える → 確定
            break
        accumulated.append(block)
        acc_bytes += block_bytes
        if acc_bytes > rollup_chunk:
            # 1エントリ単独で超過: 丸ごと昇格
            break

    if not accumulated:
        # 1エントリも取り出せなかった場合（通常ありえないが安全弁）
        return recent_body, ""

    # インデックスベースで分割（重複エントリ対策）
    promoted_blocks = blocks[:len(accumulated)]
    remaining_blocks = blocks[len(accumulated):]

    promoted_body = '\n---\n'.join(promoted_blocks)
    remaining_body = '\n---\n'.join(remaining_blocks)
    return remaining_body, promoted_body


def _extract_chunk_date_range(promoted: str) -> tuple[str, str] | None:
    """チャンクテキストから先頭・末尾エントリの日付を抽出する。

    Args:
        promoted: _rollup_recent_chunk が返した promoted テキスト

    Returns:
        成功時: (start_date, end_date) — 各要素は "YYYY-MM-DD" 形式
        失敗時（日付が1つも抽出できない）: None
    """
    import warnings
    matches = _ENTRY_HEADER_TS_RE.findall(promoted)
    if not matches:
        warnings.warn(
            f"_extract_chunk_date_range: no date headers found in chunk "
            f"(first 200 bytes: {promoted[:200]!r})",
            stacklevel=2,
        )
        return None
    start_date = matches[0].split(" ")[0]
    end_date = matches[-1].split(" ")[0]
    return (start_date, end_date)


def _check_date_coverage(
    observed_ranges: list[tuple[str, str]],
    final_text: str,
) -> list[tuple[str, str]]:
    """observed_ranges のうち、final_text にどちらの日付も出現しないものを返す。

    Args:
        observed_ranges: ロールアップで取り出した (start_date, end_date) のリスト。
                         各日付は "YYYY-MM-DD" 形式の文字列。
        final_text: assemble_memory() の出力全文。

    Returns:
        欠落しているレンジのリスト（空なら全レンジが少なくとも片端で出現）。
    """
    missed = []
    for start, end in observed_ranges:
        if start not in final_text and end not in final_text:
            missed.append((start, end))
    return missed


_ROLLUP_SUMMARY_PROMPT = """\
以下の会話ログを1行で要約してください。
- 出力は1行のみ（改行禁止）
- {target}バイト以内に収めること
- 日付は含めないこと（呼び出し元で付与する）
- 主要なトピック・決定事項・成果物を簡潔に列挙すること

---
{chunk}"""


def _compress_rollup_chunk(
    promoted: str,
    llm_call: "LlmCall",
    *,
    target: int = ROLLUP_SUMMARY_TARGET_BYTES,
    max_retries: int = ROLLUP_SUMMARY_MAX_RETRIES,
) -> str:
    """チャンクテキストをLLMで1行サマリに圧縮する。

    Args:
        promoted: 圧縮対象のチャンクテキスト
        llm_call: LLM呼び出し関数
        target: 圧縮後の目標バイト数（デフォルト: 5KB）
        max_retries: リトライ上限（デフォルト: 2）

    Returns:
        要約テキスト（改行なし、日付プレフィックスなし）。
        全リトライ失敗時は promoted をそのまま返す。
    """
    import warnings
    prompt = _ROLLUP_SUMMARY_PROMPT.format(target=target, chunk=promoted)
    for attempt in range(max_retries + 1):
        try:
            output = llm_call(prompt)
        except Exception as e:
            _log.warning("_compress_rollup_chunk: LLM call failed (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries:
                continue
            break
        # 改行を機械的に全削除
        output = output.replace("\n", " ").strip()
        # 出力長チェック: target * 3 超ならリトライ
        if len(output.encode("utf-8")) > target * 3:
            _log.warning(
                "_compress_rollup_chunk: output too long (%dB > %dB), retrying (attempt %d/%d)",
                len(output.encode("utf-8")),
                target * 3,
                attempt + 1,
                max_retries + 1,
            )
            if attempt < max_retries:
                continue
            break
        return output

    warnings.warn(
        f"_compress_rollup_chunk: all {max_retries + 1} attempts failed, "
        f"falling back to raw promoted text",
        stacklevel=2,
    )
    return promoted


def _redistribute_entries(
    entries: list["MemoryEntry"],
    now: datetime,
    config: "MemoryConfig",
    *,
    raw_days_override: int | None = None,
) -> list["MemoryEntry"]:
    """エントリの age に基づき layer を再分類する（純粋関数）。

    protected / preferences エントリはそのまま保持。
    config.timezone を使用して age を計算する（diary 固有の依存なし）。
    """
    tz = ZoneInfo(config.timezone)
    effective_raw_days = raw_days_override if raw_days_override is not None else config.raw_days
    mid_threshold_days = config.mid_weeks * 7

    result = []
    for entry in entries:
        if (entry.layer is not None and entry.layer in config.protected_layers) or entry.source_tag == "preferences":
            result.append(entry)
            continue
        ts = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
            try:
                ts = datetime.strptime(entry.timestamp, fmt)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=tz)
                break
            except ValueError:
                continue
        if ts is None:
            result.append(entry)
            continue
        age_days = (now.astimezone(tz) - ts.astimezone(tz)).total_seconds() / 86400
        if age_days <= effective_raw_days:
            new_layer = "recent"
        elif age_days <= mid_threshold_days:
            new_layer = "mid_term"
        else:
            new_layer = "long_term"
        if entry.layer == new_layer:
            result.append(entry)
        else:
            result.append(MemoryEntry(
                timestamp=entry.timestamp,
                role=entry.role,
                content=entry.content,
                source_tag=entry.source_tag,
                layer=new_layer,
                dedupe_key=entry.dedupe_key,
            ))
    return result


def _entry_to_block(e: "MemoryEntry") -> str:
    """MemoryEntry をテキストブロック形式に変換する（rollup 処理用）。

    タイムスタンプは YYYY-MM-DD HH:MM 形式（_ENTRY_HEADER_TS_RE 互換）に正規化する。
    ISO 8601 形式（2026-04-20T10:00:00+09:00 等）も変換する。
    """
    ts = e.timestamp
    # ISO 8601 の 'T' を空白に、末尾のタイムゾーン・秒を除去
    if "T" in ts:
        ts = ts.replace("T", " ")
    # タイムゾーン (+09:00 等) を除去
    if "+" in ts:
        ts = ts[:ts.index("+")]
    elif ts.endswith("Z"):
        ts = ts[:-1]
    # 秒部分 (:SS) を除去: HH:MM:SS → HH:MM
    parts = ts.split(":")
    if len(parts) >= 3:
        ts = ":".join(parts[:2])
    return f"## {ts} — {e.role}\n\n{e.content}"


def _entries_to_body(entries: list["MemoryEntry"]) -> str:
    """MemoryEntry リストをテキスト本文に変換する（rollup 処理用）。"""
    if not entries:
        return ""
    return "\n---\n".join(_entry_to_block(e) for e in entries)


def compact(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    llm_call: LlmCall,
    dry_run: bool = False,
    max_retries: int = 3,
    skip_min_ratio: bool = False,
) -> CompactionResult:
    """メモリファイルをコンパクションする（per-section cap 方式）。

    llm_call はプロンプト文字列を受け取り、コンパクション後のテキストを返す callable。
    dry_run=True のときはファイル書き込みを行わない。

    **per-section cap 方式**:
    - preferences: cap 超過時のみ統合圧縮（重複削除のみ、max_ratio=0.90）
    - long_term: cap 超過時のみ統合圧縮（max_ratio=0.90）
    - mid_term: recent からの昇格を受ける通過バッファ。cap 超過時は _promote_mid_to_long で long_term へ玉突き昇格。昇格後も超過なら LLM 圧縮
    - recent: cap 超過時のみチャンク分割 LLM 圧縮。圧縮後も cap 超過なら raw_days 短縮で早期昇格

    **インクリメンタル保存**: long_term・mid_term 各処理後と、recent のチャンクごとに
    ファイルを書き込む。大きなファイルでも少しずつ縮小され、中断時もそこまでの圧縮結果が保持される。

    新パラメータ max_retries / skip_min_ratio はすべてデフォルト値付きのため、
    既存の compact(config, stem, llm_call=fn) 呼び出しは変更不要（後方互換）。
    """
    from mltgnt.memory import memory_file_path, persona_memory_lock

    path = memory_file_path(config, persona_stem)
    if not path.exists():
        raise FileNotFoundError(f"Memory file not found: {path}")

    tz = ZoneInfo(config.timezone)

    with persona_memory_lock(config, persona_stem) as ok:
        if not ok:
            raise TimeoutError(f"Failed to acquire memory lock for {persona_stem}")

        original_text = path.read_text(encoding="utf-8")
        before_bytes = len(original_text.encode("utf-8"))

        warnings_list: list[str] = []
        observed_date_ranges: list[tuple[str, str]] = []
        new_text = original_text

        for attempt in range(max_retries):
            entries = parse_jsonl(path)

            protected_entries = [e for e in entries if e.layer is not None and e.layer in config.protected_layers]
            prefs_entries = [e for e in entries if e.source_tag == "preferences"]
            _classified_ids = set(id(e) for e in protected_entries + prefs_entries)
            other_entries = [e for e in entries if id(e) not in _classified_ids]

            long_entries = [e for e in other_entries if e.layer == "long_term"]
            mid_entries = [e for e in other_entries if e.layer == "mid_term"]
            recent_entries = [e for e in other_entries if e.layer not in ("long_term", "mid_term")]

            prefs_body = "\n".join(e.content for e in prefs_entries)
            long_term_body = _entries_to_body(long_entries)
            mid_term_body = _entries_to_body(mid_entries)
            recent_body = _entries_to_body(recent_entries)

            # --- per-section cap 計算 ---
            compact_target = config.compact_target_bytes
            prefs_cap = int(compact_target * PREFS_CAP_RATIO)
            long_term_cap = int(compact_target * LONG_TERM_CAP_RATIO)
            mid_term_cap = int(compact_target * MID_TERM_CAP_RATIO)

            prefs_size = len(prefs_body.encode("utf-8"))
            long_term_size = len(long_term_body.encode("utf-8"))
            recent_size = len(recent_body.encode("utf-8"))

            # recent の目標: compact_target から他3セクションの実サイズ（cap 以下）と
            # mid_term の cap を差し引いた残り
            prefs_budget = min(prefs_size, prefs_cap)
            long_term_budget = min(long_term_size, long_term_cap)
            recent_target = max(
                compact_target - prefs_budget - long_term_budget - mid_term_cap,
                ROLLUP_MIN_KEEP_BYTES,
            )

            # compacted 辞書: 処理済みセクションを追跡。未処理は原文を保持。
            compacted: dict[str, str] = {
                "long_term": long_term_body,
                "mid_term": mid_term_body,
                "recent": recent_body,
            }

            # [slack-observe] エントリは容量に関わらず先に除去する
            stripped = _strip_observe_entries(recent_body)
            if stripped != recent_body:
                recent_body = stripped
                recent_size = len(recent_body.encode("utf-8"))
                compacted["recent"] = recent_body
                print(
                    f"compact: stripped observe entries for {persona_stem}, "
                    f"recent={recent_size/1024:.1f}KB",
                    flush=True,
                )

            # --- recent 容量超過時のロールアップ ---
            # ① ロールアップループ → ② Phase 1 を1回
            if recent_size > recent_target:
                # ロールアップループ（observe 削除後もまだ超過の場合）
                promoted_text_parts: list[str] = []
                if recent_size > recent_target:
                    for _iter in range(ROLLUP_MAX_ITER):
                        current_recent = compacted["recent"]
                        if not current_recent:
                            break
                        current_recent_size = len(current_recent.encode("utf-8"))
                        if current_recent_size <= recent_target:
                            break

                        # 最小保持量ガード
                        if current_recent_size <= ROLLUP_MIN_KEEP_BYTES:
                            break

                        # chunk_size の動的決定
                        if current_recent_size <= ROLLUP_CHUNK:
                            chunk_size = min(ROLLUP_FINE_CHUNK,
                                             current_recent_size - ROLLUP_MIN_KEEP_BYTES)
                            if chunk_size <= 0:
                                break
                        else:
                            chunk_size = ROLLUP_CHUNK

                        remaining, promoted = _rollup_recent_chunk(current_recent, chunk_size)
                        if not promoted:
                            break

                        # 日付抽出
                        dates = _extract_chunk_date_range(promoted)
                        if dates is None:
                            _log.warning(
                                "compact: skipping undateable chunk for %s (first 200 bytes: %r)",
                                persona_stem,
                                promoted[:200],
                            )
                            compacted["recent"] = remaining
                            remaining_kb = len(remaining.encode("utf-8")) / 1024
                            print(
                                f"compact: rollup iter={_iter + 1} for {persona_stem}: "
                                f"skipped (no dates), remaining={remaining_kb:.1f}KB",
                                flush=True,
                            )
                            continue

                        observed_date_ranges.append(dates)

                        # LLM圧縮
                        summary = _compress_rollup_chunk(promoted, llm_call,
                                                          target=ROLLUP_SUMMARY_TARGET_BYTES,
                                                          max_retries=ROLLUP_SUMMARY_MAX_RETRIES)

                        # フォーマット
                        if summary is promoted:
                            # LLM 圧縮フォールバック: Phase 1 入力には積まない（過大入力を防ぐ）
                            final = promoted
                            warnings_list.append(
                                f"rollup chunk LLM compression failed for "
                                f"{dates[0]}-{dates[1]}, "
                                f"excluded from Phase 1 input"
                            )
                        else:
                            final = f"{dates[0]} - {dates[1]} {summary}"
                            promoted_text_parts.append(final)  # 整形済み1行サマリのみ積む

                        # mid_term に append
                        existing_mid = compacted["mid_term"]
                        if existing_mid:
                            compacted["mid_term"] = existing_mid + "\n---\n" + final
                        else:
                            compacted["mid_term"] = final
                        compacted["recent"] = remaining

                        promoted_kb = len(promoted.encode("utf-8")) / 1024
                        remaining_kb = len(remaining.encode("utf-8")) / 1024
                        chunk_kb = chunk_size / 1024
                        print(
                            f"compact: rollup iter={_iter + 1} for {persona_stem}: "
                            f"chunk={chunk_kb:.1f}KB, promoted={promoted_kb:.1f}KB, remaining={remaining_kb:.1f}KB",
                            flush=True,
                        )

                # ③ ロールアップ完了後、累積昇格テキスト全体で Phase 1 を1回だけ実行
                if promoted_text_parts:
                    all_promoted = "\n---\n".join(promoted_text_parts)
                    _log.info(
                        "Phase 1 input size: %d bytes (%d chunks)",
                        len(all_promoted.encode()), len(promoted_text_parts),
                    )
                    prefs_body, p1_warning = _extract_and_merge_preferences(
                        prefs_body, all_promoted, prefs_cap, llm_call,
                        skip_min_ratio=skip_min_ratio,
                    )
                    if p1_warning:
                        warnings_list.append(f"[attempt {attempt + 1}] {p1_warning}")
                    prefs_size = len(prefs_body.encode("utf-8"))
                    print(
                        f"compact: Phase 1 done for {persona_stem}, "
                        f"prefs={prefs_size/1024:.1f}KB",
                        flush=True,
                    )

            # --- [B] mid_term → long_term 玉突き昇格 ---
            mid_term_size_now = len(compacted["mid_term"].encode("utf-8"))
            if mid_term_size_now > mid_term_cap:
                promote_warnings = _promote_mid_to_long(
                    compacted, mid_term_cap, long_term_cap, llm_call,
                    skip_min_ratio=skip_min_ratio,
                )
                for w in promote_warnings:
                    warnings_list.append(f"[attempt {attempt + 1}] {w}")

            # --- [C] preferences cap 超過時の統合圧縮（重複削除、max_ratio=0.90）---
            if prefs_size > prefs_cap:
                body, warning = _promote_with_compression(
                    "preferences", prefs_body, "", prefs_cap, llm_call,
                    max_ratio=PREFS_MAX_RATIO,
                    skip_min_ratio=skip_min_ratio,
                )
                if warning:
                    warnings_list.append(f"[attempt {attempt + 1}] {warning}")
                prefs_body = body

            # --- [D] long_term（保険: B の max_iter 打ち切り時に long_term が cap 超過の場合）---
            long_term_size_now = len(compacted["long_term"].encode("utf-8"))
            if long_term_size_now > long_term_cap:
                body, warning = _promote_with_compression(
                    "long_term", compacted["long_term"], "", long_term_cap, llm_call,
                    max_ratio=LONG_TERM_MAX_RATIO,
                    skip_min_ratio=skip_min_ratio,
                )
                if warning:
                    warnings_list.append(f"[attempt {attempt + 1}] {warning}")
                compacted["long_term"] = body

            # --- [E] mid_term フォールバック（B が max_iter 打ち切り後もまだ cap 超過の場合）---
            mid_term_size_now = len(compacted["mid_term"].encode("utf-8"))
            if mid_term_size_now > mid_term_cap:
                body, warning = _compact_section(
                    "mid_term",
                    "## mid_term\n" + compacted["mid_term"],
                    mid_term_cap,
                    llm_call,
                    skip_min_ratio=skip_min_ratio,
                )
                if warning:
                    warnings_list.append(f"[attempt {attempt + 1}] {warning}")
                compacted["mid_term"] = body

            # JSONL 形式でエントリを組み立て
            now_ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
            _recent_by_key = {(e.timestamp, e.role): e for e in recent_entries}

            def _block_to_entry(block: str, default_layer: str) -> "MemoryEntry":
                m = re.match(r'^## (\S+ \S+) — (user|assistant)\n\n(.*)', block, re.DOTALL)
                if m:
                    ts_str, role, content = m.group(1), m.group(2), m.group(3).strip()
                    key = (ts_str, role)
                    if default_layer == "recent" and key in _recent_by_key:
                        return _recent_by_key[key]
                    return MemoryEntry(timestamp=ts_str, role=role, content=content, source_tag="compaction", layer=default_layer)
                return MemoryEntry(timestamp=now_ts, role="assistant", content=block.strip(), source_tag="compaction", layer=default_layer)

            def _text_to_entries(body: str, layer: str) -> list["MemoryEntry"]:
                if not body.strip():
                    return []
                blocks = [b.strip() for b in re.split(r'\n---\n', body) if b.strip()]
                return [_block_to_entry(b, layer) for b in blocks]

            final_prefs = [MemoryEntry(timestamp=now_ts, role="assistant", content=prefs_body.strip(), source_tag="preferences")] if prefs_body.strip() else []
            final_long = _text_to_entries(compacted["long_term"], "long_term")
            final_mid = _text_to_entries(compacted["mid_term"], "mid_term")
            final_recent = _text_to_entries(compacted["recent"], "recent")

            all_final_entries = protected_entries + final_prefs + final_long + final_mid + final_recent
            new_text = "".join(serialize_entry(e) + "\n" for e in all_final_entries if e.content.strip())
            after_bytes = len(new_text.encode("utf-8"))

            if after_bytes <= config.compact_target_bytes * 1.3:
                break

            _log.warning(
                "compact: attempt %d/%d result still large for %s (%dB > %dB), retrying",
                attempt + 1,
                max_retries,
                persona_stem,
                after_bytes,
                int(config.compact_target_bytes * 1.3),
            )
            # JSONL 形式では parse_jsonl がファイルから読むため、リトライ前に書き込む
            if not dry_run:
                path.write_text(new_text, encoding="utf-8")

        after_bytes = len(new_text.encode("utf-8"))

        # サイズ下限チェック: [slack-observe] を除いた元サイズの 5% 未満は異常
        MIN_RATIO = 0.05
        effective_before = _effective_bytes_for_ratio(original_text)
        ratio = after_bytes / effective_before if effective_before > 0 else 1.0
        if after_bytes < effective_before * MIN_RATIO:
            if skip_min_ratio:
                if ratio < 0.01:
                    _log.warning(
                        "compact: extreme ratio for %s (%.4f) but accepted (--no-min-ratio-guard)",
                        persona_stem,
                        ratio,
                    )
                # ガード解除: 通常パスを継続（書き込みへ）
            else:
                if not dry_run:
                    path.write_text(original_text, encoding="utf-8")
                    _log.warning(
                        "compact: restored original text for %s due to near-empty result "
                        "(%dB -> %dB, effective_before=%dB, ratio %.3f < %.2f)",
                        persona_stem,
                        before_bytes,
                        after_bytes,
                        effective_before,
                        ratio,
                        MIN_RATIO,
                    )
                raise ValueError(
                    f"Compaction produced near-empty result for {persona_stem}: "
                    f"{before_bytes}B -> {after_bytes}B "
                    f"(effective_before={effective_before}B, "
                    f"ratio {ratio:.3f} < {MIN_RATIO}) "
                    f"— aborting to prevent data loss"
                )

        # 日付カバレッジ事後検知
        if observed_date_ranges:
            missed = _check_date_coverage(observed_date_ranges, new_text)
            if missed:
                missed_str = ", ".join(f"{s}..{e}" for s, e in missed)
                _log.warning(
                    "compact: post-check missed dates for %s: [%s]",
                    persona_stem, missed_str,
                )
                warnings_list.append(f"post-check missed dates: [{missed_str}]")

        if not dry_run:
            path.write_text(new_text, encoding="utf-8")

        return CompactionResult(
            before_bytes=before_bytes,
            after_bytes=after_bytes,
            summary=f"compacted {persona_stem}: {before_bytes}B -> {after_bytes}B",
            warnings=warnings_list,
            promote_candidates=[],
        )
