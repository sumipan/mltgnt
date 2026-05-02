"""mltgnt.persona.compress

重量ブロックから軽量ブロックを LLM 圧縮で生成し、ペルソナファイルに固定保存する。

公開 API:
    compress_heavy_to_light(heavy_text, *, engine, model, timeout) -> str
    compute_block_hash(text)                                         -> str
    regenerate_light_block(persona_path, *, engine, model, timeout) -> RegenerationResult
    RegenerationResult                                               (dataclass)
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

LIGHT_BLOCK_MAX_CHARS = 1500

_COMPRESS_PROMPT_TEMPLATE = """以下のペルソナの重量ブロックから、v2.1 形式の軽量ブロックを生成してください。

## 出力形式

1. リード文（1〜2文で人物の本質を要約）
2. 必須サブセクション（太字見出し）:
   - **口調** — 話し方の特徴
   - **価値観** — 大切にしていること
   - **好意的反応** — どんなとき喜ぶか
   - **引っかかる** — どんなとき不快になるか
3. 推奨（任意）:
   - **発言例** — 引用ブロック（> ）形式で1〜3例

## 制約
- 1500文字以内（厳守）
- 太字見出しは上記の名前をそのまま使う
- 発言例がある場合は必ず引用ブロック形式にする

## 重量ブロック:
{heavy_text}"""


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class RegenerationResult:
    """軽量ブロック再生成の結果。"""

    persona_name: str
    old_hash: str
    new_hash: str
    light_text: str
    changed: bool

    @property
    def is_first_generation(self) -> bool:
        return self.old_hash == ""


# ---------------------------------------------------------------------------
# 公開関数
# ---------------------------------------------------------------------------


def compute_block_hash(text: str) -> str:
    """テキストの sha256 ハッシュを返す。

    正規化: 前後の空白を strip し、改行を LF に統一してからハッシュする。

    Args:
        text: ハッシュ対象テキスト

    Returns:
        sha256 の hex digest 文字列
    """
    normalized = text.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compress_heavy_to_light(
    heavy_text: str,
    *,
    engine: str = "claude",
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """重量ブロックのテキストを LLM 圧縮して軽量ブロック用サマリを返す。

    Args:
        heavy_text: 重量ブロックの全テキスト（H3 以下を含む）
        engine: LLM エンジン名（デフォルト: "claude"）
        model: モデル名。None の場合はエンジンのデフォルトを使用
        timeout: LLM 呼び出しタイムアウト秒数

    Returns:
        1500文字以内の v2.1 形式テキスト（リード文 + 必須サブセクション）

    Raises:
        RuntimeError: LLM 呼び出しが失敗した場合、または heavy_text が空の場合
    """
    from ghdag.llm import call as ghdag_llm_call

    if not heavy_text.strip():
        raise RuntimeError("heavy_text が空です。圧縮対象のテキストを指定してください。")

    prompt = _COMPRESS_PROMPT_TEMPLATE.format(heavy_text=heavy_text)

    kwargs: dict = {"engine": engine, "timeout": timeout}
    if model is not None:
        kwargs["model"] = model

    try:
        result = ghdag_llm_call(prompt, **kwargs)
    except Exception as e:
        raise RuntimeError(f"LLM 呼び出しが失敗しました: {e}") from e

    if not result.ok:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"LLM が ok=False を返しました: {stderr}")

    return (result.stdout or "").strip()


def regenerate_light_block(
    persona_path: Path,
    *,
    engine: str = "claude",
    model: str | None = None,
    timeout: int = 120,
) -> RegenerationResult:
    """ペルソナファイルの重量ブロックから軽量ブロックを再生成し、ファイルに書き戻す。

    処理フロー:
    1. ペルソナファイルを読み込み、H2 ブロックを分割
    2. 既存の軽量ブロックの sha256 を記録
    3. 重量ブロックを LLM 圧縮して新しい軽量ブロックを生成
    4. 生成結果を v2.1 形式でバリデーション
    5. 新しい軽量ブロックでファイルを上書き保存
    6. sha256 を比較し、変更があれば warning をログ出力

    Args:
        persona_path: ペルソナファイルのパス
        engine: LLM エンジン名
        model: モデル名
        timeout: LLM 呼び出しタイムアウト秒数

    Returns:
        RegenerationResult

    Raises:
        ValueError: v2 形式でないファイル（## 重量 が存在しない）、または生成結果が v2.1 形式に適合しない場合
        RuntimeError: LLM 圧縮に失敗した場合
    """
    from mltgnt.persona.frontmatter import split_yaml_frontmatter

    raw = persona_path.read_text(encoding="utf-8")
    frontmatter_text, body = split_yaml_frontmatter(raw)

    blocks = _split_h2_blocks(body)

    if "重量" not in blocks:
        raise ValueError(
            f"v2 形式ではありません: {persona_path.name} に '## 重量' ブロックが存在しません"
        )

    heavy_text = blocks["重量"]
    existing_light = blocks.get("軽量", "")

    # 初回生成判定（軽量ブロックが空かどうか）
    old_hash = "" if not existing_light.strip() else compute_block_hash(existing_light)

    # LLM 圧縮
    new_light = compress_heavy_to_light(heavy_text, engine=engine, model=model, timeout=timeout)

    # v2.1 形式バリデーション
    _validate_v21_light_block(new_light)

    new_hash = compute_block_hash(new_light)

    changed = old_hash != new_hash

    if changed and old_hash != "":
        logger.warning(
            "[compress] drift detected for %r: old_hash=%s new_hash=%s",
            persona_path.stem,
            old_hash,
            new_hash,
        )

    # ファイルに書き戻す
    new_content = _rebuild_file(raw, body, blocks, new_light)
    persona_path.write_text(new_content, encoding="utf-8")

    return RegenerationResult(
        persona_name=persona_path.stem,
        old_hash=old_hash,
        new_hash=new_hash,
        light_text=new_light,
        changed=changed,
    )


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------


def _validate_v21_light_block(text: str) -> None:
    """生成された軽量ブロックが v2.1 形式に準拠しているか検証する。

    v2.1 形式の要件:
    - リード文（1行以上の非空行）が最初の ** より前に存在すること
    - 必須サブセクション: **口調**、**価値観**、**好意的反応**、**引っかかる**
    - **発言例** が存在する場合は直後（同セクション内）に > で始まる行が必要

    Args:
        text: 検証対象テキスト

    Raises:
        ValueError: 形式要件を満たさない場合
    """
    lines = text.strip().splitlines()

    # リード文チェック: 最初の ** 見出し行より前に非空行が必要
    first_bold_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("**"):
            first_bold_index = i
            break

    if first_bold_index is None:
        # ** 見出しが一切ない場合、必須セクションチェックで捕捉される
        # リード文チェックとしてはとりあえず通過
        pass
    else:
        # 最初の ** 行より前に非空行があるか確認
        lead_lines = [l for l in lines[:first_bold_index] if l.strip()]
        if not lead_lines:
            raise ValueError(
                "v2.1 形式エラー: リード文がありません。最初の太字見出し（**）より前に人物紹介の文章を記述してください。"
            )

    # 必須サブセクションチェック
    required_sections = ["**口調**", "**価値観**", "**好意的反応**", "**引っかかる**"]
    for section in required_sections:
        # ** で始まる行にセクション名が含まれているか（行中位置は問わない）
        found = any(section in line for line in lines)
        if not found:
            section_name = section.strip("*")
            raise ValueError(
                f"v2.1 形式エラー: 必須サブセクション {section} がありません。"
                f"（{section_name} — ... の形式で記述してください）"
            )

    # 発言例チェック: **発言例** がある場合は > で始まる行が後続に必要
    for i, line in enumerate(lines):
        if "**発言例**" in line:
            # 後続の行に > で始まる行があるか確認（次の太字見出しまでの範囲）
            has_quote = False
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip().startswith("**") and "**発言例**" not in next_line:
                    # 別のセクションに入ったので終了
                    break
                if next_line.strip().startswith("> ") or next_line.strip() == ">":
                    has_quote = True
                    break
            if not has_quote:
                raise ValueError(
                    "v2.1 形式エラー: **発言例** の後に引用ブロック（> で始まる行）がありません。"
                )
            break


def _split_h2_blocks(body: str) -> dict[str, str]:
    """本文を H2 見出しで分割し、{見出し名: テキスト} を返す。

    v2 形式では "軽量", "重量", "参照" の3ブロックが期待される。
    """
    blocks: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        m = re.match(r"^##\s+(.+)", line)
        if m:
            if current_key is not None:
                blocks[current_key] = "\n".join(current_lines).strip()
            current_key = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        blocks[current_key] = "\n".join(current_lines).strip()

    return blocks


def _rebuild_file(
    original_raw: str,
    original_body: str,
    blocks: dict[str, str],
    new_light: str,
) -> str:
    """軽量ブロックを new_light で置き換えてファイル内容全体を再構築する。

    frontmatter は original_raw から取り出す（変更しない）。
    H2 ブロックの順序は 軽量→重量→参照 を維持する。
    """

    # frontmatter 部分（--- ... --- の行を含む）を抽出
    # split_yaml_frontmatter はメタ辞書とボディを返すが、
    # 元の frontmatter テキストを保持するために original_raw から直接取得する
    fm_end = _find_frontmatter_end(original_raw)
    if fm_end == -1:
        frontmatter_section = ""
        separator = ""
    else:
        frontmatter_section = original_raw[:fm_end]
        separator = "\n"

    # 新しいボディを構築（ブロック順序を維持）
    section_order = list(blocks.keys())
    new_sections: list[str] = []
    for key in section_order:
        if key == "軽量":
            text = new_light
        else:
            text = blocks[key]
        if text:
            new_sections.append(f"## {key}\n\n{text}\n")
        else:
            new_sections.append(f"## {key}\n")

    new_body = "\n".join(new_sections)
    return f"{frontmatter_section}{separator}{new_body}"


def _find_frontmatter_end(raw: str) -> int:
    """raw テキストから frontmatter の終端位置（2番目の --- 行の後）を返す。

    frontmatter がない場合は -1 を返す。
    """
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return -1

    pos = len(lines[0])
    for line in lines[1:]:
        pos += len(line)
        if line.strip() == "---":
            return pos

    return -1
