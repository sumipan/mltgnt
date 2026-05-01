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

LIGHT_BLOCK_MAX_CHARS = 400

_COMPRESS_PROMPT_TEMPLATE = """\
以下のペルソナの重量ブロック（詳細な人物像）を、400文字以内のプレーンテキストに圧縮してください。

制約:
- 400文字以内（厳守）
- マークダウン記法は使わない
- 人物の核心的な特徴（価値観・反応パターン・口調）を優先して残す
- 具体例やエピソードは省略し、本質を要約する

重量ブロック:
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
        400文字以内に圧縮されたプレーンテキスト

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
    4. 新しい軽量ブロックでファイルを上書き保存
    5. sha256 を比較し、変更があれば warning をログ出力

    Args:
        persona_path: ペルソナファイルのパス
        engine: LLM エンジン名
        model: モデル名
        timeout: LLM 呼び出しタイムアウト秒数

    Returns:
        RegenerationResult

    Raises:
        ValueError: v2 形式でないファイル（## 重量 が存在しない）
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
    from mltgnt.persona.frontmatter import split_yaml_frontmatter

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
