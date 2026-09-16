"""mltgnt.persona.extractor — H2 セクション解析・light/heavy テキスト抽出（#3318）。"""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def parse_sections(text: str) -> dict[str, str]:
    """H2 見出し名をキー、配下テキストを値とする辞書に変換する。

    H1（``# ``）は無視する。frontmatter 除去済みの body を渡すこと。
    """
    if not text:
        return {}

    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def extract(
    sections: dict[str, str],
    mode: Literal["light", "heavy"],
    *,
    body: str = "",
    name: str = "",
) -> str:
    """mode に応じたテキストを返す。フォールバック規則に従う。"""
    if mode == "light":
        if "軽量" in sections:
            return sections["軽量"]
        if "基本情報" in sections:
            return sections["基本情報"]
        logger.warning(
            "ペルソナ '%s': 軽量・基本情報セクションが見つかりません。"
            "body 全文にフォールバックします。",
            name,
        )
        return body[:500]
    if "重量" in sections:
        return sections["重量"]
    return body
