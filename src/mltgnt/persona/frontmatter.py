"""mltgnt.persona.frontmatter

人物像 Markdown 先頭の YAML フロントマター（オプション）を解釈する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def split_yaml_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """先頭が `---` … `---` の YAML フロントマターなら除去して本文を返す。

    それ以外は meta は空 dict、本文は元テキストのまま。
    YAML パースに失敗した場合は (None, body) を返す（エラー判定は呼び出し側で行う）。
    """
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return {}, text
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    yaml_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    try:
        meta = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        # None を返してエラーを示す（呼び出し側で PersonaValidationError に変換）
        return None, body  # type: ignore[return-value]
    if not isinstance(meta, dict):
        return {}, text
    return meta, body


def slack_post_kwargs_from_meta(meta: dict[str, Any]) -> dict[str, str]:
    """フロントマターの `slack:` から chat.postMessage 用キーワード引数を作る。"""
    slack = meta.get("slack")
    if not isinstance(slack, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("username", "icon_emoji", "icon_url"):
        val = slack.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            out[key] = s
    return out


def delegate_ack_from_meta(meta: dict[str, Any]) -> str | None:
    """`slack.delegate_ack` があれば delegate 時の一次応答に使う。"""
    slack = meta.get("slack")
    if not isinstance(slack, dict):
        return None
    val = slack.get("delegate_ack")
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def read_persona_markdown(path: Path) -> tuple[str, dict[str, Any], str | None]:
    """人物像ファイルを読み、(プロンプト用本文, meta, 読み込みエラー文言) を返す。

    ファイル欠落時は ("", {}, "…") 。存在するが読めない場合も error を返す。
    """
    try:
        if not path.exists():
            return "", {}, "ファイルが存在しません"
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return "", {}, str(e)
    meta, body = split_yaml_frontmatter(raw)
    if meta is None:
        meta = {}
    return body.strip(), meta, None
