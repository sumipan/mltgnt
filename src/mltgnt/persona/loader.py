"""mltgnt.persona.loader

エージェントファイルの読み込み・解釈を担当する。

公開するのは load() のみ。呼び出し側は mltgnt.persona.load_persona() 経由で使う。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mltgnt.persona.frontmatter import split_yaml_frontmatter
from mltgnt.persona.schema import PersonaFM, ValidationResult, parse_fm, validate_fm

_TZ = ZoneInfo("Asia/Tokyo")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persona オブジェクト
# ---------------------------------------------------------------------------


@dataclass
class Persona:
    """エージェントファイルの内容を保持するオブジェクト。

    Attributes:
        name:        ペルソナ名（FM の persona.name / ファイル stem）
        fm:          パース済み PersonaFM
        sections:    本文セクション辞書（"基本情報" → テキスト など）
        body:        FM を除いた本文全体
        path:        元ファイルパス
    """

    name: str
    fm: PersonaFM
    sections: dict[str, str]
    body: str
    path: Path

    # ------------------------------------------------------------------
    # 後方互換プロパティ
    # ------------------------------------------------------------------

    @property
    def ops_config(self) -> "_OpsConfig":
        return _OpsConfig(self.fm)

    def slack_post_kwargs(self) -> dict[str, str]:
        """chat.postMessage に渡す辞書（username / icon_emoji / icon_url）。"""
        out: dict[str, str] = {}
        if self.fm.slack_username:
            out["username"] = self.fm.slack_username
        if self.fm.slack_icon_emoji:
            out["icon_emoji"] = self.fm.slack_icon_emoji
        if self.fm.slack_icon_url:
            out["icon_url"] = self.fm.slack_icon_url
        return out

    def delegate_ack(self) -> str | None:
        return self.fm.slack_delegate_ack

    def format_prompt(self, instruction: str) -> str:
        now = datetime.now(_TZ)
        datetime_line = f"現在日時: {now.strftime('%Y-%m-%d %H:%M:%S')} (JST)\n\n"
        return (
            f"あなたは以下のキャラクターになりきり、その口調・性格で応答してください。\n\n"
            f"{datetime_line}"
            f"{self.body}\n\n"
            f"--- ユーザーからの指示 ---\n\n"
            f"{instruction}"
        )

    def build_review_prompt(self, op_mode: str = "critique") -> str:
        """レビューシステム向けプロンプト断片を返す。"""
        output_section = self.sections.get("アウトプット形式", "")
        base = self.body
        if output_section:
            return (
                f"{base}\n\n"
                f"## 指示\n以下の op_mode でレビューしてください: {op_mode}\n\n"
                f"{output_section}"
            )
        return base


@dataclass
class _OpsConfig:
    """ops namespace へのアクセスヘルパー。"""

    _fm: PersonaFM

    @property
    def slack(self) -> dict[str, str | None]:
        return {
            "username": self._fm.slack_username,
            "icon_emoji": self._fm.slack_icon_emoji,
            "icon_url": self._fm.slack_icon_url,
            "delegate_ack": self._fm.slack_delegate_ack,
        }

    @property
    def chat_model(self) -> str | None:
        return self._fm.chat_model


# ---------------------------------------------------------------------------
# 読み込み関数
# ---------------------------------------------------------------------------


def load(path: Path) -> Persona:
    """ファイルパスからペルソナを読み込んで Persona を返す。

    ファイルが存在しない場合は FileNotFoundError を送出する。
    YAML フロントマターのパースに失敗した場合は PersonaValidationError を送出する。
    """
    from mltgnt.persona import PersonaValidationError

    if not path.exists():
        raise FileNotFoundError(f"ペルソナファイルが見つかりません: {path}")

    raw = path.read_text(encoding="utf-8")
    meta, body = split_yaml_frontmatter(raw)

    # meta が None の場合は YAML パース失敗
    if meta is None:
        raise PersonaValidationError(
            f"YAML フロントマターのパースに失敗しました: {path}"
        )

    body = body.strip()
    fm = parse_fm(meta, file_stem=path.stem)

    # FM バリデーション（エラーをログに記録）
    result: ValidationResult = validate_fm(fm)
    for err in result.errors:
        logger.warning("[persona] %s: %s", path.name, err)

    sections = _parse_sections(body)

    logger.info(
        "[persona] loaded %r (sections: %s, fm_keys: %s)",
        fm.name,
        list(sections.keys()),
        [k for k in meta if meta[k] is not None],
    )

    return Persona(
        name=fm.name or path.stem,
        fm=fm,
        sections=sections,
        body=body,
        path=path,
    )


def _parse_sections(body: str) -> dict[str, str]:
    """本文を ## 見出しでセクション分割する。

    見出し行自体はセクション本文に含めない。
    "## 1. 基本情報" などの番号付き見出しの場合、キーは "基本情報" として正規化する。
    """
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        m = re.match(r"^##\s+(?:\d+\.\s+)?(.+)", line)
        if m:
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            raw_title = m.group(1).strip()
            # 【必須】などの注釈を除去
            current_key = re.sub(r"\s*【[^】]*】", "", raw_title).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections
