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
from typing import ClassVar
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

    WEIGHT_MAP: ClassVar[dict[str, str]] = {
        "基本情報": "heavy",
        "価値観": "heavy",
        "反応パターン": "heavy",
        "口調": "heavy",
        "アウトプット形式": "reference",
        "軽量": "light",
    }

    DEFAULT_OP_MODE: ClassVar[str] = "critique"

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

    def format_prompt(self, instruction: str, *, weight: str = "heavy") -> str:
        now = datetime.now(_TZ)
        datetime_line = f"現在日時: {now.strftime('%Y-%m-%d %H:%M:%S')} (JST)\n\n"

        def _weight_for(key: str) -> str | None:
            """WEIGHT_MAP の前方一致でセクションの weight を返す。マッチなしは None。"""
            for wk, wv in self.WEIGHT_MAP.items():
                if key == wk or key.startswith(wk):
                    return wv
            return None

        # WEIGHT_MAP に対応しないセクションがあれば warning + フォールバック
        unknown = [k for k in self.sections if _weight_for(k) is None]
        if unknown:
            logger.warning(
                "[persona] %r: WEIGHT_MAP に未定義のセクション %s — 全セクションを embed します",
                self.name,
                unknown,
            )
            body_part = self.body
        else:
            selected = [
                f"## {key}\n\n{text}"
                for key, text in self.sections.items()
                if _weight_for(key) == weight
            ]
            body_part = "\n\n".join(selected)

        return (
            f"あなたは以下のキャラクターになりきり、その口調・性格で応答してください。\n\n"
            f"{datetime_line}"
            f"{body_part}\n\n"
            f"--- ユーザーからの指示 ---\n\n"
            f"{instruction}"
        )

    def extract_output_format(self, op_mode: str | None = None) -> str | None:
        """アウトプット形式セクションから指定 op_mode の H4 ブロックを返す。"""
        op_mode = op_mode or self.DEFAULT_OP_MODE
        section = self.sections.get("アウトプット形式")
        if section is None:
            return None
        blocks = re.split(r"^#### ", section, flags=re.MULTILINE)
        for block in blocks:
            if block.startswith(op_mode):
                text = block[len(op_mode):].strip()
                return text if text else None
        return None

    def build_review_prompt(self, op_mode: str = "critique") -> str:
        """レビューシステム向けプロンプト断片を返す。"""
        output_fmt = self.extract_output_format(op_mode)
        parts = [self.body]
        if output_fmt:
            parts.append(f"## アウトプット形式\n{output_fmt}")
        return "\n\n".join(parts)


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


def _expand_h3_sections(section_text: str) -> dict[str, str]:
    """H3 (###) でサブセクションに分割してフラット dict を返す。

    各ブロックの 1 行目を見出し名、残りを本文とする。
    H3 の前にある pre-H3 コンテンツ（空文字の場合は破棄）は無視する。
    """
    result: dict[str, str] = {}
    parts = re.split(r"^###\s+", section_text, flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue
        lines = part.split("\n", 1)
        key = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if key:
            result[key] = body
    return result


def _parse_sections(body: str) -> dict[str, str]:
    """本文を ## 見出しでセクション分割する。

    見出し行自体はセクション本文に含めない。
    "## 1. 基本情報" などの番号付き見出しの場合、キーは "基本情報" として正規化する。
    "## 0. ..." で始まるセクション（§0）は除外する。

    v2 形式では `## 重量` と `## 参照` の内容を H3 (###) でさらに展開し、
    H3 見出し名をキーとするフラット dict に統合する。
    """
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    skip_current: bool = False

    for line in body.splitlines():
        m = re.match(r"^##\s+(\d+\.\s+)?(.+)", line)
        if m:
            if current_key is not None and not skip_current:
                sections[current_key] = "\n".join(current_lines).strip()
            num_prefix = m.group(1) or ""
            raw_title = m.group(2).strip()
            # §0 除外
            if num_prefix.strip() == "0.":
                skip_current = True
                current_key = None
                current_lines = []
                continue
            skip_current = False
            # 【必須】などの注釈を除去
            current_key = re.sub(r"\s*【[^】]*】", "", raw_title).strip()
            current_lines = []
        else:
            if not skip_current:
                current_lines.append(line)

    if current_key is not None and not skip_current:
        sections[current_key] = "\n".join(current_lines).strip()

    # v2 対応: ## 重量 / ## 参照 を H3 展開してフラット化
    for h2_key in ("重量", "参照"):
        if h2_key in sections:
            h3_sections = _expand_h3_sections(sections.pop(h2_key))
            sections.update(h3_sections)

    return sections
