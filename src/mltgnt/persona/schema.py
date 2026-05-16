"""mltgnt.persona.schema

人物像フロントマターのスキーマ定義とバリデーション。

FM 構造:
    spec_version: str  # 任意。ペルソナスキーマのバージョン（例: "2.2.0"）

    persona:
      name: str          # 必須。ファイル stem と一致
      aliases: list[str] # 任意
      description: str   # 任意

    ops:
      slack:
        username: str
        icon_emoji: str
        icon_url: str
        delegate_ack: str
      chat_model: str
      engine: str
      model: str
      skills: list[str]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 既知の許容キー
# ---------------------------------------------------------------------------

_KNOWN_PERSONA_KEYS: frozenset[str] = frozenset({"name", "aliases", "description"})

_KNOWN_OPS_KEYS: frozenset[str] = frozenset({"slack", "chat_model", "engine", "model", "skills"})

_KNOWN_OPS_SLACK_KEYS: frozenset[str] = frozenset(
    {"username", "icon_emoji", "icon_url", "delegate_ack", "channel",
     "secondary_channels", "nickname"}
)

# 必須セクション（本文中に ## <name> が存在すること）
REQUIRED_SECTIONS: tuple[str, ...] = (
    "基本情報",
    "価値観",
    "反応パターン",
    "口調",
    "アウトプット形式",
)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class PersonaFM:
    """パース済みフロントマターを保持する。"""

    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    spec_version: str | None = None

    # ops namespace
    chat_model: str | None = None
    engine: str = ""
    model: str = ""
    skills: list[str] = field(default_factory=list)
    slack_username: str | None = None
    slack_icon_emoji: str | None = None
    slack_icon_url: str | None = None
    slack_delegate_ack: str | None = None
    slack_channel: str | None = None
    slack_secondary_channels: list[str] = field(default_factory=list)
    slack_nickname: str | None = None

    # 未知のキー（バリデーション用に保持）
    unknown_keys: list[str] = field(default_factory=list)

    # 旧形式（flat キー）使用フラグ（P2-B バリデーション用）
    legacy_keys: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# パース関数
# ---------------------------------------------------------------------------


def parse_fm(meta: dict[str, Any], file_stem: str = "") -> PersonaFM:
    """フロントマター dict から PersonaFM を生成する。

    新スキーマ（persona: / ops: 分離）と旧スキーマ（flat キー）の両方を受け付ける。
    旧キーを使用した場合は deprecation warning を出す。
    """
    unknown: list[str] = []

    # ── トップレベル spec_version ────────────────────────────────────────────
    _sv_raw = meta.get("spec_version")
    spec_version: str | None = str(_sv_raw).strip() if _sv_raw is not None else None

    # ── 新スキーマの persona: namespace ──────────────────────────────────────
    persona_ns: dict[str, Any] = meta.get("persona") or {}
    if isinstance(persona_ns, dict):
        name = str(persona_ns.get("name") or file_stem)
        aliases_raw = persona_ns.get("aliases") or []
        aliases = list(aliases_raw) if isinstance(aliases_raw, list) else []
        description = str(persona_ns.get("description") or "")
        for k in persona_ns:
            if k not in _KNOWN_PERSONA_KEYS:
                unknown.append(f"persona.{k}")
    else:
        name = file_stem
        aliases = []
        description = ""

    # ── 新スキーマの ops: namespace ──────────────────────────────────────────
    ops_ns: dict[str, Any] = meta.get("ops") or {}
    chat_model: str | None = None
    engine: str = ""
    model: str = ""
    skills: list[str] = []
    slack_username: str | None = None
    slack_icon_emoji: str | None = None
    slack_icon_url: str | None = None
    slack_delegate_ack: str | None = None
    slack_channel: str | None = None
    slack_secondary_channels: list[str] = []
    slack_nickname: str | None = None

    if isinstance(ops_ns, dict):
        chat_model = _str_or_none(ops_ns.get("chat_model"))
        if chat_model is not None:
            warnings.warn(
                "ops.chat_model は非推奨です。ops.engine / ops.model に移行してください。"
                " 将来のバージョンで削除されます。",
                DeprecationWarning,
                stacklevel=2,
            )
        engine = _str_or_none(ops_ns.get("engine")) or ""
        model = _str_or_none(ops_ns.get("model")) or ""
        _skills_raw = ops_ns.get("skills")
        skills = list(_skills_raw) if isinstance(_skills_raw, list) else []
        slack_ops = ops_ns.get("slack") or {}
        if isinstance(slack_ops, dict):
            slack_username = _str_or_none(slack_ops.get("username"))
            slack_icon_emoji = _str_or_none(slack_ops.get("icon_emoji"))
            slack_icon_url = _str_or_none(slack_ops.get("icon_url"))
            slack_delegate_ack = _str_or_none(slack_ops.get("delegate_ack"))
            slack_channel = _str_or_none(slack_ops.get("channel"))
            _sec_ch = slack_ops.get("secondary_channels")
            slack_secondary_channels = list(_sec_ch) if isinstance(_sec_ch, list) else []
            slack_nickname = _str_or_none(slack_ops.get("nickname"))
            for k in slack_ops:
                if k not in _KNOWN_OPS_SLACK_KEYS:
                    unknown.append(f"ops.slack.{k}")
        for k in ops_ns:
            if k not in _KNOWN_OPS_KEYS:
                unknown.append(f"ops.{k}")

    # ── 未知トップレベルキー ─────────────────────────────────────────────────
    _legacy_flat: frozenset[str] = frozenset({"chat_model", "slack"})
    legacy_keys_list: list[str] = [k for k in meta if k in _legacy_flat]

    for key in legacy_keys_list:
        warnings.warn(
            f"トップレベルの FM キー '{key}' は非推奨です。"
            f"ops: namespace に移行してください（例: ops.{key}）。"
            " 将来のバージョンで削除されます。",
            DeprecationWarning,
            stacklevel=2,
        )

    # legacy_flat は unknown とは別扱い（validate_fm で個別エラー）
    known_top: frozenset[str] = frozenset({"persona", "ops", "spec_version"}) | _legacy_flat
    for k in meta:
        if k not in known_top:
            unknown.append(k)

    # name フォールバック: 旧スキーマでも persona.name がない場合は file_stem
    if not name:
        name = file_stem

    return PersonaFM(
        name=name,
        aliases=aliases,
        description=description,
        spec_version=spec_version,
        chat_model=chat_model,
        engine=engine,
        model=model,
        skills=skills,
        slack_username=slack_username,
        slack_icon_emoji=slack_icon_emoji,
        slack_icon_url=slack_icon_url,
        slack_delegate_ack=slack_delegate_ack,
        slack_channel=slack_channel,
        slack_secondary_channels=slack_secondary_channels,
        slack_nickname=slack_nickname,
        unknown_keys=unknown,
        legacy_keys=legacy_keys_list,
    )


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_fm(fm: PersonaFM) -> ValidationResult:
    """FM のスキーマ違反・未知キーを検査する。"""
    errors: list[str] = []
    warns: list[str] = []

    if not fm.name:
        errors.append("persona.name が未設定です")

    for k in fm.unknown_keys:
        errors.append(f"未定義の FM キー: {k!r}（スキーマに追加してから使用してください）")

    if fm.legacy_keys:
        errors.append(
            f"旧形式の FM キー {fm.legacy_keys} が使用されています。"
            "ops: namespace（ops.chat_model / ops.slack.*）に移行してください。"
        )

    if fm.chat_model is not None:
        warns.append(
            "ops.chat_model は非推奨です。ops.engine / ops.model に移行してください。"
        )

    return ValidationResult(ok=len(errors) == 0, warnings=warns, errors=errors)


def validate_sections(body: str, fm: PersonaFM) -> ValidationResult:
    """本文に必須セクションが含まれるかチェックする。"""
    warns: list[str] = []
    errors: list[str] = []

    for sec in REQUIRED_SECTIONS:
        # "## N. <名前>" 形式と "## <名前>" 形式の両方を許容
        if f"## {sec}" not in body and f"## 0. {sec}" not in body:
            # 部分一致も試みる（"## 1. 基本情報" など）
            import re
            if not re.search(rf"##\s+(?:\d+\.\s+)?(?:\S+)?{re.escape(sec)}", body):
                warns.append(f"必須セクション「{sec}」が見つかりません")

    return ValidationResult(ok=len(errors) == 0, warnings=warns, errors=errors)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _str_or_none(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# エンジン・コマンドビルダー
# ---------------------------------------------------------------------------

VALID_ENGINES: frozenset[str] = frozenset({"gemini", "claude", "cursor"})

SYSTEM_DEFAULT_ENGINE: str = "claude"
SYSTEM_DEFAULT_MODEL: str = ""


