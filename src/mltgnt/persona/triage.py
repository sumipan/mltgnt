"""mltgnt.persona.triage

トリアージ用プロンプト構築ロジック。
"""
from __future__ import annotations

import re

TRIAGE_PROFILE_MAX_CHARS = 6000


def extract_triage_section(markdown: str) -> str | None:
    """人物像 Markdown 内の `## トリアージ用` セクション本文を返す。無ければ None。"""
    m = re.search(r"^##\s+トリアージ用\s*$", markdown, re.MULTILINE)
    if not m:
        return None
    after = markdown[m.end():].lstrip("\n")
    m2 = re.search(r"^##\s+", after, re.MULTILINE)
    if m2:
        body = after[: m2.start()].rstrip()
    else:
        body = after.rstrip()
    return body if body else None


def prepare_profile_for_triage(profile_content: str | None, logger) -> str | None:
    """トリアージ用に人物像を短縮する。"""
    if not profile_content or not profile_content.strip():
        return None
    raw = profile_content.strip()
    section = extract_triage_section(raw)
    if section and section.strip():
        text = section.strip()
        source = "triage_section"
    else:
        text = raw
        source = "full_persona"
    orig_len = len(text)
    truncated = 0
    if len(text) > TRIAGE_PROFILE_MAX_CHARS:
        text = (
            text[:TRIAGE_PROFILE_MAX_CHARS].rstrip()
            + "\n…（以降省略。`## トリアージ用` セクションで要約を置くと安定します）"
        )
        truncated = 1
    logger.info(
        "[slack-triage] triage_profile source=%s original_chars=%d embedded_chars=%d truncated=%d",
        source,
        orig_len,
        len(text),
        truncated,
    )
    return text


def build_triage_prompt(
    instruction: str,
    profile_content: str | None,
    memory_tail: str | None = None,
) -> str:
    """トリアージ用プロンプトを組み立てて返す。"""
    if profile_content:
        profile_block = (
            "【キャラクター設定（この口調・価値観で応答すること）】\n\n"
            f"{profile_content}\n"
        )
    else:
        profile_block = (
            "（エージェントファイルが読み込めなかった。フレンドリーなエージェントキャラとして応答すること。）\n"
        )
    mem_block = ""
    if memory_tail:
        mem_block = (
            "--- 長期メモリ（末尾抜粋・参考） ---\n"
            f"{memory_tail}\n\n"
        )
    return f"""{profile_block}{mem_block}
Slack でユーザーからメンションされた。次を判定し、**JSON 1 オブジェクトだけ**を出力せよ（前置き・コードフェンス禁止）。

- mode が **direct** … 挨拶・雑談・キャラになりきった短〜中文、一般知識で足りる質問など、**リポジトリを触らずその場で答えられる**もの。
  - 例（direct）: 「Asana 連携を作った」「できるようになった」「実装したよ」など **報告・雑談**（実データの取得依頼ではない）。
- mode が **delegate** … 日記・タスク・コードベースの調査・複数ファイル編集・queue/exec や日記ファイルの変更、長文レポート、実行計画、「リポジトリを調べて」「ファイルを読んで」等。
- **必ず delegate**: **今日やること / タスク一覧 / Asana の実タスク**など、**外部 API やリポジトリ上の実データがないと答えられない**内容。推測でタスク名や期日を書いてはならない（CURSOR.md §5.0.2: 一覧全文を flash に渡して代用も禁止）。
  - 例（delegate）: 「今日のタスク一覧を」「Asana のタスクを期日つきで」など **一覧・同期・ファイル反映の依頼**。

キー:
- "mode": "direct" または "delegate"
- mode が direct のとき必須 "reply": キャラクター口調の回答全文（Slack にそのまま投稿）
- mode が delegate のとき任意 "ack": キャラ口調の短い「待ってね」一言（省略可）

--- ユーザーのメッセージ ---
{instruction}
"""
