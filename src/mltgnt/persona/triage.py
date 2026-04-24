"""mltgnt.persona.triage

トリアージ用プロンプト構築ロジック。
"""
from __future__ import annotations

import json
import re
import subprocess

TRIAGE_PROFILE_MAX_CHARS = 6000
GEMINI_TIMEOUT_SEC = 25
DEFAULT_TIMEOUT_SEC = 60


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


def extract_json_object(text: str) -> dict | None:
    """LLM の stdout から JSON オブジェクトを1つ取り出す。

    - 空文字なら None
    - ``` で囲まれていればフェンス行を除去
    - 最初の { から最後の } までを json.loads でパース
    """
    s = text.strip()
    if not s:
        return None
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start: end + 1])
    except json.JSONDecodeError:
        return None


def run_triage_once(
    cmd: list[str],
    prompt: str,
    logger,
    *,
    timeout: int,
    use_stdin: bool = False,
) -> dict | None:
    """エンジンコマンドを1回実行して結果を dict で返す。

    エラー・タイムアウト・不正 JSON の場合は None を返す。
    """
    try:
        proc = subprocess.run(
            cmd,
            input=prompt if use_stdin else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Slack triage: timeout (%s)", cmd[0])
        return None
    except OSError as e:
        logger.warning("Slack triage: spawn failed (%s): %s", cmd[0], e)
        return None

    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        logger.warning(
            "Slack triage: %s exit %s stderr=%s", cmd[0], proc.returncode, err[:500]
        )
    data = extract_json_object(out)
    if not isinstance(data, dict):
        return None
    return data


def run_slack_triage(
    instruction: str,
    profile_content: str | None,
    logger,
    *,
    memory_tail: str | None = None,
    engine: str = "",
    model: str = "",
) -> dict | None:
    """トリアージし direct / delegate を返す。1回失敗時に1回リトライする。失敗時は None。"""
    from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, build_engine_command

    engine = engine or SYSTEM_DEFAULT_ENGINE
    prepared = prepare_profile_for_triage(profile_content, logger)
    prompt = build_triage_prompt(instruction, prepared, memory_tail)
    cmd = build_engine_command(engine, model, prompt)

    timeout = GEMINI_TIMEOUT_SEC if engine == "gemini" else DEFAULT_TIMEOUT_SEC

    data = run_triage_once(cmd, prompt, logger, timeout=timeout)
    if data is None:
        logger.warning("Slack triage: invalid JSON, retrying once")
        data = run_triage_once(cmd, prompt, logger, timeout=timeout)
        if data is None:
            logger.warning("Slack triage: retry also failed, falling back to delegate")
            return None
        logger.info("Slack triage: retry succeeded")
    return data
