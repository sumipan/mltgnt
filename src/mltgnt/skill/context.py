"""mltgnt.skill.context — knowledge + 記憶から extra_context を組み立てる。

設計: Issue #3030
"""
from __future__ import annotations

from pathlib import Path

from mltgnt.skill.models import SkillMeta


def build_extra_context(
    skill_meta: SkillMeta,
    repo_root: Path,
    persona_name: str,
    *,
    knowledge_count: int = 5,
    memory_max_bytes: int = 4096,
) -> str | None:
    """knowledge ファイルと記憶ファイルから extra_context 文字列を組み立てる。

    引数:
        skill_meta: SkillMeta.knowledge_paths に index 済みパスが入っていること
        repo_root: リポジトリルート（memory ファイルの解決に使う）
        persona_name: ペルソナ名（memory ファイル名に使う）
        knowledge_count: 末尾から取るパラグラフ数（空行区切り）
        memory_max_bytes: 末尾から読む記憶バイト数

    戻り値:
        knowledge と記憶の両方が空なら None（extra_context=None で後方互換）
        いずれかあれば「### knowledge（直近 N 件）\\n\\n...」形式の文字列
    """
    knowledge_text = _read_knowledge(skill_meta.knowledge_paths, knowledge_count)
    memory_text = _read_memory(repo_root, persona_name, memory_max_bytes)

    parts: list[str] = []
    if knowledge_text:
        parts.append(f"### knowledge（直近 {knowledge_count} 件）\n\n{knowledge_text}")
    if memory_text:
        parts.append(f"### 記憶（末尾）\n\n{memory_text}")
    return "\n\n".join(parts) if parts else None


def _read_knowledge(knowledge_paths: list[Path], knowledge_count: int) -> str:
    """knowledge_paths の全ファイルを連結し、末尾 knowledge_count パラグラフを返す。"""
    if not knowledge_paths or knowledge_count <= 0:
        return ""
    texts: list[str] = []
    for p in knowledge_paths:
        if p.is_file():
            texts.append(p.read_text(encoding="utf-8"))
    combined = "\n\n".join(texts)
    paragraphs = [seg for seg in combined.split("\n\n") if seg.strip()]
    return "\n\n".join(paragraphs[-knowledge_count:])


def _read_memory(repo_root: Path, persona_name: str, max_bytes: int) -> str:
    """ペルソナ記憶ファイルの末尾 max_bytes を読む。"""
    memory_file = repo_root / "chat" / "memory" / f"{persona_name}.jsonl"
    if not memory_file.is_file() or max_bytes <= 0:
        return ""
    file_size = memory_file.stat().st_size
    start = max(0, file_size - max_bytes)
    with memory_file.open("rb") as f:
        f.seek(start)
        data = f.read(max_bytes)
    text = data.decode("utf-8", errors="replace")
    if start > 0:
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
    return text.lstrip("\n")
