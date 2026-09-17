"""mltgnt.persona.phrases — read ops.phrases (#3318).

Read PersonaFM-unsupported extension keys from raw YAML. Storage via persona_dir injection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_persona_phrases(
    persona_name: str,
    *,
    persona_dir: Path | None = None,
) -> dict[str, str]:
    """Return the ops.phrases dict for agents/<name>.md. {} if not found."""
    if not persona_name or not str(persona_name).strip():
        return {}
    if persona_dir is None:
        return {}

    path = Path(persona_dir) / f"{persona_name}.md"
    if not path.is_file():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}

    fm_text = text[3:end]
    try:
        import yaml

        data: Any = yaml.safe_load(fm_text)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}
    ops = data.get("ops")
    if not isinstance(ops, dict):
        return {}
    phrases = ops.get("phrases")
    if not isinstance(phrases, dict):
        return {}

    out: dict[str, str] = {}
    for key, value in phrases.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            out[key] = value
    return out
