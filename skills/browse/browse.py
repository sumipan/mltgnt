"""skills/browse/browse.py — CLI エントリポイント（サブコマンド: fetch / search）"""
from __future__ import annotations

import json
import sys

from skills.browse.extractor import extract
from skills.browse.fetcher import fetch
from skills.browse.searcher import search

_MAX_CONTENT_LEN = 30_000


def _extract_title(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else ""
    except Exception:
        return ""


def _output(data: dict, *, exit_code: int = 0) -> None:
    print(json.dumps(data, ensure_ascii=False))
    sys.exit(exit_code)


def main(argv: list[str]) -> None:
    """サブコマンドを解析し、結果を JSON で stdout に出力する。"""
    if not argv:
        _output({"error": "subcommand required: fetch <url> | search <query>"}, exit_code=1)

    subcmd = argv[0]

    if subcmd == "fetch":
        if len(argv) < 2:
            _output({"error": "usage: fetch <url>"}, exit_code=1)

        url = argv[1]
        try:
            html = fetch(url)
        except (ValueError, RuntimeError, TimeoutError) as e:
            _output({"error": str(e)}, exit_code=1)

        title = _extract_title(html)
        content = extract(html, url=url)

        truncated = len(content) > _MAX_CONTENT_LEN
        if truncated:
            content = content[:_MAX_CONTENT_LEN]

        _output({"url": url, "title": title, "content": content, "truncated": truncated})

    elif subcmd == "search":
        if len(argv) < 2:
            _output({"error": "usage: search <query> [--top N]"}, exit_code=1)

        # --top パースは簡易実装
        top = 3
        args_rest = argv[1:]
        query_parts = []
        i = 0
        while i < len(args_rest):
            if args_rest[i] == "--top" and i + 1 < len(args_rest):
                try:
                    top = int(args_rest[i + 1])
                except ValueError:
                    _output({"error": f"--top must be an integer, got {args_rest[i+1]!r}"}, exit_code=1)
                i += 2
            else:
                query_parts.append(args_rest[i])
                i += 1

        query = " ".join(query_parts)
        if not query:
            _output({"error": "query is required"}, exit_code=1)

        try:
            results = search(query, top=top)
        except (ValueError, RuntimeError) as e:
            _output({"error": str(e)}, exit_code=1)

        _output({"query": query, "results": results})

    else:
        _output({"error": f"unknown subcommand: {subcmd!r}. Use fetch or search."}, exit_code=1)


if __name__ == "__main__":
    main(sys.argv[1:])
