---
name: browse
description: URL のテキスト取得・Web 検索を行うスキル
argument_hint: "fetch <url> | search <query> [--top N]"
triggers:
  - "URL"
  - "ページを読"
  - "検索して"
  - "サイトの内容"
---

`python $SKILL_DIR/browse.py $ARGUMENTS` を実行し、結果の JSON を返す。

- `fetch <url>`: 指定 URL の本文を Markdown テキストで取得する
- `search <query> [--top N]`: DuckDuckGo で検索し、上位 N 件（デフォルト 3）の本文を取得する

出力は JSON 形式。fetch は `url`, `title`, `content`, `truncated` を含む。search は `query`, `results` を含む。
