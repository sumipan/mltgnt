"""matrix.markdown.to_html (#6101)."""

from __future__ import annotations

from mltgnt.media.matrix.markdown import to_html


def test_heading_list_link_and_escape() -> None:
    html = to_html("# T\n\n- **a** <b>\n\n[x](https://e.example)")
    assert "<h1>T</h1>" in html
    assert "<li><strong>a</strong> &lt;b&gt;</li>" in html
    assert '<a href="https://e.example">x</a>' in html
    assert "<b>" not in html


def test_table_fence_and_unsafe_link() -> None:
    html = to_html("|a|b|\n|---|---|\n|1|2|\n\n```py\nx\n```\n\n[y](javascript:alert(1))")
    assert "<table>" in html
    assert "<pre><code>" in html
    assert "<a " not in html


def test_inline_escapes_once() -> None:
    html = to_html("`a && b` **x<y** [l](https://e.example/?a=1&b=2)")
    assert "<code>a &amp;&amp; b</code>" in html
    assert "<strong>x&lt;y</strong>" in html
    assert '<a href="https://e.example/?a=1&amp;b=2">l</a>' in html
