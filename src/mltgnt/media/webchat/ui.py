"""Static single-page UI of the WebChat medium (served by ``GET /``)."""

from __future__ import annotations

__all__ = ["INDEX_HTML"]

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>webchat</title>
<style>
  body { font-family: sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }
  #log { flex: 1; overflow-y: auto; padding: 8px; }
  .msg { margin: 4px 0; padding: 6px 8px; border-radius: 6px; background: #f2f2f2; white-space: pre-wrap; }
  .msg.reply { margin-left: 24px; background: #eaf1fb; }
  .msg.selected { outline: 2px solid #3b73c4; }
  .meta { font-size: 0.75em; color: #666; }
  .status { margin-left: 6px; font-weight: bold; }
  form { display: flex; gap: 6px; padding: 8px; border-top: 1px solid #ccc; }
  textarea { flex: 1; min-height: 2.5em; }
</style>
</head>
<body>
<div id="log"></div>
<form id="form">
  <textarea id="text" placeholder="message (click a message to reply in its thread)"></textarea>
  <button type="submit">send</button>
</form>
<script>
(function () {
  var log = document.getElementById("log");
  var form = document.getElementById("form");
  var input = document.getElementById("text");
  var threadTs = null;

  function render(row) {
    var id = "m-" + row.message_id;
    var el = document.getElementById(id);
    if (!el) {
      el = document.createElement("div");
      el.id = id;
      el.className = "msg" + (row.thread_ts ? " reply" : "");
      el.addEventListener("click", function () {
        var root = row.thread_ts || row.message_id;
        threadTs = threadTs === root ? null : root;
        var nodes = log.querySelectorAll(".msg");
        for (var i = 0; i < nodes.length; i++) { nodes[i].classList.remove("selected"); }
        if (threadTs) { el.classList.add("selected"); }
      });
      var parent = row.thread_ts ? document.getElementById("m-" + row.thread_ts) : null;
      if (parent) {
        var after = parent;
        while (after.nextSibling && after.nextSibling.dataset.thread === row.thread_ts) {
          after = after.nextSibling;
        }
        log.insertBefore(el, after.nextSibling);
      } else {
        log.appendChild(el);
      }
    }
    el.dataset.thread = row.thread_ts || "";
    el.textContent = "";
    var meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = (row.author || "") + " " + (row.ts || "");
    if (row.status) {
      var status = document.createElement("span");
      status.className = "status";
      status.textContent = "[" + row.status + "]";
      meta.appendChild(status);
    }
    var body = document.createElement("div");
    body.textContent = row.text || "";
    el.appendChild(meta);
    el.appendChild(body);
    log.scrollTop = log.scrollHeight;
  }

  fetch("messages").then(function (r) { return r.json(); }).then(function (rows) {
    rows.forEach(render);
    var source = new EventSource("stream");
    ["message", "update", "status"].forEach(function (kind) {
      source.addEventListener(kind, function (e) { render(JSON.parse(e.data)); });
    });
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value.trim();
    if (!text) { return; }
    var body = { text: text };
    if (threadTs) { body.thread_ts = threadTs; }
    fetch("messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) { if (r.ok) { input.value = ""; } });
  });
})();
</script>
</body>
</html>
"""
