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
  * { box-sizing: border-box; }
  body { font-family: sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }
  #layout { flex: 1; display: flex; min-height: 0; }
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; border-right: 1px solid #ccc; }
  #log { flex: 1; overflow-y: auto; padding: 8px; }
  #panel { width: 360px; display: none; flex-direction: column; min-width: 0; }
  #panel.open { display: flex; }
  #panel-header { padding: 8px; border-bottom: 1px solid #ccc; font-weight: bold; display: flex; justify-content: space-between; }
  #panel-body { flex: 1; overflow-y: auto; padding: 8px; }
  .msg { margin: 4px 0; padding: 6px 8px; border-radius: 6px; background: #f2f2f2; cursor: pointer; }
  .msg.selected { outline: 2px solid #3b73c4; }
  .msg .head { display: flex; align-items: center; gap: 6px; }
  .avatar { width: 24px; height: 24px; border-radius: 4px; object-fit: cover; background: #ddd; flex-shrink: 0; }
  .meta { font-size: 0.75em; color: #666; flex: 1; }
  .body { white-space: pre-wrap; margin-top: 4px; }
  .body code { background: #e8e8e8; padding: 0 3px; border-radius: 3px; }
  .status { margin-left: 6px; font-size: 0.7em; padding: 1px 5px; border-radius: 4px; background: #dde; font-weight: bold; }
  .status.working { background: #ffe9a8; }
  .status.done { background: #c8f0c8; }
  .status.failed { background: #f5c2c2; }
  .replies { font-size: 0.75em; color: #3b73c4; margin-top: 2px; }
  .reactions { font-size: 0.85em; margin-top: 2px; }
  .toolbar { display: flex; gap: 4px; margin-top: 4px; }
  .toolbar button { font-size: 0.75em; padding: 2px 6px; cursor: pointer; }
  #tabs { display: flex; gap: 4px; padding: 6px 8px; border-bottom: 1px solid #ccc; }
  #tabs button { padding: 4px 10px; cursor: pointer; }
  #tabs button.active { font-weight: bold; background: #eaf1fb; border: 1px solid #3b73c4; }
  form { display: flex; gap: 6px; padding: 8px; border-top: 1px solid #ccc; }
  textarea { flex: 1; min-height: 2.5em; resize: vertical; }
</style>
</head>
<body>
<div id="tabs">
  <button type="button" id="tab-chat" class="active">Chat</button>
  <button type="button" id="tab-bookmarks">Bookmarks</button>
</div>
<div id="layout">
  <div id="main">
    <div id="log"></div>
    <form id="form">
      <textarea id="text" placeholder="message"></textarea>
      <button type="submit">send</button>
    </form>
  </div>
  <div id="panel">
    <div id="panel-header">
      <span id="panel-title">Thread</span>
      <button type="button" id="panel-close">x</button>
    </div>
    <div id="panel-body"></div>
  </div>
</div>
<script>
(function () {
  var log = document.getElementById("log");
  var panel = document.getElementById("panel");
  var panelBody = document.getElementById("panel-body");
  var panelTitle = document.getElementById("panel-title");
  var form = document.getElementById("form");
  var input = document.getElementById("text");
  var threadTs = null;
  var panelMode = null;
  var config = { avatars: {}, display_names: {} };
  var rowsById = {};

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function md(text) {
    var s = esc(text || "");
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
    s = s.replace(/\\*([^*]+)\\*/g, "<em>$1</em>");
    s = s.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, function (_match, label, href) {
      var value = href.trim();
      if (!/^(https?:\\/\\/|mailto:|\\/|#)/i.test(value)) { return label; }
      value = value.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
      return '<a href="' + value + '" rel="noopener noreferrer">' + label + "</a>";
    });
    return s;
  }

  function reactionLabel(name) {
    return ":" + name + ":";
  }

  function displayName(author) {
    return config.display_names[author] || author || "";
  }

  function avatarSrc(author) {
    var file = config.avatars[author];
    return file ? "assets/" + file : "";
  }

  function statusClass(status) {
    if (!status) { return ""; }
    if (status === "working") { return " working"; }
    if (status === "done") { return " done"; }
    if (status === "failed") { return " failed"; }
    return "";
  }

  function renderReactions(el, reactions) {
    var old = el.querySelector(".reactions");
    if (old) { old.remove(); }
    if (!reactions || !reactions.length) { return; }
    var box = document.createElement("div");
    box.className = "reactions";
    box.textContent = reactions.map(reactionLabel).join(" ");
    el.appendChild(box);
  }

  function renderReplies(el, row) {
    var old = el.querySelector(".replies");
    if (old) { old.remove(); }
    if (!row.reply_count) { return; }
    var link = document.createElement("div");
    link.className = "replies";
    link.textContent = row.reply_count + (row.reply_count === 1 ? " reply" : " replies");
    link.addEventListener("click", function (e) {
      e.stopPropagation();
      openThread(row.message_id);
    });
    el.appendChild(link);
  }

  function renderToolbar(el, row) {
    var old = el.querySelector(".toolbar");
    if (old) { old.remove(); }
    if (row.thread_ts) { return; }
    var bar = document.createElement("div");
    bar.className = "toolbar";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = row.bookmarked ? "unbookmark" : "bookmark";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      fetch("messages/" + row.message_id + "/bookmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bookmarked: !row.bookmarked })
      });
    });
    bar.appendChild(btn);
    el.appendChild(bar);
  }

  function fillMessage(el, row) {
    el.textContent = "";
    var head = document.createElement("div");
    head.className = "head";
    var src = avatarSrc(row.author);
    if (src) {
      var img = document.createElement("img");
      img.className = "avatar";
      img.src = src;
      img.alt = "";
      head.appendChild(img);
    } else {
      var ph = document.createElement("div");
      ph.className = "avatar";
      head.appendChild(ph);
    }
    var meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = displayName(row.author) + " " + (row.ts || "");
    if (row.status) {
      var status = document.createElement("span");
      status.className = "status" + statusClass(row.status);
      status.textContent = row.status;
      meta.appendChild(status);
    }
    head.appendChild(meta);
    var body = document.createElement("div");
    body.className = "body";
    body.innerHTML = md(row.text || "");
    el.appendChild(head);
    el.appendChild(body);
    renderReactions(el, row.reactions);
    renderReplies(el, row);
    renderToolbar(el, row);
  }

  function upsertRow(row) {
    rowsById[row.message_id] = row;
    if (panelMode === "thread" && (row.message_id === threadTs || row.thread_ts === threadTs)) {
      render(row, panelBody);
    }
    if (panelMode === "bookmarks" && !row.thread_ts) {
      var panelRow = document.getElementById("panel-body-m-" + row.message_id);
      if (row.bookmarked) { render(row, panelBody); }
      else if (panelRow) { panelRow.remove(); }
    }
    if (!row.thread_ts) {
      render(row, log);
    }
  }

  function render(row, container) {
    var id = container.id + "-m-" + row.message_id;
    var el = document.getElementById(id);
    if (!el) {
      el = document.createElement("div");
      el.id = id;
      el.className = "msg";
      el.addEventListener("click", function () {
        if (row.thread_ts) { return; }
        openThread(row.message_id);
      });
      container.appendChild(el);
    }
    fillMessage(el, row);
    if (threadTs === row.message_id) {
      el.classList.add("selected");
    } else {
      el.classList.remove("selected");
    }
  }

  function openThread(rootId) {
    threadTs = rootId;
    panelMode = "thread";
    panel.classList.add("open");
    panelTitle.textContent = "Thread";
    panelBody.textContent = "";
    fetch("threads/" + rootId).then(function (r) { return r.json(); }).then(function (rows) {
      panelBody.textContent = "";
      rows.forEach(function (row) { render(row, panelBody); });
    });
    var nodes = log.querySelectorAll(".msg");
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].classList.toggle("selected", nodes[i].id === "log-m-" + rootId);
    }
  }

  function closePanel() {
    threadTs = null;
    panelMode = null;
    panel.classList.remove("open");
    panelBody.textContent = "";
    var nodes = log.querySelectorAll(".msg");
    for (var i = 0; i < nodes.length; i++) { nodes[i].classList.remove("selected"); }
    input.focus();
  }

  function loadBookmarks() {
    panelMode = "bookmarks";
    panel.classList.add("open");
    panelTitle.textContent = "Bookmarks";
    panelBody.textContent = "";
    fetch("bookmarks").then(function (r) { return r.json(); }).then(function (rows) {
      panelBody.textContent = "";
      rows.forEach(function (row) { render(row, panelBody); });
    });
  }

  document.getElementById("panel-close").addEventListener("click", closePanel);
  document.getElementById("tab-chat").addEventListener("click", function () {
    document.getElementById("tab-chat").classList.add("active");
    document.getElementById("tab-bookmarks").classList.remove("active");
    closePanel();
  });
  document.getElementById("tab-bookmarks").addEventListener("click", function () {
    document.getElementById("tab-bookmarks").classList.add("active");
    document.getElementById("tab-chat").classList.remove("active");
    loadBookmarks();
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { closePanel(); }
  });

  fetch("config").then(function (r) { return r.json(); }).then(function (c) {
    config = c;
  });
  fetch("messages").then(function (r) { return r.json(); }).then(function (rows) {
    log.textContent = "";
    rows.filter(function (row) { return !row.thread_ts; }).forEach(function (row) {
      rowsById[row.message_id] = row;
      render(row, log);
    });
    var source = new EventSource("stream");
    ["message", "update", "status", "bookmark", "reaction"].forEach(function (kind) {
      source.addEventListener(kind, function (e) {
        var row = JSON.parse(e.data);
        if (kind === "bookmark" || kind === "reaction") {
          var base = rowsById[row.message_id];
          if (base) {
            if (kind === "bookmark") { base.bookmarked = row.bookmarked; }
            if (kind === "reaction") {
              base.reactions = base.reactions || [];
              base.reactions.push(row.reaction);
            }
            upsertRow(base);
          }
          return;
        }
        upsertRow(row);
      });
    });
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value;
    if (!text.trim()) { return; }
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
