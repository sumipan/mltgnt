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
  #header { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; border-bottom: 1px solid #ccc; }
  #header span { font-weight: bold; }
  #saved-link { font-size: 0.85em; color: #3b73c4; text-decoration: none; cursor: pointer; }
  #layout { flex: 1; display: flex; min-height: 0; }
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; border-right: 1px solid #ccc; }
  #log { flex: 1; overflow-y: auto; padding: 8px; }
  #panel { width: 360px; display: none; flex-direction: column; min-width: 0; }
  #panel.open { display: flex; }
  #panel-header { padding: 8px; border-bottom: 1px solid #ccc; font-weight: bold; display: flex; justify-content: space-between; }
  #panel-body { flex: 1; overflow-y: auto; padding: 8px; }
  .msg { margin: 4px 0; padding: 6px 8px; border-radius: 6px; background: #f2f2f2; cursor: pointer; position: relative; }
  .msg:hover .save { opacity: 1; }
  .msg.selected { outline: 2px solid #3b73c4; }
  .msg .head { display: flex; align-items: center; gap: 6px; }
  .avatar { width: 24px; height: 24px; border-radius: 50%; object-fit: cover; flex-shrink: 0; }
  .avatar.initial { display: flex; align-items: center; justify-content: center; background: #7c5cfc; color: #fff; font-size: 0.7em; font-weight: bold; }
  .avatar.tiny { width: 18px; height: 18px; font-size: 0.6em; }
  .meta { font-size: 0.75em; color: #666; flex: 1; }
  .time { margin-left: 4px; }
  .body { white-space: pre-wrap; margin-top: 4px; }
  .body code { background: #e8e8e8; padding: 0 3px; border-radius: 3px; }
  .thread-summary { font-size: 0.8em; color: #3b73c4; margin-top: 4px; display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
  .thread-summary .avatars { display: flex; gap: 2px; }
  .reactions { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .chip { display: inline-flex; align-items: center; gap: 3px; padding: 1px 6px; border-radius: 10px; border: 1px solid #ccc; background: #fff; font-size: 0.8em; }
  .state { font-size: 0.85em; margin-left: 4px; }
  .save { position: absolute; top: 6px; right: 8px; border: none; background: none; cursor: pointer; font-size: 1em; opacity: 0; padding: 0; line-height: 1; color: #888; }
  .save.on { opacity: 1; color: #f5a623; }
  .divider { text-align: center; font-size: 0.75em; color: #888; margin: 8px 0; border-top: 1px solid #ddd; padding-top: 6px; }
  form { display: flex; gap: 6px; padding: 8px; border-top: 1px solid #ccc; }
  textarea { flex: 1; min-height: 2.5em; resize: vertical; }
  #panel-form.hidden { display: none; }
</style>
</head>
<body>
<div id="header">
  <span>webchat</span>
  <a href="#" id="saved-link">Saved</a>
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
    <form id="panel-form">
      <textarea id="panel-text" placeholder="reply"></textarea>
      <button type="submit">send</button>
    </form>
  </div>
</div>
<script>
(function () {
  var log = document.getElementById("log");
  var panel = document.getElementById("panel");
  var panelBody = document.getElementById("panel-body");
  var panelTitle = document.getElementById("panel-title");
  var panelForm = document.getElementById("panel-form");
  var panelInput = document.getElementById("panel-text");
  var form = document.getElementById("form");
  var input = document.getElementById("text");
  var threadTs = null;
  var panelMode = null;
  var config = { avatars: {}, display_names: {} };
  var rowsById = {};
  var EMOJI = {
    thumbsup: "\\uD83D\\uDC4D",
    bulb: "\\uD83D\\uDCA1",
    eyes: "\\uD83D\\uDC40",
    thinking_face: "\\uD83E\\uDD14",
    white_check_mark: "\\u2705",
    x: "\\u274C",
    hourglass: "\\u23F3"
  };

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

  function displayName(author) {
    return config.display_names[author] || author || "";
  }

  function avatarInitial(author) {
    var name = displayName(author);
    return name ? name.charAt(0).toUpperCase() : "?";
  }

  function avatarSrc(author) {
    var file = config.avatars[author];
    return file ? "assets/" + file : "";
  }

  function formatClock(ts) {
    if (!ts) { return ""; }
    var d = new Date(ts);
    if (isNaN(d.getTime())) { return ts; }
    var hh = String(d.getHours()).padStart(2, "0");
    var mm = String(d.getMinutes()).padStart(2, "0");
    return hh + ":" + mm;
  }

  function formatLastReply(ts) {
    if (!ts) { return ""; }
    var d = new Date(ts);
    if (isNaN(d.getTime())) { return ""; }
    var now = new Date();
    var hh = String(d.getHours()).padStart(2, "0");
    var mm = String(d.getMinutes()).padStart(2, "0");
    var sameDay = d.getFullYear() === now.getFullYear()
      && d.getMonth() === now.getMonth()
      && d.getDate() === now.getDate();
    if (sameDay) {
      return "Last reply today at " + hh + ":" + mm;
    }
    var y = d.getFullYear();
    var mo = String(d.getMonth() + 1).padStart(2, "0");
    var day = String(d.getDate()).padStart(2, "0");
    return "Last reply " + y + "-" + mo + "-" + day + " " + hh + ":" + mm;
  }

  function reactionEmoji(name) {
    return EMOJI[name] || (":" + name + ":");
  }

  function makeAvatar(author, tiny) {
    var src = avatarSrc(author);
    if (src) {
      var img = document.createElement("img");
      img.className = "avatar" + (tiny ? " tiny" : "");
      img.src = src;
      img.alt = "";
      return img;
    }
    var el = document.createElement("div");
    el.className = "avatar initial" + (tiny ? " tiny" : "");
    el.textContent = avatarInitial(author);
    return el;
  }

  function renderState(meta, status) {
    var old = meta.querySelector(".state");
    if (old) { old.remove(); }
    if (status === "working") {
      var span = document.createElement("span");
      span.className = "state";
      span.textContent = "\\u23F3";
      meta.appendChild(span);
    } else if (status === "failed") {
      var fail = document.createElement("span");
      fail.className = "state";
      fail.textContent = "\\u274C";
      meta.appendChild(fail);
    } else if (status === "cancelled") {
      var cancel = document.createElement("span");
      cancel.className = "state";
      cancel.style.color = "#888";
      cancel.textContent = "\\u274C";
      meta.appendChild(cancel);
    }
  }

  function renderReactions(el, reactions) {
    var old = el.querySelector(".reactions");
    if (old) { old.remove(); }
    if (!reactions || !reactions.length) { return; }
    var counts = {};
    reactions.forEach(function (name) {
      counts[name] = (counts[name] || 0) + 1;
    });
    var box = document.createElement("div");
    box.className = "reactions";
    Object.keys(counts).forEach(function (name) {
      var chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = reactionEmoji(name) + " " + counts[name];
      box.appendChild(chip);
    });
    el.appendChild(box);
  }

  function renderThreadSummary(el, row) {
    var old = el.querySelector(".thread-summary");
    if (old) { old.remove(); }
    if (!row.reply_count) { return; }
    var summary = document.createElement("div");
    summary.className = "thread-summary";
    var avatars = document.createElement("span");
    avatars.className = "avatars";
    var parts = row.participants || [];
    for (var i = 0; i < parts.length && i < 3; i++) {
      avatars.appendChild(makeAvatar(parts[i], true));
    }
    summary.appendChild(avatars);
    var label = document.createElement("span");
    label.textContent = row.reply_count + (row.reply_count === 1 ? " reply" : " replies");
    if (row.last_reply_ts) {
      label.textContent += " \\u00b7 " + formatLastReply(row.last_reply_ts);
    }
    summary.appendChild(label);
    summary.addEventListener("click", function (e) {
      e.stopPropagation();
      openThread(row.message_id);
    });
    el.appendChild(summary);
  }

  function renderSave(el, row) {
    var old = el.querySelector(".save");
    if (old) { old.remove(); }
    if (row.thread_ts) { return; }
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "save" + (row.bookmarked ? " on" : "");
    btn.textContent = row.bookmarked ? "\\u2605" : "\\u2606";
    btn.title = "save";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      fetch("messages/" + row.message_id + "/bookmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bookmarked: !row.bookmarked })
      });
    });
    el.appendChild(btn);
  }

  function fillMessage(el, row, container) {
    el.textContent = "";
    var head = document.createElement("div");
    head.className = "head";
    head.appendChild(makeAvatar(row.author, false));
    var meta = document.createElement("div");
    meta.className = "meta";
    var name = document.createElement("span");
    name.textContent = displayName(row.author);
    meta.appendChild(name);
    var time = document.createElement("span");
    time.className = "time";
    time.textContent = formatClock(row.ts);
    if (row.ts) { time.title = row.ts; }
    meta.appendChild(time);
    renderState(meta, row.status);
    head.appendChild(meta);
    var body = document.createElement("div");
    body.className = "body";
    body.innerHTML = md(row.text || "");
    el.appendChild(head);
    el.appendChild(body);
    renderReactions(el, row.reactions);
    if (container === log && !row.thread_ts) {
      renderThreadSummary(el, row);
    }
    renderSave(el, row);
  }

  function upsertRow(row) {
    rowsById[row.message_id] = row;
    if (panelMode === "thread" && (row.message_id === threadTs || row.thread_ts === threadTs)) {
      render(row, panelBody);
    }
    if (panelMode === "saved" && !row.thread_ts) {
      var panelRow = document.getElementById("panel-body-m-" + row.message_id);
      if (row.bookmarked) { render(row, panelBody); }
      else if (panelRow) { panelRow.remove(); }
    }
    if (!row.thread_ts) {
      render(row, log);
    }
  }

  function refreshParent(threadId) {
    fetch("threads/" + threadId).then(function (r) { return r.json(); }).then(function (rows) {
      rows.forEach(function (row) {
        if (row.message_id === threadId) {
          rowsById[row.message_id] = row;
          render(row, log);
        }
      });
    });
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
    fillMessage(el, row, container);
    if (threadTs === row.message_id) {
      el.classList.add("selected");
    } else {
      el.classList.remove("selected");
    }
  }

  function renderThreadDivider(count) {
    var old = document.getElementById("panel-divider");
    if (old) { old.remove(); }
    if (!count) { return; }
    var div = document.createElement("div");
    div.id = "panel-divider";
    div.className = "divider";
    div.textContent = count + (count === 1 ? " reply" : " replies");
    panelBody.appendChild(div);
  }

  function openThread(rootId) {
    threadTs = rootId;
    panelMode = "thread";
    panel.classList.add("open");
    panelTitle.textContent = "Thread";
    panelForm.classList.remove("hidden");
    panelBody.textContent = "";
    fetch("threads/" + rootId).then(function (r) { return r.json(); }).then(function (rows) {
      panelBody.textContent = "";
      var replyCount = 0;
      rows.forEach(function (row) {
        rowsById[row.message_id] = row;
        if (row.message_id === rootId) {
          render(row, panelBody);
        } else if (row.thread_ts === rootId) {
          replyCount += 1;
        }
      });
      renderThreadDivider(replyCount);
      rows.forEach(function (row) {
        if (row.thread_ts === rootId) {
          render(row, panelBody);
        }
      });
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
    panelForm.classList.add("hidden");
    var nodes = log.querySelectorAll(".msg");
    for (var i = 0; i < nodes.length; i++) { nodes[i].classList.remove("selected"); }
    input.focus();
  }

  function loadSaved() {
    panelMode = "saved";
    threadTs = null;
    panel.classList.add("open");
    panelTitle.textContent = "Saved";
    panelForm.classList.add("hidden");
    panelBody.textContent = "";
    fetch("bookmarks").then(function (r) { return r.json(); }).then(function (rows) {
      panelBody.textContent = "";
      rows.forEach(function (row) {
        rowsById[row.message_id] = row;
        render(row, panelBody);
      });
    });
    var nodes = log.querySelectorAll(".msg");
    for (var i = 0; i < nodes.length; i++) { nodes[i].classList.remove("selected"); }
  }

  document.getElementById("panel-close").addEventListener("click", closePanel);
  document.getElementById("saved-link").addEventListener("click", function (e) {
    e.preventDefault();
    loadSaved();
  });

  function bindEnterSubmit(textarea, targetForm) {
    textarea.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        targetForm.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
  }
  bindEnterSubmit(input, form);
  bindEnterSubmit(panelInput, panelForm);

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
            if (row.bookmarked !== undefined) { base.bookmarked = row.bookmarked; }
            if (row.reactions !== undefined) { base.reactions = row.reactions; }
            if (row.reply_count !== undefined) { base.reply_count = row.reply_count; }
            if (row.last_reply_ts !== undefined) { base.last_reply_ts = row.last_reply_ts; }
            if (row.participants !== undefined) { base.participants = row.participants; }
            upsertRow(base);
          }
          return;
        }
        if (row.thread_ts) {
          if (panelMode === "thread" && row.thread_ts === threadTs) {
            upsertRow(row);
          }
          refreshParent(row.thread_ts);
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
    fetch("messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    }).then(function (r) { if (r.ok) { input.value = ""; } });
  });

  panelForm.addEventListener("submit", function (e) {
    e.preventDefault();
    if (!threadTs) { return; }
    var text = panelInput.value;
    if (!text.trim()) { return; }
    fetch("messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, thread_ts: threadTs })
    }).then(function (r) { if (r.ok) { panelInput.value = ""; } });
  });
})();
</script>
</body>
</html>
"""
