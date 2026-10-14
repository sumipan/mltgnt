"""Static single-page UI of the WebChat medium (served by ``GET /``).

One channel in a dark, Slack-like layout: channel header and tabs, the message list
with day dividers, and a thread pane on the right (full screen on narrow screens).
Images are attached with the + button, drag and drop or paste, and sent through
``POST /files``. The page is ASCII only; non-ASCII glyphs are ``\\u`` escapes in JS.
"""

from __future__ import annotations

__all__ = ["INDEX_HTML"]

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>general</title>
<style>
  :root {
    --bg: #1a1d21; --bg-hover: #222529; --bg-raised: #222529; --bg-chip: #2c2d31;
    --border: #35373b; --border-strong: #565856;
    --text: #d1d2d3; --text-strong: #f8f8f8; --muted: #9a9b9e;
    --link: #1d9bd1; --mention-bg: rgba(29, 155, 209, 0.12); --selected: rgba(242, 199, 68, 0.08);
    --send: #007a5a;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: Lato, "Slack-Lato", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic UI", "Meiryo UI", "Segoe UI", "Noto Sans JP", sans-serif;
    font-size: 15px; line-height: 1.46668; -webkit-font-smoothing: antialiased;
  }
  button { font: inherit; color: inherit; background: none; border: none; cursor: pointer; padding: 0; }
  svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; display: block; }
  #app { display: flex; height: 100vh; height: 100dvh; }
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; position: relative; }
  #header { display: flex; align-items: center; gap: 8px; height: 49px; padding: 0 12px 0 16px; flex-shrink: 0; }
  #header .star { color: var(--muted); }
  #channel-title { font-size: 18px; font-weight: 900; color: var(--text-strong); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #channel-title .hash { font-weight: 400; margin-right: 2px; }
  .spacer { flex: 1; }
  .hicon { display: inline-flex; align-items: center; gap: 4px; height: 28px; padding: 0 6px; border-radius: 6px; color: var(--text); }
  .hicon:hover { background: var(--bg-hover); }
  .hicon svg { width: 18px; height: 18px; }
  .hicon .count { font-size: 13px; }
  #channel-tabs { display: flex; gap: 2px; padding: 0 12px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .tab { display: inline-flex; align-items: center; gap: 6px; padding: 8px 10px; font-size: 13px; font-weight: 700; color: var(--muted); text-decoration: none; border-bottom: 2px solid transparent; margin-bottom: -1px; }
  .tab svg { width: 15px; height: 15px; }
  .tab:hover { color: var(--text-strong); }
  .tab.active { color: var(--text-strong); border-bottom-color: var(--text-strong); }
  .scroll { flex: 1; overflow-y: auto; overflow-x: hidden; }
  #log { padding: 8px 0 12px; }
  .channel-intro { padding: 40px 20px 12px; }
  .channel-intro .title { font-size: 22px; font-weight: 900; color: var(--text-strong); }
  .channel-intro .sub { color: var(--muted); margin-top: 4px; }
  .history-loading { text-align: center; font-size: 13px; color: var(--muted); padding: 12px 0; }
  .day-divider { position: sticky; top: 0; z-index: 2; display: flex; justify-content: center; margin: 16px 0 4px; }
  .day-divider::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; border-top: 1px solid var(--border); z-index: -1; }
  .day-pill { background: var(--bg); border: 1px solid var(--border); border-radius: 24px; padding: 2px 16px; font-size: 13px; font-weight: 700; color: var(--text-strong); }
  .msg { display: flex; gap: 8px; padding: 8px 20px; position: relative; }
  .msg.cont { padding-top: 2px; padding-bottom: 2px; }
  .msg:hover { background: var(--bg-hover); }
  .msg.selected { background: var(--selected); }
  .gutter { width: 36px; flex-shrink: 0; }
  .gutter-time { display: none; font-size: 11px; color: var(--muted); text-align: right; padding-top: 4px; }
  .msg.cont:hover .gutter-time { display: block; }
  .content { flex: 1; min-width: 0; }
  .avatar { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; display: block; }
  .avatar.initial { display: flex; align-items: center; justify-content: center; color: #fff; font-size: 15px; font-weight: 900; }
  .avatar.tiny { width: 24px; height: 24px; border-radius: 4px; font-size: 11px; }
  .meta { display: flex; align-items: baseline; gap: 8px; line-height: 1.3; }
  .meta .name { font-weight: 900; color: var(--text-strong); }
  .meta .time { font-size: 12px; color: var(--muted); }
  .state { font-size: 12px; }
  .body { white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere; }
  .body a { color: var(--link); text-decoration: none; }
  .body a:hover { text-decoration: underline; }
  .body code { font-family: Monaco, Menlo, Consolas, monospace; font-size: 12px; color: #e8912d; background: var(--bg-chip); border: 1px solid var(--border); border-radius: 3px; padding: 1px 3px; }
  .body pre { font-family: Monaco, Menlo, Consolas, monospace; font-size: 12px; background: var(--bg-raised); border: 1px solid var(--border); border-radius: 4px; padding: 8px; margin: 4px 0; white-space: pre-wrap; }
  .body blockquote { margin: 2px 0; padding-left: 12px; border-left: 4px solid var(--border-strong); }
  .mention { color: var(--link); background: var(--mention-bg); border-radius: 3px; padding: 0 2px; }
  .files { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
  .file-name { font-size: 13px; color: var(--muted); margin-bottom: 4px; }
  .file img { display: block; max-width: min(360px, 100%); max-height: 360px; border-radius: 8px; border: 1px solid var(--border); cursor: zoom-in; background: var(--bg-raised); }
  .reactions { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .chip { display: inline-flex; align-items: center; gap: 4px; height: 24px; padding: 0 8px; border-radius: 12px; background: var(--bg-chip); border: 1px solid transparent; font-size: 12px; font-weight: 700; color: var(--text); }
  .chip.mine { background: rgba(29, 155, 209, 0.15); border-color: var(--link); color: var(--link); }
  .chip .emoji { font-size: 15px; }
  .thread-summary { display: inline-flex; align-items: center; gap: 8px; margin-top: 4px; padding: 4px 8px 4px 4px; border-radius: 6px; border: 1px solid transparent; cursor: pointer; font-size: 13px; max-width: 100%; }
  .thread-summary:hover { background: var(--bg); border-color: var(--border); }
  .thread-summary .avatars { display: flex; gap: 4px; }
  .thread-summary .count { color: var(--link); font-weight: 700; white-space: nowrap; }
  .thread-summary .last { color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .actions { display: none; position: absolute; top: -14px; right: 20px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 2px; z-index: 3; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3); }
  .msg:hover .actions { display: flex; }
  .actions button { width: 32px; height: 32px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; color: var(--muted); }
  .actions button:hover { background: var(--bg-hover); color: var(--text-strong); }
  .save.on, .save.on:hover { color: #e01e5a; }
  .save.on svg { fill: currentColor; }
  .divider { display: flex; align-items: center; gap: 12px; margin: 4px 20px; font-size: 13px; color: var(--muted); }
  .divider::after { content: ""; flex: 1; border-top: 1px solid var(--border); }
  .composer { padding: 0 20px 20px; flex-shrink: 0; }
  .composer-box { border: 1px solid var(--border-strong); border-radius: 8px; background: var(--bg-raised); position: relative; }
  .composer-box:focus-within { border-color: #8e8e8e; }
  .composer-box.dragging { border-color: var(--link); box-shadow: 0 0 0 1px var(--link); }
  .fmt-bar { display: flex; align-items: center; gap: 2px; padding: 4px 6px 0; color: var(--muted); }
  .composer-box.no-fmt .fmt-bar { display: none; }
  .fmt-bar button { width: 28px; height: 28px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-weight: 700; }
  .fmt-bar button:hover { background: var(--bg-chip); color: var(--text-strong); }
  .fmt-bar .sep { width: 1px; height: 20px; background: var(--border); margin: 0 4px; }
  .composer textarea { display: block; width: 100%; min-height: 40px; max-height: 40vh; resize: none; border: none; outline: none; background: transparent; color: var(--text); font: inherit; padding: 8px 12px; }
  .composer textarea::placeholder { color: var(--muted); }
  .attachments { display: none; flex-wrap: wrap; gap: 8px; padding: 8px 12px 0; }
  .attachments.show { display: flex; }
  .attach { position: relative; width: 64px; height: 64px; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
  .attach img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .attach.uploading img { opacity: 0.4; }
  .attach.failed { border-color: #e01e5a; }
  .attach.failed img { opacity: 0.3; }
  .attach .remove { position: absolute; top: 2px; right: 2px; width: 20px; height: 20px; border-radius: 50%; background: rgba(0, 0, 0, 0.7); color: #fff; display: flex; align-items: center; justify-content: center; }
  .attach .remove svg { width: 12px; height: 12px; stroke-width: 2.5; }
  .composer-actions { display: flex; align-items: center; gap: 2px; padding: 4px 6px 6px; color: var(--muted); }
  .composer-actions > button { width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; }
  .composer-actions > button:hover { background: var(--bg-chip); color: var(--text-strong); }
  .composer-actions .attach-btn { background: var(--bg-chip); }
  .composer-actions .aa { font-size: 14px; font-weight: 700; text-decoration: underline; border-radius: 4px; }
  .send { width: 32px; height: 28px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; color: var(--muted); }
  .send:not(:disabled) { background: var(--send); color: #fff; }
  .send:disabled { cursor: default; }
  .notice { font-size: 12px; color: #e01e5a; padding: 0 12px 4px; min-height: 0; }
  .emoji-pop { display: none; position: absolute; bottom: 100%; left: 0; margin-bottom: 4px; background: var(--bg-raised); border: 1px solid var(--border); border-radius: 8px; padding: 6px; z-index: 5; flex-wrap: wrap; width: 260px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); }
  .emoji-pop.open { display: flex; }
  .emoji-pop button { width: 32px; height: 32px; border-radius: 4px; font-size: 20px; }
  .emoji-pop button:hover { background: var(--bg-chip); }
  #panel { width: 440px; max-width: 45%; display: none; flex-direction: column; min-width: 0; border-left: 1px solid var(--border); background: var(--bg); }
  #panel.open { display: flex; }
  #panel-header { display: flex; align-items: center; gap: 8px; height: 49px; padding: 0 12px 0 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  #panel-title { font-size: 18px; font-weight: 900; color: var(--text-strong); }
  #panel-sub { font-size: 13px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #panel-back { display: none; }
  #panel-body { padding: 8px 0 12px; }
  #panel-form.hidden { display: none; }
  #lightbox { display: none; position: fixed; inset: 0; z-index: 20; background: rgba(0, 0, 0, 0.85); align-items: center; justify-content: center; flex-direction: column; gap: 8px; cursor: zoom-out; }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 92vw; max-height: 86vh; border-radius: 4px; }
  #lightbox .caption { color: var(--text); font-size: 13px; }
  @media (max-width: 760px) {
    #panel { position: fixed; inset: 0; width: 100%; max-width: none; z-index: 10; border-left: none; }
    #panel-back { display: inline-flex; }
    #panel-close { display: none; }
    .hide-narrow { display: none; }
    .msg { padding-left: 12px; padding-right: 12px; }
    .composer { padding: 0 8px 8px; }
    .actions { right: 12px; }
  }
</style>
</head>
<body>
<div id="app">
  <div id="main">
    <div id="header">
      <span class="star"><svg viewBox="0 0 24 24"><polygon points="12 3 14.8 8.9 21 9.6 16.4 13.9 17.7 20 12 16.9 6.3 20 7.6 13.9 3 9.6 9.2 8.9"/></svg></span>
      <span id="channel-title"><span class="hash">#</span><span id="channel-name">general</span></span>
      <span class="spacer"></span>
      <span class="hicon hide-narrow" title="Members"><svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18 14.3c2 .7 3.5 2.7 3.5 5.7"/></svg><span class="count" id="member-count">1</span></span>
      <span class="hicon hide-narrow" title="Huddle"><svg viewBox="0 0 24 24"><path d="M4 15v-3a8 8 0 0 1 16 0v3"/><rect x="3" y="14" width="4" height="6" rx="1.5"/><rect x="17" y="14" width="4" height="6" rx="1.5"/></svg></span>
      <span class="hicon" title="Notifications"><svg viewBox="0 0 24 24"><path d="M6 16v-5a6 6 0 0 1 12 0v5l1.5 2h-15z"/><path d="M10 20.5a2 2 0 0 0 4 0"/></svg></span>
      <span class="hicon" title="Search"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l5 5"/></svg></span>
      <span class="hicon" title="More"><svg viewBox="0 0 24 24"><circle cx="12" cy="5" r="1" fill="currentColor"/><circle cx="12" cy="12" r="1" fill="currentColor"/><circle cx="12" cy="19" r="1" fill="currentColor"/></svg></span>
    </div>
    <div id="channel-tabs">
      <a href="#" class="tab active" id="messages-tab"><svg viewBox="0 0 24 24"><path d="M4 5h16v11H9l-5 4z"/></svg>Messages</a>
      <a href="#" class="tab" id="saved-link"><svg viewBox="0 0 24 24"><path d="M7 4h10v16l-5-4-5 4z"/></svg>Saved</a>
    </div>
    <div id="log" class="scroll"></div>
    <form id="form" class="composer">
      <div class="composer-box">
        <div class="fmt-bar">
          <button type="button" data-fmt="bold" title="Bold">B</button>
          <button type="button" data-fmt="italic" title="Italic"><i>I</i></button>
          <button type="button" data-fmt="strike" title="Strikethrough"><s>S</s></button>
          <span class="sep"></span>
          <button type="button" data-fmt="link" title="Link"><svg viewBox="0 0 24 24"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/></svg></button>
          <button type="button" data-fmt="ol" title="Ordered list"><svg viewBox="0 0 24 24"><path d="M10 6h10M10 12h10M10 18h10M4 5h1v4M4 18h2l-2 2h2"/></svg></button>
          <button type="button" data-fmt="ul" title="Bulleted list"><svg viewBox="0 0 24 24"><path d="M10 6h10M10 12h10M10 18h10"/><circle cx="5" cy="6" r="1" fill="currentColor"/><circle cx="5" cy="12" r="1" fill="currentColor"/><circle cx="5" cy="18" r="1" fill="currentColor"/></svg></button>
          <span class="sep"></span>
          <button type="button" data-fmt="code" title="Code"><svg viewBox="0 0 24 24"><path d="M9 7l-5 5 5 5M15 7l5 5-5 5"/></svg></button>
          <button type="button" data-fmt="codeblock" title="Code block"><svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M10 10l-2 2 2 2M14 10l2 2-2 2"/></svg></button>
        </div>
        <div class="attachments" id="attachments"></div>
        <textarea id="text" rows="1" placeholder="Message #general"></textarea>
        <div class="notice" id="notice"></div>
        <div class="composer-actions">
          <button type="button" class="attach-btn" id="attach-btn" title="Attach image"><svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></button>
          <button type="button" class="aa" title="Formatting">Aa</button>
          <button type="button" class="emoji-btn" title="Emoji"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8.5 14.5a4.5 4.5 0 0 0 7 0"/><circle cx="9" cy="10" r=".6" fill="currentColor"/><circle cx="15" cy="10" r=".6" fill="currentColor"/></svg></button>
          <button type="button" class="at-btn" title="Mention"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.5"/><path d="M15.5 12v1.5a2.5 2.5 0 0 0 5 0V12a8.5 8.5 0 1 0-3.4 6.8"/></svg></button>
          <span class="spacer"></span>
          <button type="submit" class="send" title="Send" disabled><svg viewBox="0 0 24 24"><path d="M4 20l16-8L4 4l2 8z"/><path d="M6 12h7"/></svg></button>
        </div>
        <div class="emoji-pop"></div>
      </div>
      <input type="file" id="file-input" accept="image/png,image/jpeg,image/gif,image/webp" multiple hidden>
    </form>
  </div>
  <div id="panel">
    <div id="panel-header">
      <button type="button" class="hicon" id="panel-back" title="Back"><svg viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg></button>
      <span id="panel-title">Thread</span>
      <span id="panel-sub"></span>
      <span class="spacer"></span>
      <button type="button" class="hicon" id="panel-close" title="Close"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
    </div>
    <div id="panel-body" class="scroll"></div>
    <form id="panel-form" class="composer hidden">
      <div class="composer-box no-fmt">
        <div class="fmt-bar">
          <button type="button" data-fmt="bold" title="Bold">B</button>
          <button type="button" data-fmt="italic" title="Italic"><i>I</i></button>
          <button type="button" data-fmt="strike" title="Strikethrough"><s>S</s></button>
          <span class="sep"></span>
          <button type="button" data-fmt="link" title="Link"><svg viewBox="0 0 24 24"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/></svg></button>
          <button type="button" data-fmt="ul" title="Bulleted list"><svg viewBox="0 0 24 24"><path d="M10 6h10M10 12h10M10 18h10"/><circle cx="5" cy="6" r="1" fill="currentColor"/><circle cx="5" cy="12" r="1" fill="currentColor"/><circle cx="5" cy="18" r="1" fill="currentColor"/></svg></button>
          <button type="button" data-fmt="code" title="Code"><svg viewBox="0 0 24 24"><path d="M9 7l-5 5 5 5M15 7l5 5-5 5"/></svg></button>
        </div>
        <div class="attachments" id="panel-attachments"></div>
        <textarea id="panel-text" rows="1" placeholder="Reply..."></textarea>
        <div class="notice" id="panel-notice"></div>
        <div class="composer-actions">
          <button type="button" class="attach-btn" id="panel-attach-btn" title="Attach image"><svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></button>
          <button type="button" class="aa" title="Formatting">Aa</button>
          <button type="button" class="emoji-btn" title="Emoji"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8.5 14.5a4.5 4.5 0 0 0 7 0"/><circle cx="9" cy="10" r=".6" fill="currentColor"/><circle cx="15" cy="10" r=".6" fill="currentColor"/></svg></button>
          <button type="button" class="at-btn" title="Mention"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.5"/><path d="M15.5 12v1.5a2.5 2.5 0 0 0 5 0V12a8.5 8.5 0 1 0-3.4 6.8"/></svg></button>
          <span class="spacer"></span>
          <button type="submit" class="send" title="Send" disabled><svg viewBox="0 0 24 24"><path d="M4 20l16-8L4 4l2 8z"/><path d="M6 12h7"/></svg></button>
        </div>
        <div class="emoji-pop"></div>
      </div>
      <input type="file" id="panel-file-input" accept="image/png,image/jpeg,image/gif,image/webp" multiple hidden>
    </form>
  </div>
</div>
<div id="lightbox"><img alt=""><div class="caption"></div></div>
<script>
(function () {
  "use strict";
  var log = document.getElementById("log");
  var panel = document.getElementById("panel");
  var panelBody = document.getElementById("panel-body");
  var panelTitle = document.getElementById("panel-title");
  var panelSub = document.getElementById("panel-sub");
  var panelForm = document.getElementById("panel-form");
  var savedLink = document.getElementById("saved-link");
  var messagesTab = document.getElementById("messages-tab");
  var lightbox = document.getElementById("lightbox");
  var threadTs = null;
  var panelMode = null;
  var config = { avatars: {}, display_names: {}, channel_name: "general", max_upload_bytes: 20971520 };
  var rowsById = {};
  var stick = { "log": true, "panel-body": true };
  var touch = window.matchMedia && window.matchMedia("(hover: none)").matches;
  var GROUP_MS = 5 * 60 * 1000;
  var PAGE = 100;
  var logOrder = [];
  var rebuilding = false;
  var backlog = { loading: false, more: true };
  var IMAGE_TYPES = /^image\/(png|jpeg|gif|webp)$/;
  var AVATAR_COLORS = ["#7c5cfc", "#2bac76", "#e8912d", "#1d9bd1", "#e01e5a", "#9b59b6", "#4a9b8e", "#c7743c"];
  var EMOJI = {
    thumbsup: "\uD83D\uDC4D", "+1": "\uD83D\uDC4D", thumbsdown: "\uD83D\uDC4E", "-1": "\uD83D\uDC4E",
    bulb: "\uD83D\uDCA1", eyes: "\uD83D\uDC40", thinking_face: "\uD83E\uDD14", white_check_mark: "\u2705",
    heavy_check_mark: "\u2714\uFE0F", x: "\u274C", hourglass: "\u231B", hourglass_flowing_sand: "\u23F3",
    pray: "\uD83D\uDE4F", clap: "\uD83D\uDC4F", tada: "\uD83C\uDF89", raised_hands: "\uD83D\uDE4C",
    ok_hand: "\uD83D\uDC4C", muscle: "\uD83D\uDCAA", fire: "\uD83D\uDD25", sparkles: "\u2728", star: "\u2B50",
    heart: "\u2764\uFE0F", smile: "\uD83D\uDE04", smiley: "\uD83D\uDE03", grin: "\uD83D\uDE01", joy: "\uD83D\uDE02",
    sweat_smile: "\uD83D\uDE05", wink: "\uD83D\uDE09", blush: "\uD83D\uDE0A", relieved: "\uD83D\uDE0C",
    sob: "\uD83D\uDE2D", cry: "\uD83D\uDE22", scream: "\uD83D\uDE31", rocket: "\uD83D\uDE80", memo: "\uD83D\uDCDD",
    pencil2: "\u270F\uFE0F", books: "\uD83D\uDCDA", book: "\uD83D\uDCD6", mag: "\uD83D\uDD0D", warning: "\u26A0\uFE0F",
    question: "\u2753", exclamation: "\u2757", bow: "\uD83D\uDE47", wave: "\uD83D\uDC4B", point_up: "\u261D\uFE0F",
    100: "\uD83D\uDCAF", calendar: "\uD83D\uDCC5", spiral_calendar_pad: "\uD83D\uDDD3\uFE0F", clipboard: "\uD83D\uDCCB",
    pushpin: "\uD83D\uDCCC", link: "\uD83D\uDD17", bell: "\uD83D\uDD14", coffee: "\u2615", sunny: "\u2600\uFE0F",
    zap: "\u26A1", seedling: "\uD83C\uDF31", chart_with_upwards_trend: "\uD83D\uDCC8", saluting_face: "\uD83E\uDEE1",
    slightly_smiling_face: "\uD83D\uDE42", upside_down_face: "\uD83D\uDE43", sweat: "\uD83D\uDE13", ok: "\uD83C\uDD97"
  };
  var PICKER = ["thumbsup", "pray", "bow", "white_check_mark", "eyes", "bulb", "thinking_face", "sweat_smile",
    "smile", "joy", "sob", "clap", "tada", "fire", "sparkles", "heart", "muscle", "ok_hand", "memo", "warning",
    "rocket", "100", "coffee", "saluting_face"];
  var ICON_THREAD = '<svg viewBox="0 0 24 24"><path d="M4 5h16v11H9l-5 4z"/></svg>';
  var ICON_SAVE = '<svg viewBox="0 0 24 24"><path d="M7 4h10v16l-5-4-5 4z"/></svg>';
  var ICON_X = '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function anchor(href, label) {
    return '<a href="' + href.replace(/'/g, "&#39;") + '" target="_blank" rel="noopener noreferrer">' + label + "</a>";
  }

  function md(text) {
    var keep = [];
    function stash(html) { keep.push(html); return "\u0000" + (keep.length - 1) + "\u0000"; }
    var s = esc(text || "");
    s = s.replace(/```(?:[A-Za-z0-9_+-]*\n)?([\s\S]*?)```\n?/g, function (_m, code) {
      return stash("<pre>" + code.replace(/\n$/, "") + "</pre>");
    });
    s = s.replace(/`([^`\n]+)`/g, function (_m, code) { return stash("<code>" + code + "</code>"); });
    s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, function (_m, label, href) {
      var value = href.trim();
      if (!/^(https?:\/\/|mailto:|\/|#)/i.test(value)) { return label; }
      return stash(anchor(value, label));
    });
    s = s.replace(/https?:\/\/[^\s<>"'\u0000]+/g, function (url) {
      var trail = (url.match(/[.,;:!?)\]]+$/) || [""])[0];
      var bare = url.slice(0, url.length - trail.length);
      return stash(anchor(bare, bare)) + trail;
    });
    s = s.replace(/^([ \t]*)[-*] (?=\S)/gm, "$1\u2022 ");
    s = s.replace(/^#{1,6} +(.+)$/gm, "<strong>$1</strong>");
    s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
    s = s.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    s = s.replace(/(^|[\s(])@([^\s@<>()\u0000]+)/g, '$1<span class="mention">@$2</span>');
    s = s.replace(/^(?:&gt; ?.*(?:\n|$))+/gm, function (block) {
      return "<blockquote>" + block.replace(/\n$/, "").replace(/^&gt; ?/gm, "") + "</blockquote>";
    });
    for (var i = 0; i < 3 && s.indexOf("\u0000") !== -1; i++) {
      s = s.replace(/\u0000(\d+)\u0000/g, function (_m, n) { return keep[Number(n)]; });
    }
    return s;
  }

  function displayName(author) {
    return config.display_names[author] || author || "";
  }

  function avatarSrc(author) {
    var file = config.avatars[author];
    if (!file) { return ""; }
    if (/^(https?:\/\/|data:image\/)/i.test(file)) { return file; }
    return "assets/" + encodeURIComponent(file);
  }

  function avatarColor(author) {
    var h = 0;
    var s = String(author || "");
    for (var i = 0; i < s.length; i++) { h = (h * 31 + s.charCodeAt(i)) | 0; }
    return AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length];
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
    el.style.background = avatarColor(author);
    var name = displayName(author);
    el.textContent = name ? name.charAt(0).toUpperCase() : "?";
    return el;
  }

  function pad2(n) { return String(n).padStart(2, "0"); }

  function sameDay(a, b) {
    return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  }

  function formatClock(ts) {
    if (!ts) { return ""; }
    var d = new Date(ts);
    if (isNaN(d.getTime())) { return ts; }
    return pad2(d.getHours()) + ":" + pad2(d.getMinutes());
  }

  function dayLabel(ts) {
    var d = new Date(ts);
    if (isNaN(d.getTime())) { return ""; }
    var now = new Date();
    var yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
    if (sameDay(d, now)) { return "Today"; }
    if (sameDay(d, yesterday)) { return "Yesterday"; }
    var WD = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    var MO = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    var n = d.getDate();
    var suffix = (n % 10 === 1 && n !== 11) ? "st" : (n % 10 === 2 && n !== 12) ? "nd" : (n % 10 === 3 && n !== 13) ? "rd" : "th";
    var label = WD[d.getDay()] + ", " + MO[d.getMonth()] + " " + n + suffix;
    return d.getFullYear() === now.getFullYear() ? label : label + ", " + d.getFullYear();
  }

  function formatLastReply(ts) {
    if (!ts) { return ""; }
    var d = new Date(ts);
    if (isNaN(d.getTime())) { return ""; }
    var now = new Date();
    var clock = pad2(d.getHours()) + ":" + pad2(d.getMinutes());
    if (sameDay(d, now)) { return "Last reply today at " + clock; }
    var yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
    if (sameDay(d, yesterday)) { return "Last reply yesterday at " + clock; }
    return "Last reply " + d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate()) + " " + clock;
  }

  function reactionEmoji(name) {
    return EMOJI[name] || (":" + name + ":");
  }

  function atBottom(c) { return c.scrollHeight - c.scrollTop - c.clientHeight < 80; }
  function scrollBottom(c) { c.scrollTop = c.scrollHeight; }
  [log, panelBody].forEach(function (c) {
    c.addEventListener("scroll", function () {
      stick[c.id] = atBottom(c);
      if (c === log && c.scrollTop < 300) { loadOlder(); }
    });
  });

  function renderState(meta, status) {
    var old = meta.querySelector(".state");
    if (old) { old.remove(); }
    var glyph = status === "working" ? "\u23F3" : (status === "failed" || status === "cancelled") ? "\u274C" : "";
    if (!glyph) { return; }
    var span = document.createElement("span");
    span.className = "state";
    span.title = status;
    span.textContent = glyph;
    if (status === "cancelled") { span.style.opacity = "0.5"; }
    meta.appendChild(span);
  }

  function renderReactions(el, reactions) {
    var old = el.querySelector(".reactions");
    if (old) { old.remove(); }
    if (!reactions || !reactions.length) { return; }
    var counts = {};
    var order = [];
    reactions.forEach(function (name) {
      if (!counts[name]) { order.push(name); }
      counts[name] = (counts[name] || 0) + 1;
    });
    var box = document.createElement("div");
    box.className = "reactions";
    order.forEach(function (name) {
      var chip = document.createElement("span");
      chip.className = "chip";
      chip.title = ":" + name + ":";
      var emoji = document.createElement("span");
      emoji.className = "emoji";
      emoji.textContent = reactionEmoji(name);
      chip.appendChild(emoji);
      chip.appendChild(document.createTextNode(String(counts[name])));
      box.appendChild(chip);
    });
    el.appendChild(box);
  }

  function renderFiles(el, files) {
    if (!files || !files.length) { return; }
    var box = document.createElement("div");
    box.className = "files";
    files.forEach(function (f) {
      var fig = document.createElement("div");
      fig.className = "file";
      var label = document.createElement("div");
      label.className = "file-name";
      label.textContent = (f.name || "image") + " \u25BE";
      var img = document.createElement("img");
      img.src = "files/" + f.id;
      img.alt = f.name || "";
      img.loading = "lazy";
      img.addEventListener("load", function () {
        var sc = img.closest(".scroll");
        if (sc && stick[sc.id]) { scrollBottom(sc); }
      });
      img.addEventListener("click", function (e) {
        e.stopPropagation();
        openLightbox(img.src, f.name || "");
      });
      fig.appendChild(label);
      fig.appendChild(img);
      box.appendChild(fig);
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
    for (var i = 0; i < parts.length && i < 5; i++) {
      avatars.appendChild(makeAvatar(parts[i], true));
    }
    summary.appendChild(avatars);
    var count = document.createElement("span");
    count.className = "count";
    count.textContent = row.reply_count + (row.reply_count === 1 ? " reply" : " replies");
    summary.appendChild(count);
    if (row.last_reply_ts) {
      var last = document.createElement("span");
      last.className = "last";
      last.textContent = formatLastReply(row.last_reply_ts);
      summary.appendChild(last);
    }
    summary.addEventListener("click", function (e) {
      e.stopPropagation();
      openThread(row.message_id);
    });
    el.appendChild(summary);
  }

  function renderSave(bar, row) {
    var old = bar.querySelector(".save");
    if (old) { old.remove(); }
    if (row.thread_ts) { return; }
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "save" + (row.bookmarked ? " on" : "");
    btn.innerHTML = ICON_SAVE;
    btn.title = row.bookmarked ? "Remove from saved items" : "Save for later";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var current = rowsById[row.message_id] || row;
      fetch("messages/" + current.message_id + "/bookmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bookmarked: !current.bookmarked })
      });
    });
    bar.appendChild(btn);
  }

  function isContinuation(prev, row) {
    if (!prev || !prev.classList.contains("msg") || prev.dataset.author !== (row.author || "")) { return false; }
    var a = new Date(prev.dataset.ts).getTime();
    var b = new Date(row.ts).getTime();
    return !isNaN(a) && !isNaN(b) && b - a >= 0 && b - a < GROUP_MS && sameDay(new Date(a), new Date(b));
  }

  function fillMessage(el, row, container) {
    el.textContent = "";
    el.dataset.author = row.author || "";
    el.dataset.ts = row.ts || "";
    var cont = isContinuation(el.previousElementSibling, row);
    el.classList.toggle("cont", cont);
    var gutter = document.createElement("div");
    gutter.className = "gutter";
    if (cont) {
      var gt = document.createElement("div");
      gt.className = "gutter-time";
      gt.textContent = formatClock(row.ts);
      gutter.appendChild(gt);
    } else {
      gutter.appendChild(makeAvatar(row.author, false));
    }
    var content = document.createElement("div");
    content.className = "content";
    var meta = document.createElement("div");
    meta.className = "meta";
    if (!cont) {
      var name = document.createElement("span");
      name.className = "name";
      name.textContent = displayName(row.author);
      meta.appendChild(name);
      var time = document.createElement("span");
      time.className = "time";
      time.textContent = formatClock(row.ts);
      if (row.ts) { time.title = row.ts; }
      meta.appendChild(time);
    }
    renderState(meta, row.status);
    if (meta.childNodes.length) { content.appendChild(meta); }
    if (row.text) {
      var body = document.createElement("div");
      body.className = "body";
      body.innerHTML = md(row.text);
      content.appendChild(body);
    }
    renderFiles(content, row.files);
    renderReactions(content, row.reactions);
    if (container === log && !row.thread_ts) {
      renderThreadSummary(content, row);
    }
    el.appendChild(gutter);
    el.appendChild(content);
    var bar = document.createElement("div");
    bar.className = "actions";
    if (container === log && !row.thread_ts) {
      var reply = document.createElement("button");
      reply.type = "button";
      reply.title = "Reply in thread";
      reply.innerHTML = ICON_THREAD;
      reply.addEventListener("click", function (e) {
        e.stopPropagation();
        openThread(row.message_id);
      });
      bar.appendChild(reply);
    }
    renderSave(bar, row);
    if (bar.childNodes.length) { el.appendChild(bar); }
  }

  function insertDayDivider(container, row) {
    var msgs = container.querySelectorAll(".msg");
    var last = msgs.length ? msgs[msgs.length - 1] : null;
    var d = new Date(row.ts);
    if (isNaN(d.getTime())) { return; }
    if (last && sameDay(new Date(last.dataset.ts), d)) { return; }
    var div = document.createElement("div");
    div.className = "day-divider";
    var pill = document.createElement("span");
    pill.className = "day-pill";
    pill.textContent = dayLabel(row.ts);
    div.appendChild(pill);
    container.appendChild(div);
  }

  function updateMemberCount() {
    var seen = {};
    Object.keys(rowsById).forEach(function (id) { seen[rowsById[id].author || ""] = true; });
    document.getElementById("member-count").textContent = String(Math.max(1, Object.keys(seen).length));
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
    var created = false;
    if (!el) {
      if (container === log) { insertDayDivider(container, row); }
      el = document.createElement("div");
      el.id = id;
      el.className = "msg";
      var messageId = row.message_id;
      el.addEventListener("click", function (e) {
        if (!touch || container !== log) { return; }
        var current = rowsById[messageId];
        if (!current || current.thread_ts) { return; }
        if (e.target.closest("a, img, button")) { return; }
        openThread(messageId);
      });
      container.appendChild(el);
      created = true;
    }
    fillMessage(el, row, container);
    el.classList.toggle("selected", panelMode === "thread" && threadTs === row.message_id && container === log);
    if (created && !rebuilding) {
      if (container === log) { logOrder.push(row.message_id); }
      updateMemberCount();
      if (stick[container.id]) { scrollBottom(container); }
    }
  }

  function channelIntro() {
    var box = document.createElement("div");
    box.className = "channel-intro";
    var title = document.createElement("div");
    title.className = "title";
    title.textContent = "# " + config.channel_name;
    var sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = "This is the very beginning of the #" + config.channel_name + " channel.";
    box.appendChild(title);
    box.appendChild(sub);
    return box;
  }

  function rebuildLog() {
    rebuilding = true;
    log.textContent = "";
    if (!backlog.more) { log.appendChild(channelIntro()); }
    logOrder.forEach(function (id) {
      if (rowsById[id]) { render(rowsById[id], log); }
    });
    rebuilding = false;
    updateMemberCount();
  }

  function addRoots(rows) {
    var ids = [];
    rows.forEach(function (row) {
      if (row.thread_ts || logOrder.indexOf(row.message_id) !== -1 || ids.indexOf(row.message_id) !== -1) { return; }
      rowsById[row.message_id] = row;
      ids.push(row.message_id);
    });
    return ids;
  }

  function loadOlder() {
    if (backlog.loading || !backlog.more || !logOrder.length) { return; }
    var oldest = rowsById[logOrder[0]];
    if (!oldest || !oldest.ts) { return; }
    backlog.loading = true;
    var spinner = document.createElement("div");
    spinner.className = "history-loading";
    spinner.textContent = "Loading older messages...";
    log.insertBefore(spinner, log.firstChild);
    fetch("messages?limit=" + PAGE + "&before=" + encodeURIComponent(oldest.ts)).then(function (r) {
      if (!r.ok) { throw new Error("HTTP " + r.status); }
      return r.json();
    }).then(function (rows) {
      var ids = addRoots(rows);
      if (rows.length < PAGE) { backlog.more = false; }
      var height = log.scrollHeight;
      var top = log.scrollTop;
      logOrder = ids.concat(logOrder);
      rebuildLog();
      log.scrollTop = log.scrollHeight - height + top;
      markSelected();
    }).catch(function () {
      spinner.remove();
    }).then(function () {
      backlog.loading = false;
      if (backlog.more && log.scrollHeight <= log.clientHeight) { loadOlder(); }
    });
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

  function markSelected() {
    var nodes = log.querySelectorAll(".msg");
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].classList.toggle("selected", panelMode === "thread" && nodes[i].id === "log-m-" + threadTs);
    }
  }

  function openPanel(mode, title) {
    panelMode = mode;
    panel.classList.add("open");
    panelTitle.textContent = title;
    panelSub.textContent = mode === "thread" ? "# " + config.channel_name : "";
    panelBody.textContent = "";
    stick["panel-body"] = true;
    savedLink.classList.toggle("active", mode === "saved");
    messagesTab.classList.toggle("active", mode !== "saved");
  }

  function openThread(rootId) {
    threadTs = rootId;
    openPanel("thread", "Thread");
    panelForm.classList.remove("hidden");
    clearComposer(panelComposer);
    markSelected();
    fetch("threads/" + rootId).then(function (r) { return r.json(); }).then(function (rows) {
      if (threadTs !== rootId) { return; }
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
      scrollBottom(panelBody);
      if (!touch) { panelComposer.input.focus(); }
    });
  }

  function closePanel() {
    threadTs = null;
    panelMode = null;
    panel.classList.remove("open");
    panelBody.textContent = "";
    panelForm.classList.add("hidden");
    savedLink.classList.remove("active");
    messagesTab.classList.add("active");
    markSelected();
    if (!touch) { mainComposer.input.focus(); }
  }

  function loadSaved() {
    threadTs = null;
    openPanel("saved", "Saved items");
    panelForm.classList.add("hidden");
    markSelected();
    fetch("bookmarks").then(function (r) { return r.json(); }).then(function (rows) {
      panelBody.textContent = "";
      rows.forEach(function (row) {
        rowsById[row.message_id] = row;
        render(row, panelBody);
      });
      panelBody.scrollTop = 0;
    });
  }

  function openLightbox(src, caption) {
    lightbox.querySelector("img").src = src;
    lightbox.querySelector(".caption").textContent = caption;
    lightbox.classList.add("open");
  }

  function closeLightbox() {
    lightbox.classList.remove("open");
    lightbox.querySelector("img").removeAttribute("src");
  }

  lightbox.addEventListener("click", closeLightbox);
  document.getElementById("panel-close").addEventListener("click", closePanel);
  document.getElementById("panel-back").addEventListener("click", closePanel);
  savedLink.addEventListener("click", function (e) {
    e.preventDefault();
    if (panelMode === "saved") { closePanel(); } else { loadSaved(); }
  });
  messagesTab.addEventListener("click", function (e) {
    e.preventDefault();
    if (panelMode === "saved") { closePanel(); }
  });

  function insertAtCursor(input, text) {
    var s = input.selectionStart;
    var e = input.selectionEnd;
    input.value = input.value.slice(0, s) + text + input.value.slice(e);
    input.selectionStart = input.selectionEnd = s + text.length;
    input.focus();
    input.dispatchEvent(new Event("input"));
  }

  function applyFormat(input, fmt) {
    var s = input.selectionStart;
    var e = input.selectionEnd;
    var v = input.value;
    if (fmt === "ul" || fmt === "ol") {
      var lineStart = v.lastIndexOf("\n", s - 1) + 1;
      var prefix = fmt === "ul" ? "- " : "1. ";
      input.value = v.slice(0, lineStart) + prefix + v.slice(lineStart);
      input.selectionStart = input.selectionEnd = e + prefix.length;
    } else {
      var marks = {
        bold: ["**", "**"], italic: ["*", "*"], strike: ["~~", "~~"], code: ["`", "`"],
        codeblock: ["```\n", "\n```"], link: ["[", "](https://)"]
      }[fmt];
      if (!marks) { return; }
      input.value = v.slice(0, s) + marks[0] + v.slice(s, e) + marks[1] + v.slice(e);
      input.selectionStart = s + marks[0].length;
      input.selectionEnd = e + marks[0].length;
    }
    input.focus();
    input.dispatchEvent(new Event("input"));
  }

  function autoGrow(input) {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.4) + "px";
  }

  function notice(c, text) {
    c.notice.textContent = text;
    clearTimeout(c.noticeTimer);
    if (text) { c.noticeTimer = setTimeout(function () { c.notice.textContent = ""; }, 5000); }
  }

  function isUploading(c) {
    return c.pending.some(function (p) { return !p.id && !p.failed; });
  }

  function readyFiles(c) {
    return c.pending.filter(function (p) { return p.id; });
  }

  function updateSend(c) {
    c.send.disabled = c.sending || isUploading(c) || (!c.input.value.trim() && !readyFiles(c).length);
  }

  function renderPending(c) {
    c.preview.textContent = "";
    c.preview.classList.toggle("show", c.pending.length > 0);
    c.pending.forEach(function (item) {
      var box = document.createElement("div");
      box.className = "attach" + (item.failed ? " failed" : item.id ? "" : " uploading");
      box.title = item.failed ? "Upload failed: " + item.name : item.name;
      var img = document.createElement("img");
      img.src = item.url;
      img.alt = "";
      box.appendChild(img);
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove";
      remove.title = "Remove";
      remove.innerHTML = ICON_X;
      remove.addEventListener("click", function () {
        URL.revokeObjectURL(item.url);
        c.pending.splice(c.pending.indexOf(item), 1);
        renderPending(c);
        updateSend(c);
      });
      box.appendChild(remove);
      c.preview.appendChild(box);
    });
  }

  function uploadOne(c, item) {
    fetch("files?name=" + encodeURIComponent(item.name), {
      method: "POST",
      headers: { "Content-Type": item.file.type },
      body: item.file
    }).then(function (r) {
      if (!r.ok) { throw new Error("HTTP " + r.status); }
      return r.json();
    }).then(function (entry) {
      item.id = entry.id;
      item.name = entry.name;
    }).catch(function (err) {
      item.failed = true;
      notice(c, "Upload failed (" + err.message + "): " + item.name);
    }).then(function () {
      renderPending(c);
      updateSend(c);
    });
  }

  function addFiles(c, list) {
    for (var i = 0; i < list.length; i++) {
      var f = list[i];
      if (!IMAGE_TYPES.test(f.type)) {
        notice(c, "Only PNG, JPEG, GIF and WebP images can be attached.");
        continue;
      }
      if (f.size > config.max_upload_bytes) {
        notice(c, "Too large (max " + Math.round(config.max_upload_bytes / 1048576) + " MB): " + f.name);
        continue;
      }
      var item = { file: f, name: f.name || "image", url: URL.createObjectURL(f), id: null, failed: false };
      c.pending.push(item);
      uploadOne(c, item);
    }
    renderPending(c);
    updateSend(c);
  }

  function clearComposer(c) {
    c.pending.forEach(function (p) { URL.revokeObjectURL(p.url); });
    c.pending = [];
    c.input.value = "";
    autoGrow(c.input);
    renderPending(c);
    notice(c, "");
    updateSend(c);
  }

  function submit(c) {
    if (c.threaded && !threadTs) { return; }
    if (c.sending || isUploading(c)) { return; }
    var text = c.input.value;
    var files = readyFiles(c);
    if (!text.trim() && !files.length) { return; }
    var body = { text: text };
    if (files.length) {
      body.files = files.map(function (p) { return { id: p.id, name: p.name }; });
    }
    if (c.threaded) { body.thread_ts = threadTs; }
    c.sending = true;
    updateSend(c);
    fetch("messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      if (r.ok) {
        clearComposer(c);
        stick[c.threaded ? "panel-body" : "log"] = true;
      } else {
        notice(c, "Send failed (HTTP " + r.status + ")");
      }
    }).catch(function () {
      notice(c, "Send failed (network)");
    }).then(function () {
      c.sending = false;
      updateSend(c);
    });
  }

  function makeComposer(form, input, preview, fileInput, attachBtn, noticeEl, threaded) {
    var box = form.querySelector(".composer-box");
    if (touch) { box.classList.add("no-fmt"); }
    var pop = form.querySelector(".emoji-pop");
    var c = {
      form: form, input: input, preview: preview, notice: noticeEl, pending: [], threaded: threaded,
      send: form.querySelector(".send"), sending: false, noticeTimer: null
    };
    attachBtn.addEventListener("click", function () { fileInput.click(); });
    fileInput.addEventListener("change", function () {
      addFiles(c, fileInput.files);
      fileInput.value = "";
    });
    input.addEventListener("paste", function (e) {
      var data = e.clipboardData;
      if (!data || !data.files || !data.files.length || data.getData("text/plain")) { return; }
      var images = [];
      for (var i = 0; i < data.files.length; i++) {
        if (/^image\//.test(data.files[i].type)) { images.push(data.files[i]); }
      }
      if (images.length) {
        e.preventDefault();
        addFiles(c, images);
      }
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
        e.preventDefault();
        submit(c);
      }
    });
    input.addEventListener("input", function () {
      autoGrow(input);
      updateSend(c);
    });
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      submit(c);
    });
    var fmtButtons = form.querySelectorAll("[data-fmt]");
    for (var i = 0; i < fmtButtons.length; i++) {
      (function (btn) {
        btn.addEventListener("mousedown", function (e) { e.preventDefault(); });
        btn.addEventListener("click", function () { applyFormat(input, btn.getAttribute("data-fmt")); });
      })(fmtButtons[i]);
    }
    form.querySelector(".aa").addEventListener("click", function () { box.classList.toggle("no-fmt"); });
    form.querySelector(".at-btn").addEventListener("click", function () { insertAtCursor(input, "@"); });
    PICKER.forEach(function (name) {
      var b = document.createElement("button");
      b.type = "button";
      b.title = ":" + name + ":";
      b.textContent = reactionEmoji(name);
      b.addEventListener("click", function () {
        pop.classList.remove("open");
        insertAtCursor(input, reactionEmoji(name));
      });
      pop.appendChild(b);
    });
    form.querySelector(".emoji-btn").addEventListener("click", function (e) {
      e.stopPropagation();
      pop.classList.toggle("open");
    });
    document.addEventListener("click", function (e) {
      if (!pop.contains(e.target)) { pop.classList.remove("open"); }
    });
    var dropTarget = threaded ? panel : document.getElementById("main");
    var depth = 0;
    dropTarget.addEventListener("dragenter", function (e) {
      if (!e.dataTransfer || Array.prototype.indexOf.call(e.dataTransfer.types, "Files") === -1) { return; }
      if (threaded && panelMode !== "thread") { return; }
      e.preventDefault();
      depth += 1;
      box.classList.add("dragging");
    });
    dropTarget.addEventListener("dragover", function (e) {
      if (e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types, "Files") !== -1) { e.preventDefault(); }
    });
    dropTarget.addEventListener("dragleave", function () {
      depth = Math.max(0, depth - 1);
      if (!depth) { box.classList.remove("dragging"); }
    });
    dropTarget.addEventListener("drop", function (e) {
      depth = 0;
      box.classList.remove("dragging");
      if (!e.dataTransfer || !e.dataTransfer.files.length) { return; }
      if (threaded && panelMode !== "thread") { return; }
      e.preventDefault();
      e.stopPropagation();
      addFiles(c, e.dataTransfer.files);
    });
    return c;
  }

  var mainComposer = makeComposer(
    document.getElementById("form"), document.getElementById("text"), document.getElementById("attachments"),
    document.getElementById("file-input"), document.getElementById("attach-btn"), document.getElementById("notice"), false
  );
  var panelComposer = makeComposer(
    panelForm, document.getElementById("panel-text"), document.getElementById("panel-attachments"),
    document.getElementById("panel-file-input"), document.getElementById("panel-attach-btn"),
    document.getElementById("panel-notice"), true
  );
  document.addEventListener("dragover", function (e) { e.preventDefault(); });
  document.addEventListener("drop", function (e) { e.preventDefault(); });

  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") { return; }
    if (lightbox.classList.contains("open")) { closeLightbox(); return; }
    closePanel();
  });

  function applyConfig(c) {
    config.avatars = c.avatars || {};
    config.display_names = c.display_names || {};
    config.channel_name = c.channel_name || "general";
    config.max_upload_bytes = c.max_upload_bytes || config.max_upload_bytes;
    document.title = config.channel_name;
    document.getElementById("channel-name").textContent = config.channel_name;
    mainComposer.input.placeholder = "Message #" + config.channel_name;
  }

  fetch("config").then(function (r) { return r.json(); }).catch(function () { return {}; }).then(function (c) {
    applyConfig(c);
    return fetch("messages?limit=" + PAGE).then(function (r) { return r.json(); });
  }).then(function (rows) {
    logOrder = addRoots(rows);
    backlog.more = rows.length >= PAGE;
    rebuildLog();
    scrollBottom(log);
    if (backlog.more && log.scrollHeight <= log.clientHeight) { loadOlder(); }
    if (!touch) { mainComposer.input.focus(); }
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
})();
</script>
</body>
</html>
"""
