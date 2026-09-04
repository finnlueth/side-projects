# Claude Tree

A Firefox extension that adds a tree icon to the Claude web app's header and opens a side pane
showing the current conversation as an interactive tree — including every branch created by
editing a prompt or regenerating a reply.

Claude stores a conversation as a tree, not a list: editing a message or asking for another
answer adds a *sibling* rather than replacing anything. The chat UI only ever shows one path
through that tree, with a small `‹ 2/3 ›` switcher. This pane shows the whole thing at once.

## Features

- **Full branch tree** — every message in the conversation, not just the visible path.
- **Active path highlighted** — the branch the chat is currently showing is drawn in Claude's
  accent colour; everything else is dimmed. Fork points carry a badge with the number of
  alternatives.
- **Message details** — click a node for the full text, timestamp, extended thinking, tools
  used and attachment count, plus `‹ 2/3 ›` navigation between the variants at that fork.
- **Find in chat** — jumps to and flashes the message in the live conversation, scrolling to
  hunt it down if Claude has not rendered that far yet.
- **Follows where you are** — the message you are currently reading is outlined in the tree, so
  you always know your place. The selected message's accent outline takes precedence over it.
- **Keeps up on its own** — refreshes when you send a message, when Claude answers, and when an
  edit or a regeneration creates a new branch.
- **Two directions** — top-to-bottom or left-to-right, with pan, zoom and a resizable pane.
  Both preferences are remembered.
- **A real pane, not an overlay** — the page reserves width for it, so the chat reflows beside
  it instead of disappearing underneath.
- **Matches the app** — docked into Claude's own header next to Share, styled after Claude's
  document pane, using Claude's palette and, where the page has already loaded them, Claude's
  own fonts. Follows light and dark mode automatically.

## Install

Firefox 140 or newer.

**Temporarily (survives until you quit Firefox):**

1. Open `about:debugging#/runtime/this-firefox`.
2. *Load Temporary Add-on…* → pick `manifest.json` in this directory.
3. Open a chat on claude.ai. The tree icon appears in the chat header, just left of **Share**.

**Permanently:** the extension has to be signed by Mozilla. Package it with
[`web-ext`](https://github.com/mozilla/web-ext) and submit it, or run it in
[Firefox Developer Edition / Nightly](https://www.mozilla.org/firefox/developer/) with
`xpinstall.signatures.required` set to `false` in `about:config`.

```sh
npx web-ext lint      # validate
npx web-ext build     # -> web-ext-artifacts/*.zip
npx web-ext run       # launch a scratch profile with the extension loaded
```

## Using it

| Action | How |
| --- | --- |
| Open / close the pane | The tree icon in Claude's header, the toolbar icon, or `Alt+Shift+T` |
| Pan | Drag the canvas, or scroll |
| Zoom | `Ctrl`/`⌘` + scroll, trackpad pinch, or the `−` / `+` buttons |
| Reset the view | The frame button in the toolbar, or double-click the canvas |
| Select a message | Click a node; arrow keys walk parents, children and siblings |
| Resize the pane | Drag its left edge |
| Reload the tree | The refresh button in the pane header (it also refreshes itself) |

The tree reloads when you switch conversations, refreshes itself when you come back to the tab
after being away for more than 30 seconds, and updates in place — keeping your view, zoom and
selection — whenever the chat changes.

Matching a tree node to a message in the page is done on letters and digits only, so markdown,
punctuation and whitespace differences between the API text and the rendered page do not matter.
If a message has not been rendered yet, *find in chat* walks the chat scroller until Claude
renders it, starting from an estimate of where along the branch it sits.

## How it works

| File | Role |
| --- | --- |
| `src/api.js` | Reads the conversation from claude.ai's own web API |
| `src/model.js` | Turns the flat message list into a tree and lays it out |
| `src/chat.js` | Locating messages in the page, reading position, change detection |
| `src/panel.js` | The shadow-DOM side pane: rendering, pan/zoom, detail drawer |
| `src/content.js` | Docks the toggle, reserves page width, follows client-side navigation |
| `src/background.js` | Toolbar button, keyboard shortcut, and a fetch fallback |

The content script runs on claude.ai and requests
`/api/organizations/{org}/chat_conversations/{id}?tree=True&rendering_mode=messages`, the same
endpoint the app itself uses. Requests are same-origin, so your existing session cookie is
attached by the browser; the extension never reads, stores or transmits credentials, and talks
to no server other than claude.ai. Panel preferences are the only thing it stores.

The organisation id is taken from the `lastActiveOrg` cookie and only looked up over the
network if that guess is wrong. All UI lives in a shadow root, so nothing here can affect
claude.ai's own styles or be affected by them.

To make room for the pane, the extension narrows the page's `<body>` box. That is enough for a
normal-flow app shell; if anything positioned against the viewport still reaches under the
pane, `<body>` additionally gets layout containment so it becomes their containing block. That
second step is applied only on evidence, and never while the document itself scrolls — where it
would make fixed elements scroll away with the page. Closing the pane removes all of it.

## Limitations

- **Unofficial.** It depends on a private API. If Anthropic changes the response shape the
  pane falls back through several older request forms, and worst case shows only the visible
  path instead of the full tree — if you see one branch where the chat offers `‹ 2/3 ›`
  switchers, that fallback is what happened.
- **Read-only.** It never writes to your conversations, so it cannot switch the chat to a
  different branch. *Find in chat* can reach any message on the branch the chat is showing, but
  not one on another branch — use Claude's own `‹ ›` switchers for that.
- **The header button is placed against markup, not an API.** It anchors on Claude's own
  `wiggle-controls-actions-*` test ids and is re-inserted whenever Claude re-renders the header.
  If those disappear it falls back to locating the Share control by accessible name, and failing
  that to a floating pill in the bottom-right corner — never to nothing.
- Projects, artifacts and other non-chat pages have no tree to show.
