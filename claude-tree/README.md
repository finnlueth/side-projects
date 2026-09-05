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
- **Show in chat** — double-click any node, or use the one button in the drawer, to take the
  chat to that message. It scrolls, hunting for the message if Claude has not rendered that far
  yet; and when the message is on another branch it moves the conversation there first, so the
  same button reads *Show this branch in the chat* and there is nothing else to find.
- **Follows where you are** — every message currently on screen in the chat is outlined in the
  tree, so the run you are reading is visible at a glance. Short turns from either side are
  matched from the start of their text, so a "Thanks" is tracked as readily as a long answer.
  The selected message's accent outline takes precedence over the outline.
- **Keeps up on its own** — refreshes when you send a message, when Claude answers, and when an
  edit or a regeneration creates a new branch.
- **Two directions** — top-to-bottom or left-to-right, with pan and zoom. The pane and the
  message drawer are both resizable, and every preference is remembered.
- **Sized to its content** — nodes show up to four lines of a message and no more, and a
  one-line message takes one line's worth of space rather than being padded out to match.
- **A real pane, not an overlay** — the page reserves width for it, so the chat reflows beside
  it instead of disappearing underneath.
- **Built from Claude's own design system** — docked into Claude's header next to Share, shaped
  like Claude's document pane, and drawn with the page's live design tokens, control metrics,
  typefaces and icon font rather than a private imitation of them. Theme changes and even
  restyles of claude.ai carry through on their own.

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
| Leave a conversation | Starting a new chat closes the pane and hides the button |
| Pan | Drag the canvas, or scroll |
| Zoom | Trackpad pinch, `Ctrl`/`⌘` + scroll, or the `−` / `+` buttons |
| Reset the view | The frame button, or double-click the canvas — press again to fit the whole tree |
| Resize the message drawer | Drag its top edge |
| Select a message | Click a node — click it again to close the drawer |
| Take the chat to a message | Double-click a node, or *Show in chat* in the drawer |
| Walk the tree | Arrow keys move between parents, children and siblings |
| Resize the pane | Drag its left edge (the width is remembered) |
| Reload the tree | The refresh button in the pane header (it also refreshes itself) |

The pane opens at 1:1, centred on whichever message the chat is currently showing, so it starts
where you are; pressing reset a second time zooms out to fit the whole
tree. The tree reloads when you switch conversations, refreshes itself when you come back to the
tab after being away for more than 30 seconds, and updates in place — keeping your view, zoom and
selection — whenever the chat changes.

Messages are matched to the page by **position, not by text**. Text is not an identity: two
turns can read exactly the same, and one the page renders differently matches nothing at all.
Claude renders a contiguous window of the branch and tags each row with its distance from the
end, so the whole alignment is a single offset. Several candidate offsets are tried rather than
one, because the page and the tree can legitimately disagree about how many messages exist —
while a reply is streaming the page runs a message ahead, and trusting the tail alone would
discard every highlight until the tree caught up — after sorting rows by where
they are on screen, since a virtualised list recycles them out of document order. Every row's
sender, and the text of the first few, are checked against the messages they land on; a
disagreement discards the alignment rather than reporting something wrong. Text matching remains
only as a fallback for markup without those attributes, and there it compares the message body
rather than the whole row, so the action bar cannot swamp a two-word turn.
If a message has not been rendered yet, *find in chat* walks the chat scroller until Claude
renders it, starting from an estimate of where along the branch it sits.

## How it works

| File | Role |
| --- | --- |
| `src/api.js` | Reads the conversation from claude.ai's own web API |
| `src/model.js` | Turns the flat message list into a tree and lays it out |
| `src/chat.js` | Locating messages in the page, reading position, change detection |
| `src/panel.js` | The shadow-DOM side pane: rendering, pan/zoom, detail drawer |
| `src/panel.css` | Token bindings and pane styling |
| `test/` | Regression suites — not part of the packaged extension |
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

Custom properties do cross that shadow boundary, though, which is how the pane reuses Claude's
design system rather than reimplementing it: colours come from the page's `--bg-*`, `--text-*`,
`--border-*` and `--accent-brand` scales — composited the way Claude composites them, which for
`--border-300` means with an alpha, since the raw triplet is a near-black or near-white meant to
be used at around 20% — metrics from `--cds-radius`, `--cds-h-control` and
`--cds-pad-*`, motion from `--cds-dur-*` and `--cds-ease-*`, and type from `--font-sans`. Icons
are glyphs from the Anthropicons font the page already loads, their codepoints read off Claude's
own buttons by accessible name at runtime so they survive the font being renumbered. Every one
of these has a fallback, so the pane still renders correctly if a token is renamed or removed.

To make room for the pane, the extension narrows the page's `<body>` box. That is enough for a
normal-flow app shell; if anything positioned against the viewport still reaches under the
pane, `<body>` additionally gets layout containment so it becomes their containing block. That
second step is applied only on evidence, and never while the document itself scrolls — where it
would make fixed elements scroll away with the page. Closing the pane removes all of it.

## Tests

```sh
node test/run.mjs              # every suite
node test/run.mjs duplicate    # suites whose name matches
```

`test/suites/tree-model.mjs` exercises the tree building and layout directly in Node. The rest
drive the real `src/` files in Chrome against `test/support/harness.html`, which stands in for
claude.ai: it serves the conversation API, virtualises the transcript, carries the same row
attributes (including the `data-perf-row-from-tail` clamp), mounts the message action bar from
an inner node on hover as Claude does, and includes the cases that have actually caused bugs —
identical messages, a two-character turn, a KaTeX fraction that reads as `1/6`, a reply still
streaming, and a branch switcher wired to pointer events rather than `click`. Chrome and
puppeteer are required; nothing under `test/` is included in the packaged extension.

## Limitations

- **Unofficial.** It depends on a private API. If Anthropic changes the response shape the
  pane falls back through several older request forms, and worst case shows only the visible
  path instead of the full tree — if you see one branch where the chat offers `‹ 2/3 ›`
  switchers, that fallback is what happened.
- **Branch switching is a real write.** Selecting a branch moves the conversation's current
  message, using the endpoint claude.ai uses for its own `‹ 2/3 ›` control
  (`PUT .../chat_conversations/{id}/current_leaf_message_uuid`). Nothing else is ever written:
  no messages, edits or deletions. The in-page control is tried first, because when it works the chat
  updates with no reload at all — it is Claude's own control, so Claude re-renders itself.
  Only Claude's `action-bar-previous-version` / `action-bar-next-version` buttons count, or a
  pair labelled "previous version" / "next version", and the button is pressed with a full
  pointer sequence rather than `element.click()`, because a lone click event never reaches
  handlers built on pointer events. Nothing is matched by the *shape* of an `n / m` readout: a
  message containing `p(ω) = 1/6` renders exactly that shape, and the buttons beside it are
  Copy and Retry. Claude's whole action bar — copy, retry and the switcher — exists in
  the DOM only while a message is hovered, and the handler that mounts it sits on an inner
  node, so events dispatched at the row never reach it. The fork is therefore scrolled into
  view and then hovered the way a real mouse does: from the deepest element under the point,
  bubbling up. Every press is
  checked; one that does not move the chat is undone and the walk stops. Once the branch has
  moved the tree is reloaded before the message is located, because the alignment still
  describes the branch just left — and a message with little or no text could not be found by
  text either. When the fallback reload is used, the message is carried across it, so you land
  on what you asked for rather than at the top of the conversation.
  Only when no control can be driven does it fall back to the API, which
  needs a reload — and that reload waits until the move is readable, because the write is
  accepted before it has propagated and reloading into that gap is what leaves the chat showing
  the branch you just left.
- **Two things are behind flags in `src/panel.js`**, not deleted: `SHOW_STATS` hides the
  summary pills above the tree, and `SHOW_TOASTS` controls the pane's notifications. Toasts
  are currently on, and now name the reason a branch switch failed rather than just that it
  did; the pane also warns in the console when claude.ai returns a conversation flat that the
  page is visibly branching.
- **The pane steps aside for Claude's own overlays.** Dialogs are found by role; Claude's
  in-chat find bar is not a dialog, so anything holding keyboard focus inside a very high
  stacking layer counts too. The pane drops to just under whatever is on top — not to the back,
  so it does not vanish behind the chat while a small find bar is open — and returns to the
  front when it closes.
- **The header button is placed against markup, not an API.** It anchors on Claude's own
  `wiggle-controls-actions-*` test ids and is re-inserted whenever Claude re-renders the header.
  If those disappear it falls back to locating the Share control by accessible name, and failing
  that to a floating pill in the bottom-right corner — never to nothing.
- **Only the tree glyph is ours.** Icons are taken from Claude's icon font wherever an
  equivalent exists there; where one cannot be confirmed at runtime the pane falls back to a
  plain SVG of its own rather than guessing at a codepoint and drawing the wrong picture.
- Projects, artifacts and other non-chat pages have no tree to show.
