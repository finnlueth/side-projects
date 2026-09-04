/**
 * Everything that touches claude.ai's own message DOM: locating a message, following the
 * one you are reading, and noticing when the conversation changes.
 *
 * Claude renders messages lazily, so a message can be part of the conversation without
 * existing in the page at all. Nothing here assumes an element is present.
 */
(() => {
  'use strict';

  const CT = (globalThis.CT ||= {});

  /** One element per message on the branch the chat is showing. */
  const ROW_SELECTOR = '[data-testid="transcript-row"]';
  /** Older / alternative markup, in case Claude's transcript testids change. */
  const FALLBACK_SELECTOR = [
    '[data-testid="user-message"]',
    '[data-testid="chat-message"]',
    '.font-claude-response',
    '.font-claude-message',
  ].join(', ');
  const MESSAGE_SELECTOR = `${ROW_SELECTOR}, ${FALLBACK_SELECTOR}`;

  /** How much of a message to match on, in comparable characters. */
  const NEEDLE_LENGTH = 48;
  /**
   * Shortest comparable text worth matching on. It is deliberately low: a floor of a dozen
   * characters silently drops "Thanks", "yes please" and every other short turn, from
   * either sender, so those messages could never be found or highlighted at all. Needles
   * this short are matched from the start of the message instead of anywhere inside it,
   * which keeps "ok" from matching the middle of a longer one.
   */
  const MIN_NEEDLE = 2;
  const LOOSE_MATCH = 12;
  /** Scroll positions to try when hunting for a message that has not been rendered. */
  const SWEEP_STEPS = 6;

  /** The message itself inside a transcript row, without the action bar around it. */
  const BODY_SELECTOR = '[data-testid="user-message"], .font-claude-response, .font-claude-message';

  /** Does this element's text belong to a message starting with `needle`? */
  function textMatches(text, needle) {
    return needle.length >= LOOSE_MATCH ? text.includes(needle) : text.startsWith(needle);
  }

  /**
   * Comparable text for one message.
   *
   * A transcript row also holds its action bar — "Copy", "Retry", the "1 / 4" variant
   * readout — and that trailing chrome swamps a two-word message, which is why the shortest
   * turns could not be matched at all. The body element carries the message and nothing else.
   */
  function messageText(el) {
    const body = el.querySelector?.(BODY_SELECTOR);
    return elementText(body || el);
  }

  /**
   * Reduce text to letters and digits.
   *
   * The panel's text comes from the API and the page's text comes from rendered markdown,
   * so `**bold**`, headings, list bullets, smart quotes and collapsed whitespace all differ
   * between the two. Comparing only alphanumerics sidesteps every one of those.
   */
  function looseText(value) {
    return String(value ?? '').toLowerCase().replace(/[^a-z0-9]+/g, '');
  }

  const looseCache = new WeakMap();

  function elementText(el) {
    const raw = el.textContent || '';
    const cached = looseCache.get(el);
    if (cached && cached.raw === raw.length) return cached.loose;
    const loose = looseText(raw);
    looseCache.set(el, { raw: raw.length, loose });
    return loose;
  }

  function needleFor(node) {
    return looseText(node?.text || node?.preview || '').slice(0, NEEDLE_LENGTH);
  }

  function messageElements() {
    const scope = document.querySelector('main') || document.body;
    // Transcript rows are one-per-message and carry the sender, so prefer them outright.
    const rows = scope.querySelectorAll(ROW_SELECTOR);
    if (rows.length) return Array.from(rows);
    const tagged = scope.querySelectorAll(FALLBACK_SELECTOR);
    if (tagged.length) return Array.from(tagged);
    // Claude's markup changed: sweep a bounded set of blocks instead of giving up.
    return Array.from(scope.querySelectorAll('p, div')).slice(0, 3000);
  }

  /** Claude tags each transcript row with who sent it — a free check against mismatches. */
  function senderOf(el) {
    const row = el.closest?.('[data-perf-row]');
    const sender = row?.getAttribute('data-perf-row');
    return sender === 'human' || sender === 'assistant' ? sender : null;
  }

  /**
   * Claude clamps this attribute: the last ten rows carry their true distance from the end
   * of the branch, everything older reports 10.
   */
  const FROM_TAIL_CLAMP = 10;

  /** The messages on the branch the chat is showing, in order, with their match needles. */
  let branch = [];

  /** @param {object[]} nodes ordered messages of the branch currently displayed */
  function setBranch(nodes) {
    branch = (nodes || []).map((node) => ({ node, needle: needleFor(node) }));
  }

  /**
   * Line the rendered rows up with the messages on the branch.
   *
   * Text is not an identity: two messages can read exactly the same, and a message whose
   * wording the page renders differently matches nothing at all. Position is an identity.
   * Claude renders a contiguous window of the branch, so the entire alignment is one offset,
   * and one anchor fixes it — taken from `data-perf-row-from-tail` where that is exact, and
   * from a text match only when every rendered row is beyond the clamp.
   *
   * The sender of every row is checked against the message it lands on; a single mismatch
   * discards the alignment rather than reporting something wrong.
   *
   * @returns {{el: Element, node: object}[]} one entry per rendered row, in document order
   */
  function alignRows() {
    // Ordered by where they are on screen, not by where they sit in the document: a
    // virtualised list recycles rows, so the DOM order of a scrolled transcript is not the
    // order the reader sees, and the offset below depends on the visual order being right.
    const rows = messageElements()
      .map((el) => ({ el, top: el.getBoundingClientRect().top }))
      .filter((row) => Number.isFinite(row.top))
      .sort((a, b) => a.top - b.top)
      .map((row) => row.el);
    if (!rows.length || !branch.length) return [];

    let atRow = -1;
    let atIndex = -1;

    for (let j = 0; j < rows.length && atRow < 0; j++) {
      const tail = Number(rows[j].getAttribute('data-perf-row-from-tail'));
      if (Number.isFinite(tail) && tail < FROM_TAIL_CLAMP) {
        atRow = j;
        atIndex = branch.length - 1 - tail;
      }
    }

    if (atRow < 0) {
      // Every row is beyond the clamp: fall back to text for the anchor alone, and let
      // position carry the rest — including any duplicates around it.
      const votes = new Map();
      for (let j = 0; j < rows.length; j++) {
        const text = messageText(rows[j]);
        for (let i = 0; i < branch.length; i++) {
          if (!branch[i].needle || !textMatches(text, branch[i].needle)) continue;
          const offset = i - j;
          votes.set(offset, (votes.get(offset) ?? 0) + 1);
        }
      }
      let best = null;
      for (const [offset, count] of votes) {
        if (!best || count > best.count) best = { offset, count };
      }
      if (!best) return [];
      atRow = 0;
      atIndex = best.offset;
    }

    const aligned = [];
    let checked = 0;
    let agreed = 0;
    for (let j = 0; j < rows.length; j++) {
      const entry = branch[atIndex + (j - atRow)];
      if (!entry) continue;
      const sender = rows[j].getAttribute('data-perf-row');
      if (sender && sender !== entry.node.sender) return []; // clearly wrong; report nothing

      // Senders alternate, so they alone cannot catch an alignment that is off by an even
      // number of rows. Confirm with the text of the first few rows that have a needle.
      if (checked < 3 && entry.needle.length >= MIN_NEEDLE) {
        checked += 1;
        if (textMatches(messageText(rows[j]), entry.needle)) agreed += 1;
      }
      aligned.push({ el: rows[j], node: entry.node });
    }
    if (checked && !agreed) return []; // the offset is wrong; better to show nothing
    return aligned;
  }

  /** The row showing a given message, or null when the chat is not showing it. */
  function findElement(node) {
    if (!node) return null;
    const rows = alignRows();
    for (const row of rows) {
      if (row.node.id === node.id) return row.el;
    }
    // A working alignment is authoritative: if the message is not in it, it is not on
    // screen, and guessing by text here would hand back somebody else's row. Text is only
    // for when there is no alignment at all.
    return rows.length ? null : findByText(node);
  }

  /** Smallest rendered element containing the message — the tightest match wins. */
  function findByText(node) {
    const needle = needleFor(node);
    if (needle.length < MIN_NEEDLE) return null;

    let best = null;
    let bestLength = Infinity;
    for (const el of messageElements()) {
      const sender = senderOf(el);
      if (sender && node.sender && sender !== node.sender) continue;
      const text = messageText(el);
      if (text.length >= bestLength || !textMatches(text, needle)) continue;
      best = el;
      bestLength = text.length;
    }
    return best;
  }

  function findScroller() {
    const sample = document.querySelector(MESSAGE_SELECTOR);
    for (let el = sample?.parentElement; el; el = el.parentElement) {
      if (el.scrollHeight - el.clientHeight < 40) continue;
      const overflow = getComputedStyle(el).overflowY;
      if (overflow === 'auto' || overflow === 'scroll') return el;
    }
    const root = document.scrollingElement;
    return root && root.scrollHeight - root.clientHeight > 40 ? root : null;
  }

  /** Scroll fractions to try: the estimate first, then outwards from it. */
  function* sweep(estimate) {
    const anchor = Math.min(1, Math.max(0, Number.isFinite(estimate) ? estimate : 0.5));
    yield anchor;
    for (let step = 1; step <= SWEEP_STEPS; step++) {
      const delta = step / SWEEP_STEPS;
      if (anchor - delta >= 0) yield anchor - delta;
      if (anchor + delta <= 1) yield anchor + delta;
    }
  }

  const settle = () => new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(resolve, 90)));
  });

  /** Bring a message's first line into view, rather than centring its middle. */
  function scrollToStart(el) {
    const scroller = findScroller();
    if (!scroller || scroller === document.scrollingElement) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    const TOP_INSET = 16;
    const top = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
    scroller.scrollTo({ top: scroller.scrollTop + top - TOP_INSET, behavior: 'smooth' });
  }

  const controlName = (el) =>
    (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '').trim();

  /** Claude's own variant controls. Anything else is guesswork and gets nothing clicked. */
  const PREV_SELECTOR = '[data-testid="action-bar-previous-version"]';
  const NEXT_SELECTOR = '[data-testid="action-bar-next-version"]';

  function namedVersionControls(row) {
    const buttons = Array.from(row.querySelectorAll('button, [role="button"]'));
    const prev = buttons.find((b) => /\bprevious\b.*\bversion\b|\bversion\b.*\bprevious\b/i.test(controlName(b)));
    const next = buttons.find((b) => /\bnext\b.*\bversion\b|\bversion\b.*\bnext\b/i.test(controlName(b)));
    return prev && next && prev !== next ? [prev, next] : null;
  }

  /**
   * Locate Claude's variant switcher on a rendered message.
   *
   * Only Claude's own controls count — its `action-bar-*-version` test ids, or a pair
   * labelled "previous version" / "next version". An earlier version of this matched the
   * *shape* of an "n / m" readout with two buttons near it, which is not safe: a message
   * containing "p(ω) = 1/6" renders exactly that shape, and the buttons nearby are Copy and
   * Retry. Guessing there does not mis-navigate, it regenerates an answer.
   *
   * @returns {?{index: ?number, count: ?number, prev: Element, next: Element}}
   */
  function findVariantSwitcher(el) {
    if (!el) return null;
    const row = el.closest('[data-perf-row]') || el.parentElement || el;

    let prev = row.querySelector(PREV_SELECTOR);
    let next = row.querySelector(NEXT_SELECTOR);
    if (!prev || !next) {
      const named = namedVersionControls(row);
      if (!named) return null;
      [prev, next] = named;
    }

    const readout = readoutBetween(prev, next);
    return { index: readout?.index ?? null, count: readout?.count ?? null, prev, next };
  }

  /** The "n / m" readout sitting between two switcher controls. */
  function readoutBetween(prev, next) {
    let scope = prev.parentElement;
    for (let up = 0; up < 3 && scope; up += 1, scope = scope.parentElement) {
      if (!scope.contains(next)) continue;
      for (const candidate of scope.querySelectorAll('*')) {
        const readout = readVariantCount(candidate);
        if (readout) return readout;
      }
    }
    return null;
  }

  /**
   * Claude reveals a message's controls on hover, so a switcher that is not in the DOM yet
   * is not necessarily absent — the message just has not been pointed at.
   */
  async function hover(el) {
    if (!el) return;
    const row = el.closest('[data-perf-row]') || el;
    const rect = row.getBoundingClientRect();
    const at = { clientX: rect.left + rect.width / 2, clientY: rect.top + Math.min(24, rect.height / 2) };
    for (const [type, bubbles] of [['pointerover', true], ['pointerenter', false],
      ['mouseover', true], ['mouseenter', false], ['mousemove', true]]) {
      row.dispatchEvent(new MouseEvent(type, { bubbles, cancelable: true, ...at }));
    }
    await settle();
  }

  /** Find the switcher, pointing at the message first if it is not already showing one. */
  async function locateSwitcher(el) {
    if (!el) return null;
    return findVariantSwitcher(el) ?? (await hover(el), findVariantSwitcher(el));
  }

  /**
   * Press a control the way a person does.
   *
   * `element.click()` dispatches a lone click event. Claude's controls are built on handlers
   * that can key off the pointer sequence instead, and those never fire — the button looks
   * pressed, nothing happens, and the switch silently fails. Sending the whole sequence
   * covers both.
   */
  function press(button) {
    const rect = button.getBoundingClientRect();
    const at = {
      bubbles: true,
      cancelable: true,
      composed: true,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
      button: 0,
      buttons: 1,
    };
    button.dispatchEvent(new PointerEvent('pointerdown', { ...at, pointerId: 1, isPrimary: true }));
    button.dispatchEvent(new MouseEvent('mousedown', at));
    button.focus?.();
    button.dispatchEvent(new PointerEvent('pointerup', { ...at, buttons: 0, pointerId: 1, isPrimary: true }));
    button.dispatchEvent(new MouseEvent('mouseup', { ...at, buttons: 0 }));
    button.dispatchEvent(new MouseEvent('click', { ...at, buttons: 0 }));
  }

  /**
   * Does the page itself show a "n / m" variant readout anywhere?
   *
   * If it does while the tree has no forks, the API returned only the branch on screen —
   * which is worth saying out loud, because nothing else about the pane would reveal it.
   */
  function switcherOnScreen() {
    if (document.querySelector(`${PREV_SELECTOR}, ${NEXT_SELECTOR}`)) return true;
    return messageElements().some((row) => namedVersionControls(row));
  }

  async function hasVisibleSwitcher() {
    if (switcherOnScreen()) return true;
    // Claude only renders a message's controls while it is pointed at, so a switcher that is
    // not on screen has to be asked for before its absence means anything. Each probe costs
    // a frame or two of synthetic pointer events over the reader's messages, so only a few
    // rows are tried, and the caller runs this once per conversation rather than per load.
    for (const row of messageElements().slice(-4)) {
      await hover(row);
      if (switcherOnScreen()) return true;
    }
    return false;
  }

  /**
   * The shallowest fork on the way to `node` where the chat is showing a different variant.
   *
   * Which sibling is showing comes from the tree's own active path, not from what happens to
   * be rendered — the branch point is usually far off screen, and its absence from the DOM
   * says nothing about which variant is selected.
   */
  function findDivergence(node) {
    let found = null;
    for (let n = node; n && n.parent; n = n.parent) {
      const siblings = n.parent.children;
      if (siblings.length < 2) continue;
      const showing = siblings.find((sibling) => sibling.onPath);
      if (!showing || showing === n) continue; // this fork already shows the right variant
      found = { wanted: n, siblings, showing }; // keep going: prefer the shallowest fork
    }
    return found;
  }

  /**
   * Walk the chat onto the branch a message lives on by clicking the same controls a person
   * would. Every click is verified by text — the row alignment describes the branch that was
   * showing before the click, so it cannot be trusted to judge the result — and a click that
   * changes nothing is undone.
   *
   * @returns {Promise<boolean>} whether the displayed branch was changed
   */
  async function switchToBranch(node) {
    let switched = false;

    for (let hop = 0; hop < 4; hop++) {
      const divergence = switched ? findDivergenceByText(node) : findDivergence(node);
      if (!divergence) break;
      const { wanted, siblings, showing } = divergence;

      // The fork has to be on screen for Claude to have rendered its controls.
      const row = findElement(showing) || (await hunt(showing));
      if (!row) break;

      const switcher = await locateSwitcher(row);
      if (!switcher) break;

      const from = switcher.index ?? siblings.indexOf(showing);
      const total = switcher.count ?? siblings.length;
      const forward = wanted.siblingIndex > from;
      const steps = switcher.index === null ? total - 1 : Math.abs(wanted.siblingIndex - from);

      let moved = 0;
      for (let i = 0; i < steps; i++) {
        const before = findByText(showing);
        const control = await locateSwitcher(before || row);
        if (!control) break;

        const button = forward ? control.next : control.prev;
        if (button.disabled || button.getAttribute('aria-disabled') === 'true') break;
        press(button);
        switched = true;
        moved += 1;
        await settle();

        if (findByText(wanted)) break;           // the variant we wanted is now showing
        if (before && findByText(showing) === before && moved >= total) break;
      }
      if (!moved) break;
      // The alignment describes the branch we just left; drop it until the tree reloads.
      setBranch([]);
      if (findByText(node)) break;
    }
    return switched;
  }

  /** After a switch the tree's active path is stale, so re-read the fork from the page. */
  function findDivergenceByText(node) {
    let found = null;
    for (let n = node; n && n.parent; n = n.parent) {
      const siblings = n.parent.children;
      if (siblings.length < 2) continue;
      const showing = siblings.find((sibling) => findByText(sibling));
      if (!showing || showing === n) continue;
      found = { wanted: n, siblings, showing };
    }
    return found;
  }

  /** Scroll the chat until Claude has rendered `node`, or give up. */
  async function hunt(node, position) {
    let el = findElement(node);
    if (el) return el;

    const scroller = findScroller();
    if (!scroller) return null;

    const previous = scroller.scrollTop;
    const travel = scroller.scrollHeight - scroller.clientHeight;
    for (const fraction of sweep(position)) {
      scroller.scrollTop = travel * fraction;
      await settle();
      el = findElement(node);
      if (el) return el;
    }
    scroller.scrollTop = previous;
    return null;
  }

  /**
   * Scroll the chat to a message.
   *
   * Claude renders lazily, and the branch it is showing is not always the branch the API
   * last recorded, so this searches before it draws any conclusion — and if the message
   * really is on another branch, it asks Claude to show that branch first.
   *
   * @param {object} node tree node to look for
   * @param {{onPath: boolean, position: number}} hint what the API believes about the
   *   message, and roughly how far down its branch it sits (0–1)
   * @returns {Promise<'found'|'switched'|'off-path'|'not-found'>}
   */
  async function revealNode(node, { onPath, position } = {}) {
    // Sweep the scroller only for a message that should be on the branch already showing.
    // For one the tree places elsewhere, a cheap look is enough before switching — sweeping
    // the whole conversation just to fail first costs a second and a half.
    let el = onPath ? await hunt(node, position) : findElement(node);
    if (el) {
      scrollToStart(el);
      return 'found';
    }

    if (await switchToBranch(node)) {
      el = await hunt(node, position);
      if (el) {
        scrollToStart(el);
        return 'switched';
      }
    }

    return onPath ? 'not-found' : 'off-path';
  }

  /* ----------------------------------------------------------- reading position -- */

  /**
   * Report which message is currently being read, so the tree can show where you are.
   * @param {(id: string|null) => void} onChange
   */
  function trackReading(onChange) {
    let currentKey = '';
    let frame = 0;

    const evaluate = () => {
      frame = 0;
      const ids = [];
      for (const { el, node } of alignRows()) {
        const rect = el.getBoundingClientRect();
        if (!rect.height || rect.bottom <= 0 || rect.top >= window.innerHeight) continue;
        ids.push(node.id);
      }

      const key = ids.join(',');
      if (key === currentKey) return;
      currentKey = key;
      onChange(ids);
    };

    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(evaluate);
    };

    // Capture phase catches scrolling in whichever container Claude is using.
    document.addEventListener('scroll', schedule, { capture: true, passive: true });
    window.addEventListener('resize', schedule);

    return {
      /** @param {object[]} nodes messages on the branch the chat is showing */
      setNodes(nodes) {
        setBranch(nodes);
        currentKey = '';
        schedule();
      },
      refresh: schedule,
      stop() {
        document.removeEventListener('scroll', schedule, { capture: true });
        window.removeEventListener('resize', schedule);
        cancelAnimationFrame(frame);
      },
    };
  }

  /* ------------------------------------------------------------------ updates ---- */

  /**
   * Call back when the conversation looks like it has changed.
   *
   * Two triggers, because they catch different things: a change in the number of rendered
   * messages catches sending and receiving, while a burst of DOM churn settling down
   * catches edits and regenerations, which leave the count alone but create a new branch.
   *
   * @param {(reason: 'messages'|'settled') => void} onChange
   */
  function watchForUpdates(onChange) {
    const QUIET_MS = 1200;
    const THROTTLE_MS = 2500;

    /**
     * A cheap fingerprint of the end of the conversation: how many messages are rendered,
     * and how long the last one is. Sending, receiving and regenerating all change it —
     * streaming changes it on every tick, which is why a change alone does not trigger a
     * reload; it has to have stopped changing.
     */
    const signature = () => {
      const rows = document.querySelectorAll(MESSAGE_SELECTOR);
      const tail = document.querySelector('[data-perf-row-from-tail="0"]') || rows[rows.length - 1];
      return `${rows.length}:${tail ? (tail.textContent || '').length : 0}`;
    };

    let settleTimer = 0;
    let lastSignature = signature();
    let changing = false;
    let lastFired = 0;

    const fire = (reason) => {
      const now = Date.now();
      if (now - lastFired < THROTTLE_MS) return;
      lastFired = now;
      onChange(reason);
    };

    // Catches edits and regenerations, which rewrite a branch without touching the tail.
    const observer = new MutationObserver(() => {
      clearTimeout(settleTimer);
      settleTimer = setTimeout(() => fire('settled'), QUIET_MS);
    });
    observer.observe(document.querySelector('main') || document.body, {
      childList: true,
      subtree: true,
    });

    const poll = setInterval(() => {
      const current = signature();
      if (current !== lastSignature) {
        lastSignature = current;
        changing = true;      // mid-exchange; wait for it to come to rest
        return;
      }
      if (!changing) return;
      changing = false;
      fire('messages');
    }, 1000);

    return () => {
      observer.disconnect();
      clearTimeout(settleTimer);
      clearInterval(poll);
    };
  }

  CT.chat = { revealNode, hasVisibleSwitcher, trackReading, watchForUpdates, findElement, alignRows };
})();
