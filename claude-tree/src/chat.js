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
  const MIN_NEEDLE = 4;
  const LOOSE_MATCH = 12;

  /** Does this element's text belong to a message starting with `needle`? */
  function textMatches(text, needle) {
    return needle.length >= LOOSE_MATCH ? text.includes(needle) : text.startsWith(needle);
  }
  /** Scroll positions to try when hunting for a message that has not been rendered. */
  const SWEEP_STEPS = 6;

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

  /** Claude may carry the message uuid in the DOM; these are the cheap shapes to test. */
  function findByUuid(id) {
    if (!id) return null;
    const value = CSS.escape(id);
    return document.querySelector(
      `[data-message-id="${value}"], [data-message-uuid="${value}"], ` +
      `[data-uuid="${value}"], [data-id="${value}"], [id="${value}"]`
    );
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
      const text = elementText(el);
      if (text.length >= bestLength || !textMatches(text, needle)) continue;
      best = el;
      bestLength = text.length;
    }
    return best;
  }

  function findElement(node) {
    return findByUuid(node.id) || findByText(node);
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

  /**
   * Is this element the "n / m" readout of a variant switcher? The digits may be split
   * across nested spans, but a readout never contains a button.
   * @returns {?{index: number, count: number}}
   */
  function readVariantCount(el) {
    if (el.querySelector('button, [role="button"]')) return null;
    const match = /^(\d+)\s*\/\s*(\d+)$/.exec((el.textContent || '').trim());
    return match ? { index: Number(match[1]) - 1, count: Number(match[2]) } : null;
  }

  /** Walk up from a readout looking for the container holding exactly two controls. */
  function pairAround(readout) {
    let scope = readout.parentElement;
    for (let up = 0; up < 3 && scope; up += 1, scope = scope.parentElement) {
      const buttons = Array.from(scope.querySelectorAll('button, [role="button"]'));
      if (buttons.length === 2) return buttons;
    }
    return null;
  }

  /**
   * Locate Claude's variant switcher on a rendered message.
   *
   * Two shapes are recognised: an "n / m" readout with a control either side, which also
   * says which variant is showing, and a bare previous/next pair, which does not. Neither
   * is matched on class names, and every click is verified afterwards, so a mis-identified
   * control changes nothing that is not immediately undone.
   */
  function findVariantSwitcher(el, expectedCount) {
    if (!el) return null;
    const row = el.closest('[data-perf-row]') || el.parentElement || el;
    const scopes = [row];
    // Claude groups per-message controls in an action bar; search that too.
    const bar = row.querySelector('[data-testid^="action-bar-"]')?.parentElement;
    if (bar && bar !== row) scopes.push(bar);

    for (const scope of scopes) {
      for (const candidate of scope.querySelectorAll('*')) {
        const readout = readVariantCount(candidate);
        if (!readout || readout.count !== expectedCount) continue;

        const buttons = pairAround(candidate);
        if (!buttons) continue;
        return { index: readout.index, prev: buttons[0], next: buttons[1] };
      }
    }

    for (const scope of scopes) {
      const buttons = Array.from(scope.querySelectorAll('button, [role="button"]'));
      const prev = buttons.find((b) => /\b(previous|prev)\b/i.test(controlName(b)));
      const next = buttons.find((b) => /\bnext\b/i.test(controlName(b)));
      if (prev && next && prev !== next) return { index: null, prev, next };
    }
    return null;
  }

  /**
   * Claude reveals a message's controls on hover, so a switcher that is not in the DOM yet
   * is not necessarily absent — the message just has not been pointed at.
   */
  async function hover(el) {
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
  async function locateSwitcher(el, expectedCount) {
    return findVariantSwitcher(el, expectedCount)
      ?? (await hover(el), findVariantSwitcher(el, expectedCount));
  }

  /**
   * Does the page itself show a "n / m" variant readout anywhere?
   *
   * If it does while the tree has no forks, the API returned only the branch on screen —
   * which is worth saying out loud, because nothing else about the pane would reveal it.
   */
  function switcherOnScreen() {
    for (const el of document.querySelectorAll('[data-perf-row] *')) {
      if (readVariantCount(el)) return true;
    }
    return false;
  }

  async function hasVisibleSwitcher() {
    if (switcherOnScreen()) return true;
    // Claude only renders a message's controls while it is pointed at, so a switcher that
    // is not on screen has to be asked for before its absence means anything.
    for (const row of messageElements().slice(0, 12)) {
      await hover(row);
      if (switcherOnScreen()) return true;
    }
    return false;
  }

  /** Which of a set of siblings the chat is currently rendering, if any. */
  function renderedSibling(siblings) {
    return siblings.find((sibling) => findElement(sibling)) ?? null;
  }

  /**
   * The shallowest fork on the way to `node` where the chat is showing a different variant,
   * together with the switcher on the variant that is currently rendered.
   */
  function findDivergence(node) {
    let found = null;
    for (let n = node; n && n.parent; n = n.parent) {
      const siblings = n.parent.children;
      if (siblings.length < 2) continue;
      const shown = renderedSibling(siblings);
      if (!shown || shown === n) continue; // this fork already shows the right variant
      found = { wanted: n, siblings, shown }; // keep going: prefer the shallowest fork
    }
    return found;
  }

  /** Click one control and report whether the chat actually moved to another variant. */
  async function clickAndVerify(control, siblings, before) {
    control.click();
    await settle();
    const now = renderedSibling(siblings);
    return Boolean(now && now !== before);
  }

  /**
   * Walk the chat onto the branch a message lives on by clicking the same switcher a person
   * would. Every click is checked; if one does not move the chat it is undone and the walk
   * stops, so nothing is left in a state the user did not ask for.
   *
   * @returns {Promise<boolean>} whether the displayed branch was changed
   */
  async function switchToBranch(node) {
    let switched = false;

    for (let hop = 0; hop < 4; hop++) {
      const divergence = findDivergence(node);
      if (!divergence) break;
      const { wanted, siblings, shown } = divergence;

      const switcher = await locateSwitcher(findElement(shown), siblings.length);
      if (!switcher) break;

      // With an "n / m" readout we know the direction and distance; without one, step in a
      // direction until the variant we want appears, then stop.
      const from = switcher.index ?? siblings.indexOf(shown);
      const steps = switcher.index === null ? siblings.length - 1 : Math.abs(wanted.siblingIndex - from);
      const forward = wanted.siblingIndex > from;

      let moved = 0;
      for (let i = 0; i < steps; i++) {
        const before = renderedSibling(siblings);
        const control = await locateSwitcher(findElement(before), siblings.length);
        if (!control) break;
        if (!(await clickAndVerify(forward ? control.next : control.prev, siblings, before))) {
          // That control did nothing; put the chat back where it was and give up here.
          const undo = await locateSwitcher(findElement(renderedSibling(siblings) || before), siblings.length);
          for (let back = 0; back < moved && undo; back++) (forward ? undo.prev : undo.next).click();
          return switched;
        }
        moved += 1;
        switched = true;
        if (renderedSibling(siblings) === wanted) break;
      }
      if (!moved) break;
    }
    return switched;
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
    let el = await hunt(node, position);
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
    let index = [];
    let currentId = null;
    let frame = 0;

    const evaluate = () => {
      frame = 0;
      if (!index.length) return;

      // A line near the top of the window: the message you are reading is the one it falls
      // in. Nothing is carried between frames, so the mark cannot drift or stick.
      const focus = window.innerHeight * 0.3;

      let best = null;
      let bestDistance = Infinity;
      for (const el of messageElements()) {
        const rect = el.getBoundingClientRect();
        if (!rect.height) continue;
        const distance = rect.top > focus ? rect.top - focus
          : rect.bottom < focus ? focus - rect.bottom
            : 0;
        if (distance >= bestDistance) continue;
        bestDistance = distance;
        best = el;
      }

      let id = null;
      if (best) {
        const text = elementText(best);
        id = index.find((entry) => textMatches(text, entry.needle))?.id ?? null;
      }
      if (id === currentId) return;
      currentId = id;
      onChange(id);
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
        index = nodes
          .map((node) => ({ id: node.id, needle: needleFor(node) }))
          .filter((entry) => entry.needle.length >= MIN_NEEDLE);
        currentId = null;
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

  CT.chat = { revealNode, switchToBranch, hasVisibleSwitcher, trackReading, watchForUpdates, findElement, looseText };
})();
