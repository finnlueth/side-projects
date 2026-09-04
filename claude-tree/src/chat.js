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
    if (needle.length < 12) return null;

    let best = null;
    let bestLength = Infinity;
    for (const el of messageElements()) {
      const sender = senderOf(el);
      if (sender && node.sender && sender !== node.sender) continue;
      const text = elementText(el);
      if (text.length >= bestLength || !text.includes(needle)) continue;
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

  function flash(el) {
    scrollToStart(el);
    const previous = el.getAttribute('style');
    el.style.transition = 'box-shadow 180ms ease';
    el.style.boxShadow = '0 0 0 3px rgba(217, 119, 87, 0.6)';
    el.style.borderRadius = '12px';
    setTimeout(() => {
      if (previous === null) el.removeAttribute('style');
      else el.setAttribute('style', previous);
    }, 1900);
  }

  /**
   * Scroll the chat to a message and flash it.
   *
   * @param {object} node tree node to look for
   * @param {{onPath: boolean, position: number}} hint whether the message is on the branch
   *   the chat is showing, and roughly how far down that branch it sits (0–1)
   * @returns {Promise<'found'|'off-path'|'not-found'>}
   */
  async function revealNode(node, { onPath, position } = {}) {
    let el = findElement(node);
    if (el) {
      flash(el);
      return 'found';
    }
    // Off the visible branch, no amount of scrolling will render it.
    if (!onPath) return 'off-path';

    const scroller = findScroller();
    if (!scroller) return 'not-found';

    // On the visible branch but not in the DOM: Claude has not rendered that far yet, so
    // walk the scroller until it does.
    const previous = scroller.scrollTop;
    const travel = scroller.scrollHeight - scroller.clientHeight;
    for (const fraction of sweep(position)) {
      scroller.scrollTop = travel * fraction;
      await settle();
      el = findElement(node);
      if (el) {
        flash(el);
        return 'found';
      }
    }
    scroller.scrollTop = previous;
    return 'not-found';
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
      // A little above centre: what you are reading, not what you have scrolled past.
      const focus = window.innerHeight * 0.28;
      const rows = messageElements()
        .map((el) => ({ el, rect: el.getBoundingClientRect() }))
        .filter((row) => row.rect.height)
        .sort((a, b) => a.rect.top - b.rect.top);
      if (!rows.length) return;

      /*
       * Each message claims a run of scrolling that starts where the previous one's ended
       * and lasts at least DWELL. Without that floor a short prompt between two long
       * answers is highlighted for only its own height — a flicker while scrolling fast.
       */
      const dwell = Math.max(240, window.innerHeight * 0.45);
      let cursor = -Infinity;
      let best = null;
      for (let i = 0; i < rows.length; i++) {
        const start = Math.max(rows[i].rect.top, cursor);
        const nextTop = i + 1 < rows.length ? rows[i + 1].rect.top : Infinity;
        const end = Math.max(nextTop, start + dwell);
        if (focus >= start && focus < end) {
          best = rows[i];
          break;
        }
        cursor = end;
      }
      // Never hold the mark on a message that has scrolled entirely out of sight.
      if (!best || best.rect.bottom < 0) {
        best = rows.find((row) => row.rect.bottom > 0 && row.rect.top < window.innerHeight) ?? null;
      }

      let id = null;
      if (best) {
        const text = elementText(best.el);
        id = index.find((entry) => text.includes(entry.needle))?.id ?? null;
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
          .filter((entry) => entry.needle.length >= 12);
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

  CT.chat = { revealNode, trackReading, watchForUpdates, findElement, looseText };
})();
