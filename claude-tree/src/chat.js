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

  const MESSAGE_SELECTOR = [
    '[data-testid="user-message"]',
    '[data-testid="chat-message"]',
    '[data-test-render-count]',
    '.font-claude-message',
    '.font-claude-response',
  ].join(', ');

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
    const tagged = scope.querySelectorAll(MESSAGE_SELECTOR);
    if (tagged.length) return Array.from(tagged);
    // Claude's markup changed: sweep a bounded set of blocks instead of giving up.
    return Array.from(scope.querySelectorAll('p, div')).slice(0, 3000);
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

  function flash(el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
      const focus = window.innerHeight * 0.34;

      let best = null;
      let bestDistance = Infinity;
      let bestHeight = Infinity;
      for (const el of messageElements()) {
        const rect = el.getBoundingClientRect();
        if (!rect.height || rect.bottom < 0 || rect.top > window.innerHeight) continue;
        const covers = rect.top <= focus && rect.bottom >= focus;
        const distance = covers
          ? 0
          : Math.min(Math.abs(rect.top - focus), Math.abs(rect.bottom - focus));
        // Closest to the focus line; on a tie prefer the tighter element.
        if (distance > bestDistance || (distance === bestDistance && rect.height >= bestHeight)) continue;
        bestDistance = distance;
        bestHeight = rect.height;
        best = el;
      }

      let id = null;
      if (best) {
        const text = elementText(best);
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

    let settleTimer = 0;
    let lastCount = document.querySelectorAll(MESSAGE_SELECTOR).length;
    let lastFired = 0;

    const fire = (reason) => {
      const now = Date.now();
      if (now - lastFired < THROTTLE_MS) return;
      lastFired = now;
      onChange(reason);
    };

    const observer = new MutationObserver(() => {
      clearTimeout(settleTimer);
      settleTimer = setTimeout(() => fire('settled'), QUIET_MS);
    });
    observer.observe(document.querySelector('main') || document.body, {
      childList: true,
      subtree: true,
    });

    // Streaming never goes quiet, so a sent message would otherwise wait for the reply.
    const poll = setInterval(() => {
      const count = document.querySelectorAll(MESSAGE_SELECTOR).length;
      if (count === lastCount) return;
      lastCount = count;
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
