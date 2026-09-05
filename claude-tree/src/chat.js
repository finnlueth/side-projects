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

  /**
   * A rendered row's position on the branch the chat is showing, counted from the first
   * message and *not* clamped — unlike the distance-from-tail above, which is useless past
   * the tenth row from the end. This is the transcript's own numbering, so it identifies a
   * row exactly, without comparing any text. Messages carrying an attachment reduce to
   * almost the same comparable text as their siblings, so text alone picks the wrong branch
   * at precisely the forks that matter.
   */
  function rowIndex(el) {
    const row = el.closest?.('[data-perf-row]') || el;
    const raw = row.getAttribute?.('data-index') ?? row.getAttribute?.('data-rs-index');
    const index = Number(raw);
    return raw !== null && raw !== undefined && Number.isInteger(index) && index >= 0 ? index : null;
  }

  /**
   * Every rendered row that states its position, with where it sits in the scroller.
   *
   * The transcript keeps rows mounted well outside the viewport and drops others from the
   * middle, so the rendered positions are not a contiguous window: 8-15, 19, 21-23 is a
   * perfectly ordinary reading. Anything that assumes a single unbroken span of rows is
   * wrong about which messages are actually available.
   */
  function renderedRows(scroller) {
    const origin = scroller ? scroller.getBoundingClientRect().top - scroller.scrollTop : 0;
    const found = [];
    for (const el of messageElements()) {
      const index = rowIndex(el);
      if (index === null) continue;
      found.push({ el, index, offset: el.getBoundingClientRect().top - origin });
    }
    return found.sort((a, b) => a.index - b.index);
  }

  /** The row showing a given position on the displayed branch, if it is mounted. */
  function rowAtIndex(index) {
    for (const el of messageElements()) if (rowIndex(el) === index) return el;
    return null;
  }

  /** Candidate branches, each an ordered path with its match needles. */
  let candidates = [];
  /** Index of the branch the page appears to be showing. */
  let chosen = -1;
  /**
   * Branches that fit the rendered rows exactly as well as the chosen one.
   *
   * Sibling replies to the same prompt often begin with the same words, so until the messages
   * below them are on screen there is genuinely nothing to tell their branches apart. Keeping
   * the ties rather than pretending the winner is certain is what stops a message being
   * declared missing because it sits on the other one.
   */
  let tied = [];

  /** @param {object[][]} paths every root-to-leaf path, most likely first */
  function setBranches(paths) {
    candidates = (paths || []).map((nodes) => ({
      leafId: nodes[nodes.length - 1]?.id ?? null,
      ids: new Set(nodes.map((n) => n.id)),
      positions: new Map(nodes.map((node, i) => [node.id, i])),
      entries: nodes.map((node) => ({ node, needle: needleFor(node) })),
    }));
    chosen = -1;
  }

  /** Is this message on the branch the chat is actually showing? */
  function onDisplayedBranch(node) {
    if (!node) return false;
    if (chosen < 0) alignRows();
    return chosen >= 0 ? candidates[chosen].ids.has(node.id) : Boolean(node.onPath);
  }

  /**
   * Line the rendered rows up with the messages of whichever branch the chat is showing.
   *
   * Which branch that is cannot be taken from the conversation's recorded leaf: the two
   * disagree in normal use — Claude's client rewrites the leaf to match what it is showing,
   * and a switch elsewhere leaves the record ahead of the page. So every candidate branch is
   * fitted against the rendered rows and the one that actually matches wins.
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
    if (!rows.length || !candidates.length) return [];

    // The branch that fitted last time is tried first; it is nearly always still right.
    const order = [];
    if (chosen >= 0) order.push(chosen);
    for (let i = 0; i < candidates.length; i++) if (i !== chosen) order.push(i);

    // Every row numbered means the transcript's own indices can be used directly, which is
    // exact. Only fall back to fitting by text and tail distance when they are missing.
    const numbered = rows.every((el) => rowIndex(el) !== null);

    let best = null;
    const scored = [];
    for (const index of order) {
      const aligned = numbered
        ? fitByIndex(rows, candidates[index].entries)
        : fitBranch(rows, candidates[index].entries);
      if (!aligned) continue;
      scored.push({ index, score: aligned.score });
      if (!best || aligned.score > best.aligned.score) best = { index, aligned };
    }
    if (!best) { chosen = -1; tied = []; return []; }
    chosen = best.index;
    tied = scored.filter((entry) => entry.score === best.aligned.score && entry.index !== chosen)
      .map((entry) => entry.index);
    return best.aligned.pairs;
  }

  /**
   * Pair rows with messages by the transcript's own row numbering.
   *
   * Every rendered row states its position on the displayed branch, so there is no offset to
   * guess: row `n` is the branch's `n`th message. What remains is deciding *which* branch is
   * on screen, and two things settle that without relying on text. A row's sender has to
   * match the message at its position, and if the last row of the conversation is rendered
   * then the branch has to end exactly there — a branch of a different length simply cannot
   * be the one being shown. Text agreement is still counted, but only to break ties, because
   * a message whose body is an attachment carries almost no comparable text of its own.
   *
   * @returns {?{pairs: {el: Element, node: object}[], score: number, exact: boolean}}
   */
  function fitByIndex(rows, entries) {
    if (!entries.length) return null;

    const pairs = [];
    let agreed = 0;
    let senders = 0;
    let ahead = 0;
    let anchored = false;

    for (const el of rows) {
      const index = rowIndex(el);
      if (index === null) return null;

      const entry = entries[index];
      if (!entry) {
        /*
         * A row past the end of this branch. While Claude is streaming a reply the page runs
         * ahead of the conversation the API returned, so the last row or two belong to
         * messages the tree has never seen. Rejecting the branch for that would throw away
         * the alignment — and every highlight with it — for as long as an answer is being
         * written. Anything more than a message or two ahead is a different branch, though.
         */
        ahead += 1;
        if (ahead > 2) return null;
        continue;
      }

      const sender = el.getAttribute('data-perf-row');
      if (sender && sender !== entry.node.sender) return null;
      if (sender) senders += 1;

      if (Number(el.getAttribute('data-perf-row-from-tail')) === 0) {
        if (entries.length !== index + 1) return null; // this branch ends somewhere else
        anchored = true;
      }

      if (entry.needle.length >= MIN_NEEDLE && textMatches(messageText(el), entry.needle)) {
        agreed += 1;
      }
      pairs.push({ el, node: entry.node });
    }

    if (!pairs.length) return null;
    return {
      pairs,
      score: agreed + senders + (anchored ? rows.length : 0),
      exact: true,
    };
  }

  /**
   * Fit one branch to the rendered rows, or return null if it cannot be made to agree.
   * @returns {?{pairs: {el: Element, node: object}[], score: number}}
   */
  function fitBranch(rows, entries) {
    if (!entries.length) return null;

    /*
     * Offsets worth trying, best first. More than one is needed because the page and the
     * tree can legitimately disagree about how many messages there are: while Claude is
     * streaming a reply the page is a message ahead, which shifts every row relative to the
     * tail and would otherwise throw the whole alignment away — and with it every highlight.
     */
    const offsets = [];
    const add = (offset) => {
      if (Number.isFinite(offset) && !offsets.includes(offset)) offsets.push(offset);
    };
    for (let j = 0; j < rows.length; j++) {
      const tail = Number(rows[j].getAttribute('data-perf-row-from-tail'));
      if (Number.isFinite(tail) && tail < FROM_TAIL_CLAMP) add(entries.length - 1 - tail - j);
    }
    for (const offset of textOffsets(rows, entries)) add(offset);

    let best = null;
    for (const offset of offsets) {
      const fit = applyOffset(rows, entries, offset);
      if (fit && (!best || fit.score > best.score)) best = fit;
    }
    return best;
  }

  /** Offsets suggested by the text of the rows, most-agreed first. */
  function textOffsets(rows, entries) {
    const votes = new Map();
    for (let j = 0; j < rows.length; j++) {
      const text = messageText(rows[j]);
      for (let i = 0; i < entries.length; i++) {
        if (entries[i].needle.length < MIN_NEEDLE) continue;
        if (!textMatches(text, entries[i].needle)) continue;
        votes.set(i - j, (votes.get(i - j) ?? 0) + 1);
      }
    }
    return [...votes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([offset]) => offset);
  }

  /**
   * Pair every row with the message `offset` places along, scoring how well they agree.
   * Senders alternate, so they alone cannot catch an alignment that is off by an even number
   * of rows; the text is what separates one branch from another past their shared prefix.
   */
  function applyOffset(rows, entries, offset) {
    const pairs = [];
    let score = 0;
    let compared = 0;

    for (let j = 0; j < rows.length; j++) {
      const entry = entries[offset + j];
      if (!entry) continue;
      const sender = rows[j].getAttribute('data-perf-row');
      if (sender && sender !== entry.node.sender) return null;

      if (entry.needle.length >= MIN_NEEDLE) {
        compared += 1;
        if (textMatches(messageText(rows[j]), entry.needle)) score += 1;
      }
      pairs.push({ el: rows[j], node: entry.node });
    }

    if (!pairs.length) return null;
    if (compared && !score) return null; // nothing agreed; this is the wrong branch
    return { pairs, score };
  }

  /** The row showing a given message, or null when the chat is not showing it. */
  function findElement(node) {
    if (!node) return null;
    const rows = alignRows();
    for (const row of rows) {
      if (row.node.id === node.id) return row.el;
    }
    /*
     * A reading of the page that fits is normally authoritative. It is not when another
     * branch fits exactly as well — two replies that open with the same words — because then
     * the one picked may simply be the wrong half of the pair. Accept a row that an equally
     * good reading places this message at, but only when the row's own text agrees, so this
     * can never hand back an unrelated message.
     */
    for (const index of tied) {
      const position = candidates[index]?.positions.get(node.id);
      if (!Number.isInteger(position)) continue;
      const el = rowAtIndex(position);
      const needle = needleFor(node);
      if (el && needle.length >= MIN_NEEDLE && textMatches(messageText(el), needle)) return el;
    }

    // Text is otherwise only for when there is no alignment at all.
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

  /**
   * Move the transcript, and tell it that it moved.
   *
   * Setting `scrollTop` raises a scroll event of its own, but Claude's transcript can end up
   * in a state — reliably reached by switching branch, and also seen just after a reload —
   * where it stops re-rendering in response, showing only the last handful of messages
   * however far the page is scrolled. Every earlier message then looks unreachable, which is
   * what "the chat breaks after switching branches a few times" is. Raising the event
   * explicitly brings it back, and is harmless when the native one arrives as well.
   */
  function scrollTranscript(scroller, top, smooth = false) {
    if (smooth) scroller.scrollTo({ top, behavior: 'smooth' });
    else scroller.scrollTop = top;
    scroller.dispatchEvent(new Event('scroll', { bubbles: true }));
  }

  /** Where `scrollToStart` parks a message: just below the top of the transcript. */
  const TOP_INSET = 16;

  /** Bring a message's first line into view, rather than centring its middle. */
  function scrollToStart(el, smooth = true) {
    const scroller = findScroller();
    if (!scroller || scroller === document.scrollingElement) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    const top = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
    scrollTranscript(scroller, scroller.scrollTop + top - TOP_INSET, smooth);
  }

  const controlName = (el) =>
    (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '').trim();

  /**
   * Claude's own variant controls. Anything else is guesswork and gets nothing clicked.
   *
   * There are two families of them. An assistant retry puts `action-bar-*-version` in the
   * row's action bar; a fork made by editing a prompt puts `user-message-*-version` on the
   * human row, which has no action bar at all. Matching only the first family made every
   * prompt-edit fork look unswitchable, and sent the panel down the API-and-reload path for
   * the most common kind of branch there is.
   */
  const PREV_SELECTOR = '[data-testid="action-bar-previous-version"], '
    + '[data-testid="user-message-previous-version"]';
  const NEXT_SELECTOR = '[data-testid="action-bar-next-version"], '
    + '[data-testid="user-message-next-version"]';

  /** Controls Claude mounts only while a message is pointed at: proof that a hover landed. */
  const CONTROLS_SELECTOR = '[data-testid^="action-bar-"], [data-testid="user-message-copy"], '
    + '[data-testid="user-message-edit"], [data-testid="user-message-retry"], '
    + PREV_SELECTOR + ', ' + NEXT_SELECTOR;

  /** The body of a message: where a real pointer has to land for Claude to react. */
  function messageBody(row) {
    return row.querySelector('[data-testid="user-message"], .font-claude-response') || row;
  }

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

  /**
   * Read an "n / m" readout. Only ever called on an element already known to sit between
   * Claude's two version controls, so a fraction in prose can never reach it.
   * @returns {?{index: number, count: number}}
   */
  function readVariantCount(el) {
    if (el.querySelector('button, [role="button"]')) return null;
    const match = /^(\d+)\s*\/\s*(\d+)$/.exec((el.textContent || '').trim());
    return match ? { index: Number(match[1]) - 1, count: Number(match[2]) } : null;
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
   * Point at a message so Claude mounts its action bar.
   *
   * The whole bar — copy, retry, and the variant switcher — exists in the DOM only while a
   * message is hovered, and two things have to be right for that. The event has to start at
   * the deepest element under the cursor and bubble up, because the handler sits on an inner
   * node and an event targeted at the row never reaches it. And the point has to be inside
   * the message body: a human message is right-aligned inside a much wider row, so a point a
   * fixed distance in from the row's left edge lands in the empty space beside the bubble
   * and mounts nothing. Several points are tried, and the hover is only believed once
   * Claude's own controls actually appear.
   */
  async function hover(el) {
    if (!el) return false;
    let row = el.closest('[data-perf-row]') || el;

    /*
     * The transcript recycles its row elements as it scrolls, so the node handed to us can
     * be detached — or reused for a different message — by the time we point at it. A
     * detached node never contains the element under the cursor, so every candidate point is
     * rejected and the hover quietly does nothing, which is what made branch switching work
     * one moment and fail the next. Re-resolving by row position after every scroll keeps
     * hold of the message rather than the element that happened to be showing it.
     */
    const index = rowIndex(row);
    const resolve = () => {
      const fresh = index === null ? null : rowAtIndex(index);
      if (fresh) row = fresh;
      return row.isConnected;
    };

    /*
     * What counts as a hover that landed. Some controls are mounted whether or not the
     * message is pointed at — a row can carry a "show message actions" button of its own —
     * so the presence of a control proves nothing. Only a control that was not there a
     * moment ago does, or the switcher itself.
     */
    const before = row.querySelectorAll(CONTROLS_SELECTOR).length;
    const landed = () => Boolean(row.querySelector(PREV_SELECTOR))
      || row.querySelectorAll(CONTROLS_SELECTOR).length > before;

    // Three attempts: as the row lies, then with it moved to the middle of the screen, then
    // again after a further settle. A row can report itself on screen while every part of it
    // that is on screen sits underneath Claude's sticky header, and a point there lands on
    // the header instead of the message.
    for (let attempt = 0; attempt < 3; attempt += 1) {
      if (!resolve()) return false;
      if (attempt === 0) {
        if (hoverBand(row) < MIN_HOVER_BAND) { scrollToStart(row); await settle(); }
      } else {
        scrollToMiddle(row);
        await settle();
      }
      if (!resolve()) return false;

      let tried = 0;
      for (const point of hoverPoints(row)) {
        const deepest = document.elementFromPoint(point.clientX, point.clientY);
        if (!deepest || !row.contains(deepest)) continue; // covered by the header or a pane

        for (let node = deepest; node && node !== document.body; node = node.parentElement) {
          for (const [type, bubbles] of [['pointerover', true], ['pointerenter', false],
            ['mouseover', true], ['mouseenter', false], ['pointermove', true], ['mousemove', true]]) {
            const Kind = type.startsWith('pointer') ? PointerEvent : MouseEvent;
            node.dispatchEvent(new Kind(type, {
              ...point, bubbles, cancelable: true, composed: true, pointerId: 1, isPrimary: true,
            }));
          }
          if (node === row) break;
        }

        await settle();
        if (landed()) return true;
        if ((tried += 1) >= 6) break;
      }
    }
    return false;
  }

  /** Least of a row that has to be reachable before pointing at it is worth trying. */
  const MIN_HOVER_BAND = 40;

  /** How much of a message is both on screen and not under Claude's header, in pixels. */
  function hoverBand(row) {
    const rect = messageBody(row).getBoundingClientRect();
    if (!rect.height) return 0;
    return Math.min(rect.bottom, window.innerHeight - 8) - Math.max(rect.top, HEADER_INSET);
  }

  /**
   * Room to leave at the top of the viewport for Claude's own header, which floats above the
   * transcript. A row overlapping it is technically on screen and completely unhoverable.
   */
  const HEADER_INSET = 96;

  /**
   * Points worth aiming at, best first: inside the message body, then the row itself.
   * Kept within the band that is actually reachable rather than clamped to the viewport
   * edge, because clamping is what put every point on top of the header.
   */
  function* hoverPoints(row) {
    for (const box of [messageBody(row).getBoundingClientRect(), row.getBoundingClientRect()]) {
      if (!box.width || !box.height) continue;
      const top = Math.max(box.top, HEADER_INSET);
      const bottom = Math.min(box.bottom, window.innerHeight - 8);
      if (bottom - top < 4) continue;

      for (const y of [top + Math.min(24, (bottom - top) / 2), (top + bottom) / 2, bottom - 4]) {
        for (const x of [box.left + box.width / 2, box.left + Math.min(40, box.width / 2),
          box.right - Math.min(40, box.width / 2)]) {
          yield {
            clientX: Math.min(window.innerWidth - 2, Math.max(2, x)),
            clientY: y,
          };
        }
      }
    }
  }

  /** Put a message in the middle of the transcript, clear of anything floating over it. */
  function scrollToMiddle(el) {
    const scroller = findScroller();
    if (!scroller) return;
    const rect = el.getBoundingClientRect();
    const within = rect.top - scroller.getBoundingClientRect().top;
    const centred = (scroller.clientHeight - Math.min(rect.height, scroller.clientHeight)) / 2;
    scrollTranscript(scroller, scroller.scrollTop + within - centred);
  }

  /** Find the switcher, pointing at the message first if it is not already showing one. */
  async function locateSwitcher(el) {
    if (!el) return null;
    const already = findVariantSwitcher(el);
    if (already) return already;
    await hover(el);
    return findVariantSwitcher(el);
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

  /** The root-to-node path through the tree. */
  function pathTo(node) {
    const path = [];
    for (let n = node; n; n = n.parent) path.unshift(n);
    return path;
  }

  /** Throw away the cached branch identification, so the next read re-decides from the page. */
  function resetAlignment() {
    chosen = -1;
  }

  /**
   * Where the branch on screen and the branch `node` lives on first part company.
   *
   * Both are read as positions rather than as text. Position `i` of the displayed branch is
   * the row numbered `i`, so the fork is the first position at which the two paths hold
   * different messages — and that same number says exactly which row carries the control
   * that switches between them. Identifying the fork by text instead cannot be relied on:
   * the two candidates at a fork are often edits of one another, and a message whose body is
   * an attachment has almost no comparable text at all.
   */
  function findFork(node) {
    const displayed = displayedBranch();
    if (!displayed) return null;

    const wanted = pathTo(node);
    for (let i = 0; i < displayed.length && i < wanted.length; i += 1) {
      if (displayed[i].node.id !== wanted[i].id) {
        return { index: i, showing: displayed[i].node, wanted: wanted[i] };
      }
    }
    return null;
  }

  /**
   * The branch the chat is showing, as an ordered list of messages.
   *
   * Read from the page where that is possible. When the rows cannot be lined up against any
   * branch — a conversation still rendering, or markup we do not recognise — fall back to
   * the branch the API says is current, which is the first candidate. Returning nothing
   * there would mean refusing to switch branches at all rather than merely guessing badly.
   */
  function displayedBranch() {
    if (chosen < 0) alignRows();
    return (candidates[chosen] ?? candidates[0])?.entries ?? null;
  }

  /** Which message the chat is showing at a position on its branch, as the tree knows it. */
  function displayedAt(index) {
    return displayedBranch()?.[index]?.node ?? null;
  }

  /** The rendered row at a position on the displayed branch, however it can be found. */
  async function rowAt(index) {
    const numbered = await rowForPosition(index);
    if (numbered) return numbered;
    const node = displayedBranch()?.[index]?.node;
    return node ? findElement(node) : null;
  }

  /**
   * A cheap summary of what the transcript is showing.
   *
   * Used to tell whether pressing a control did anything at all. Reading the variant
   * readout instead is unreliable — it can only be read while the message is hovered, and
   * straight after a press the row holding it may not have been re-rendered yet, so the old
   * number is still there. Comparing what is on screen needs neither a hover nor a
   * successful branch identification.
   */
  function transcriptFingerprint() {
    const parts = [];
    for (const el of messageElements()) {
      const row = el.closest?.('[data-perf-row]') || el;
      parts.push(`${rowIndex(row) ?? ''}:${row.getAttribute('data-perf-row') ?? ''}:${messageText(el).slice(0, 24)}`);
    }
    return parts.join('|');
  }

  /**
   * Did that press actually move the chat?
   *
   * Claude re-renders a beat after the control is pressed, so this waits rather than looking
   * once. Nothing here scrolls, so the only thing that can change what is on screen is the
   * press itself.
   */
  async function pressTook(before, timeout = 1500) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (transcriptFingerprint() !== before) return true;
      await settle();
    }
    return false;
  }

  /**
   * The row carrying the fork's switcher.
   *
   * Its position is the reliable way to find it, but the row numbering is Claude's and could
   * go away, so this still falls back to locating the message itself the older way rather
   * than losing the ability to switch branches at all.
   */
  async function forkRow(fork) {
    const byPosition = await rowForPosition(fork.index);
    if (byPosition) return byPosition;
    return findElement(fork.showing) || hunt(fork.showing);
  }

  /**
   * Walk the chat onto the branch a message lives on by pressing the same controls a person
   * would, one fork at a time.
   *
   * Claude's own controls are used rather than the API because they leave the page and the
   * server agreeing: the client rewrites the conversation's recorded leaf to match whatever
   * it is showing, so moving the leaf underneath it produces exactly the reported symptom —
   * the chat lands on a sibling, and pressing again sometimes rescues it. After every press
   * the branch on screen is identified again from scratch, because the previous reading
   * described the branch we just left.
   *
   * @returns {Promise<boolean>} whether the displayed branch was changed
   */
  async function switchToBranch(node) {
    let switched = false;
    const handled = new Set();

    for (let hop = 0; hop < 6; hop += 1) {
      const fork = findFork(node);
      if (!fork) break; // already on the branch this message lives on

      /*
       * Each fork is switched once and only once. Reading the branch back straight after a
       * press is not always possible — the transcript may line up with nothing while it
       * re-renders, and the fallback reading is the branch the API still believes is
       * current, which is the one we just left. Without this the same fork is "found" again
       * and pressed back, and the chat oscillates between two siblings.
       */
      if (handled.has(fork.index)) break;
      handled.add(fork.index);

      const row = await forkRow(fork);
      if (!row) break;

      const switcher = await locateSwitcher(row);
      if (!switcher) break;

      const siblings = fork.wanted.parent ? fork.wanted.parent.children : [];
      const want = siblings.indexOf(fork.wanted);
      if (want < 0) break;

      /*
       * Step towards the variant that was asked for, reading Claude's own "2 / 3" after every
       * press rather than working out the whole move in advance.
       *
       * Working it out in advance needs to know which variant is showing, and that cannot
       * always be read from the page: sibling replies to the same prompt routinely begin with
       * the same words, so unless the rows that differ happen to be on screen, one branch
       * looks exactly like another. A move planned from the wrong starting point lands on a
       * sibling of the branch that was wanted. The readout beside the control is authoritative
       * about which variant is showing, and both it and the tree order variants by when they
       * were made, so comparing the two says exactly which way to go — and asking again after
       * each press means a wrong guess corrects itself instead of being carried through.
       */
      let moved = 0;
      let arrived = false;
      let stalled = false;
      for (let step = 0; step <= siblings.length && !stalled; step += 1) {
        // Re-resolve every time: the transcript recycles rows as it scrolls, and the control
        // belongs to the row rather than to the element that was showing it a moment ago.
        const live = await locateSwitcher((await rowAt(fork.index)) || row);
        if (!live) break;

        const from = live.index ?? siblings.indexOf(fork.showing);
        if (from < 0) break;
        if (from === want) { arrived = true; break; } // already showing what was asked for

        const forward = want > from;
        const button = forward ? live.next : live.prev;
        if (button.disabled || button.getAttribute('aria-disabled') === 'true') break;

        const showing = transcriptFingerprint();
        press(button);
        const took = await pressTook(showing);
        resetAlignment();

        /*
         * Every press is checked against the page. A control that looked like a variant
         * switcher but changed nothing has to be undone and abandoned — otherwise the pane
         * announces a branch switch that never happened, and the reader is left looking at
         * the same conversation wondering why the tree disagrees with it.
         */
        if (!took) {
          const undo = await locateSwitcher((await rowAt(fork.index)) || row);
          const back = forward ? undo?.prev : undo?.next;
          if (back && !(back.disabled || back.getAttribute('aria-disabled') === 'true')) {
            press(back);
            await settle();
            resetAlignment();
          }
          stalled = true;
          break;
        }

        moved += 1;
      }

      /*
       * Nothing to do at this fork because it already shows the variant wanted: the reading
       * of which branch was on screen was wrong, not the request. Carry on to the next fork
       * rather than abandoning the whole thing — that reading is exactly what fails when
       * sibling replies begin with the same words.
       */
      if (!moved) {
        if (arrived) continue;
        break;
      }
      switched = true;
    }

    return switched;
  }

  /** Where a message sits on the branch the chat is showing, or null if it is not on it. */
  function branchPosition(node) {
    if (!node) return null;
    if (chosen < 0) alignRows();
    for (const index of [chosen, ...tied]) {
      const position = candidates[index]?.positions.get(node.id);
      if (Number.isInteger(position)) return position;
    }
    return null;
  }

  async function huntByIndex(node, scroller) {
    const target = branchPosition(node);
    return target === null ? null : huntToIndex(target, scroller);
  }

  /** The row at a position on the displayed branch, scrolling the chat to it if need be. */
  async function rowForPosition(index) {
    const here = rowAtIndex(index);
    if (here) return here;
    const scroller = findScroller();
    return scroller ? huntToIndex(index, scroller) : null;
  }

  /**
   * Scroll straight to a position on the branch using the transcript's own row numbering.
   *
   * The rows that are mounted state their exact positions and where they sit in the
   * scroller, so a missing one can be interpolated between its neighbours and scrolled to
   * directly. The blind sweep below cannot do this: it tries a fixed ladder of scroll
   * fractions, and on a long conversation the message falls between two rungs, so the same
   * hunt succeeds one moment and fails the next. Bisecting on the span of rendered rows does
   * not work either — the transcript drops rows out of the middle of that span, so being
   * "inside" it does not mean the row is there.
   */
  async function huntToIndex(target, scroller) {
    const travel = scroller.scrollHeight - scroller.clientHeight;
    if (travel <= 0) return rowAtIndex(target);

    let previous = -1;
    for (let step = 0; step < 10; step += 1) {
      const here = rowAtIndex(target);
      if (here) return here;

      const known = renderedRows(scroller);
      if (!known.length) return null;

      const below = [...known].reverse().find((row) => row.index < target);
      const above = known.find((row) => row.index > target);

      // Rows are not a uniform height, so interpolate between the two nearest known rows and
      // fall back to the average spacing when the target lies outside them.
      const span = known[known.length - 1].index - known[0].index;
      const spacing = span > 0
        ? (known[known.length - 1].offset - known[0].offset) / span
        : scroller.clientHeight;

      let guess;
      if (below && above) {
        const ratio = (target - below.index) / (above.index - below.index);
        guess = below.offset + (above.offset - below.offset) * ratio;
      } else if (below) {
        guess = below.offset + (target - below.index) * spacing;
      } else {
        guess = above.offset - (above.index - target) * spacing;
      }

      /*
       * Which way the message lies, taken from the row numbers rather than from the
       * interpolated offset. Once we have scrolled to that offset it sits behind us, so
       * asking whether it is above or below says "carry on the way you came" — and a message
       * above the rendered rows sends the chat downwards, away from it, then back, for as
       * many attempts as it is given. That oscillation is why a message sometimes could not
       * be reached no matter how long it tried.
       */
      const towards = target < known[0].index ? -1
        : target > known[known.length - 1].index ? 1
        : (Math.sign(guess - scroller.scrollTop) || -1);

      // Land the row a third of the way down, so it is on screen and hoverable, not flush
      // against the top edge under Claude's header.
      let want = Math.min(travel, Math.max(0, guess - scroller.clientHeight / 3));

      // Interpolating between two rows only estimates where a third one starts, because rows
      // differ in height by an order of magnitude — a one-line question next to a
      // thousand-pixel answer. When the jump lands short the row still is not mounted, so
      // creep towards it a screen at a time rather than landing on the same wrong spot and
      // declaring the message unreachable.
      if (Math.abs(want - scroller.scrollTop) < 8 || Math.abs(want - previous) < 8) {
        want = Math.min(travel, Math.max(0,
          scroller.scrollTop + towards * scroller.clientHeight * 0.75));
      }
      if (Math.abs(want - scroller.scrollTop) < 8) return rowAtIndex(target);
      previous = want;
      scrollTranscript(scroller, want);
      await settle();
    }
    return rowAtIndex(target);
  }

  /** Scroll the chat until Claude has rendered `node`, or give up. */
  async function hunt(node, position) {
    let el = findElement(node);
    if (el) return el;

    const scroller = findScroller();
    if (!scroller) return null;

    el = await huntByIndex(node, scroller);
    if (el) return el;

    const previous = scroller.scrollTop;
    const travel = scroller.scrollHeight - scroller.clientHeight;
    for (const fraction of sweep(position)) {
      scrollTranscript(scroller, travel * fraction);
      await settle();
      el = findElement(node);
      if (el) return el;
    }
    scrollTranscript(scroller, previous);
    return null;
  }

  /** Is this row parked where `scrollToStart` puts it? */
  function atStart(el) {
    const scroller = findScroller();
    if (!scroller) return false;
    const offset = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
    return Math.abs(offset - TOP_INSET) <= 24;
  }

  /**
   * Scroll to a message and keep it there.
   *
   * Claude scrolls the transcript to the end of the branch whenever it renders one, so a
   * single scroll made while it is still settling is simply undone: the reader is asked for
   * a message in the middle of a branch and is left looking at its last message instead.
   * Re-asserting the position until it holds is what makes the chat land where it was asked
   * to. It gives up quietly rather than fighting a reader who scrolls somewhere themselves,
   * because the position is only re-asserted while the message stays findable and the loop
   * is short.
   */
  async function landAt(node, position, tries = 12) {
    let held = 0;
    for (let attempt = 0; attempt < tries; attempt += 1) {
      // Claude's scroll to the end of the branch takes the message off screen again, and a
      // message that is not rendered cannot be scrolled to — so look for it once more rather
      // than concluding it has gone. This is what left the reader at the last message of the
      // branch while the pane reported that it had scrolled to theirs.
      const el = findElement(node) || await hunt(node, position);
      if (!el) return false;

      if (atStart(el)) {
        // Two readings in a row means Claude has stopped moving the transcript under us.
        if ((held += 1) >= 2) return true;
      } else {
        /*
         * Glide there the first time, because that is what a reader expects to see. After
         * that, jump: a smooth scroll is an animation, and an animation can simply not
         * happen — the tab is in the background, the reader prefers reduced motion, or
         * Claude's own scroll to the end of the branch interrupts it — leaving the message
         * never arrived at and the reader looking at the end of the branch instead.
         */
        held = 0;
        scrollToStart(el, attempt === 0);
      }
      await settle();
    }
    return Boolean(findElement(node));
  }

  /**
   * Scroll the chat to a message.
   *
   * Claude renders lazily, and the branch it is showing is not always the branch the API
   * last recorded, so this searches before it draws any conclusion — and if the message
   * really is on another branch, it asks Claude to show that branch first.
   *
   * @param {object} node tree node to look for
   * @param {{onPath: boolean, position: number, allowSwitch: boolean}} hint what the API
   *   believes about the message, roughly how far down its branch it sits (0–1), and whether
   *   changing branch is allowed. A caller that has just switched passes `allowSwitch: false`
   *   so that failing to find the message cannot set off another switch — pressing more
   *   variant controls while trying to scroll somewhere is how the chat ends up on a sibling
   *   of the branch that was asked for.
   * @returns {Promise<'found'|'switched'|'off-path'|'not-found'>}
   */
  async function revealNode(node, { onPath, position, allowSwitch = true } = {}) {
    /*
     * Sweep the scroller for a message that should be on the branch already showing. For one
     * the tree places elsewhere, a cheap look is enough before switching — sweeping the whole
     * conversation just to fail first costs a second and a half.
     *
     * When switching is not allowed there is nothing to save the sweep for, so always look
     * properly. This is the case that made showing a message take two presses: landing after
     * a switch, the tree still places the message on the branch it came from, so the cheap
     * look was used — and Claude had just scrolled to the end of the branch, taking the
     * message out of the page entirely. Nothing was found, the landing gave up, and only a
     * second press, by which time everything agreed, actually scrolled anywhere.
     */
    let el = (onPath || !allowSwitch) ? await hunt(node, position) : findElement(node);
    if (el) {
      await landAt(node, position);
      return 'found';
    }

    // Switching leaves the previous reading of the branch stale, so decide again from the
    // page and then scroll to the message. Landing on the right branch but at the top of the
    // conversation is not much use, and the caller cannot always recover it afterwards.
    if (allowSwitch && await switchToBranch(node)) {
      resetAlignment();
      await landAt(node, position);
      return 'switched';
    }

    // Nothing was switched. Before giving up on a message the tree places on another branch,
    // look for it properly — when sibling branches read alike, the branch it was thought to
    // be on can simply have been wrong, and the message is in the page after all.
    if (!onPath && await hunt(node, position)) {
      await landAt(node, position);
      return 'found';
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
      /** Re-evaluate after the tree has changed; branches come from setBranches(). */
      refreshBranches() {
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

  CT.chat = { revealNode, hasVisibleSwitcher, trackReading, watchForUpdates,
    findElement, alignRows, setBranches, onDisplayedBranch };
})();
