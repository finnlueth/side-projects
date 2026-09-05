/**
 * The side panel: shadow-DOM UI, tree rendering, pan/zoom and the message detail drawer.
 */
(() => {
  'use strict';

  const CT = (globalThis.CT ||= {});
  const ext = globalThis.browser ?? globalThis.chrome;

  const PREFS_KEY = 'prefs';
  const MIN_WIDTH = 340;
  const MAX_WIDTH = 960;
  /* Wide enough that two branches of 315px nodes stay legible without zooming out. */
  const DEFAULT_WIDTH = 560;
  /** Page width to leave for the chat itself when the pane is docked. */
  const MIN_PAGE_WIDTH = 380;
  /**
   * Toast notifications are switched off. The calls are left where they are so turning
   * them back on is a one-line change; failures are still reported through the pane itself.
   */
  const SHOW_TOASTS = false;
  /** The summary pills above the tree are switched off; the code stays in place. */
  const SHOW_STATS = false;
  /**
   * Showing one of Claude's replies in the chat scrolls to the prompt above it instead.
   *
   * A reply reads better from the question that produced it, and a long answer scrolled to
   * its own first line gives no clue what was asked. Set to false to land on the reply
   * itself. Only the scrolling changes: the message picked in the tree stays picked, and the
   * branch worked out to reach it is the reply's own.
   */
  const JUMP_TO_PROMPT = true;
  const MIN_DETAIL = 120;
  /** How long a click waits to see whether it is really the first half of a double-click. */
  const DOUBLE_CLICK_MS = 250;
  /** How long the canvas takes to glide when it follows the chat. */
  const GLIDE_MS = 260;
  /** Room left around a message the canvas has followed to. */
  const FOLLOW_PAD = 28;
  /** Share of the pane the message drawer may take before it crowds out the tree. */
  const MAX_DETAIL_RATIO = 0.8;
  const MIN_SCALE = 0.15;
  const MAX_SCALE = 2;

  /** Box + gap sizes per orientation, in the layout's (spread, depth) space. */
  /* Nodes are a fixed 315 (vertical) or 371 (horizontal) wide; their height follows their
     content, up to the four-line clamp. Only the gaps are configured here. */
  const NODE_WIDTH = { vertical: 315, horizontal: 371 };
  const LAYOUT = {
    vertical: { spreadGap: 24, depthGap: 34 },
    horizontal: { spreadGap: 18, depthGap: 60 },
  };

  /**
   * HTML-escape. Message text comes from the API, so every interpolation of conversation
   * data into markup below goes through this — in attribute and text positions alike.
   */
  const esc = (value) =>
    String(value ?? '').replace(/[&<>"']/g, (ch) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const glyph = (body, extra = '') =>
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" ` +
    `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"${extra}>${body}</svg>`;

  const TREE_PATHS =
    '<circle cx="12" cy="4.5" r="2.2"/><circle cx="6" cy="19.5" r="2.2"/>' +
    '<circle cx="18" cy="19.5" r="2.2"/>' +
    '<path d="M12 6.7v3.6c0 3.4-6 2.6-6 7M12 10.3c0 3.4 6 2.6 6 7"/>';

  /**
   * Icons.
   *
   * claude.ai draws its icons from an Anthropicons font, so the pane uses the same glyphs
   * rather than redrawing them. Codepoints are read off the live page by accessible name
   * where possible — that self-corrects if Anthropic renumbers the font — and fall back to
   * the codepoints observed in Claude's own controls, then to an SVG of our own.
   */
  const GLYPHS = { refresh: '\ue11d', copy: '\ue056', search: '\ue0d3', expand: '\ue067' };

  /** Accessible names on claude.ai's own controls that render the icon we want. */
  const DISCOVERABLE = {
    close: /^(close|dismiss)\b/i,
    refresh: /^retry$/i,
    copy: /^copy$/i,
    search: /^search$/i,
  };

  const discovered = new Map();
  let hasIconFont = null;

  function iconFontLoaded() {
    if (hasIconFont) return true;
    try {
      // Only a positive result is cached — the face may not be registered on an early call.
      hasIconFont = Array.from(document.fonts).some((face) => /anthropicons/i.test(face.family));
    } catch {
      hasIconFont = false;
    }
    return hasIconFont;
  }

  /** Harvest glyphs from claude.ai's own buttons. Cheap, and safe to call again. */
  function discoverGlyphs() {
    if (!iconFontLoaded()) return;
    const wanted = Object.entries(DISCOVERABLE).filter(([key]) => !discovered.has(key));
    if (!wanted.length) return;
    for (const el of document.querySelectorAll('[data-cds="Icon"]')) {
      const name = el.closest('[aria-label]')?.getAttribute('aria-label')?.trim();
      const glyph = el.textContent?.trim();
      if (!name || !glyph || glyph.length > 2) continue;
      for (const [key, pattern] of wanted) {
        if (!discovered.has(key) && pattern.test(name)) discovered.set(key, glyph);
      }
    }
  }

  function icon(name) {
    const glyph = discovered.get(name) ?? (iconFontLoaded() ? GLYPHS[name] : null);
    if (glyph) return `<span class="ct-glyph" aria-hidden="true">${glyph}</span>`;
    return ICON[name] ?? '';
  }

  const ICON = {
    tree: glyph(TREE_PATHS),
    treeRight: glyph(`<g transform="rotate(-90 12 12)">${TREE_PATHS}</g>`),
    close: glyph('<path d="M6 6l12 12M18 6L6 18"/>'),
    refresh: glyph('<path d="M20.5 12a8.5 8.5 0 1 1-2.6-6.1"/><path d="M20.5 4.5V10H15"/>'),
    plus: glyph('<path d="M12 5.5v13M5.5 12h13"/>'),
    minus: glyph('<path d="M5.5 12h13"/>'),
    fit: glyph('<path d="M4 9V4.5h4.5M20 9V4.5h-4.5M4 15v4.5h4.5M20 15v4.5h-4.5"/>'),
    copy: glyph('<rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M5.5 15V6.5a2 2 0 0 1 2-2H16"/>'),
    locate: glyph('<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.6-4.6"/>'),
    left: glyph('<path d="M14.5 6l-6 6 6 6"/>'),
    right: glyph('<path d="M9.5 6l6 6-6 6"/>'),
    warning: glyph('<path d="M12 8.5v5M12 17h.01"/><circle cx="12" cy="12" r="9"/>'),
    branch: glyph('<path d="M4 12h4.5c3 0 3-5 6-5H20M16.5 4.5 20 7l-3.5 2.5M8.5 12c3 0 3 5 6 5H20M16.5 14.5 20 17l-3.5 2.5"/>'),
    chat: glyph('<path d="M20 14.5a2.5 2.5 0 0 1-2.5 2.5H8l-4 3.5V6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5z"/>'),
    // A message with the tree following it up and down.
    follow: glyph('<circle cx="12" cy="12" r="2.75"/><path d="M12 2.5v4M12 17.5v4"/>'
      + '<path d="M8.75 5.75 12 2.5l3.25 3.25M8.75 18.25 12 21.5l3.25-3.25"/>'),
  };

  /* ------------------------------------------------------------------ theme ------ */

  function parseLuminance(color) {
    const match = /rgba?\(([^)]+)\)/.exec(color || '');
    if (!match) return null;
    const parts = match[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.some(Number.isNaN)) return null;
    if (parts.length >= 4 && parts[3] === 0) return null; // fully transparent tells us nothing
    const [r, g, b] = parts;
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  }

  function detectTheme() {
    const root = document.documentElement;
    if (root.classList.contains('dark')) return 'dark';
    if (root.classList.contains('light')) return 'light';
    const declared = root.dataset.theme || root.dataset.mode || document.body?.dataset?.theme;
    if (declared === 'dark' || declared === 'light') return declared;
    const luminance = parseLuminance(getComputedStyle(document.body || root).backgroundColor);
    if (luminance !== null) return luminance < 0.5 ? 'dark' : 'light';
    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  /** Borrow the font families claude.ai has already loaded so the panel matches the app. */
  function pageFonts() {
    const fonts = { sans: null, serif: null };
    try {
      fonts.sans = getComputedStyle(document.body).fontFamily || null;
    } catch { /* ignore */ }
    try {
      for (const face of document.fonts) {
        const family = String(face.family || '').replace(/^['"]|['"]$/g, '');
        if (!fonts.serif && /copernicus|tiempos|garamond|serif/i.test(family)) {
          fonts.serif = `"${family}", ui-serif, Georgia, serif`;
        }
      }
    } catch { /* ignore */ }
    return fonts;
  }

  const themeWatchers = new Set();
  let themeObserverStarted = false;

  function watchTheme(callback) {
    themeWatchers.add(callback);
    if (themeObserverStarted) return;
    themeObserverStarted = true;
    const notify = () => {
      const theme = detectTheme();
      themeWatchers.forEach((fn) => fn(theme));
    };
    new MutationObserver(notify).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'style', 'data-theme', 'data-mode'],
    });
    if (document.body) {
      new MutationObserver(notify).observe(document.body, {
        attributes: true,
        attributeFilter: ['class', 'style', 'data-theme'],
      });
    }
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', notify);
  }

  /* ------------------------------------------------------------------ styles ----- */

  let cssPromise = null;

  function styleText() {
    if (!cssPromise) {
      cssPromise = fetch(ext.runtime.getURL('src/panel.css')).then((res) => res.text());
    }
    return cssPromise;
  }

  async function attachStyles(shadowRoot) {
    const style = document.createElement('style');
    style.textContent = await styleText();
    shadowRoot.prepend(style);
  }

  /**
   * Create a shadow host that also tracks the page theme and fonts.
   *
   * Hosts default to hanging off <html> rather than <body>: content.js reshapes the body
   * box to make room for the pane, and anything inside it would be reshaped along with it.
   */
  async function createHost(hostClass, { hidden = false, parent = document.documentElement } = {}) {
    const host = document.createElement('div');
    host.className = hostClass;
    host.hidden = hidden;
    host.dataset.theme = detectTheme();
    const fonts = pageFonts();
    if (fonts.sans) host.style.setProperty('--ct-page-font', fonts.sans);
    if (fonts.serif) host.style.setProperty('--ct-page-serif', fonts.serif);
    const shadow = host.attachShadow({ mode: 'open' });
    await attachStyles(shadow);
    watchTheme((theme) => { host.dataset.theme = theme; });
    if (parent) parent.appendChild(host);
    return { host, shadow };
  }

  /* ------------------------------------------------------------------ helpers ---- */

  async function loadPrefs() {
    try {
      const stored = await ext.storage.local.get(PREFS_KEY);
      return stored?.[PREFS_KEY] ?? {};
    } catch {
      return {};
    }
  }

  /** Reload onto the branch just selected, and bring the pane back with it. */
  async function reopenAfterReload(conversationId, message) {
    try {
      await ext.storage.local.set({ resume: { conversation: conversationId, message, at: Date.now() } });
    } catch { /* the pane simply stays closed after the reload */ }
    location.reload();
  }

  function savePrefs(prefs) {
    try {
      ext.storage.local.set({ [PREFS_KEY]: prefs });
    } catch { /* storage is optional */ }
  }

  function formatTime(iso) {
    if (!iso) return '';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch { /* fall through to the legacy path */ }
    try {
      const area = document.createElement('textarea');
      area.value = text;
      area.style.cssText = 'position:fixed;top:-1000px;opacity:0';
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand('copy');
      area.remove();
      return ok;
    } catch {
      return false;
    }
  }

  /* ------------------------------------------------------------------ panel ------ */

  class TreePanel {
    constructor() {
      this.open = false;
      this.mounted = false;
      this.mounting = null;
      this.orientation = 'vertical';
      this.width = DEFAULT_WIDTH;
      this.conversationId = null;
      this.tree = null;
      this.status = 'idle'; // idle | loading | ready | error
      this.error = null;
      this.selectedId = null;
      this.currentIds = new Set();
      this.currentKey = '';
      this.loadedAt = 0;
      this.dirty = true;
      this.view = { x: 0, y: 0, scale: 1 };
      this.content = { width: 0, height: 0 };
      this.onOpenChange = null;
      this.onStatsChange = null;
      this.onTreeChange = null;
      this.toastTimer = 0;
      this.exitTimer = 0;
      this.viewIsDefault = false;
      /** Whether the canvas follows the messages the chat is showing. */
      this.sticky = false;
      this.glideFrame = 0;
      /** Where a glide in progress is heading, so follow-ups measure against that. */
      this.glideTarget = null;
      this.lastFollowTop = null;
      this.centreOnCurrent = false;
      this.shapeReportedFor = null;
      this.pendingTarget = null;
      this.readyWaiters = [];
      this.detailHeight = 0;
    }

    /**
     * Sit below claude.ai's own overlays while one is open (search, dialogs).
     * @param {number} level stacking level to sit at, or 0 to return to the front
     */
    setBehind(level) {
      if (!this.host) return;
      this.host.classList.toggle('ct-behind', level > 0);
      if (level > 0) this.host.style.setProperty('--ct-behind-z', String(level));
    }

    /** Widest the pane may get without crowding the chat out of the window. */
    maxWidth() {
      return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, window.innerWidth - MIN_PAGE_WIDTH));
    }

    /* -------------------------------------------------------------- lifecycle -- */

    async ensureMounted() {
      if (this.mounted) return;
      if (!this.mounting) {
        this.mounting = this.mount().catch((err) => {
          this.mounting = null; // allow a later click to retry
          throw err;
        });
      }
      await this.mounting;
    }

    async mount() {
      const prefs = await loadPrefs();
      if (prefs.orientation === 'horizontal' || prefs.orientation === 'vertical') {
        this.orientation = prefs.orientation;
      }
      this.width = clamp(Number(prefs.width) || DEFAULT_WIDTH, MIN_WIDTH, this.maxWidth());
      this.detailHeight = Number(prefs.detailHeight) || 0;
      this.sticky = prefs.sticky === true;

      discoverGlyphs();
      const { host, shadow } = await createHost('ct-panel-host', { hidden: true });
      this.host = host;
      this.shadow = shadow;
      host.style.setProperty('--ct-w', `${this.width}px`);
      if (this.detailHeight) this.applyDetailHeight(this.detailHeight);

      shadow.append(this.buildDom());
      this.wireEvents();
      this.mounted = true;
      this.renderAll();
    }

    buildDom() {
      const root = document.createElement('aside');
      root.className = 'ct-root';
      root.setAttribute('role', 'complementary');
      root.setAttribute('aria-label', 'Claude conversation tree');
      root.innerHTML = `
        <div class="ct-resize" role="separator" aria-orientation="vertical" title="Drag to resize"></div>
        <div class="ct-bar">
          <div class="ct-tools" data-role="toolbar">
            <div class="ct-segmented" role="group" aria-label="Tree direction">
              <button data-action="orient" data-value="vertical" title="Top to bottom" aria-label="Top to bottom">${ICON.tree}</button>
              <button data-action="orient" data-value="horizontal" title="Left to right" aria-label="Left to right">${ICON.treeRight}</button>
            </div>
            <div class="ct-zoom">
              <button class="ct-icon-btn" data-action="zoom-out" title="Zoom out" aria-label="Zoom out">${ICON.minus}</button>
              <span class="ct-zoom-value" data-role="zoom">100%</span>
              <button class="ct-icon-btn" data-action="zoom-in" title="Zoom in" aria-label="Zoom in">${ICON.plus}</button>
              <button class="ct-icon-btn" data-action="fit" title="Reset view — press again to fit the whole tree" aria-label="Reset view">${ICON.fit}</button>
              <button class="ct-icon-btn" data-action="sticky" aria-pressed="false"
                title="Follow the chat — keep the messages on screen centred in the tree"
                aria-label="Follow the chat">${ICON.follow}</button>
            </div>
          </div>
          <div class="ct-bar-end">
            <button class="ct-icon-btn" data-action="refresh" title="Reload the tree" aria-label="Reload the tree">${icon('refresh')}</button>
            <button class="ct-icon-btn" data-action="close" title="Close" aria-label="Close">${icon('close')}</button>
          </div>
        </div>
        <div class="ct-stats" data-role="stats"></div>
        <div class="ct-canvas" data-role="canvas">
          <div class="ct-viewport" data-role="viewport">
            <svg class="ct-edges" data-role="edges"></svg>
            <div class="ct-nodes" data-role="nodes"></div>
          </div>
          <div class="ct-state" data-role="state" hidden></div>
          <div class="ct-toast" data-role="toast" role="status" aria-live="${SHOW_TOASTS ? 'polite' : 'off'}"></div>
        </div>
        <section class="ct-detail" data-role="detail" hidden></section>
      `;

      this.el = {};
      for (const node of root.querySelectorAll('[data-role]')) {
        this.el[node.dataset.role] = node;
      }
      this.el.root = root;
      this.el.resize = root.querySelector('.ct-resize');
      return root;
    }

    /* ------------------------------------------------------------------ events -- */

    wireEvents() {
      const { root, canvas, nodes, detail, resize } = this.el;

      root.addEventListener('click', (event) => {
        const button = event.target.closest('[data-action]');
        if (!button) return;
        this.handleAction(button.dataset.action, button);
      });

      nodes.addEventListener('click', (event) => {
        if (this.suppressClick) return;
        const el = event.target.closest('.ct-node');
        if (!el) return;
        // Selecting only. Clicking the open message used to close the drawer again, which
        // fought the double-click that shows the message in the chat: the first of the two
        // clicks closed what the second one needed.
        this.select(el.dataset.id, { center: false });
      });

      nodes.addEventListener('dblclick', (event) => {
        const el = event.target.closest('.ct-node');
        const node = el && this.tree?.nodes.get(el.dataset.id);
        if (!node) return;
        event.stopPropagation();
        this.select(node.id, { center: false });
        this.goToMessage(node);
      });

      nodes.addEventListener('keydown', (event) => this.handleNodeKey(event));

      detail.addEventListener('click', (event) => {
        const button = event.target.closest('[data-detail]');
        if (button) this.handleDetailAction(button.dataset.detail);
      });

      // Delegated: the drawer's contents are re-rendered on every selection.
      detail.addEventListener('pointerdown', (event) => {
        if (event.target.closest('.ct-detail-resize')) this.handleDetailResize(event);
      });

      canvas.addEventListener('wheel', (event) => this.handleWheel(event), { passive: false });
      canvas.addEventListener('pointerdown', (event) => this.handlePanStart(event));
      canvas.addEventListener('click', (event) => {
        if (this.suppressClick) return;               // the end of a pan, not a click
        if (event.target.closest('.ct-node')) return; // the nodes handle their own clicks
        /*
         * Held briefly, because the first half of a double-click is a click. Clearing the
         * selection immediately would mean a double-click on the canvas could never act on a
         * selected message — the click that starts it would have just thrown it away.
         */
        clearTimeout(this.deselectTimer);
        this.deselectTimer = setTimeout(() => this.select(null), DOUBLE_CLICK_MS);
      });

      canvas.addEventListener('dblclick', (event) => {
        if (event.target.closest('.ct-node')) return; // handled as "jump to this message"
        clearTimeout(this.deselectTimer);             // this pair was not a deselect

        // Bring the selected message back into view. After panning around a large
        // conversation what is wanted is the message being worked with, not the whole tree
        // re-framed from its root.
        if (this.selectedId
          && this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(this.selectedId)}"]`)) {
          this.centerOn(this.selectedId);
          return;
        }
        // Nothing selected: go to where the conversation currently ends, which is what
        // "back to where I was" means when no single message is in hand.
        const last = this.currentPathLeaf();
        if (last && this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(last.id)}"]`)) {
          this.centerOn(last.id);
          return;
        }
        if (this.viewIsDefault) this.fitAll();
        else this.resetView();
      });

      resize.addEventListener('pointerdown', (event) => this.handleResizeStart(event));

      root.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        if (this.selectedId) this.select(null);
        else this.setOpen(false);
        event.stopPropagation();
      });

      /*
       * Hold the view still when the canvas changes size — the browser window, the pane's
       * own width, or the drawer opening underneath it.
       *
       * The tree is drawn at a fixed offset inside the canvas, so a canvas that grows leaves
       * it where it was: pinned towards the left edge with the new space all on one side.
       * Moving the view by half the change keeps whatever was in the middle of the canvas in
       * the middle of it, which for a tree that is centred keeps it exactly centred, and for
       * one larger than the canvas keeps the reader looking at the same place instead of
       * having the tree jump.
       */
      if (typeof ResizeObserver === 'function') {
        let previous = null;
        new ResizeObserver(() => {
          const { width, height } = canvas.getBoundingClientRect();
          const last = previous;
          previous = { width, height };
          if (!last || !width || !height || !last.width || !last.height) return;
          if (this.status !== 'ready' || !this.open) return;

          this.view.x += (width - last.width) / 2;
          this.view.y += (height - last.height) / 2;
          // Resizing is not the reader taking the view over, so the default framing — and
          // the fit button that toggles against it — survives one.
          this.applyView({ isDefault: this.viewIsDefault });
          // Nothing else: a resize must leave the view exactly where it was. Pulling the
          // selected message back into view here made the whole tree jump to it whenever the
          // window changed size, which is not what resizing a window is asking for.
        }).observe(canvas);
      }

      window.addEventListener('resize', () => {
        const capped = clamp(this.width, MIN_WIDTH, this.maxWidth());
        if (capped === this.width) return;
        this.width = capped;
        this.host.style.setProperty('--ct-w', `${capped}px`);
        this.onOpenChange?.(this.open, capped);
      });

      document.addEventListener('visibilitychange', () => {
        if (document.hidden || !this.open || this.status !== 'ready') return;
        if (Date.now() - this.loadedAt > 30000) this.load({ quiet: true });
      });
    }

    handleAction(action, button) {
      switch (action) {
        case 'close': this.setOpen(false); break;
        case 'refresh': this.load({ force: true }); break;
        case 'retry': this.load({ force: true }); break;
        case 'zoom-in': this.zoomBy(1.25); break;
        case 'zoom-out': this.zoomBy(1 / 1.25); break;
        case 'fit': this.viewIsDefault ? this.fitAll() : this.resetView(); break;
        case 'orient': this.setOrientation(button.dataset.value); break;
        case 'sticky': this.setSticky(!this.sticky); break;
        default: break;
      }
    }

    handleDetailAction(action) {
      const node = this.tree?.nodes.get(this.selectedId);
      if (!node) return;
      if (action === 'close') { this.select(null); return; }
      if (action === 'copy') {
        copyText(node.text || node.preview || '').then((ok) =>
          this.toast(ok ? 'Message copied' : 'Could not copy message'));
        return;
      }
      if (action === 'goto') {
        this.goToMessage(node);
        return;
      }
      if (action === 'prev' || action === 'next') {
        const siblings = node.parent ? node.parent.children : this.tree.roots;
        const target = siblings[node.siblingIndex + (action === 'next' ? 1 : -1)];
        if (target) this.select(target.id, { center: true });
      }
    }

    /**
     * Take the chat to a message, whatever that requires.
     *
     * Scrolling is enough when the message is on the branch already showing. When it is not,
     * Claude's own "‹ 2/3 ›" control is tried, and failing that the conversation's current
     * message is moved through the API — which needs a reload for Claude to re-render.
     */
    /**
     * Which message the chat is scrolled to when `node` is asked for.
     *
     * See JUMP_TO_PROMPT. The prompt is the reply's own parent, so it is on the same branch
     * and needs no separate work to reach.
     */
    scrollTargetFor(node) {
      if (!JUMP_TO_PROMPT || node.sender !== 'assistant') return node;
      return node.parent?.sender === 'human' ? node.parent : node;
    }

    async goToMessage(node) {
      const depth = Math.max(1, (this.tree?.stats.depth ?? 1) - 1);
      this.toast('Looking for the message…', { sticky: true });

      // The branch is worked out from the message that was asked for; only where the chat
      // comes to rest can differ.
      const anchorId = this.scrollTargetFor(node).id;

      const outcome = await CT.chat.revealNode(node, {
        onPath: CT.chat.onDisplayedBranch(node),
        position: node.depth / depth,
      });
      if (outcome === 'found') {
        if (anchorId !== node.id) await this.landOn(anchorId, { select: false });
        this.toast('Scrolled to the message');
        return;
      }
      if (outcome === 'switched') {
        // The chat is on the right branch now; reload so the tree agrees with it, then
        // locate the message properly rather than leaving the reader at the top.
        await this.load({ quiet: true });
        const landed = await this.landOn(node.id);
        if (landed && anchorId !== node.id) await this.landOn(anchorId, { select: false });
        this.toast(landed
          ? 'Switched branch and scrolled to the message'
          : 'Switched the chat to that branch');
        return;
      }
      if (outcome === 'not-found') {
        this.toast('Could not find that message in the page');
        return;
      }

      // On another branch and no switcher to be found: move the conversation itself.
      const leaf = CT.model.deepestLeaf(node);
      if (!leaf) {
        this.toast('That branch has no reply to switch to yet');
        return;
      }
      this.pendingTarget = node.id;
      const moved = await CT.api.setCurrentLeaf(this.conversationId, leaf.id);
      if (moved.ok) {
        // Only reload once the move is readable, or claude.ai reloads onto the old branch.
        this.toast('Switching branch…', { sticky: true });
        const settled = await CT.api.confirmLeaf(this.conversationId, moved.leafId || leaf.id);
        if (!settled) {
          console.warn('[claude-tree] branch move accepted but not yet readable; ' +
            'reloading anyway — the chat may need a further refresh');
        }
        await reopenAfterReload(this.conversationId, this.pendingTarget);
        return;
      }
      this.toast(`Could not reach that message — ${moved.reason}`);
      console.warn('[claude-tree] branch switch failed:', moved.reason,
        { conversation: this.conversationId, leaf: leaf.id });
    }

    /**
     * The last message of the branch the pane marks as current.
     *
     * `order` is a pre-order walk, so the messages on the path come out in depth order and
     * the last of them is the one the conversation currently ends on.
     */
    currentPathLeaf() {
      const path = this.tree?.order.filter((node) => node.onPath);
      return path && path.length ? path[path.length - 1] : null;
    }

    /** Resolves once a tree has loaded, so callers after a reload do not race it. */
    whenLoaded(timeout = 8000) {
      if (this.status === 'ready') return Promise.resolve(true);
      return new Promise((resolve) => {
        const done = (value) => { clearTimeout(timer); resolve(value); };
        const timer = setTimeout(() => done(false), timeout);
        this.readyWaiters.push(() => done(true));
      });
    }

    /**
     * Scroll the chat to a message that should now be on the branch it is showing.
     *
     * Tried more than once: right after a branch switch the transcript is still mounting
     * rows, and a message that is not in the page yet is indistinguishable from one that
     * cannot be found at all. Giving up on the first attempt is what left the reader at the
     * top of the conversation after switching.
     */
    async landOn(id, { attempts = 3, select = true } = {}) {
      const node = this.tree?.nodes.get(id);
      if (!node) return false;
      const depth = Math.max(1, (this.tree?.stats.depth ?? 1) - 1);

      for (let attempt = 0; attempt < attempts; attempt += 1) {
        const outcome = await CT.chat.revealNode(node, {
          onPath: CT.chat.onDisplayedBranch(node),
          position: node.depth / depth,
          allowSwitch: false, // landing scrolls; it never changes which branch is showing
        });
        if (outcome === 'found') {
          if (select) this.select(id, { center: true });
          return true;
        }
        if (outcome === 'off-path') return false; // a switch is needed, not another look
        await new Promise((resolve) => setTimeout(resolve, 220));
      }
      return false;
    }

    handleNodeKey(event) {
      const current = event.target.closest?.('.ct-node');
      if (!current) return;
      const node = this.tree?.nodes.get(current.dataset.id);
      if (!node) return;

      const vertical = this.orientation === 'vertical';
      const toParent = vertical ? 'ArrowUp' : 'ArrowLeft';
      const toChild = vertical ? 'ArrowDown' : 'ArrowRight';
      const toPrev = vertical ? 'ArrowLeft' : 'ArrowUp';
      const toNext = vertical ? 'ArrowRight' : 'ArrowDown';

      let target = null;
      if (event.key === toParent) target = node.parent;
      else if (event.key === toChild) target = node.children[0];
      else if (event.key === toPrev || event.key === toNext) {
        const siblings = node.parent ? node.parent.children : this.tree.roots;
        target = siblings[node.siblingIndex + (event.key === toNext ? 1 : -1)];
      }
      if (!target) return;

      event.preventDefault();
      this.select(target.id, { center: true });
      this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(target.id)}"]`)?.focus();
    }

    /* ------------------------------------------------------------ open / close -- */

    async toggle() {
      await this.setOpen(!this.open);
    }

    async setOpen(open) {
      await this.ensureMounted();
      if (this.open === open) return;
      this.open = open;
      clearTimeout(this.exitTimer);
      if (open) {
        discoverGlyphs();
        this.centreOnCurrent = true; // land on whatever the chat is showing
        this.host.hidden = false;
        // A frame with the pane rendered but not yet marked open, so the transition runs.
        requestAnimationFrame(() => { this.host.dataset.open = 'true'; });
      } else {
        this.host.dataset.open = 'false';
        this.exitTimer = setTimeout(() => { this.host.hidden = true; }, 220);
      }
      this.onOpenChange?.(open, this.width);
      if (open) {
        if (this.dirty) this.load();
        else this.applyView();
      }
    }

    setConversation(conversationId) {
      if (conversationId === this.conversationId) return;
      this.conversationId = conversationId;
      this.tree = null;
      this.selectedId = null;
      this.dirty = true;
      this.status = 'idle';
      if (this.mounted) this.renderAll();
      if (this.open) this.load();
    }

    /** Everything the pane remembers between visits. */
    persist() {
      savePrefs({
        orientation: this.orientation,
        width: this.width,
        detailHeight: this.detailHeight,
        sticky: this.sticky,
      });
    }

    setSticky(on) {
      this.sticky = Boolean(on);
      this.el?.root?.querySelector('[data-action="sticky"]')
        ?.setAttribute('aria-pressed', String(this.sticky));
      this.persist();
      this.lastFollowTop = null;             // the next update starts a fresh direction
      if (this.sticky) this.followCurrent([...this.currentIds || []]);
      else this.stopGlide();
    }

    setOrientation(orientation) {
      if (orientation !== 'vertical' && orientation !== 'horizontal') return;
      if (this.orientation === orientation) return;
      this.orientation = orientation;
      this.persist();
      this.renderAll();
      this.resetView();
      this.ensureVisible(this.selectedId);
    }

    /* ------------------------------------------------------------------- data --- */

    async load({ force = false, quiet = false } = {}) {
      await this.ensureMounted();
      if (!this.conversationId) {
        this.status = 'idle';
        this.dirty = true;
        this.renderAll();
        return;
      }
      if (this.status === 'loading') return;
      if (!force && !quiet && this.status === 'ready' && !this.dirty) return;

      const requestedFor = this.conversationId;
      if (!quiet) {
        this.status = 'loading';
        this.renderAll();
      }
      this.el.root.querySelector('[data-action="refresh"]')?.classList.add('is-busy');

      try {
        const raw = await CT.api.fetchConversation(requestedFor);
        if (requestedFor !== this.conversationId) return; // navigated away mid-flight
        const tree = CT.model.buildTree(raw);
        const previousSelection = this.selectedId;
        const isFirstRender = !this.tree || this.dirty;
        this.tree = tree;
        this.status = 'ready';
        this.error = null;
        this.loadedAt = Date.now();
        this.dirty = false;
        this.readyWaiters.splice(0).forEach((resolve) => resolve());
        if (this.shapeReportedFor !== this.conversationId) {
          this.shapeReportedFor = this.conversationId;
          void this.reportTreeShape(tree);
        }
        this.selectedId = tree.nodes.has(previousSelection) ? previousSelection : null;
        this.currentIds = new Set([...this.currentIds].filter((id) => tree.nodes.has(id)));
        this.currentKey = [...this.currentIds].join(',');
        this.renderAll();
        if (isFirstRender) this.resetView();
      } catch (err) {
        if (requestedFor !== this.conversationId) return;
        this.status = 'error';
        this.error = err?.message || 'Something went wrong loading this conversation.';
        this.renderAll();
      } finally {
        this.el.root.querySelector('[data-action="refresh"]')?.classList.remove('is-busy');
      }
    }

    /**
     * Note in the console what came back, and flag the one failure the pane cannot show:
     * a conversation the page is branching but the API returned flat.
     */
    async reportTreeShape(tree) {
      const via = CT.api.describeLoad();
      if (tree.stats.forks === 0 && await CT.chat.hasVisibleSwitcher()) {
        console.warn(
          `[claude-tree] claude.ai returned ${tree.stats.messages} messages with no branches ` +
          `via "${via}", but the page is showing a variant switcher. The conversation API is ` +
          'only sending the branch on screen, so the pane cannot offer the others.'
        );
        this.toast('claude.ai returned only the visible branch');
        return;
      }
      console.info(
        `[claude-tree] ${tree.stats.messages} messages, ${tree.stats.forks} forks via "${via}"`
      );
    }

    /* ----------------------------------------------------------------- render --- */

    renderAll() {
      if (!this.mounted) return;
      this.renderChrome();
      this.renderStats();
      this.renderTree();
      this.renderDetail();
      this.renderState();
      this.onStatsChange?.(this.tree?.stats ?? null);
      this.onTreeChange?.(this.tree);
    }

    renderChrome() {
      // The conversation's name is already at the top of the chat — no title here.
      // The zoom and direction controls do nothing without a tree to point them at.
      this.el.toolbar.hidden = !this.tree?.order.length;
      for (const button of this.el.root.querySelectorAll('[data-action="orient"]')) {
        button.setAttribute('aria-pressed', String(button.dataset.value === this.orientation));
      }
      this.el.root.querySelector('[data-action="sticky"]')
        ?.setAttribute('aria-pressed', String(this.sticky));
    }

    renderStats() {
      const stats = SHOW_STATS ? this.tree?.stats : null;
      if (!stats) {
        this.el.stats.innerHTML = '';
        this.el.stats.hidden = true;
        return;
      }
      this.el.stats.hidden = false;
      const plural = (n, one, many) => (n === 1 ? one : many);
      this.el.stats.innerHTML = [
        `<span class="ct-stat"><b>${stats.messages}</b> ${plural(stats.messages, 'message', 'messages')}</span>`,
        `<span class="ct-stat${stats.branches > 1 ? ' is-accent' : ''}"><b>${stats.branches}</b> ${plural(stats.branches, 'branch', 'branches')}</span>`,
        `<span class="ct-stat"><b>${stats.forks}</b> ${plural(stats.forks, 'fork', 'forks')}</span>`,
        `<span class="ct-stat"><b>${stats.pathLength}</b> on active path</span>`,
      ].join('');
    }

    renderTree() {
      if (!this.tree || !this.tree.roots.length) {
        this.el.nodes.innerHTML = '';
        this.el.edges.innerHTML = '';
        this.content = { width: 0, height: 0 };
        return;
      }

      const vertical = this.orientation === 'vertical';
      const conf = LAYOUT[this.orientation];
      const nodeW = NODE_WIDTH[this.orientation];

      // Pass one: put the cards in the document at their real width with the height left
      // to the content, so each one can be measured rather than assumed.
      this.el.nodes.innerHTML = this.tree.order
        .map((node) => this.nodeHtml(node, nodeW))
        .join('');
      const cards = new Map();
      for (const el of this.el.nodes.children) cards.set(el.dataset.id, el);
      const heightOf = (node) => cards.get(node.id)?.offsetHeight || 40;

      // Pass two: lay the tree out with those measurements.
      const extent = CT.model.computeLayout(this.tree.roots, {
        spreadOf: vertical ? () => nodeW : heightOf,
        depthOf: vertical ? heightOf : () => nodeW,
        spreadGap: conf.spreadGap,
        depthGap: conf.depthGap,
      });

      const width = vertical ? extent.spread : extent.depth;
      const height = vertical ? extent.depth : extent.spread;
      this.content = { width, height };

      const box = (node) => vertical
        ? { x: node.s, y: node.d, w: node.sSize, h: node.dSize }
        : { x: node.d, y: node.s, w: node.dSize, h: node.sSize };

      // Pass three: place them, and draw the edges between the boxes we just placed.
      const edges = [];
      const forks = [];
      for (const node of this.tree.order) {
        const at = box(node);
        const el = cards.get(node.id);
        if (el) {
          el.style.left = `${at.x}px`;
          el.style.top = `${at.y}px`;
        }

        if (node.children.length > 1) {
          const badge = vertical
            ? { x: at.x + at.w / 2, y: at.y + at.h + 7 }
            : { x: at.x + at.w + 7, y: at.y + at.h / 2 };
          forks.push(
            `<span class="ct-fork" style="left:${badge.x}px;top:${badge.y}px;transform:translate(-50%,-50%)" ` +
            `title="${node.children.length} alternative replies">${node.children.length}</span>`
          );
        }

        for (const child of node.children) {
          const to = box(child);
          const onPath = node.onPath && child.onPath;
          const d = vertical
            ? this.curve(at.x + at.w / 2, at.y + at.h, to.x + to.w / 2, to.y, true)
            : this.curve(at.x + at.w, at.y + at.h / 2, to.x, to.y + to.h / 2, false);
          edges.push(`<path class="ct-edge${onPath ? ' is-path' : ''}" d="${d}"/>`);
        }
      }

      this.el.edges.setAttribute('width', String(width));
      this.el.edges.setAttribute('height', String(height));
      this.el.edges.innerHTML = edges.join('');
      this.el.nodes.style.width = `${width}px`;
      this.el.nodes.style.height = `${height}px`;
      this.el.nodes.insertAdjacentHTML('beforeend', forks.join(''));
    }

    nodeHtml(node, nodeW) {
      const isPath = node.onPath;
      const isSelected = node.id === this.selectedId;
      const isCurrent = this.currentIds.has(node.id);
      const role = node.sender === 'human' ? 'You' : 'Claude';
      const variant = node.siblingCount > 1
        ? `<span class="ct-node-variant">${node.siblingIndex + 1}/${node.siblingCount}</span>`
        : '';
      const tag = node.tools.length
        ? `<span class="ct-node-tag" title="${esc(node.tools.join(', '))}">${esc(node.tools[0])}</span>`
        : node.attachments
          ? `<span class="ct-node-tag">${node.attachments} file${node.attachments === 1 ? '' : 's'}</span>`
          : '';
      const preview = node.preview
        ? `<span class="ct-node-text">${esc(CT.model.snippet(node.preview, 320))}</span>`
        : '<span class="ct-node-text ct-node-empty">Empty message</span>';

      return `<button type="button" class="ct-node${isPath ? ' is-path' : ''}${isCurrent ? ' is-current' : ''}${isSelected ? ' is-selected' : ''}"
        data-id="${esc(node.id)}" data-sender="${node.sender}"
        style="width:${nodeW}px"
        aria-pressed="${isSelected}">
        <span class="ct-node-meta"><span class="ct-node-who">${role}</span>${tag}${variant}</span>
        ${preview}
      </button>`;
    }

    /** Cubic bezier between two anchors, bending along the depth axis. */
    curve(x1, y1, x2, y2, vertical) {
      if (vertical) {
        const mid = y1 + (y2 - y1) / 2;
        return `M${x1} ${y1} C${x1} ${mid} ${x2} ${mid} ${x2} ${y2}`;
      }
      const mid = x1 + (x2 - x1) / 2;
      return `M${x1} ${y1} C${mid} ${y1} ${mid} ${y2} ${x2} ${y2}`;
    }

    renderState() {
      const state = this.el.state;
      const show = (html) => { state.innerHTML = html; state.hidden = false; };

      if (this.status === 'loading') {
        show('<div class="ct-spinner"></div><p>Reading the conversation…</p>');
        return;
      }
      if (this.status === 'error') {
        show(`${ICON.warning}<h2>Could not load the tree</h2><p>${esc(this.error)}</p>
              <button class="ct-btn is-primary" data-action="retry">${icon('refresh')}Try again</button>`);
        return;
      }
      if (!this.conversationId) {
        show(`${ICON.chat}<h2>No conversation open</h2>
              <p>Open a chat on claude.ai and the panel will map it, including every branch.</p>`);
        return;
      }
      if (this.status === 'ready' && !this.tree?.order.length) {
        show(`${ICON.chat}<h2>Nothing to show yet</h2><p>This conversation has no messages.</p>`);
        return;
      }
      state.hidden = true;
      state.innerHTML = '';
    }

    renderDetail() {
      const detail = this.el.detail;
      const node = this.selectedId ? this.tree?.nodes.get(this.selectedId) : null;
      if (!node) {
        detail.hidden = true;
        detail.innerHTML = '';
        return;
      }

      const siblings = node.parent ? node.parent.children : this.tree.roots;
      const nav = node.siblingCount > 1
        ? `<div class="ct-variant-nav">
             <button data-detail="prev" ${node.siblingIndex === 0 ? 'disabled' : ''} title="Previous variant" aria-label="Previous variant">${ICON.left}</button>
             <span>${node.siblingIndex + 1} / ${node.siblingCount}</span>
             <button data-detail="next" ${node.siblingIndex === siblings.length - 1 ? 'disabled' : ''} title="Next variant" aria-label="Next variant">${ICON.right}</button>
           </div>`
        : '<span style="margin-left:auto"></span>';

      const notes = [];
      if (node.tools.length) notes.push(`Tools used: ${node.tools.join(', ')}`);
      if (node.attachments) notes.push(`${node.attachments} attachment${node.attachments === 1 ? '' : 's'}`);
      // The label says what the button will actually do for this message — judged by what
      // the chat is showing, which is not always what the conversation record says.
      const shown = CT.chat.onDisplayedBranch(node);
      const gotoLabel = shown ? 'Show in chat' : 'Show this branch in the chat';

      detail.innerHTML = `
        <div class="ct-detail-resize" role="separator" aria-orientation="horizontal" title="Drag to resize"></div>
        <div class="ct-detail-head">
          <span class="ct-detail-who" data-sender="${node.sender}">${node.sender === 'human' ? 'You' : 'Claude'}</span>
          <span class="ct-detail-time">${esc(formatTime(node.createdAt))}</span>
          ${nav}
          <div class="ct-detail-actions">
            <button class="ct-btn ct-goto" data-detail="goto" title="${gotoLabel}">${ICON.branch}${gotoLabel}</button>
            <button class="ct-icon-btn" data-detail="copy" title="Copy message text" aria-label="Copy message text">${icon('copy')}</button>
            <button class="ct-icon-btn" data-detail="close" title="Close details" aria-label="Close details">${icon('close')}</button>
          </div>
        </div>
        <div class="ct-detail-body">
          ${notes.length ? `<p class="ct-detail-note">${esc(notes.join(' · '))}</p>` : ''}
          <p class="ct-detail-text">${esc(node.text || node.preview || 'Empty message')}</p>
          ${node.thinking ? `<details class="ct-thinking"><summary>Extended thinking</summary><pre>${esc(node.thinking)}</pre></details>` : ''}
        </div>
      `;
      detail.hidden = false;
    }

    /**
     * Outline every message currently on screen in the chat; selection still wins.
     * @param {string[]} ids message ids, in the order they appear
     */
    setCurrent(ids) {
      const visible = Array.isArray(ids) ? ids : (ids ? [ids] : []);
      const key = visible.join(',');
      if (this.currentKey === key) return;
      this.currentKey = key;
      this.currentIds = new Set(visible);
      if (!this.mounted) return;

      for (const el of this.el.nodes.querySelectorAll('.ct-node.is-current')) {
        el.classList.remove('is-current');
      }
      for (const id of visible) {
        this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(id)}"]`)?.classList.add('is-current');
      }

      // On opening, bring the top of what the chat is showing into the middle of the pane.
      if (this.centreOnCurrent && visible.length) {
        this.centreOnCurrent = false;
        this.centerOn(visible[0]);
        this.lastFollowTop = null;
        return;
      }
      this.followCurrent(visible);
    }

    select(id, { center = false } = {}) {
      if (this.selectedId === id) {
        if (id) this.renderDetail();
        return;
      }
      const previous = this.el.nodes.querySelector('.ct-node.is-selected');
      previous?.classList.remove('is-selected');
      previous?.setAttribute('aria-pressed', 'false');

      this.selectedId = id;
      if (id) {
        const next = this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(id)}"]`);
        next?.classList.add('is-selected');
        next?.setAttribute('aria-pressed', 'true');
        if (center) this.centerOn(id);
      }
      this.renderDetail();
    }

    /* ------------------------------------------------------------- pan / zoom --- */

    applyView({ isDefault = false } = {}) {
      this.viewIsDefault = isDefault;
      const { x, y, scale } = this.view;
      this.el.viewport.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
      this.el.zoom.textContent = `${Math.round(scale * 100)}%`;
    }

    /**
     * Frame the tree at 1:1.
     *
     * Zooming out to fit would shrink a long conversation until nothing could be read, so
     * the view starts at full size, centred on the tree if it fits and on the first message
     * if it does not. Panning and the zoom controls take it from there.
     */
    resetView() {
      if (!this.content.width || !this.content.height) {
        this.view = { x: 0, y: 0, scale: 1 };
        this.applyView({ isDefault: true });
        return;
      }
      const vertical = this.orientation === 'vertical';
      const box = this.el.canvas.getBoundingClientRect();
      const padding = 28;

      const spread = vertical ? this.content.width : this.content.height;
      const spreadBox = vertical ? box.width : box.height;
      const depth = vertical ? this.content.height : this.content.width;
      const depthBox = vertical ? box.height : box.width;

      const root = this.tree?.roots[0];
      const rootCentre = root ? root.s + root.sSize / 2 : spread / 2;
      const alongSpread = spread <= spreadBox - padding * 2
        ? (spreadBox - spread) / 2
        : spreadBox / 2 - rootCentre;
      const alongDepth = depth <= depthBox - padding * 2
        ? (depthBox - depth) / 2
        : padding;

      this.view = vertical
        ? { scale: 1, x: alongSpread, y: alongDepth }
        : { scale: 1, x: alongDepth, y: alongSpread };
      this.applyView({ isDefault: true });
    }

    /** Zoom out far enough to see the whole tree at once. */
    fitAll() {
      if (!this.content.width || !this.content.height) return;
      const box = this.el.canvas.getBoundingClientRect();
      const padding = 24;
      const scale = clamp(
        Math.min((box.width - padding * 2) / this.content.width,
                 (box.height - padding * 2) / this.content.height, 1),
        MIN_SCALE,
        1
      );
      this.view = {
        scale,
        x: (box.width - this.content.width * scale) / 2,
        y: (box.height - this.content.height * scale) / 2,
      };
      this.applyView();
    }

    zoomBy(factor, origin) {
      const box = this.el.canvas.getBoundingClientRect();
      const px = origin ? origin.x : box.width / 2;
      const py = origin ? origin.y : box.height / 2;
      const next = clamp(this.view.scale * factor, MIN_SCALE, MAX_SCALE);
      const ratio = next / this.view.scale;
      this.view = {
        scale: next,
        x: px - (px - this.view.x) * ratio,
        y: py - (py - this.view.y) * ratio,
      };
      this.applyView();
    }

    /** Pan by the smallest amount that brings a node fully into the canvas. */
    ensureVisible(id) {
      const el = id && this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(id)}"]`);
      if (!el) return;
      const box = this.el.canvas.getBoundingClientRect();
      const { scale, x, y } = this.view;
      const padding = 16;
      const left = parseFloat(el.style.left) * scale + x;
      const top = parseFloat(el.style.top) * scale + y;
      const width = el.offsetWidth * scale;
      const height = el.offsetHeight * scale;

      let dx = 0;
      let dy = 0;
      if (left < padding) dx = padding - left;
      else if (left + width > box.width - padding) dx = box.width - padding - (left + width);
      if (top < padding) dy = padding - top;
      else if (top + height > box.height - padding) dy = box.height - padding - (top + height);

      if (!dx && !dy) return;
      this.view.x += dx;
      this.view.y += dy;
      this.applyView();
    }

    /**
     * Keep the messages the chat is showing in view as it scrolls.
     *
     * Sideways, the canvas centres on the message leading the way rather than on all of them
     * together. Averaging them only works while they sit in one column: across a fork the
     * messages on screen can span further than the canvas is wide, and centring on the middle
     * of that span puts *both* ends off the edge — which is how scrolling to the top of a
     * conversation could leave the first message off the canvas with the later ones showing.
     * Consecutive messages on a branch share a column, so this still reads as the column
     * being centred.
     *
     * Up and down, the canvas only moves when the message leading the way is off it — the
     * lowest one when scrolling down, the highest when scrolling up. It is then brought just
     * inside whichever edge it fell outside, and otherwise nothing moves: a reader working
     * within one screenful should not have the tree drifting under them.
     *
     * Both edges are checked, not just the one being scrolled towards. Checking only that
     * edge leaves the leading message stranded whenever it is off the *other* one, which is
     * what happened at the top of a conversation — scrolling up to the first message left
     * the canvas showing the messages below it, because the first message was past the
     * bottom edge and the upward rule only ever looked at the top.
     */
    followCurrent(ids) {
      if (!this.sticky || !this.mounted || !ids?.length) return;
      if (this.el.canvas.classList.contains('is-panning')) return; // the reader has the wheel

      let highest = null; // the message nearest the start of the conversation
      let lowest = null;  // ...and the one nearest its end
      for (const id of ids) {
        const el = this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(id)}"]`);
        if (!el) continue;
        const x = parseFloat(el.style.left);
        const y = parseFloat(el.style.top);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const span = {
          left: x, right: x + el.offsetWidth, top: y, bottom: y + el.offsetHeight,
        };
        if (!highest || span.top < highest.top) highest = span;
        if (!lowest || span.bottom > lowest.bottom) lowest = span;
      }
      if (!highest || !lowest) return;

      const { width, height } = this.canvasSpace();
      const { scale } = this.view;

      // Up and down: nudge only when the message leading the way is off the canvas.
      // With no previous reading there is no direction yet, and guessing "down" anchors the
      // lowest message and pushes the earliest one off the top — the opposite of what someone
      // who has just arrived at the start of a conversation wants. Lead with the earliest.
      const goingDown = this.lastFollowTop !== null && this.lastFollowTop !== undefined
        && highest.top >= this.lastFollowTop;
      this.lastFollowTop = highest.top;

      const lead = goingDown ? lowest : highest;

      // Sideways: centred on that message.
      const x = width / 2 - ((lead.left + lead.right) / 2) * scale;

      /*
       * Measured against where the canvas is going, not where it happens to be this frame.
       * A glide takes a few hundred milliseconds, and scrolling produces updates faster than
       * that, so reading the live position asks "is the message on screen?" of somewhere the
       * canvas is only passing through. Answering yes about a halfway position stops the
       * follow early — and a quick scroll to the start of a conversation then finishes with
       * the first message still off the canvas.
       */
      let y = (this.glideTarget ?? this.view).y;
      const leadTop = lead.top * scale + y;
      const leadBottom = lead.bottom * scale + y;
      if (leadBottom > height - FOLLOW_PAD) y -= leadBottom - (height - FOLLOW_PAD);
      else if (leadTop < FOLLOW_PAD) y += FOLLOW_PAD - leadTop;

      this.glideTo(x, y);
    }

    /**
     * The part of the canvas a message can actually be seen in.
     *
     * The drawer is drawn over the bottom of the canvas, so anything moved into the strip
     * behind it is not on screen at all — which is what put a message being scrolled to
     * underneath the open drawer.
     */
    canvasSpace() {
      const box = this.el.canvas.getBoundingClientRect();
      const drawer = this.el.detail;
      const covered = drawer && !drawer.hidden
        ? Math.max(0, box.bottom - drawer.getBoundingClientRect().top)
        : 0;
      return { width: box.width, height: Math.max(80, box.height - covered) };
    }

    /** Ease the canvas to a position, rather than snapping it there. */
    glideTo(x, y) {
      this.stopGlide();
      this.glideTarget = { x, y };
      const fromX = this.view.x;
      const fromY = this.view.y;
      const dx = x - fromX;
      const dy = y - fromY;
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1) { this.glideTarget = null; return; }

      // A reader who has asked for less motion gets the move without the animation.
      const still = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
      if (still) {
        this.view.x = x;
        this.view.y = y;
        this.applyView({ isDefault: this.viewIsDefault });
        this.glideTarget = null;
        return;
      }

      const started = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - started) / GLIDE_MS);
        const eased = 1 - (1 - t) ** 3;
        this.view.x = fromX + dx * eased;
        this.view.y = fromY + dy * eased;
        this.applyView({ isDefault: this.viewIsDefault });
        if (t < 1) {
          this.glideFrame = requestAnimationFrame(step);
        } else {
          this.glideFrame = 0;
          this.glideTarget = null;
        }
      };
      this.glideFrame = requestAnimationFrame(step);
    }

    /** Drop any glide in progress, so it cannot fight what happens next. */
    stopGlide() {
      if (this.glideFrame) cancelAnimationFrame(this.glideFrame);
      this.glideFrame = 0;
      this.glideTarget = null;
    }

    centerOn(id) {
      this.stopGlide();
      const el = this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(id)}"]`);
      if (!el) return;
      const box = this.el.canvas.getBoundingClientRect();
      const { scale } = this.view;
      const cx = (parseFloat(el.style.left) + el.offsetWidth / 2) * scale;
      const cy = (parseFloat(el.style.top) + el.offsetHeight / 2) * scale;
      this.view.x = box.width / 2 - cx;
      this.view.y = box.height / 2 - cy;
      this.applyView();
    }

    handleWheel(event) {
      this.stopGlide();
      event.preventDefault();
      const box = this.el.canvas.getBoundingClientRect();
      // Trackpad pinch and ctrl/⌘ + wheel zoom; a plain wheel pans, like a canvas editor.
      if (event.ctrlKey || event.metaKey) {
        // A trackpad pinch arrives as many small deltas, a mouse wheel as few large ones;
        // the clamp keeps the pinch responsive without a wheel notch jumping the view.
        const factor = clamp(Math.exp(-event.deltaY * 0.01), 0.8, 1.25);
        this.zoomBy(factor, { x: event.clientX - box.left, y: event.clientY - box.top });
        return;
      }
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? box.height : 1;
      this.view.x -= event.deltaX * unit;
      this.view.y -= event.deltaY * unit;
      this.applyView();
    }

    handlePanStart(event) {
      if (event.button !== 0) return;
      this.stopGlide();
      const start = { x: event.clientX, y: event.clientY, vx: this.view.x, vy: this.view.y };
      let moved = false;
      this.suppressClick = false;

      const move = (moveEvent) => {
        const dx = moveEvent.clientX - start.x;
        const dy = moveEvent.clientY - start.y;
        if (!moved && Math.hypot(dx, dy) < 4) return;
        if (!moved) {
          moved = true;
          this.el.canvas.classList.add('is-panning');
          this.el.canvas.setPointerCapture(moveEvent.pointerId);
        }
        this.view.x = start.vx + dx;
        this.view.y = start.vy + dy;
        this.applyView();
      };

      const up = (upEvent) => {
        this.el.canvas.removeEventListener('pointermove', move);
        this.el.canvas.removeEventListener('pointerup', up);
        this.el.canvas.removeEventListener('pointercancel', up);
        this.el.canvas.classList.remove('is-panning');
        if (this.el.canvas.hasPointerCapture?.(upEvent.pointerId)) {
          this.el.canvas.releasePointerCapture(upEvent.pointerId);
        }
        if (moved) {
          this.suppressClick = true;
          setTimeout(() => { this.suppressClick = false; }, 0);
        }
      };

      this.el.canvas.addEventListener('pointermove', move);
      this.el.canvas.addEventListener('pointerup', up);
      this.el.canvas.addEventListener('pointercancel', up);
    }

    /** Pin the drawer to a height; both properties move so the cap cannot block the drag. */
    applyDetailHeight(height) {
      const value = `${Math.round(height)}px`;
      this.host.style.setProperty('--ct-detail-h', value);
      this.host.style.setProperty('--ct-detail-max', value);
    }

    /** Drag the drawer's top edge to resize it. */
    handleDetailResize(event) {
      if (event.button !== 0) return;
      event.preventDefault();
      const handle = event.target.closest('.ct-detail-resize');
      const startY = event.clientY;
      const startHeight = this.el.detail.getBoundingClientRect().height;
      const limit = this.el.root.getBoundingClientRect().height * MAX_DETAIL_RATIO;
      handle.classList.add('is-active');
      handle.setPointerCapture(event.pointerId);

      const move = (moveEvent) => {
        this.detailHeight = clamp(startHeight + (startY - moveEvent.clientY), MIN_DETAIL, limit);
        this.applyDetailHeight(this.detailHeight);
        this.ensureVisible(this.selectedId);
      };
      const up = () => {
        handle.removeEventListener('pointermove', move);
        handle.removeEventListener('pointerup', up);
        handle.classList.remove('is-active');
        this.persist();
      };
      handle.addEventListener('pointermove', move);
      handle.addEventListener('pointerup', up);
    }

    handleResizeStart(event) {
      if (event.button !== 0) return;
      event.preventDefault();
      const startX = event.clientX;
      const startWidth = this.width;
      this.el.resize.classList.add('is-active');
      this.el.resize.setPointerCapture(event.pointerId);

      const move = (moveEvent) => {
        this.width = clamp(startWidth + (startX - moveEvent.clientX), MIN_WIDTH, this.maxWidth());
        this.host.style.setProperty('--ct-w', `${this.width}px`);
        this.onOpenChange?.(this.open, this.width);
      };
      const up = () => {
        this.el.resize.removeEventListener('pointermove', move);
        this.el.resize.removeEventListener('pointerup', up);
        this.el.resize.classList.remove('is-active');
        this.persist();
      };
      this.el.resize.addEventListener('pointermove', move);
      this.el.resize.addEventListener('pointerup', up);
    }

    /* ------------------------------------------------------------------ toast --- */

    toast(message, { sticky = false } = {}) {
      const el = this.el.toast;
      // The message is written either way, so turning notifications back on is the one
      // flag and nothing else; it is simply never revealed while they are off.
      el.textContent = message;
      if (!SHOW_TOASTS) return;
      el.classList.add('is-visible');
      clearTimeout(this.toastTimer);
      if (sticky) return;
      this.toastTimer = setTimeout(() => el.classList.remove('is-visible'), 2600);
    }
  }

  CT.ui = { TreePanel, createHost, ICON };
})();
