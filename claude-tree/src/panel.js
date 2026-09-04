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
  const DEFAULT_WIDTH = 470;
  /** Page width to leave for the chat itself when the pane is docked. */
  const MIN_PAGE_WIDTH = 380;
  const MIN_SCALE = 0.15;
  /** Never auto-zoom below this — smaller than that, node text stops being legible. */
  const READABLE_SCALE = 0.45;
  const MAX_SCALE = 2;

  /** Box + gap sizes per orientation, in the layout's (spread, depth) space. */
  const LAYOUT = {
    vertical: { spread: 180, spreadGap: 22, depth: 72, depthGap: 32 },
    horizontal: { spread: 70, spreadGap: 16, depth: 212, depthGap: 58 },
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
    chat: glyph('<path d="M20 14.5a2.5 2.5 0 0 1-2.5 2.5H8l-4 3.5V6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5z"/>'),
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
      this.currentId = null;
      this.loadedAt = 0;
      this.dirty = true;
      this.view = { x: 0, y: 0, scale: 1 };
      this.content = { width: 0, height: 0 };
      this.onOpenChange = null;
      this.onStatsChange = null;
      this.onTreeChange = null;
      this.toastTimer = 0;
      this.exitTimer = 0;
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

      discoverGlyphs();
      const { host, shadow } = await createHost('ct-panel-host', { hidden: true });
      this.host = host;
      this.shadow = shadow;
      host.style.setProperty('--ct-w', `${this.width}px`);

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
              <button class="ct-icon-btn" data-action="fit" title="Reset view" aria-label="Reset view">${ICON.fit}</button>
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
          <div class="ct-toast" data-role="toast" role="status" aria-live="polite"></div>
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
        const node = event.target.closest('.ct-node');
        if (node) this.select(node.dataset.id, { center: false });
      });

      nodes.addEventListener('keydown', (event) => this.handleNodeKey(event));

      detail.addEventListener('click', (event) => {
        const button = event.target.closest('[data-detail]');
        if (button) this.handleDetailAction(button.dataset.detail);
      });

      canvas.addEventListener('wheel', (event) => this.handleWheel(event), { passive: false });
      canvas.addEventListener('pointerdown', (event) => this.handlePanStart(event));
      canvas.addEventListener('dblclick', () => this.resetView());

      resize.addEventListener('pointerdown', (event) => this.handleResizeStart(event));

      root.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        if (this.selectedId) this.select(null);
        else this.setOpen(false);
        event.stopPropagation();
      });

      // Re-fit if the panel is resized while a tree is on screen.
      if (typeof ResizeObserver === 'function') {
        let firstObservation = true;
        new ResizeObserver(() => {
          if (firstObservation) { firstObservation = false; return; }
          if (this.status !== 'ready' || !this.open) return;
          // Opening the detail drawer shrinks the canvas — keep the selection in view.
          this.ensureVisible(this.selectedId);
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
        case 'fit': this.resetView(); break;
        case 'orient': this.setOrientation(button.dataset.value); break;
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
      if (action === 'locate') {
        this.locate(node);
        return;
      }
      if (action === 'prev' || action === 'next') {
        const siblings = node.parent ? node.parent.children : this.tree.roots;
        const target = siblings[node.siblingIndex + (action === 'next' ? 1 : -1)];
        if (target) this.select(target.id, { center: true });
      }
    }

    /**
     * Scroll the chat to a message. Claude renders lazily, so a message on the visible
     * branch may still need hunting for — say so rather than blaming the branch.
     */
    async locate(node) {
      const depth = Math.max(1, (this.tree?.stats.depth ?? 1) - 1);
      this.toast('Looking for the message…', { sticky: true });
      const outcome = await CT.chat.revealNode(node, {
        onPath: node.onPath,
        position: node.depth / depth,
      });
      this.toast({
        found: 'Scrolled to the message',
        'off-path': 'That message is on a different branch than the chat is showing',
        'not-found': 'Could not find that message in the page',
      }[outcome]);
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

    setOrientation(orientation) {
      if (orientation !== 'vertical' && orientation !== 'horizontal') return;
      if (this.orientation === orientation) return;
      this.orientation = orientation;
      savePrefs({ orientation, width: this.width });
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
        this.selectedId = tree.nodes.has(previousSelection) ? previousSelection : null;
        if (!tree.nodes.has(this.currentId)) this.currentId = null;
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
    }

    renderStats() {
      const stats = this.tree?.stats;
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
      const extent = CT.model.computeLayout(this.tree.roots, {
        spreadSize: conf.spread,
        spreadGap: conf.spreadGap,
        depthSize: conf.depth,
        depthGap: conf.depthGap,
      });

      const nodeW = vertical ? conf.spread : conf.depth;
      const nodeH = vertical ? conf.depth : conf.spread;
      const width = vertical ? extent.spread : extent.depth;
      const height = vertical ? extent.depth : extent.spread;
      this.content = { width, height };

      const at = (node) => (vertical ? { x: node.s, y: node.d } : { x: node.d, y: node.s });

      const edges = [];
      const forks = [];
      const cards = [];

      for (const node of this.tree.order) {
        const pos = at(node);
        cards.push(this.nodeHtml(node, pos, nodeW, nodeH));

        if (node.children.length > 1) {
          const badge = vertical
            ? { x: pos.x + nodeW / 2, y: pos.y + nodeH + 7 }
            : { x: pos.x + nodeW + 7, y: pos.y + nodeH / 2 };
          forks.push(
            `<span class="ct-fork" style="left:${badge.x}px;top:${badge.y}px;transform:translate(-50%,-50%)" ` +
            `title="${node.children.length} alternative replies">${node.children.length}</span>`
          );
        }

        for (const child of node.children) {
          const to = at(child);
          const onPath = node.onPath && child.onPath;
          const d = vertical
            ? this.curve(pos.x + nodeW / 2, pos.y + nodeH, to.x + nodeW / 2, to.y, true)
            : this.curve(pos.x + nodeW, pos.y + nodeH / 2, to.x, to.y + nodeH / 2, false);
          edges.push(`<path class="ct-edge${onPath ? ' is-path' : ''}" d="${d}"/>`);
        }
      }

      this.el.edges.setAttribute('width', String(width));
      this.el.edges.setAttribute('height', String(height));
      this.el.edges.innerHTML = edges.join('');
      this.el.nodes.style.width = `${width}px`;
      this.el.nodes.style.height = `${height}px`;
      this.el.nodes.innerHTML = cards.join('') + forks.join('');
    }

    nodeHtml(node, pos, nodeW, nodeH) {
      const isPath = node.onPath;
      const isSelected = node.id === this.selectedId;
      const isCurrent = node.id === this.currentId;
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
        ? `<span class="ct-node-text">${esc(CT.model.snippet(node.preview, 120))}</span>`
        : '<span class="ct-node-text ct-node-empty">Empty message</span>';

      return `<button type="button" class="ct-node${isPath ? ' is-path' : ''}${isCurrent ? ' is-current' : ''}${isSelected ? ' is-selected' : ''}"
        data-id="${esc(node.id)}" data-sender="${node.sender}"
        style="left:${pos.x}px;top:${pos.y}px;width:${nodeW}px;height:${nodeH}px"
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
      if (!node.onPath) notes.push('This message is not on the branch currently shown in the chat.');

      detail.innerHTML = `
        <div class="ct-detail-head">
          <span class="ct-detail-who">${node.sender === 'human' ? 'You' : 'Claude'}</span>
          <span class="ct-detail-time">${esc(formatTime(node.createdAt))}</span>
          ${nav}
          <div class="ct-detail-actions">
            <button class="ct-icon-btn" data-detail="locate" title="Find this message in the chat" aria-label="Find this message in the chat">${icon('search')}</button>
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

    /** Outline the message the chat is currently showing; selection still wins. */
    setCurrent(id) {
      if (this.currentId === id) return;
      this.currentId = id;
      if (!this.mounted) return;
      for (const el of this.el.nodes.querySelectorAll('.ct-node.is-current')) {
        el.classList.remove('is-current');
      }
      if (!id) return;
      this.el.nodes.querySelector(`.ct-node[data-id="${CSS.escape(id)}"]`)?.classList.add('is-current');
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

    applyView() {
      const { x, y, scale } = this.view;
      this.el.viewport.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
      this.el.zoom.textContent = `${Math.round(scale * 100)}%`;
    }

    /**
     * Frame the tree at a readable scale.
     *
     * Only the axis the tree spreads along is fitted to the panel; the axis it grows along
     * is left to panning. Fitting both would shrink a long conversation — especially a
     * left-to-right one in a narrow panel — until nothing could be read.
     */
    resetView() {
      if (!this.content.width || !this.content.height) {
        this.view = { x: 0, y: 0, scale: 1 };
        this.applyView();
        return;
      }
      const vertical = this.orientation === 'vertical';
      const box = this.el.canvas.getBoundingClientRect();
      const padding = 28;

      const spread = vertical ? this.content.width : this.content.height;
      const spreadBox = vertical ? box.width : box.height;
      const depth = vertical ? this.content.height : this.content.width;
      const depthBox = vertical ? box.height : box.width;

      const scale = clamp(Math.min(1, (spreadBox - padding * 2) / spread), READABLE_SCALE, 1);
      const alongSpread = (spreadBox - spread * scale) / 2;
      const alongDepth = depth * scale <= depthBox - padding * 2
        ? (depthBox - depth * scale) / 2
        : padding;

      this.view = vertical
        ? { scale, x: alongSpread, y: alongDepth }
        : { scale, x: alongDepth, y: alongSpread };
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

    centerOn(id) {
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
      event.preventDefault();
      const box = this.el.canvas.getBoundingClientRect();
      // Trackpad pinch and ctrl/⌘ + wheel zoom; a plain wheel pans, like a canvas editor.
      if (event.ctrlKey || event.metaKey) {
        const factor = Math.exp(-event.deltaY * 0.0022);
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
        savePrefs({ orientation: this.orientation, width: this.width });
      };
      this.el.resize.addEventListener('pointermove', move);
      this.el.resize.addEventListener('pointerup', up);
    }

    /* ------------------------------------------------------------------ toast --- */

    toast(message, { sticky = false } = {}) {
      const el = this.el.toast;
      el.textContent = message;
      el.classList.add('is-visible');
      clearTimeout(this.toastTimer);
      if (sticky) return;
      this.toastTimer = setTimeout(() => el.classList.remove('is-visible'), 2600);
    }
  }

  CT.ui = { TreePanel, createHost, ICON, icon, discoverGlyphs };
})();
