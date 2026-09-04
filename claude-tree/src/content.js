/**
 * Bootstrap: docks the toggle into Claude's header, reserves page width for the pane, and
 * follows client-side navigation.
 */
(() => {
  'use strict';

  const CT = globalThis.CT;
  const ext = globalThis.browser ?? globalThis.chrome;

  if (window.__claudeTreeLoaded) return;
  window.__claudeTreeLoaded = true;

  const CHAT_PATH = /^\/chat\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i;
  /** Gap between the pane and the window edge — mirrors --ct-pane-gap in panel.css. */
  const PANE_GAP = 8;
  const PANE_STYLE_ID = 'claude-tree-pane-style';
  const FAB_GAP = 18;

  const panel = new CT.ui.TreePanel();
  CT.panel = panel;

  /* ------------------------------------------------------------------ toggle ---- */

  let toggle = null;
  let slotObserver = null;

  async function mountToggle() {
    const { host, shadow } = await CT.ui.createHost('ct-toggle-host', { parent: null });
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ct-toggle';
    button.setAttribute('aria-pressed', 'false');
    button.innerHTML =
      `${CT.ui.ICON.tree}<span class="ct-toggle-label">Tree</span>` +
      '<span class="ct-toggle-count" hidden></span><span class="ct-toggle-dot" hidden></span>';
    button.addEventListener('click', () => panel.toggle());
    shadow.append(button);

    toggle = { host, button, count: button.querySelector('.ct-toggle-count'), dot: button.querySelector('.ct-toggle-dot') };
    describeToggle(null);
    placeToggle();
    syncToggle(panel.open, panel.width);
  }

  /** Accessible name of a control, however claude.ai happens to label it. */
  function controlName(el) {
    return (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '')
      .trim()
      .toLowerCase();
  }

  /**
   * Find where in claude.ai's header the button belongs: immediately before Share.
   *
   * Nothing about that header is a stable API, so this matches on the accessible name and
   * on position near the top of the window rather than on class names, and gives up
   * cleanly — `placeToggle` then falls back to a floating button.
   */
  function findHeaderSlot() {
    const scopes = [
      document.querySelector('header'),
      document.querySelector('[role="banner"]'),
      document.body,
    ];

    for (const scope of scopes) {
      if (!scope) continue;
      for (const el of scope.querySelectorAll('button, a[role="button"], [role="button"]')) {
        if (el.closest('.ct-toggle-host')) continue;
        const name = controlName(el);
        if (name !== 'share' && !name.startsWith('share ')) continue;

        const rect = el.getBoundingClientRect();
        if (!rect.width || rect.top > 160) continue; // must be a visible control in the top bar

        // Step outside any single-purpose wrapper around the Share button (a tooltip trigger,
        // say) so we land in the header's action row rather than inside Share's own box.
        // Capped at two hops: climbing further risks leaving the action row altogether.
        let node = el;
        for (let hop = 0; hop < 2; hop++) {
          const parent = node.parentElement;
          if (!parent || parent === scope || parent.children.length !== 1) break;
          node = parent;
        }
        if (!node.parentElement) continue;

        return {
          parent: node.parentElement,
          before: node,
          size: Math.min(44, Math.max(28, Math.round(rect.height))),
        };
      }
    }
    return null;
  }

  function placeToggle() {
    if (!toggle) return;
    const slot = findHeaderSlot();

    if (!slot) {
      // No header to dock into (yet): float instead, and keep looking on each tick.
      slotObserver?.disconnect();
      slotObserver = null;
      toggle.host.classList.remove('ct-header-host');
      toggle.host.classList.add('ct-fab-host');
      if (toggle.host.parentElement !== document.documentElement) {
        document.documentElement.appendChild(toggle.host);
      }
      return;
    }

    toggle.host.classList.remove('ct-fab-host');
    toggle.host.classList.add('ct-header-host');
    toggle.host.style.setProperty('--ct-btn-size', `${slot.size}px`);
    if (toggle.host.parentElement !== slot.parent || toggle.host.nextElementSibling !== slot.before) {
      slot.parent.insertBefore(toggle.host, slot.before);
    }

    // Claude re-renders its header; re-insert immediately rather than waiting for the poll.
    if (slotObserver?.target !== slot.parent) {
      slotObserver?.disconnect();
      slotObserver = new MutationObserver(() => {
        if (!toggle.host.isConnected) placeToggle();
      });
      slotObserver.observe(slot.parent, { childList: true });
      slotObserver.target = slot.parent;
    }
  }

  /** Cheap per-tick check: only re-search the DOM when we are not already docked. */
  function ensureTogglePlaced() {
    if (!toggle) return;
    const docked = toggle.host.isConnected && toggle.host.classList.contains('ct-header-host');
    if (!docked) placeToggle();
  }

  function syncToggle(open, width) {
    if (!toggle) return;
    toggle.button.setAttribute('aria-pressed', String(open));
    toggle.host.style.setProperty('--ct-fab-right', `${(open ? width + PANE_GAP * 2 : 0) + FAB_GAP}px`);
    setPane(open, width);
  }

  function describeToggle(stats) {
    if (!toggle) return;
    const branches = stats?.branches ?? 0;
    const many = branches > 1;
    toggle.count.hidden = !many;
    toggle.count.textContent = many ? `${branches}` : '';
    toggle.dot.hidden = !many;
    toggle.button.title = many
      ? `Conversation tree — ${branches} branches (Alt+Shift+T)`
      : 'Conversation tree (Alt+Shift+T)';
    toggle.button.setAttribute('aria-label', toggle.button.title);
  }

  panel.onOpenChange = syncToggle;
  panel.onStatsChange = describeToggle;

  /* -------------------------------------------------------------------- pane ---- */

  /**
   * Reserve width on the page so the pane sits beside the chat instead of over it.
   *
   * Narrowing the body box is enough for a normal-flow app shell. Elements positioned
   * against the viewport ignore it, so if any turn out to reach under the pane, body also
   * gets layout containment, which makes it their containing block. That is only safe when
   * the document itself does not scroll — otherwise contained fixed elements would scroll
   * away with the page — so it is applied on evidence rather than by default.
   */
  function setPane(open, width) {
    const root = document.documentElement;
    let style = document.getElementById(PANE_STYLE_ID);

    if (!open) {
      style?.remove();
      root.classList.remove('claude-tree-paned', 'claude-tree-contained');
      return;
    }

    if (!style) {
      style = document.createElement('style');
      style.id = PANE_STYLE_ID;
      (document.head || root).appendChild(style);
    }
    const reserved = width + PANE_GAP * 2;
    style.textContent = `
      html.claude-tree-paned > body {
        margin-right: ${reserved}px !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: none !important;
      }
      html.claude-tree-contained > body {
        contain: layout !important;
        min-height: 100dvh !important;
      }
    `;
    root.classList.add('claude-tree-paned');
    requestAnimationFrame(() => updateContainment(reserved));
  }

  function updateContainment(reserved) {
    const root = document.documentElement;
    const documentScrolls = root.scrollHeight - window.innerHeight > 4;
    root.classList.toggle(
      'claude-tree-contained',
      !documentScrolls && hasFixedUnderPane(reserved)
    );
  }

  /** Look for a viewport-anchored element that still reaches into the reserved strip. */
  function hasFixedUnderPane(reserved) {
    if (!document.body) return false;
    const limit = window.innerWidth - reserved + 8;
    const queue = Array.from(document.body.children, (el) => [el, 0]);

    while (queue.length) {
      const [el, depth] = queue.shift();
      if (!(el instanceof HTMLElement) || el.classList.contains('ct-toggle-host')) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width > window.innerWidth * 0.5 && rect.right > limit) {
        if (getComputedStyle(el).position === 'fixed') return true;
      }
      if (depth < 3) for (const child of el.children) queue.push([child, depth + 1]);
    }
    return false;
  }

  /* -------------------------------------------------------------- navigation --- */

  function conversationIdFromLocation() {
    const match = CHAT_PATH.exec(location.pathname);
    return match ? match[1] : null;
  }

  let lastHref = '';
  function checkLocation() {
    if (location.href === lastHref) return;
    lastHref = location.href;
    panel.setConversation(conversationIdFromLocation());
  }

  // claude.ai is a client-side-routed app, so there is no navigation event to hook;
  // a cheap poll alongside popstate keeps the panel and the docked button in sync.
  window.addEventListener('popstate', checkLocation);
  setInterval(() => {
    checkLocation();
    ensureTogglePlaced();
  }, 500);

  window.addEventListener('resize', () => {
    if (panel.open) setPane(true, panel.width);
  });

  /* ----------------------------------------------------------------- messages -- */

  ext?.runtime?.onMessage?.addListener((message) => {
    if (message?.type !== 'ct:toggle') return;
    panel.toggle();
    return Promise.resolve({ ok: true });
  });

  /* --------------------------------------------------------------------- go ---- */

  checkLocation();
  mountToggle().catch((err) => console.error('[claude-tree] could not mount the toggle:', err));
})();
