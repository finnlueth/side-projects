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
    host.hidden = !conversationIdFromLocation();
    describeToggle(null);
    placeToggle();
    syncToggle(panel.open, panel.width);
  }

  /**
   * Claude's own header markup. Stable enough to prefer, not stable enough to rely on —
   * `inferredHeaderSlot` covers it changing.
   */
  const HEADER = {
    group: '[data-testid="wiggle-controls-actions-group"]',
    share: '[data-testid="wiggle-controls-actions-share"]',
    icon: '[data-testid="wiggle-controls-actions-toggle"], [data-cds-icon-only]',
  };

  /** Walk up from `el` until its parent is `ancestor`, so we insert at the right level. */
  function childOf(ancestor, el) {
    let node = el;
    while (node && node.parentElement !== ancestor) node = node.parentElement;
    return node;
  }

  function metricsFor(icon, share) {
    const source = icon || share;
    const rect = source?.getBoundingClientRect();
    if (!rect?.height) return { size: 34, radius: '8px' };
    return {
      size: Math.min(48, Math.max(24, Math.round(rect.height))),
      radius: icon ? getComputedStyle(icon).borderRadius : '8px',
    };
  }

  /** Accessible name of a control, however claude.ai happens to label it. */
  function controlName(el) {
    return (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '')
      .trim()
      .toLowerCase();
  }

  /**
   * Fallback for when Claude's testids change: find the button belonging immediately before
   * Share by accessible name and by sitting near the top of the window, never by class name.
   * Gives up cleanly — `placeToggle` then falls back to a floating button.
   */
  function findHeaderSlot() {
    return knownHeaderSlot() || inferredHeaderSlot();
  }

  function knownHeaderSlot() {
    const group = document.querySelector(HEADER.group);
    const share = group?.querySelector(HEADER.share);
    const before = share && childOf(group, share);
    if (!before) return null;
    return { parent: group, before, ...metricsFor(group.querySelector(HEADER.icon), share) };
  }

  function inferredHeaderSlot() {
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

        const neighbour = iconButtonMetrics(node.parentElement, el);
        return {
          parent: node.parentElement,
          before: node,
          size: neighbour?.size ?? Math.min(44, Math.max(28, Math.round(rect.height))),
          radius: neighbour?.radius ?? '8px',
        };
      }
    }
    return null;
  }

  /**
   * Measure a square icon button already in the header row, so the tree icon takes exactly
   * the same box. Share itself is a wide pill, so its width is no guide.
   */
  function iconButtonMetrics(row, share) {
    for (const child of row.children) {
      if (child.classList.contains('ct-toggle-host') || child.contains(share)) continue;
      const control = child.matches('button, [role="button"]')
        ? child
        : child.querySelector('button, [role="button"]');
      if (!control) continue;

      const rect = control.getBoundingClientRect();
      if (!rect.width || !rect.height) continue;
      const ratio = rect.width / rect.height;
      if (ratio < 0.8 || ratio > 1.3) continue; // not an icon button

      return {
        size: Math.min(48, Math.max(24, Math.round(rect.height))),
        radius: getComputedStyle(control).borderRadius,
      };
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
    toggle.host.style.setProperty('--ct-btn-radius', slot.radius);
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

  panel.onOpenChange = (open, width) => {
    syncToggle(open, width);
    if (open) startWatching();
    else stopWatching();
  };
  panel.onStatsChange = describeToggle;

  /* ------------------------------------------------------------ chat watching -- */

  let reading = null;
  let stopUpdates = null;

  function startWatching() {
    if (!reading) {
      reading = CT.chat.trackReading((ids) => panel.setCurrent(ids));
      panel.onTreeChange = (tree) => {
        reading?.setNodes(tree ? tree.order.filter((node) => node.onPath) : []);
      };
      panel.onTreeChange(panel.tree);
    }
    if (!stopUpdates) {
      // Refresh in place: no spinner, no lost view, no lost selection.
      stopUpdates = CT.chat.watchForUpdates(() => {
        if (panel.open && !document.hidden) panel.load({ quiet: true });
      });
    }
  }

  function stopWatching() {
    reading?.stop();
    reading = null;
    panel.onTreeChange = null;
    stopUpdates?.();
    stopUpdates = null;
  }

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

  /* ---------------------------------------------------------------- overlays --- */

  /**
   * claude.ai's search and dialogs open in portals with a z-index of their own. The pane
   * sits above ordinary page content, which would put it over them, so it steps down for
   * as long as one is on screen.
   */
  const DIALOG_SELECTOR = '[role="dialog"], [aria-modal="true"], [role="search"]';
  /** Layers at or above this are Claude's overlays, not its page furniture. */
  const OVERLAY_Z = 500;

  const isVisible = (el) => (el.checkVisibility ? el.checkVisibility() : el.offsetParent !== null);
  const ours = (el) => el.closest('.ct-panel-host, .ct-toggle-host');

  /** The stacking level of the nearest positioned ancestor that declares one. */
  function layerZ(el) {
    for (let node = el; node && node !== document.body; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.position === 'static') continue;
      const z = Number(style.zIndex);
      if (Number.isFinite(z)) return z;
    }
    return 0;
  }

  /**
   * How high claude.ai is currently painting, if it is painting above the pane.
   *
   * Dialogs are found by role, but Claude's in-chat find bar is not a dialog, so anything
   * holding keyboard focus inside a very high layer counts too — which is what an open
   * search field always does. Returns 0 when nothing is above us.
   */
  function overlayLevel() {
    let level = 0;

    for (const el of document.querySelectorAll(DIALOG_SELECTOR)) {
      if (ours(el) || !isVisible(el)) continue;
      level = Math.max(level, layerZ(el) || OVERLAY_Z);
    }

    const focused = document.activeElement;
    if (focused && !ours(focused)) {
      const z = layerZ(focused);
      if (z >= OVERLAY_Z) level = Math.max(level, z);
    }
    return level;
  }

  function watchOverlays() {
    // Sit just under whatever is on top rather than dropping to the back, so the pane does
    // not vanish behind the chat while a small find bar is open.
    const check = () => {
      const level = overlayLevel();
      panel.setBehind(level ? Math.max(1, level - 1) : 0);
    };

    let frame = 0;
    const schedule = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => { frame = 0; check(); });
    };
    new MutationObserver(schedule).observe(document.body, {
      childList: true,
      subtree: true,
      attributeFilter: ['role', 'aria-modal', 'hidden', 'data-state'],
    });
    // An overlay that only takes focus (Claude's find bar) shows up here and nowhere else.
    document.addEventListener('focusin', schedule, true);
    document.addEventListener('focusout', schedule, true);
    document.addEventListener('keydown', schedule, true);

    check();
    return check;
  }

  const checkOverlays = watchOverlays();

  /* -------------------------------------------------------------- navigation --- */

  function conversationIdFromLocation() {
    const match = CHAT_PATH.exec(location.pathname);
    return match ? match[1] : null;
  }

  let lastHref = '';
  function checkLocation() {
    if (location.href === lastHref) return;
    lastHref = location.href;

    const conversation = conversationIdFromLocation();
    panel.setConversation(conversation);

    // Starting a new chat leaves nothing to map: close the pane and take the button away
    // until there is a conversation again.
    if (!conversation) panel.setOpen(false);
    if (toggle) toggle.host.hidden = !conversation;
  }

  // claude.ai is a client-side-routed app, so there is no navigation event to hook;
  // a cheap poll alongside popstate keeps the panel and the docked button in sync.
  window.addEventListener('popstate', checkLocation);
  setInterval(() => {
    checkLocation();
    ensureTogglePlaced();
    checkOverlays();
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

  /** A branch switch reloads the page; bring the pane straight back when it does. */
  async function resumeAfterBranchSwitch() {
    let stored;
    try {
      stored = await ext.storage.local.get('resume');
      if (stored?.resume) await ext.storage.local.remove('resume');
    } catch {
      return;
    }
    const resume = stored?.resume;
    if (!resume || Date.now() - resume.at > 30000) return;
    if (resume.conversation !== conversationIdFromLocation()) return;
    panel.setOpen(true);
  }

  checkLocation();
  mountToggle()
    .then(resumeAfterBranchSwitch)
    .catch((err) => console.error('[claude-tree] could not mount the toggle:', err));
})();
