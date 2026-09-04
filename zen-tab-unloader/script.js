(function () {
  "use strict";

  const PREF_MAX_TABS = "mod.zen-tab-unloader.max_loaded_tabs";
  const DEFAULT_MAX_TABS = 5;

  function getMaxTabs() {
    try {
      const raw = Services.prefs.getStringPref(PREF_MAX_TABS, String(DEFAULT_MAX_TABS));
      return Math.max(1, parseInt(raw, 10) || DEFAULT_MAX_TABS);
    } catch (_) {
      return DEFAULT_MAX_TABS;
    }
  }

  // Index 0 = most recently used, last index = LRU discard candidate
  let lruQueue = [];

  function removeFromQueue(tab) {
    const idx = lruQueue.indexOf(tab);
    if (idx !== -1) lruQueue.splice(idx, 1);
  }

  function promoteTab(tab) {
    removeFromQueue(tab);
    lruQueue.unshift(tab);
  }

  function isDiscardable(tab) {
    if (!tab) return false;
    if (tab.pinned) return false;
    if (tab.hidden) return false;
    if (tab === gBrowser.selectedTab) return false;
    if (tab.getAttribute("pending") === "true") return false;
    if (!tab.linkedBrowser) return false;
    return true;
  }

  function enforceLimit() {
    const maxTabs = getMaxTabs();
    const loaded = lruQueue.filter(
      (t) => t && !t.pinned && t.getAttribute("pending") !== "true"
    ).length;

    if (loaded <= maxTabs) return;

    let toDiscard = loaded - maxTabs;
    for (let i = lruQueue.length - 1; i >= 0 && toDiscard > 0; i--) {
      const candidate = lruQueue[i];
      if (isDiscardable(candidate)) {
        try {
          gBrowser.discardBrowser(candidate.linkedBrowser, false);
          toDiscard--;
        } catch (_) {
          // tab may be in a transitional state; skip silently
        }
      }
    }
  }

  function initQueue() {
    const selected = gBrowser.selectedTab;
    lruQueue = Array.from(gBrowser.tabs).filter((t) => t !== selected);
    if (selected) lruQueue.unshift(selected);
    enforceLimit();
  }

  function onTabSelect(event) {
    promoteTab(event.target);
    enforceLimit();
  }

  function onTabClose(event) {
    removeFromQueue(event.target);
  }

  function onTabOpen(event) {
    promoteTab(event.target);
    enforceLimit();
  }

  function init() {
    initQueue();
    gBrowser.tabContainer.addEventListener("TabSelect", onTabSelect);
    gBrowser.tabContainer.addEventListener("TabClose", onTabClose);
    gBrowser.tabContainer.addEventListener("TabOpen", onTabOpen);
  }

  if (gBrowser) {
    init();
  } else {
    window.addEventListener("load", init, { once: true });
  }
})();
