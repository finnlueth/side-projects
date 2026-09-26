// MARK: - Daily Limit
//
// Time is only counted, and the site only blocked, while the user is on one of the site's
// feed pages (feedPaths in storageConstants.js). Sites are SPAs, so the URL is watched and
// the state re-evaluated on every navigation.

let intervalId = null;
let siteBlocked = false;
let previousHtmlOverflow = null;
let lastCheckedHref = location.href;

document.addEventListener("DOMContentLoaded", () => {
  if (websiteObject) {
    evaluateDailyLimitState();
    startLocationWatcher();
  }
});

function stopTimer() {
  clearInterval(intervalId);
  intervalId = null;
}

function isSiteBlocked() {
  return siteBlocked;
}

// MARK: - State

function getDailyLimitState(callback) {
  const websiteName = websiteObject.name;
  const durationKey = getConst.dailyLimitDuration[websiteName];
  const lastedTimeKey = getConst.dailyLimitLastedTime[websiteName];
  const currentDateKey = getConst.dailyCurrentDate[websiteName];
  const masterToggleKey = `socialFocus_${websiteName}_master_toggle`;

  browser.storage.local.get(
    [durationKey, lastedTimeKey, currentDateKey, masterToggleKey],
    function (obj) {
      const duration = obj[durationKey];
      const masterToggle = obj[masterToggleKey] ?? false;
      const dateNow = formatDate();

      let lasted = Number(obj[lastedTimeKey] ?? 0);

      // New day: reset the time already spent
      if (obj[currentDateKey] !== dateNow) {
        lasted = 0;

        browser.storage.local.set({
          [currentDateKey]: dateNow,
          [lastedTimeKey]: 0,
        });
      }

      const hasLimit =
        duration !== undefined &&
        duration !== "noLimit" &&
        !masterToggle &&
        !isFreeTimeNow();

      callback({
        duration,
        lasted,
        hasLimit,
        timeIsUp: hasLimit && lasted >= getSecondsFromMinutes(duration),
      });
    }
  );
}

// Decide for the current page and moment: count, block, or leave the site alone
function evaluateDailyLimitState() {
  if (!websiteObject) {
    return;
  }

  getDailyLimitState(function (state) {
    const onFeedPage = isOnFeedPage();

    if (state.hasLimit && onFeedPage && state.timeIsUp) {
      stopTimer();
      showBlockingOverlay();
    } else {
      hideBlockingOverlay();

      if (
        state.hasLimit &&
        onFeedPage &&
        document.visibilityState === "visible"
      ) {
        startSiteBlockingTimer();
      } else {
        stopTimer();
      }
    }

    refreshTimerModalForPage();
  });
}

// MARK: - Counting

function startSiteBlockingTimer() {
  if (intervalId !== null) {
    return;
  }

  const lastedTimeKey = getConst.dailyLimitLastedTime[websiteObject.name];

  intervalId = setInterval(() => {
    if (!isOnFeedPage() || document.visibilityState !== "visible") {
      return;
    }

    getDailyLimitState(function (state) {
      if (!state.hasLimit) {
        stopTimer();
        refreshTimerModalForPage();
        return;
      }

      const lasted = state.lasted + 1;

      browser.storage.local.set({ [lastedTimeKey]: lasted });

      if (lasted >= getSecondsFromMinutes(state.duration)) {
        stopTimer();
        showBlockingOverlay();
        refreshTimerModalForPage();
      }
    });
  }, 1000);
}

// MARK: - Blocking Overlay
//
// Instead of wiping the page, an opaque overlay is placed on top, so leaving the feed page
// through SPA navigation (back button, search, ...) brings the site straight back.

function showBlockingOverlay() {
  siteBlocked = true;

  if (document.getElementById("socialFocusBlockOverlay")) {
    return;
  }

  whenBodyReady(() => {
    if (!siteBlocked || document.getElementById("socialFocusBlockOverlay")) {
      return;
    }

    (async () => {
      const content = await getBlockingContent();

      if (!siteBlocked || document.getElementById("socialFocusBlockOverlay")) {
        return;
      }

      const overlay = document.createElement("div");
      overlay.id = "socialFocusBlockOverlay";

      const shadowRoot = overlay.attachShadow({ mode: "open" });
      shadowRoot.innerHTML = `
        <div style="position: fixed; inset: 0; z-index: 2147483647; overflow: hidden; background: ${
          isPageDark() ? "#0f0f0f" : "#ffffff"
        };">
          ${content}
        </div>
      `;

      document.documentElement.appendChild(overlay);

      previousHtmlOverflow = document.documentElement.style.overflow;
      document.documentElement.style.setProperty("overflow", "hidden", "important");
    })();
  });
}

function hideBlockingOverlay() {
  siteBlocked = false;

  const overlay = document.getElementById("socialFocusBlockOverlay");

  if (overlay) {
    overlay.remove();

    if (previousHtmlOverflow) {
      document.documentElement.style.overflow = previousHtmlOverflow;
    } else {
      document.documentElement.style.removeProperty("overflow");
    }

    previousHtmlOverflow = null;
  }
}

// Kept for the popup, which sends these after the daily limit select changes
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (
    message.action === "startBlockingSite" ||
    message.action === "startBlockingSiteTimer"
  ) {
    evaluateDailyLimitState();
  }
});

// Limit changed from the popup (in any tab)
browser.storage.onChanged.addListener((changes, area) => {
  if (
    area === "local" &&
    websiteObject &&
    changes[getConst.dailyLimitDuration[websiteObject.name]]
  ) {
    evaluateDailyLimitState();
  }
});

// MARK: - Location Watcher (SPA navigation)

function startLocationWatcher() {
  const checkLocation = () => {
    if (location.href !== lastCheckedHref) {
      lastCheckedHref = location.href;
      evaluateDailyLimitState();
    }
  };

  window.addEventListener("popstate", checkLocation);
  window.addEventListener("hashchange", checkLocation);
  setInterval(checkLocation, 500);
}

function formatDate() {
  const date = new Date();

  const day = String(date.getDate()).padStart(2, "0");
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const year = date.getFullYear();

  return `${day}.${month}.${year}`;
}

function handleVisibilityChange() {
  if (document.visibilityState === "visible") {
    evaluateDailyLimitState();
  } else {
    stopTimer();
  }
}

document.addEventListener("visibilitychange", handleVisibilityChange);
