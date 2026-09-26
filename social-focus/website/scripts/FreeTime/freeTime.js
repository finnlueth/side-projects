// MARK: - Free Time
//
// A daily window (default 7:00 PM - 11:00 PM) in which every SocialFocus feature is paused:
// the site CSS (all gated on html[socialFocus_global_enable="true"]), the daily limit
// countdown and the blocking of feed pages.

let freeTimeSettings = { ...FREE_TIME_DEFAULTS };
let freeTimeLastState = null;

function isFreeTimeNow() {
  return isFreeTimeForSettings(freeTimeSettings);
}

// The global enable attribute is the one switch all site CSS hangs on. During free time it is
// forced to false without touching the stored value, so everything comes back afterwards.
function applyGlobalEnableAttribute(storedValue) {
  const isEnabled = storedValue === true || storedValue === "true";

  document.documentElement.setAttribute(
    getConstNotSyncing.extensionIsEnabledData,
    isEnabled && !isFreeTimeNow()
  );
}

function refreshGlobalEnableAttribute() {
  browser.storage.local.get(
    getConstNotSyncing.extensionIsEnabledData,
    function (obj) {
      applyGlobalEnableAttribute(
        obj[getConstNotSyncing.extensionIsEnabledData] ?? true
      );
    }
  );
}

function checkFreeTimeStateChange() {
  const state = isFreeTimeNow();

  if (state === freeTimeLastState) {
    return;
  }

  freeTimeLastState = state;

  refreshGlobalEnableAttribute();

  // Daily limit scripts are loaded after this file; re-evaluate them if present
  if (typeof evaluateDailyLimitState === "function") {
    evaluateDailyLimitState();
  }
}

function loadFreeTimeSettings() {
  getFreeTimeSettings(function (settings) {
    freeTimeSettings = settings;
    freeTimeLastState = null;
    checkFreeTimeStateChange();
  });
}

loadFreeTimeSettings();

browser.storage.onChanged.addListener((changes, area) => {
  if (
    area === "local" &&
    (changes[getConst.freeTimeIsActiveData] ||
      changes[getConst.freeTimeStartData] ||
      changes[getConst.freeTimeEndData])
  ) {
    loadFreeTimeSettings();
  }
});

setInterval(checkFreeTimeStateChange, 15 * 1000);
