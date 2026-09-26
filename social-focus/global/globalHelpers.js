// MARK: - Specify Browser

var browser = browser || chrome;

// MARK: - Special variable

const channelPageUrlPart1 = "youtube.com/@";
const channelPageUrlPart2 = "youtube.com/channel";

// MARK: - System Design Methods

function queryById(name) {
  return document.getElementById(name);
}

function querySelector(selector) {
  return document.querySelector(selector);
}

function querySelectorAll(selector) {
  return document.querySelectorAll(selector);
}

function isUserAgentMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

function getSecondsFromMinutes(seconds) {
  const SECONDS_IN_MINUTE = 60;

  return Number(seconds * SECONDS_IN_MINUTE);
}

// MARK: - Free Time helpers (shared by popup and content scripts)

// Is `nowMinutes` inside [start, end)? Windows may wrap past midnight (e.g. 22:00 - 02:00).
function isWithinTimeWindow(nowMinutes, startMinutes, endMinutes) {
  if (startMinutes === endMinutes) {
    return false;
  }

  if (startMinutes < endMinutes) {
    return nowMinutes >= startMinutes && nowMinutes < endMinutes;
  }

  return nowMinutes >= startMinutes || nowMinutes < endMinutes;
}

function getFreeTimeSettings(callback) {
  browser.storage.local.get(
    [
      getConst.freeTimeIsActiveData,
      getConst.freeTimeStartData,
      getConst.freeTimeEndData,
    ],
    function (obj) {
      callback({
        isActive:
          obj[getConst.freeTimeIsActiveData] ?? FREE_TIME_DEFAULTS.isActive,
        start: Number(
          obj[getConst.freeTimeStartData] ?? FREE_TIME_DEFAULTS.start
        ),
        end: Number(obj[getConst.freeTimeEndData] ?? FREE_TIME_DEFAULTS.end),
      });
    }
  );
}

function isFreeTimeForSettings(settings) {
  if (!settings || !settings.isActive) {
    return false;
  }

  const now = new Date();

  return isWithinTimeWindow(
    now.getHours() * 60 + now.getMinutes(),
    settings.start,
    settings.end
  );
}

function formatMinutesAsTime(minutes) {
  const date = new Date();
  date.setHours(Math.floor(minutes / 60), minutes % 60, 0, 0);

  return date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}
