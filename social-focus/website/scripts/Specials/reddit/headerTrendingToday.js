function handleRedditTrendingToday(isInit, checkedValue, deviceTypeAttribute) {
  browser.storage.local.get("socialFocus_reddit_master_toggle", function (obj) {
    const masterToogle = obj["socialFocus_reddit_master_toggle"] ?? false;

    function getTrendingTodayContainer() {
      const redditSearch = document.querySelector(
        `reddit-search-${deviceTypeAttribute === "mobile" ? "small" : "large"}`
      );

      if (!redditSearch || !redditSearch.shadowRoot) {
        return null;
      }

      return [
        redditSearch.shadowRoot.querySelector(
          `div[data-faceplate-tracking-context] div #reddit-trending-searches-partial-container`
        ),
        redditSearch.shadowRoot.querySelector(
          `div[data-faceplate-tracking-context] div.text-neutral-content-weak`
        ),
      ];
    }

    if (isInit) {
      const observer = new MutationObserver(() => {
        const trendingTodayContainer = getTrendingTodayContainer();

        if (trendingTodayContainer && trendingTodayContainer[0]) {
          observer.disconnect();
          hideCheckRedditTrendingToday(
            masterToogle ? false : checkedValue,
            trendingTodayContainer
          );
        }
      });

      observer.observe(document, { childList: true, subtree: true });
    } else {
      const trendingTodayContainer = getTrendingTodayContainer();

      if (trendingTodayContainer && trendingTodayContainer[0]) {
        hideCheckRedditTrendingToday(
          masterToogle ? false : checkedValue,
          trendingTodayContainer
        );
      }
    }
  });
}

function hideCheckRedditTrendingToday(checkedValue, trendingTodayContainer) {
  for (const sectionToHide of trendingTodayContainer) {
    sectionToHide.style.cssText = checkedValue
      ? "display: none !important;"
      : "display: block !important;";
  }
}

browser.storage.onChanged.addListener((changes, area) => {
  if (changes["socialFocus_reddit_master_toggle"]) {
    const { newValue } = changes["socialFocus_reddit_master_toggle"];

    const deviceTypeAttribute = document.documentElement.getAttribute(
      "socialFocus_device_type"
    );

    browser.storage.local.get(
      "socialFocus_reddit_header_hide_trending_today",
      function (obj) {
        const trendingValue =
          obj["socialFocus_reddit_header_hide_trending_today"] ?? false;

        handleRedditTrendingToday(
          false,
          newValue ? false : trendingValue,
          deviceTypeAttribute
        );
      }
    );
  }
});
