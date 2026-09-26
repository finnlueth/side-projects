// MARK: - Set Site Attribute

const websiteObject = getSiteInfo();
const hackerNewsOptionsWithVerticalLinesList = [
  "socialFocus_hacker-news_navigation_hide_new",
  "socialFocus_hacker-news_navigation_hide_past",
  "socialFocus_hacker-news_navigation_hide_comments",
  "socialFocus_hacker-news_navigation_hide_ask",
  "socialFocus_hacker-news_navigation_hide_show",
  "socialFocus_hacker-news_navigation_hide_jobs",
  "socialFocus_hacker-news_post_hide_hide_button",
  "socialFocus_hacker-news_navigation_hide_submit",
];

if (websiteObject) {
  document
    .querySelector("html")
    .setAttribute("socialFocus_site_is", websiteObject.name);
}

// MARK: - Set Device Attribute

// First set depend on user agent

if (isUserAgentMobile()) {
  document.documentElement.setAttribute("socialFocus_device_type", "mobile");
} else {
  document.documentElement.setAttribute("socialFocus_device_type", "desktop");
}

// Start observer by desktop and mobile tags

var detectInterval = setInterval(function () {
  // Check mobile selectors

  for (const selector of websiteObject.mobileSelectorCheck) {
    if (selector != "") {
      const checkSelect = document.querySelector(selector);
      if (checkSelect) {
        document.documentElement.setAttribute(
          "socialFocus_device_type",
          "mobile"
        );
        clearInterval(detectInterval);
        break;
      }
    }
  }

  // Check desktop selectors

  for (const selector of websiteObject.desktopSelectorCheck) {
    if (selector != "") {
      const checkSelect = document.querySelector(selector);
      if (checkSelect) {
        document.documentElement.setAttribute(
          "socialFocus_device_type",
          "desktop"
        );
        clearInterval(detectInterval);
        break;
      }
    }
  }
}, 10);

setTimeout(function () {
  clearInterval(detectInterval);
}, 5000);

const deviceTypeAttribute = document.documentElement.getAttribute(
  "socialFocus_device_type"
);

// MARK: - Set options values to HTML

function setAttributesToSettingsDiv() {
  for (const option of getAllOptions(getWebsiteCategoriesFromWebsite())) {
    const optionId = option.id;
    const optionDefault = option.defaultValue;
    browser.storage.local.get(optionId, function (obj) {
      const value = obj[optionId] ?? optionDefault;

      // Set in HTML (global enable is free-time aware)

      if (optionId === getConstNotSyncing.extensionIsEnabledData) {
        applyGlobalEnableAttribute(value);
      } else {
        document.documentElement.setAttribute(optionId, value);
      }

      if (hackerNewsOptionsWithVerticalLinesList.includes(optionId)) {
        setTimeout(() => {
          formatHackerNewsNavigationElements(optionId, value);
        });
      }

      if (
        optionId ===
          `socialFocus_${websiteObject.name}_daily_limit_show_timer_draggable` &&
        deviceTypeAttribute === "desktop"
      ) {
        if (value) {
          createTimerModal({ isDesktop: true, isInit: true });
        } else {
          removeTimerModal();
        }
      }

      if (
        optionId ===
          `socialFocus_${websiteObject.name}_daily_limit_show_timer` &&
        deviceTypeAttribute === "mobile"
      ) {
        if (value) {
          createTimerModal({ isDesktop: false, isInit: true });
        } else {
          removeTimerModal();
        }
      }

      if (option.id === "socialFocus_reddit_header_hide_trending_today") {
        handleRedditTrendingToday(true, value, deviceTypeAttribute);
      }
    });
  }
}

setAttributesToSettingsDiv();

// MARK: - Update in html

function updateSettingAttribute(settingId, attribute) {
  // Set in html (global enable is free-time aware)

  if (settingId === getConstNotSyncing.extensionIsEnabledData) {
    applyGlobalEnableAttribute(attribute);
  } else {
    document.documentElement.setAttribute(settingId, attribute);
  }

  // Special for facebook feed posts

  if (
    settingId == "socialFocus_facebook_feed_hide_sponsored_posts" ||
    settingId == "socialFocus_facebook_feed_hide_group_posts" ||
    settingId == "socialFocus_facebook_feed_hide_section_people_you_may_know" ||
    settingId ==
      "socialFocus_facebook_feed_hide_section_reels_and_short_videos" ||
    settingId == "socialFocus_facebook_feed_hide_suggested_groups" ||
    settingId == "socialFocus_facebook_feed_hide_posts_from_people_to_follow"
  ) {
    triggerFilterFacebookPosts();
  }

  if (hackerNewsOptionsWithVerticalLinesList.includes(settingId)) {
    formatHackerNewsNavigationElements(settingId, attribute);
  }

  if (
    settingId ==
      `socialFocus_${websiteObject.name}_daily_limit_show_timer_draggable` &&
    deviceTypeAttribute === "desktop"
  ) {
    if (attribute) {
      createTimerModal({ isDesktop: true, isInit: false });
    } else {
      removeTimerModal();
    }
  }

  if (
    settingId == `socialFocus_${websiteObject.name}_daily_limit_show_timer` &&
    deviceTypeAttribute === "mobile"
  ) {
    if (attribute) {
      createTimerModal({ isDesktop: false, isInit: false });
    } else {
      removeTimerModal();
    }
  }

  if (settingId == `socialFocus_${websiteObject.name}_master_toggle`) {
    evaluateDailyLimitState();
  }

  if (settingId === "socialFocus_reddit_header_hide_trending_today") {
    handleRedditTrendingToday(false, attribute, deviceTypeAttribute);
  }

  if (settingId === "socialFocus_youtube_hide_shorts") {
    hideMenuTabs();
  }
}

// MARK: - Receive Requests from popup

browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Toggle

  if (message.type == "toggle") {
    const objectId = message.id;

    browser.storage.local.get(objectId, function (obj) {
      const defaultObject = findOptionById(
        objectId,
        getWebsiteCategoriesFromWebsite()
      );

      if (!defaultObject) {
        return;
      }

      const currentValue = obj[objectId] ?? defaultObject.defaultValue;
      updateSettingAttribute(objectId, currentValue);
    });
  } else if (message.type == "checkSelectors") {
    sendResponse({ isDesktop: !isMobileVersion() });
  }
});
