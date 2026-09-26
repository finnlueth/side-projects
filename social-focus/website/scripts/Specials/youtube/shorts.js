function hideMenuTabs() {
  const shortsButtonId = "socialFocus_youtube_hide_shorts";

  browser.storage.local.get([shortsButtonId], function (obj) {
    const isHideShortsMenuButton = obj[shortsButtonId] ?? false;

    var interval = setInterval(function () {
      const tabs = document.querySelectorAll(
        "ytd-browse[page-subtype='channels'] yt-tab-group-shape yt-tab-shape"
      );
      const tabsMobile = document.querySelectorAll(
        "ytm-browse:has(yt-decorated-avatar-view-model) yt-tab-group-shape yt-tab-shape"
      );

      const actualTabs = !tabs.length ? tabsMobile : tabs;

      if (actualTabs.length) {
        clearInterval(interval);

        for (const tab of actualTabs) {
          const inTab = tab.querySelector(".yt-tab-shape-wiz__tab");

          if (inTab) {
            const preparedTitle = inTab.innerHTML.trim().toUpperCase();
            if (shortsButtonTitles.includes(preparedTitle)) {
              tab.style.display = isHideShortsMenuButton
                ? "none"
                : "inline-flex";
            }
          }
        }
      }
    }, 100);
  });
}

function runContentChangesObserver() {
  new MutationObserver(() => {
    const url = location.href;

    if (
      url.includes(channelPageUrlPart1) ||
      url.includes(channelPageUrlPart2)
    ) {
      hideMenuTabs();
    }
  }).observe(document.body, { subtree: true, childList: true });
}

window.onload = function () {
  browser.storage.local.get(
    getConstNotSyncing.extensionIsEnabledData,
    function (obj) {
      const isEnabled = obj[getConstNotSyncing.extensionIsEnabledData] ?? true;

      if (isEnabled) {
        runContentChangesObserver();
      }
    }
  );
};

const shortsButtonTitles = [
  "KORTVIDEO'S",
  "SHORTS",
  "CURTS",
  "LÜHIVIDEOD",
  "COURTES VIDÉOS",
  "SHORTS VIDEOZAPISI",
  "AMA-SHORT",
  "VIDEO FUPI",
  "SHORTS-VIDEOER",
  "КЫСКА ВИДЕОЛОР",
  "YOUTUBE SHORTS",
  "SHORTS",
  "ԿԱՐՃ ՀՈԼՈՎԱԿՆԵՐ",
  "فیلم‌های کوتاه YOUTUBE",
  "ᲛᲝᲙᲚᲔ ᲕᲘᲓᲔᲝᲔᲑᲘ",
  "ショート",
  "SHORTS",
];
