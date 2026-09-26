const watchedKeys = [
  "socialFocus_reddit_post_hide_comments",
  "socialFocus_reddit_post_hide_up_down_vote_buttons",
  "socialFocus_reddit_post_hide_votes_count",
  "socialFocus_reddit_post_hide_shares_button",
  "socialFocus_reddit_post_hide_award_button",
  "socialFocus_reddit_master_toggle",
];

const observerConfig = { childList: true, subtree: true };

function getPostHideButtonsContainers() {
  const redditPostContainers = document.querySelectorAll("shreddit-post");

  if (!redditPostContainers || redditPostContainers.length === 0) {
    return [];
  }

  return Array.from(redditPostContainers)
    .map((container) =>
      container.shadowRoot
        ? container.shadowRoot.querySelector(
            `div:has(span[data-post-click-location="vote"])`
          )
        : null
    )
    .filter((postButtonSection) => postButtonSection !== null);
}

function handlePostHideButtons() {
  browser.storage.local.get(watchedKeys, function (obj) {
    const isHideUpDownVote =
      obj["socialFocus_reddit_post_hide_up_down_vote_buttons"] ?? false;
    const isHideComments =
      obj["socialFocus_reddit_post_hide_comments"] ?? false;
    const isHideVotesCount =
      obj["socialFocus_reddit_post_hide_votes_count"] ?? false;
    const isHideSharesButton =
      obj["socialFocus_reddit_post_hide_shares_button"] ?? false;
    const isHideAwardButton =
      obj["socialFocus_reddit_post_hide_award_button"] ?? false;
    const masterToogle = obj["socialFocus_reddit_master_toggle"] ?? false;

    const postButtonSections = getPostHideButtonsContainers();

    postButtonSections.forEach((postButtonSection) => {
      if (postButtonSection) {
        hideCheckPostButtons(
          postButtonSection,
          masterToogle ? false : isHideUpDownVote,
          "upDownVote"
        );
        hideCheckPostButtons(
          postButtonSection,
          masterToogle ? false : isHideComments,
          "comments"
        );
        hideCheckPostButtons(
          postButtonSection,
          masterToogle ? false : isHideVotesCount,
          "voteCounts"
        );
        hideCheckPostButtons(
          postButtonSection,
          masterToogle ? false : isHideSharesButton,
          "share"
        );
        hideCheckPostButtons(
          postButtonSection,
          masterToogle ? false : isHideAwardButton,
          "awards"
        );
      }
    });
  });
}

function handleLandingMobileCheckPostButtons(
  postsButtonSection,
  checkedValue,
  typeButtonToHide
) {
  const isHideButtonValue = checkedValue
    ? "display: none !important;"
    : "display: block !important;";

  const currentShredditPost = postsButtonSection.getRootNode().host;

  const menuShadowRoot = currentShredditPost.shadowRoot.querySelector(
    `div[data-testid="action-row"]`
  );

  if (menuShadowRoot) {
    const commentsButton = menuShadowRoot.querySelector(
      `a[data-post-click-location="comments-button"]`
    );

    const upDownVoteButtons = postsButtonSection.querySelector(
      `span:has(span[data-post-click-location="vote"])`
    );

    const voteCounts = postsButtonSection.querySelector(
      `span > span[data-post-click-location="vote"] span:has(faceplate-number)`
    );

    const shareButton = menuShadowRoot.querySelector(
      `slot[name="share-button"]`
    );

    const awardsButton = menuShadowRoot.querySelector(`award-button`);

    switch (typeButtonToHide) {
      case "comments":
        commentsButton.style.cssText = isHideButtonValue;

        break;

      case "upDownVote":
        upDownVoteButtons.style.cssText = checkedValue
          ? "opacity: 0 !important;"
          : "opacity: 1 !important;";
        break;

      case "voteCounts":
        voteCounts.style.cssText = checkedValue
          ? "opacity: 0 !important;"
          : "opacity: 1 !important;";
        break;

      case "share":
        if (shareButton) {
          shareButton.style.cssText = isHideButtonValue;
        }
        break;

      case "awards":
        if (awardsButton) {
          awardsButton.style.cssText = isHideButtonValue;
        }
        break;

      default:
        break;
    }
  }
}

function hideCheckPostButtons(
  postsButtonSection,
  checkedValue,
  typeButtonToHide
) {
  const isHideButtonValue = checkedValue
    ? "display: none !important;"
    : "display: block !important;";
  const isLandingPageAreMobile =
    document.querySelector("shreddit-app").getAttribute("routename") ===
      "frontpage" &&
    document.querySelector("shreddit-app").getAttribute("devicetype") ===
      "mobile";

  if (isLandingPageAreMobile) {
    handleLandingMobileCheckPostButtons(
      postsButtonSection,
      checkedValue,
      typeButtonToHide
    );
  } else {
    const commentsButton = postsButtonSection.querySelector(
      `button[data-post-click-location="comments-button"], a[name="comments-action-button"]`
    );

    const upDownVoteButtons = postsButtonSection.querySelector(
      `span:has(span[data-post-click-location="vote"])`
    );

    const voteCounts = postsButtonSection.querySelector(
      `span > span[data-post-click-location="vote"] span:has(faceplate-number)`
    );

    const shareButton = postsButtonSection.querySelector(
      `slot[name="share-button"]`
    );

    const awardsButton = postsButtonSection.querySelector("award-button");

    switch (typeButtonToHide) {
      case "comments":
        commentsButton.style.cssText = isHideButtonValue;
        break;

      case "upDownVote":
        upDownVoteButtons.style.cssText = isHideButtonValue;
        break;

      case "voteCounts":
        voteCounts.style.cssText = isHideButtonValue;
        break;

      case "share":
        shareButton.style.cssText = isHideButtonValue;
        break;

      case "awards":
        awardsButton.style.cssText = isHideButtonValue;
        break;

      default:
        break;
    }
  }
}

browser.storage.onChanged.addListener((changes, area) => {
  if (area === "local") {
    watchedKeys.forEach((key) => {
      if (changes[key]) {
        handlePostHideButtons();
      }
    });
  }
});

const observer = new MutationObserver(() => {
  handlePostHideButtons();
});

const bodyObserver = new MutationObserver(() => {
  if (document.body) {
    bodyObserver.disconnect();
    observer.observe(document.body, observerConfig);
  }
});

bodyObserver.observe(document, observerConfig);
