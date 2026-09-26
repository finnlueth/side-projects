function getSiteInfo() {
  outerLoop: for (const website of WEBSITES) {
    for (const domain of website.domain) {
      if (location.href.includes(domain)) {
        return website;
      }
    }
  }
  return null;
}

function isMobileVersion() {
  const mobileSelectors = getSiteInfo().mobileSelectorCheck;

  if (mobileSelectors) {
    var found = false;

    for (const mobileSelector of mobileSelectors) {
      if (mobileSelector != "") {
        const checkSelect = document.querySelector(mobileSelector);
        if (checkSelect) {
          found = true;
          break;
        }
      }
    }

    if (found) {
      return true;
    } else {
      return false;
    }
  }
}

// Return both desktop and mobile to dont spend ms on launch to detect which is correct
function getWebsiteCategoriesFromWebsite() {
  var returnCategories = [];

  const currentHref = window.location.href;

  const relevantCategory = CATEGORIES.find(
    (category) => category.categoryId == getSiteInfo().name
  );

  if (!relevantCategory) {
    return returnCategories;
  }

  const desktopCategories = {
    categoryId: relevantCategory.categoryId,
    categoryName: relevantCategory.categoryDesktopName,
    categoryGroups: relevantCategory.categoryDesktopGroups,
  };

  returnCategories.push(desktopCategories);

  if (relevantCategory.categoryMobileGroups.length > 0) {
    const mobileCategories = {
      categoryId: relevantCategory.categoryId,
      categoryName: relevantCategory.categoryMobileName,
      categoryGroups: relevantCategory.categoryMobileGroups,
    };

    returnCategories.push(mobileCategories);
  }

  return returnCategories;
}

// Is the current URL one of the site's "primary / feed" pages (see feedPaths in storageConstants.js)?
function isOnFeedPage() {
  const site = getSiteInfo();

  if (!site) {
    return false;
  }

  const feedPaths = site.feedPaths ?? [];

  if (feedPaths.length === 0) {
    return true;
  }

  const path = location.pathname.replace(/\/+$/, "") || "/";

  return feedPaths.some((pattern) => pattern.test(path));
}

// Run `callback` once document.body exists (content scripts start before it does)
function whenBodyReady(callback) {
  if (document.body) {
    callback();
    return;
  }

  const observer = new MutationObserver(() => {
    if (document.body) {
      observer.disconnect();
      callback();
    }
  });

  observer.observe(document.documentElement, { childList: true });
}
