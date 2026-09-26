function getLocalizedOptionName(optionObjectName) {
  if (optionObjectName.hasOwnProperty(app_language)) {
    return optionObjectName[app_language];
  } else if (optionObjectName.hasOwnProperty("en")) {
    return optionObjectName.en;
  } else {
    return optionObjectName;
  }
}

function getLocalizedGroupName(groupObjectName) {
  if (groupObjectName.hasOwnProperty(app_language)) {
    return groupObjectName[app_language];
  } else if (groupObjectName.hasOwnProperty("en")) {
    return groupObjectName.en;
  } else {
    return groupObjectName;
  }
}

function getLocalizedCategoryName(categoryObjectName) {
  if (categoryObjectName.hasOwnProperty(app_language)) {
    return categoryObjectName[app_language];
  } else if (categoryObjectName.hasOwnProperty("en")) {
    return categoryObjectName.en;
  } else {
    return categoryObjectName;
  }
}

function findOptionById(id, categories) {
  // Loop through the categories
  for (const category of categories) {
    for (const group of category.categoryGroups) {
      // Check if the group has "options"
      if (group.options) {
        for (const option of group.options) {
          // Check if the option has the given ID
          if (option.id === id) {
            return option;
          }
        }
      }
    }
  }

  return null; // Return null if no element with the given ID is found
}

function getAllOptions(categories) {
  const allOptions = [];

  // Loop through the categories
  for (const category of categories) {
    for (const group of category.categoryGroups) {
      // Check if the group has "options"
      if (group.options) {
        allOptions.push(...group.options);
      }
    }
  }

  allOptions.push(OTHER_SETTINGS[0]);

  return allOptions;
}

function isBrowserSafari() {
  let userAgent = window.navigator.userAgent;

  if (userAgent.includes("Safari") && !userAgent.includes("Chrome")) {
    return true;
  } else {
    return false;
  }
}
