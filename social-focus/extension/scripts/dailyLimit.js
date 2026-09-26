function createDailyLimitOptions(categoryId, isInit) {
  browser.storage.local.get(
    getConst.dailyLimitDuration[categoryId],

    function (obj) {
      const selected = obj[getConst.dailyLimitDuration[categoryId]];

      const dailyLimitSelects = document.querySelectorAll(
        "#socialFocus_daily_limit"
      );

      const currentSelect = document.querySelector(
        `#mainScreen .collapsibleSection[categoryId=${categoryId}] .optionWrapper select`
      );

      if (isInit) {
        for (const select of dailyLimitSelects) {
          for (let i = 5; i <= 500; i += 5) {
            const option = document.createElement("option");

            option.value = i;

            option.innerHTML = i;

            select.appendChild(option);
          }
        }
      }

      for (const option of currentSelect.options) {
        if (option.value == selected) {
          option.selected = true;
        }
      }
    }
  );
}

function setBlockingSiteTimer(pickedTimeLimit, categoryId) {
  browser.storage.local.get(
    [getConst.dailyLimitLastedTime[categoryId]],

    function (obj) {
      const dailyLimitLastedTime =
        obj[getConst.dailyLimitLastedTime[categoryId]];

      if (pickedTimeLimit === "noLimit") {
        browser.storage.local.set({
          [getConst.dailyLimitDuration[categoryId]]: pickedTimeLimit,
        });
      } else if (
        getSecondsFromMinutes(pickedTimeLimit) < Number(dailyLimitLastedTime)
      ) {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          chrome.tabs.sendMessage(tabs[0].id, {
            action: "startBlockingSite",
          });
        });
      } else {
        browser.storage.local.set({
          [getConst.dailyLimitDuration[categoryId]]: pickedTimeLimit,
        });

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          chrome.tabs.sendMessage(tabs[0].id, {
            action: "startBlockingSiteTimer",
          });
        });
      }
    }
  );
}
