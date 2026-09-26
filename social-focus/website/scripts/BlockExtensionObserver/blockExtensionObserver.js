(function () {
  let intervalId;

  const extensionBlockTimeKey = getConstNotSyncing.extensionBlockTime;
  const extensionBlockLastedTimeKey =
    getConstNotSyncing.extensionBlockLastedTime;

  document.addEventListener("DOMContentLoaded", () => {
    browser.storage.local.get(
      [extensionBlockTimeKey, extensionBlockLastedTimeKey],
      function (obj) {
        const extensionBlockLimitDuration = obj[extensionBlockTimeKey];
        let extensionBlockLimitLastedTime = obj[extensionBlockLastedTimeKey];

        if (
          extensionBlockLimitDuration !== undefined &&
          extensionBlockLimitLastedTime !== undefined &&
          extensionBlockLimitDuration !== 0
        ) {
          if (
            extensionBlockLimitLastedTime <=
            getSecondsFromMinutes(extensionBlockLimitDuration)
          ) {
            stopTimer();
            startExtensionBlockingTimer();
          }
        }
      }
    );
  });

  browser.runtime.onMessage.addListener((message) => {
    if (message.action === "startBlockingExtension") {
      stopTimer();
      startExtensionBlockingTimer();
    }
  });

  function stopTimer() {
    clearInterval(intervalId);
  }

  function startExtensionBlockingTimer() {
    applyGlobalEnableAttribute(false);

    intervalId = setInterval(() => {
      browser.storage.local.get(
        [extensionBlockTimeKey, extensionBlockLastedTimeKey],
        function (obj) {
          const extensionBlockLimitDuration = obj[extensionBlockTimeKey];
          let extensionBlockLimitLastedTime = obj[extensionBlockLastedTimeKey];

          if (extensionBlockLimitDuration === 0) {
            stopTimer();
          } else {
            extensionBlockLimitLastedTime++;
            browser.storage.local.set({
              [getConstNotSyncing.extensionBlockLastedTime]:
                extensionBlockLimitLastedTime,
            });

            if (
              extensionBlockLimitLastedTime >=
              getSecondsFromMinutes(extensionBlockLimitDuration)
            ) {
              blockingExtension();
            }
          }
        }
      );
    }, 1000);
  }

  function blockingExtension() {
    stopTimer();

    browser.storage.local.set({
      [getConstNotSyncing.extensionIsEnabledData]: true,
    });

    browser.storage.local.set({
      [extensionBlockTimeKey]: 0,
    });

    browser.storage.local.set({
      [extensionBlockLastedTimeKey]: 0,
    });

    applyGlobalEnableAttribute(true);
  }
})();
