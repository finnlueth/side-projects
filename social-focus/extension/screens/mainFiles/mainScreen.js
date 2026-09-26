(function () {
  // MARK: - Category Picker

  queryById("activeCategoryButton").onclick = function () {
    var isShowing = queryById("mainScreen").getAttribute(
      "categoryPickerIsShowing"
    );
    queryById("mainScreen").setAttribute(
      "categoryPickerIsShowing",
      isShowing == "false"
    );
  };

  // MARK: - Router

  function showUnlockScreen() {
    queryById("unlockPasswordTextField").classList.remove("error");
    queryById("unlockPasswordTextField").value = "";
    queryById("wrongProtectionPasswordError").style.display = "none";
    showScreen("unlockScreen");
  }

  // MARK: - Power Button

  function blockExtensionListener(changes, area) {
    if (
      area === "local" &&
      changes[getConstNotSyncing.extensionIsEnabledData]
    ) {
      const { newValue } = changes[getConstNotSyncing.extensionIsEnabledData];

      document.documentElement.setAttribute("disabled", !newValue);
    }
  }

  function onOffClickHandler(turnType, element) {
    browser.storage.onChanged.removeListener(blockExtensionListener);

    const blockTimeInMinutes = +element.dataset.value;

    const popUp = document.querySelector(
      `#mainScreen .turn${turnType}PickerPopup`
    );

    popUp.removeAttribute("active");

    if (blockTimeInMinutes !== 0) {
      browser.storage.onChanged.addListener(blockExtensionListener);
    }

    extensionBlockTimerHandler(blockTimeInMinutes, turnType);

    if (turnType === "Off") {
      setToStorage(
        getConstNotSyncing.extensionBlockTime,
        blockTimeInMinutes,
        function () {
          sendCommand(
            getConstNotSyncing.extensionBlockTime,
            blockTimeInMinutes
          );
        }
      );
    }

    if (turnType === "On") {
      setToStorage(
        getConstNotSyncing.extensionEnableTime,
        blockTimeInMinutes,
        function () {
          sendCommand(
            getConstNotSyncing.extensionEnableTime,
            blockTimeInMinutes
          );
        }
      );
    }
  }

  // Set State

  browser.storage.local.get(
    getConstNotSyncing.extensionIsEnabledData,
    function (obj) {
      const globalDefault = true;
      const status =
        obj[getConstNotSyncing.extensionIsEnabledData] ?? globalDefault;

      document.documentElement.setAttribute("disabled", status != true);
    }
  );

  // Click

  queryById("powerButton").onclick = function () {
    browser.storage.local.get(
      getConstNotSyncing.extensionIsEnabledData,
      function (obj) {
        const globalDefault = true;
        const status =
          obj[getConstNotSyncing.extensionIsEnabledData] ?? globalDefault;

        if (status) {
          const turnOffPopUp = document.querySelector(
            "#mainScreen .turnOffPickerPopup"
          );
          const isVisible = turnOffPopUp.hasAttribute("active");

          if (isVisible) {
            turnOffPopUp.removeAttribute("active");
          } else {
            turnOffPopUp.setAttribute("active", "");
          }
        } else {
          const turnOnPopUp = document.querySelector(
            "#mainScreen .turnOnPickerPopup"
          );
          const isVisible = turnOnPopUp.hasAttribute("active");

          if (isVisible) {
            turnOnPopUp.removeAttribute("active");
          } else {
            turnOnPopUp.setAttribute("active", "");
          }
        }
      }
    );
  };

  // Select time for block or enable extension

  const turnOffElements = document
    .querySelector(".turnOffPickerPopup")
    .querySelectorAll(".turnOffElement");

  for (const element of turnOffElements) {
    element.onclick = () => onOffClickHandler("Off", element);
  }

  const turnOnElements = document
    .querySelector(".turnOnPickerPopup")
    .querySelectorAll(".turnOnElement");

  for (const element of turnOnElements) {
    element.onclick = () => onOffClickHandler("On", element);
  }
})();
