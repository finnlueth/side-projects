(function () {
  const STEP_MINUTES = 15;

  // MARK: - Prepare Screen

  function fillTimeSelect(select, selectedMinutes) {
    select.innerHTML = "";

    for (let minutes = 0; minutes < 24 * 60; minutes += STEP_MINUTES) {
      const option = document.createElement("option");
      option.value = minutes;
      option.textContent = formatMinutesAsTime(minutes);

      if (minutes == selectedMinutes) {
        option.selected = true;
      }

      select.appendChild(option);
    }
  }

  function setFreeTimeState(settings) {
    const stateInfo = queryById("freeTimeState");
    const bottomButtons = queryById("freeTime-bottomButtons");
    const nowInfo = queryById("freeTimeNowInfo");
    const banner = queryById("freeTimeBanner");

    if (settings.isActive) {
      stateInfo.setAttribute("active", "");
      bottomButtons.setAttribute("active", "");
    } else {
      stateInfo.removeAttribute("active");
      bottomButtons.removeAttribute("active");
    }

    // Is free time running right now?

    if (isFreeTimeForSettings(settings)) {
      nowInfo.setAttribute("active", "");
      banner.setAttribute("active", "");
      queryById("freeTimeBannerUntil").textContent = formatMinutesAsTime(
        settings.end
      );
    } else {
      nowInfo.removeAttribute("active");
      banner.removeAttribute("active");
    }
  }

  function prepareFreeTimeScreen() {
    getFreeTimeSettings(function (settings) {
      fillTimeSelect(queryById("freeTimeStartSelect"), settings.start);
      fillTimeSelect(queryById("freeTimeEndSelect"), settings.end);

      setFreeTimeState(settings);
    });
  }

  prepareFreeTimeScreen();

  // MARK: - Actions

  // Activate / Update

  document
    .querySelectorAll("#freeTimeSetButton, #freeTimeUpdateButton")
    .forEach((element) => {
      element.onclick = function () {
        const start = Number(queryById("freeTimeStartSelect").value);
        const end = Number(queryById("freeTimeEndSelect").value);

        setToStorage(getConst.freeTimeStartData, start);
        setToStorage(getConst.freeTimeEndData, end);
        setToStorage(getConst.freeTimeIsActiveData, true, function () {
          setFreeTimeState({ isActive: true, start: start, end: end });
        });
      };
    });

  // Deactivate

  queryById("freeTimeDestructButton").onclick = function () {
    setToStorage(getConst.freeTimeIsActiveData, false, function () {
      getFreeTimeSettings(setFreeTimeState);
    });
  };

  // Click on row with select

  const intervalItems = querySelectorAll(
    "#freeTimeScreen .modernFormBlockItemsWrapper:has(select)"
  );

  for (const index in intervalItems) {
    const item = intervalItems[index];
    item.onclick = function () {
      showDropdown(item.querySelector("select"));
    };
  }
})();
