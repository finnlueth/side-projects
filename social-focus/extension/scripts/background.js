// MARK: - One-time settings seed
//
// migration/seed.json carries settings over from another install (e.g. the original
// SocialFocus). It is applied once per seedId, on install / update. Delete the file
// once applied.

const SEED_APPLIED_KEY = "socialFocus_seedApplied";

async function applySeedSettings() {
  let seed;

  try {
    const response = await fetch(browser.runtime.getURL("migration/seed.json"));

    if (!response.ok) {
      return;
    }

    seed = await response.json();
  } catch (error) {
    return;
  }

  if (!seed || !seed.seedId || !seed.settings) {
    return;
  }

  const stored = await browser.storage.local.get(SEED_APPLIED_KEY);

  if (stored[SEED_APPLIED_KEY] === seed.seedId) {
    return;
  }

  await browser.storage.local.set({
    ...seed.settings,
    [SEED_APPLIED_KEY]: seed.seedId,
  });

  console.log(`SocialFocus: applied settings seed ${seed.seedId}`);
}

browser.runtime.onInstalled.addListener(() => {
  applySeedSettings();
});
