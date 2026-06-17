const STATUS_URL_RE = /^https?:\/\/(x|twitter)\.com\/[^/]+\/status\/\d+/;

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.url || !STATUS_URL_RE.test(tab.url)) return;

  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });
  } catch (err) {
    console.error('[x2pdf] Injection failed:', err);
  }
});
