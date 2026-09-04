/**
 * Background event page.
 *
 * Two jobs only:
 *  1. Toolbar button / keyboard shortcut -> tell the content script to toggle the panel.
 *  2. Act as a fetch relay for the content script. The content script normally talks to
 *     claude.ai directly (same origin, cookies attached), but if that request is blocked
 *     the extension can retry from here, where host permissions apply.
 */

const ext = globalThis.browser ?? globalThis.chrome;

const CONTENT_SCRIPTS = ['src/api.js', 'src/model.js', 'src/chat.js', 'src/panel.js', 'src/content.js'];

ext.action.onClicked.addListener(async (tab) => {
  if (!tab?.id) return;
  try {
    await ext.tabs.sendMessage(tab.id, { type: 'ct:toggle' });
    return;
  } catch {
    // No content script yet: either this is not claude.ai, or the page was already open
    // when the extension was installed or reloaded.
  }

  if (!/^https:\/\/claude\.ai\//.test(tab.url || '')) {
    await ext.tabs.create({ url: 'https://claude.ai/' });
    return;
  }

  try {
    await ext.scripting.executeScript({ target: { tabId: tab.id }, files: CONTENT_SCRIPTS });
    await ext.tabs.sendMessage(tab.id, { type: 'ct:toggle' });
  } catch (err) {
    console.error('[claude-tree] could not inject into tab', tab.id, err);
  }
});

ext.runtime.onMessage.addListener((message, sender) => {
  if (message?.type !== 'ct:fetch') return;
  if (!/^https:\/\/claude\.ai\//.test(message.url || '')) {
    return Promise.resolve({ error: 'Refusing to fetch a non-claude.ai URL.' });
  }
  return relay(message.url);
});

async function relay(url) {
  try {
    const res = await fetch(url, {
      credentials: 'include',
      headers: { accept: 'application/json' },
    });
    if (!res.ok) return { ok: false, status: res.status };
    return { ok: true, status: res.status, body: await res.json() };
  } catch (err) {
    return { error: String(err?.message || err) };
  }
}
