/**
 * Thin wrapper around the claude.ai web API.
 *
 * Everything runs from a content script on claude.ai, so requests are same-origin and the
 * session cookie is attached automatically — no tokens are read, stored or forwarded anywhere.
 */
(() => {
  'use strict';

  const CT = (globalThis.CT ||= {});
  const ext = globalThis.browser ?? globalThis.chrome;

  /**
   * Query strings to try, most informative first. The API has changed shape before, so we
   * fall back until one returns something with a `chat_messages` array.
   */
  const QUERIES = [
    // Exactly what claude.ai's own client sends. `consistency=strong` matters here: the
    // pane refetches immediately after moving a branch, and a relaxed read can still be
    // serving the previous one.
    'tree=True&rendering_mode=messages&render_all_tools=true&include_inline_comparison=true&consistency=strong',
    'tree=True&rendering_mode=messages&render_all_tools=true',
    'tree=True&rendering_mode=messages',
    'tree=True&rendering_mode=raw',
    'tree=True',
    '',
  ];

  class ApiError extends Error {
    constructor(message, status = 0) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  }

  function describeStatus(status) {
    if (status === 401 || status === 403) {
      return 'claude.ai rejected the request — the session may have expired. Reload the page and sign in again.';
    }
    if (status === 404) return 'That conversation could not be found.';
    if (status === 429) return 'claude.ai is rate limiting requests. Try again in a moment.';
    if (status >= 500) return `claude.ai returned a server error (HTTP ${status}).`;
    return `claude.ai returned HTTP ${status}.`;
  }

  function readCookie(name) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const match = document.cookie.match(new RegExp(`(?:^|;\\s*)${escaped}=([^;]*)`));
    if (!match) return null;
    try {
      return decodeURIComponent(match[1]);
    } catch {
      return match[1];
    }
  }

  /** Ask the background page to make the request instead — used only if a direct fetch fails. */
  async function viaBackground(url) {
    if (!ext?.runtime?.sendMessage) return null;
    let result;
    try {
      result = await ext.runtime.sendMessage({ type: 'ct:fetch', url });
    } catch {
      return null;
    }
    if (!result) return null;
    if (result.error) return null;
    if (!result.ok) throw new ApiError(describeStatus(result.status), result.status);
    return result.body;
  }

  async function requestJson(path) {
    const url = new URL(path, location.origin).href;
    let res;
    try {
      res = await fetch(url, { credentials: 'include', headers: { accept: 'application/json' } });
    } catch {
      const fallback = await viaBackground(url);
      if (fallback) return fallback;
      throw new ApiError('Could not reach claude.ai. Check your connection and try again.', 0);
    }
    if (!res.ok) throw new ApiError(describeStatus(res.status), res.status);
    try {
      return await res.json();
    } catch {
      throw new ApiError('claude.ai returned a response the extension could not read.', 0);
    }
  }

  /** Organisation uuid that last worked, so repeat loads skip the discovery step. */
  let knownOrgId = null;
  /** Which query string last returned a conversation — useful when branches are missing. */
  let lastQuery = null;

  /**
   * Load one conversation with its full message tree.
   *
   * The organisation uuid is not in the page URL, so it is guessed from the `lastActiveOrg`
   * cookie first and only looked up over the network if that guess turns out to be wrong.
   *
   * @param {string} conversationId
   * @returns {Promise<object>} the raw conversation payload
   */
  async function fetchConversation(conversationId) {
    const attempted = new Set();
    let lastError = null;

    const tryOrgs = async (orgs) => {
      for (const org of orgs) {
        if (!org || attempted.has(org)) continue;
        attempted.add(org);
        for (const query of QUERIES) {
          const path = `/api/organizations/${org}/chat_conversations/${conversationId}${query ? `?${query}` : ''}`;
          try {
            const data = await requestJson(path);
            if (data && Array.isArray(data.chat_messages)) {
              knownOrgId = org;
              lastQuery = query || '(no parameters)';
              return data;
            }
          } catch (err) {
            lastError = err;
            if (err.status === 401 || err.status === 403) throw err;
            if (err.status === 404) break; // wrong organisation — move on to the next
          }
        }
      }
      return null;
    };

    const guessed = await tryOrgs([knownOrgId, readCookie('lastActiveOrg')]);
    if (guessed) return guessed;

    let listed = [];
    try {
      const orgs = await requestJson('/api/organizations');
      if (Array.isArray(orgs)) listed = orgs.map((org) => org?.uuid);
    } catch (err) {
      lastError = err;
    }

    const found = await tryOrgs(listed);
    if (found) return found;

    throw lastError ?? new ApiError('claude.ai did not return any messages for this conversation.', 0);
  }

  /**
   * Point the conversation at a different message, which is what selecting a branch means.
   *
   * This is the same request claude.ai makes when you click its "‹ 2/3 ›" switcher:
   *   PUT /api/organizations/{org}/chat_conversations/{id}/current_leaf_message_uuid
   * Authentication is the session cookie the browser already attaches.
   *
   * @param {string} conversationId
   * @param {string} leafId message the conversation should now end at
   * @returns {Promise<{ok: boolean, reason?: string}>}
   */
  async function setCurrentLeaf(conversationId, leafId) {
    const org = knownOrgId || readCookie('lastActiveOrg');
    if (!org) return { ok: false, reason: 'no organisation id' };
    if (!conversationId || !leafId) return { ok: false, reason: 'no conversation or message id' };

    const url = new URL(
      `/api/organizations/${org}/chat_conversations/${conversationId}/current_leaf_message_uuid`,
      location.origin
    ).href;

    try {
      const res = await fetch(url, {
        method: 'PUT',
        credentials: 'include',
        headers: { 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({ current_leaf_message_uuid: leafId }),
      });
      if (res.ok) return { ok: true };
      return { ok: false, reason: `claude.ai answered HTTP ${res.status}` };
    } catch (err) {
      return { ok: false, reason: `request blocked (${err?.message || 'network error'})` };
    }
  }

  /**
   * Wait until the branch move is actually readable before anything reloads on it.
   *
   * `setCurrentLeaf` resolving only means the write was accepted. Reloading straight away
   * races its propagation: claude.ai's own page fetches the conversation, is handed the
   * previous leaf, and renders the branch you just moved away from — while this pane, which
   * asks for a strongly consistent read, shows the new one. That is the split where the
   * chat looks stale and a second manual refresh fixes it.
   *
   * @returns {Promise<boolean>} whether the new leaf became readable in time
   */
  async function confirmLeaf(conversationId, leafId, { attempts = 8, delay = 250 } = {}) {
    for (let attempt = 0; attempt < attempts; attempt++) {
      try {
        const data = await fetchConversation(conversationId);
        if (data?.current_leaf_message_uuid === leafId) return true;
      } catch {
        // keep waiting; a failed read here is not a failed write
      }
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
    return false;
  }

  CT.api = { fetchConversation, setCurrentLeaf, confirmLeaf, ApiError, describeLoad: () => lastQuery };
})();
