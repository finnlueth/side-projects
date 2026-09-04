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

  CT.api = { fetchConversation, ApiError };
})();
