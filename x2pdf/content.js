(async () => {
  if (window.__x2pdfRunning) return;
  window.__x2pdfRunning = true;

  const STATUS_INDICATOR_ID = '__x2pdf_status';

  function showStatus(msg) {
    let el = document.getElementById(STATUS_INDICATOR_ID);
    if (!el) {
      el = document.createElement('div');
      el.id = STATUS_INDICATOR_ID;
      Object.assign(el.style, {
        position: 'fixed', top: '12px', right: '12px', zIndex: '99999',
        background: '#1d9bf0', color: '#fff', fontFamily: 'system-ui, sans-serif',
        fontSize: '13px', fontWeight: '600', padding: '8px 14px',
        borderRadius: '20px', boxShadow: '0 2px 12px rgba(0,0,0,0.25)',
        pointerEvents: 'none', transition: 'opacity 0.3s',
      });
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.style.opacity = '1';
  }

  function hideStatus() {
    const el = document.getElementById(STATUS_INDICATOR_ID);
    if (el) {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 400);
    }
  }

  function waitForElement(selector, timeout = 8000) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(selector);
      if (existing) return resolve(existing);
      const observer = new MutationObserver(() => {
        const el = document.querySelector(selector);
        if (el) { observer.disconnect(); clearTimeout(timer); resolve(el); }
      });
      observer.observe(document.body, { childList: true, subtree: true });
      const timer = setTimeout(() => {
        observer.disconnect();
        reject(new Error('Timeout waiting for tweet content. Is this a tweet page?'));
      }, timeout);
    });
  }

  function delay(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  async function loadFullThread() {
    const MAX_ITERATIONS = 40;
    const STALE_THRESHOLD = 3;
    const SCROLL_WAIT_MS = 1300;

    let stale = 0;
    let prevCount = 0;

    for (let i = 0; i < MAX_ITERATIONS && stale < STALE_THRESHOLD; i++) {
      window.scrollTo(0, document.body.scrollHeight);
      await delay(SCROLL_WAIT_MS);

      const count = document.querySelectorAll('[data-testid="tweet"]').length;
      showStatus(`x2pdf: loading thread… ${count} tweets found`);

      if (count === prevCount) {
        stale++;
      } else {
        stale = 0;
        prevCount = count;
      }
    }

    window.scrollTo(0, 0);
    await delay(400);
  }

  async function expandShowMore() {
    // X truncates long tweets with a "Show more" link using this specific testid.
    // Only use the exact testid — broad role/text selectors risk clicking unintended
    // buttons (navigation, compose, etc.) and breaking page state before extraction.
    for (let pass = 0; pass < 5; pass++) {
      const buttons = document.querySelectorAll('[data-testid="tweet-text-show-more-link"]');
      if (!buttons.length) break;
      buttons.forEach(btn => { try { btn.click(); } catch {} });
      await delay(600);
    }
  }

  function isArticleMode() {
    const primaryCol = document.querySelector('[data-testid="primaryColumn"]') || document.body;

    // Only trust explicit article testids. Draft.js markers ([data-contents], [data-block],
    // [contenteditable]) are also used by X's tweet compose box on every page — do not use them.
    return !!(
      primaryCol.querySelector('[data-testid="twitterArticleRichTextView"]') ||
      primaryCol.querySelector('[data-testid="article-content"]') ||
      primaryCol.querySelector('[data-testid="article-body"]')
    );
  }

  function extractTextContent(el) {
    let result = '';
    for (const node of el.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        result += node.textContent;
      } else if (node.nodeName === 'BR') {
        result += '\n';
      } else if (node.nodeName === 'A') {
        const expanded = node.getAttribute('data-expanded-url') || node.href || node.textContent;
        result += expanded;
      } else if (node.nodeName === 'IMG') {
        result += node.getAttribute('alt') || '';
      } else {
        result += extractTextContent(node);
      }
    }
    return result;
  }

  function extractTweetData(tweetEl) {
    const userNameEl = tweetEl.querySelector('[data-testid="User-Name"]');
    let authorName = '', authorHandle = '';
    if (userNameEl) {
      const spans = Array.from(userNameEl.querySelectorAll('span')).map(s => s.textContent.trim()).filter(Boolean);
      authorHandle = spans.find(t => t.startsWith('@')) || '';
      authorName = spans.find(t => t && !t.startsWith('@') && t !== '·' && t !== '·') || '';
    }

    const textEl = tweetEl.querySelector('[data-testid="tweetText"]');
    const tweetText = textEl ? extractTextContent(textEl) : '';

    const timeEl = tweetEl.querySelector('time[datetime]');
    const timestamp = timeEl ? timeEl.getAttribute('datetime') : '';
    const displayTime = timeEl ? timeEl.textContent.trim() : '';

    const tweetLink = timeEl ? timeEl.closest('a') : null;
    const tweetUrl = tweetLink ? tweetLink.href : window.location.href;

    const imageUrls = Array.from(tweetEl.querySelectorAll('[data-testid="tweetPhoto"] img'))
      .map(img => img.src || img.getAttribute('src'))
      .filter(Boolean);

    return { authorName, authorHandle, tweetText, timestamp, displayTime, tweetUrl, imageUrls };
  }

  function extractThreadTweets() {
    const primaryCol = document.querySelector('[data-testid="primaryColumn"]') || document.body;
    const allTweets = Array.from(primaryCol.querySelectorAll('[data-testid="tweet"]'));
    if (!allTweets.length) return [];

    const firstData = extractTweetData(allTweets[0]);
    const originalHandle = firstData.authorHandle;

    const tweets = [];
    let gapCount = 0;
    const GAP_TOLERANCE = 1;

    for (const el of allTweets) {
      const data = extractTweetData(el);
      // If we couldn't determine the author (no User-Name element), include the tweet
      const sameAuthor = !data.authorHandle || !originalHandle || data.authorHandle === originalHandle;
      if (sameAuthor) {
        gapCount = 0;
        tweets.push(data);
      } else {
        gapCount++;
        if (gapCount > GAP_TOLERANCE) break;
      }
    }

    return tweets;
  }

  function extractArticle() {
    const primaryCol = document.querySelector('[data-testid="primaryColumn"]') || document.body;
    const focalTweet = primaryCol.querySelector('[data-testid="tweet"]');
    const meta = focalTweet ? extractTweetData(focalTweet) : {};

    let bodyEl = null;

    // 1. Specific article testids (twitterArticleRichTextView confirmed from DOM inspection)
    bodyEl = primaryCol.querySelector('[data-testid="twitterArticleRichTextView"]') ||
             primaryCol.querySelector('[data-testid="article-content"]') ||
             primaryCol.querySelector('[data-testid="article-body"]');

    // 2. Draft.js rich text editor (X Articles use this internally)
    if (!bodyEl) {
      const draftRoot = primaryCol.querySelector('[data-contents="true"]');
      if (draftRoot) {
        // Walk up to the contenteditable wrapper or the nearest meaningful ancestor
        bodyEl = draftRoot.closest('[contenteditable]') ||
                 draftRoot.parentElement ||
                 draftRoot;
      }
    }

    // 3. Find container by Draft.js block markers
    if (!bodyEl) {
      const blocks = primaryCol.querySelectorAll('[data-block="true"]');
      if (blocks.length >= 2) {
        // Common ancestor of all blocks (they should share a parent chain)
        let ancestor = blocks[0].parentElement;
        while (ancestor && ancestor !== primaryCol) {
          if (ancestor.contains(blocks[blocks.length - 1])) break;
          ancestor = ancestor.parentElement;
        }
        if (ancestor && ancestor !== primaryCol) bodyEl = ancestor;
      }
    }

    // 4. Text-volume heuristic: find the biggest text container that's not a UI section
    if (!bodyEl) {
      let bestLen = 500;
      for (const el of primaryCol.querySelectorAll('div, section')) {
        if (el === primaryCol || el === focalTweet) continue;
        if (el.closest('[role="navigation"]') || el.closest('[role="banner"]')) continue;
        const len = el.textContent.trim().length;
        if (len > bestLen) { bestLen = len; bodyEl = el; }
      }
    }

    if (!bodyEl && focalTweet) bodyEl = focalTweet;

    // Title priority: X's specific title testid → first heading in body → first line of tweet text
    const titleEl =
      primaryCol.querySelector('[data-testid="twitter-article-title"]') ||
      primaryCol.querySelector('[data-testid="article-title"]') ||
      (bodyEl && (bodyEl.querySelector('h1') || bodyEl.querySelector('h2')));
    const articleTitle =
      (titleEl?.textContent.trim()) ||
      (meta.tweetText?.split('\n')[0].trim()) ||
      '';

    const bodyImages = bodyEl
      ? Array.from(bodyEl.querySelectorAll('img')).map(img => img.src).filter(Boolean)
      : [];

    // Cover image: tweetPhoto images on the focal tweet that are NOT inside the article body.
    // extractTweetData collects all tweetPhoto imgs within focalTweet, which includes every
    // article body image — filter those out so only the true cover image(s) remain.
    const coverImageUrls = focalTweet
      ? Array.from(focalTweet.querySelectorAll('[data-testid="tweetPhoto"] img'))
          .filter(img => !bodyEl || !bodyEl.contains(img))
          .map(img => img.src || img.getAttribute('src'))
          .filter(Boolean)
      : [];

    return { ...meta, articleTitle, bodyEl, bodyImages, coverImageUrls };
  }

  function cleanArticleClone(clone) {
    // Remove all SVGs — every SVG on the page is a UI icon (verified badge, action icons, etc.)
    clone.querySelectorAll('svg').forEach(el => el.remove());

    // Remove UI/interactive elements that should never appear in the document.
    // IMPORTANT: use precise selectors — broad wildcards like [data-testid*="card"] risk
    // matching the article body container itself and deleting all content.
    [
      'button', '[role="button"]', '[role="group"]', '[role="toolbar"]',
      '[data-testid="User-Name"]', '[data-testid*="Avatar"]', '[data-testid*="avatar"]',
      '[data-testid="like"]', '[data-testid="unlike"]', '[data-testid="reply"]',
      '[data-testid="retweet"]', '[data-testid="unretweet"]',
      '[data-testid="bookmark"]', '[data-testid="share"]',
      '[data-testid*="analytics"]',
      // Tweet-preview cards use the "card." namespace — safe to target by prefix
      '[data-testid^="card."]',
      'script', 'style',
    ].forEach(sel => clone.querySelectorAll(sel).forEach(el => el.remove()));

    // Strip X's dark-theme inline color/background styles from every element.
    // X applies light colors (e.g. rgb(231,233,234)) designed for its dark UI —
    // these are invisible on white paper. Remove them and let the CSS take over.
    clone.querySelectorAll('*').forEach(el => {
      el.style.removeProperty('color');
      el.style.removeProperty('background-color');
      el.style.removeProperty('background');
      // Also strip border-color if it references a dark-theme palette
      el.style.removeProperty('border-color');
    });

    // Strip href from X-internal and relative links — prevents garbage URL expansion in print
    clone.querySelectorAll('a[href]').forEach(a => {
      const href = a.getAttribute('href') || '';
      if (href.startsWith('/') || /https?:\/\/(x|twitter)\.com/.test(href)) {
        a.removeAttribute('href');
      }
    });

    // Fix code-language labels — X renders them with inline style="color: rgb(83,100,113)"
    // which CSS classes can't override. Directly set the style on the element and all its
    // children so it renders as readable dark text regardless of X's inline styling.
    clone.querySelectorAll('pre').forEach(pre => {
      const prev = pre.previousElementSibling;
      if (prev && prev.textContent.trim().length < 40 && !prev.querySelector('img, code, pre')) {
        const applyLabelStyle = el => {
          el.style.cssText = [
            'color: #0f0f0f',
            'font-weight: 700',
            'font-size: 8.5pt',
            'font-family: system-ui, sans-serif',
            'text-transform: uppercase',
            'letter-spacing: 0.05em',
            'display: block',
            'margin-bottom: 2pt',
          ].join(';');
        };
        applyLabelStyle(prev);
        prev.querySelectorAll('*').forEach(applyLabelStyle);
      }
    });

    // Fix images and wrap each one in a break-safe div so page breaks never cut through them
    clone.querySelectorAll('img').forEach(img => {
      img.removeAttribute('srcset');
      img.removeAttribute('sizes');
      img.loading = 'eager';
      const wrap = document.createElement('div');
      wrap.className = 'img-wrap';
      img.parentNode.insertBefore(wrap, img);
      wrap.appendChild(img);
    });

    // Fix link line-breaks — two-pass approach:
    //
    // Pass 1: Merge link-only and punctuation-only divs into the preceding sibling.
    // X's Draft.js editor creates separate [data-block] divs for each inline entity
    // (links, trailing punctuation). Each is display:block by default, so each gets
    // its own line. Merging moves their content into the preceding paragraph.
    //
    // KEY BUG FIX: the old check included 'div' in the block-descendant test, which
    // caused Draft.js link blocks (which always contain nested data-offset-key divs)
    // to be skipped. Now we only check for truly block-level semantic elements.
    let mergedAny = true;
    while (mergedAny) {
      mergedAny = false;
      Array.from(clone.querySelectorAll('div')).forEach(div => {
        if (!div.parentNode) return;
        const prev = div.previousElementSibling;
        if (!prev) return; // nothing to merge into

        // Skip if this div contains truly block-level semantic elements
        // NOTE: 'div' is intentionally NOT in this list — Draft.js nests divs inside
        // link wrappers ([data-offset-key]) and we must not skip those.
        if (div.querySelector('p, pre, ul, ol, blockquote, h1, h2, h3, h4, h5, h6, table, figure, .img-wrap')) return;
        // Don't merge into structural blocks
        if (prev.querySelector('pre, ul, ol, table, h1, h2, h3, h4, h5, h6, .img-wrap')) return;

        const text = div.textContent.trim();
        const links = Array.from(div.querySelectorAll('a'));
        const linkText = links.map(a => a.textContent.trim()).join('');
        const nonLinkText = text.replace(linkText, '').replace(/\s/g, '');

        const isLinkWrapper = links.length > 0 && nonLinkText.length <= 3;
        const isShortPunct  = text.length <= 3 && links.length === 0 && !div.querySelector('img');

        if (isLinkWrapper || isShortPunct) {
          while (div.firstChild) prev.appendChild(div.firstChild);
          div.remove();
          mergedAny = true;
        }
      });
    }

    // Pass 2: After merging, make all non-structural nested divs inline so the merged
    // content (e.g. data-offset-key wrappers from Draft.js) flows inline within paragraphs.
    clone.querySelectorAll('div').forEach(div => {
      if (div.classList.contains('img-wrap')) return;
      if (div.hasAttribute('data-block') || div.hasAttribute('data-contents')) return;
      if (Array.from(clone.children).includes(div)) return; // outermost container
      div.style.setProperty('display', 'inline', 'important');
    });

    // Remove elements that have no children at all (truly empty structural tags).
    // Only 3 passes; check childNodes (not textContent) so we never discard text containers.
    for (let pass = 0; pass < 3; pass++) {
      let removed = 0;
      clone.querySelectorAll('div, span, aside, section, header, footer').forEach(el => {
        if (el.parentNode && el.childNodes.length === 0) {
          el.remove();
          removed++;
        }
      });
      if (!removed) break;
    }

    return clone;
  }

  async function imageUrlToDataUrl(url) {
    try {
      const highRes = url.replace(/([?&]name=)(small|medium|large|thumb)(\b|&|$)/, '$1orig$3');
      const res = await fetch(highRes, { credentials: 'include' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch {
      return url;
    }
  }

  async function convertImages(urls) {
    const unique = [...new Set(urls)];
    const map = {};
    for (let i = 0; i < unique.length; i += 4) {
      const chunk = unique.slice(i, i + 4);
      const results = await Promise.all(chunk.map(imageUrlToDataUrl));
      chunk.forEach((url, idx) => { map[url] = results[idx]; });
    }
    return map;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function textToHtml(str) {
    return escapeHtml(str).replace(/\n/g, '<br>');
  }

  function printStyles() {
    return `
      :root {
        --font-body: Georgia, 'Times New Roman', serif;
        --font-ui: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        --color-text: #0f0f0f;
        --color-meta: #536471;
        --color-border: #cfd9de;
        --color-accent: #1d9bf0;
      }
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      body {
        font-family: var(--font-body);
        font-size: 12pt;
        line-height: 1.65;
        color: var(--color-text);
        background: #fff;
      }
      .container {
        max-width: 680px;
        margin: 0 auto;
        padding: 0 20pt;
      }
      .doc-header {
        border-bottom: 2px solid var(--color-text);
        padding: 20pt 0 14pt;
        margin-bottom: 20pt;
      }
      .author-name {
        font-family: var(--font-ui);
        font-size: 16pt;
        font-weight: 700;
        display: block;
      }
      .author-handle {
        font-family: var(--font-ui);
        font-size: 11pt;
        color: var(--color-meta);
      }
      .doc-source {
        font-family: var(--font-ui);
        font-size: 9pt;
        color: var(--color-meta);
        margin-top: 6pt;
        word-break: break-all;
      }
      .tweet-card {
        padding: 14pt 0;
        border-bottom: 1px solid var(--color-border);
      }
      .tweet-card:last-of-type { border-bottom: none; }
      .tweet-meta {
        font-family: var(--font-ui);
        font-size: 9pt;
        color: var(--color-meta);
        margin-bottom: 6pt;
      }
      .tweet-meta .name { font-weight: 600; color: var(--color-text); }
      .tweet-text {
        font-size: 12pt;
        line-height: 1.65;
        word-wrap: break-word;
        white-space: pre-wrap;
      }
      .tweet-images {
        display: flex;
        flex-wrap: wrap;
        gap: 6pt;
        margin-top: 10pt;
        page-break-inside: avoid;
        break-inside: avoid;
      }
      .tweet-images img {
        max-width: 100%;
        max-height: 280pt;
        object-fit: contain;
        display: block;
        border-radius: 4pt;
      }
      .tweet-images.single img { width: 100%; }
      .tweet-images.multi img { max-width: calc(50% - 3pt); }
      .article-title {
        font-family: var(--font-ui);
        font-size: 22pt;
        font-weight: 800;
        color: var(--color-text);
        line-height: 1.25;
        margin: 20pt 0 16pt;
        page-break-after: avoid;
        break-after: avoid;
      }
      .article-cover {
        margin: 0 0 20pt;
        page-break-inside: avoid;
        break-inside: avoid;
      }
      .article-cover img {
        width: 100%;
        max-height: 360pt;
        object-fit: cover;
        border-radius: 6pt;
        display: block;
      }
      .article-body { font-size: 12pt; line-height: 1.7; }
      .article-body h1, .article-body h2, .article-body h3,
      .article-body h4, .article-body h5, .article-body h6 {
        font-family: var(--font-ui);
        font-weight: 700;
        /* X wraps headings in <a> anchor tags — force black regardless of link color */
        color: var(--color-text) !important;
        margin: 16pt 0 6pt;
        line-height: 1.3;
        page-break-after: avoid;
        break-after: avoid;
      }
      /* Headings that are or contain links should still appear black */
      .article-body h1 a, .article-body h2 a, .article-body h3 a,
      .article-body h4 a, .article-body h5 a, .article-body h6 a {
        color: inherit !important;
      }
      .article-body h1 { font-size: 20pt; }
      .article-body h2 { font-size: 16pt; }
      .article-body h3 { font-size: 13pt; }
      .article-body p { margin: 8pt 0; }
      /* Code block language label — X renders it as a small label above <pre> */
      .article-body pre {
        background: #1e1e1e;
        color: #e8e8e8;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
        font-size: 9.5pt;
        line-height: 1.5;
        padding: 10pt 12pt;
        border-radius: 6pt;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
        margin: 8pt 0;
        page-break-inside: avoid;
        break-inside: avoid;
      }
      /* Language label that X shows above code blocks (sibling before <pre>) */
      .article-body pre + pre,
      .article-body :is(div, span)[class*="lang"],
      .article-body :is(div, span)[class*="language"],
      .article-body :is(div, span)[class*="code-lang"] {
        font-family: var(--font-ui);
        font-size: 8.5pt;
        color: var(--color-text);
        font-weight: 600;
        background: none;
        padding: 0;
        margin-bottom: 2pt;
      }
      /* Language label detected and marked by cleanArticleClone */
      .article-body .code-lang-label {
        font-family: var(--font-ui);
        font-size: 8.5pt;
        font-weight: 700;
        color: var(--color-text);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 1pt;
        display: block;
      }
      /* Generic inline code */
      .article-body code {
        font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
        font-size: 9.5pt;
        background: #f1f3f4;
        color: #c7254e;
        padding: 1pt 4pt;
        border-radius: 3pt;
      }
      .article-body pre code {
        background: none;
        color: inherit;
        padding: 0;
        border-radius: 0;
        font-size: inherit;
      }
      /* Images are wrapped in .img-wrap by cleanArticleClone for reliable page break control */
      .img-wrap {
        display: block;
        margin: 12pt 0;
        page-break-inside: avoid;
        break-inside: avoid;
      }
      .img-wrap img, .article-body img {
        max-width: 100%;
        max-height: 340pt;
        width: auto;
        object-fit: contain;
        display: block;
        border-radius: 4pt;
      }
      .article-body blockquote {
        border-left: 3px solid var(--color-border);
        margin: 12pt 0 12pt 16pt;
        padding-left: 12pt;
        color: var(--color-meta);
        font-style: italic;
      }
      /* Links in body text only — headings override this with color: inherit above */
      .article-body a { color: var(--color-accent); text-decoration: none; }
      /* Also ensure the language label text before a code block is readable */
      .article-body *:has(+ pre), .article-body *:has(> pre) {
        color: var(--color-text);
        font-weight: 600;
      }
      .article-body ul, .article-body ol { margin: 8pt 0 8pt 24pt; }
      .article-body li { margin: 4pt 0; }
      .article-body hr {
        border: none;
        border-top: 1px solid var(--color-border);
        margin: 16pt 0;
      }
      .doc-footer {
        margin-top: 24pt;
        padding-top: 8pt;
        border-top: 1px solid var(--color-border);
        font-family: var(--font-ui);
        font-size: 8pt;
        color: var(--color-meta);
      }
      @page {
        margin: 1.5cm 2cm;
        size: A4 portrait;
      }
      /* No print URL expansion — internal X links were stripped of href in cleanArticleClone */
    `;
  }

  function buildThreadHtml(tweets, imageMap) {
    if (!tweets.length) throw new Error('No tweets found on this page.');

    const first = tweets[0];
    const dateStr = first.timestamp
      ? new Date(first.timestamp).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
      : '';

    const tweetCards = tweets.map(t => {
      const resolvedImages = t.imageUrls.map(u => imageMap[u] || u);
      const imagesHtml = resolvedImages.length
        ? `<div class="tweet-images ${resolvedImages.length === 1 ? 'single' : 'multi'}">
            ${resolvedImages.map(u => `<img src="${escapeHtml(u)}" alt="Tweet image" loading="eager">`).join('')}
           </div>`
        : '';

      const metaLine = [
        t.authorName ? `<span class="name">${escapeHtml(t.authorName)}</span>` : '',
        t.authorHandle ? escapeHtml(t.authorHandle) : '',
        t.displayTime ? `· ${escapeHtml(t.displayTime)}` : '',
      ].filter(Boolean).join(' ');

      return `
      <div class="tweet-card">
        ${metaLine ? `<div class="tweet-meta">${metaLine}</div>` : ''}
        <div class="tweet-text">${textToHtml(t.tweetText)}</div>
        ${imagesHtml}
      </div>`;
    }).join('\n');

    return buildDocument({
      title: `${first.authorName} on X — ${dateStr}`,
      authorName: first.authorName,
      authorHandle: first.authorHandle,
      sourceUrl: first.tweetUrl,
      dateStr,
      body: `<div class="thread-body">${tweetCards}</div>`,
    });
  }

  function buildArticleHtml(data, imageMap) {
    const dateStr = data.timestamp
      ? new Date(data.timestamp).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
      : '';

    // Cover image: tweetPhoto images outside the article body (pre-filtered in extractArticle)
    const headerImageUrls = (data.coverImageUrls || []).map(u => imageMap[u] || u);

    let articleContent = '';
    if (data.bodyEl) {
      const clone = cleanArticleClone(data.bodyEl.cloneNode(true));

      // Replace image src with data URLs (cleanArticleClone already wrapped them in .img-wrap)
      clone.querySelectorAll('img').forEach(img => {
        const src = img.src || img.getAttribute('src') || '';
        if (src && imageMap[src]) img.src = imageMap[src];
      });

      articleContent = clone.innerHTML;
    }

    return buildDocument({
      title: data.articleTitle || `${data.authorName} — X Article`,
      authorName: data.authorName,
      authorHandle: data.authorHandle,
      sourceUrl: data.tweetUrl || window.location.href,
      dateStr,
      articleTitle: data.articleTitle,
      headerImages: headerImageUrls,
      body: `<div class="article-body">${articleContent}</div>`,
    });
  }

  function buildDocument({ title, authorName, authorHandle, sourceUrl, dateStr, articleTitle, headerImages, body }) {
    const generated = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width">
  <title>${escapeHtml(title)}</title>
  <style>${printStyles()}</style>
</head>
<body>
  <div class="container">
    <header class="doc-header">
      <span class="author-name">${escapeHtml(authorName)}</span>
      <span class="author-handle">${escapeHtml(authorHandle)}</span>
      ${dateStr ? `<div class="doc-source">${escapeHtml(dateStr)}</div>` : ''}
      <div class="doc-source">${escapeHtml(sourceUrl)}</div>
    </header>

    ${articleTitle ? `<h1 class="article-title">${escapeHtml(articleTitle)}</h1>` : ''}

    ${headerImages && headerImages.length
      ? `<div class="article-cover">${headerImages.map(u => `<img src="${escapeHtml(u)}" alt="Article cover image" loading="eager">`).join('')}</div>`
      : ''}

    <main>${body}</main>

    <footer class="doc-footer">
      Generated by x2pdf on ${escapeHtml(generated)}
    </footer>
  </div>
  <script>
    window.addEventListener('load', () => {
      const images = Array.from(document.images);
      if (!images.length) { setTimeout(() => window.print(), 200); return; }
      let loaded = 0;
      const check = () => { if (++loaded >= images.length) setTimeout(() => window.print(), 200); };
      images.forEach(img => {
        if (img.complete) check();
        else { img.addEventListener('load', check); img.addEventListener('error', check); }
      });
    });
  </script>
</body>
</html>`;
  }

  try {
    showStatus('x2pdf: waiting for page…');
    await waitForElement('[data-testid="tweet"]', 10000);

    const articleMode = isArticleMode();

    if (!articleMode) {
      showStatus('x2pdf: loading full thread…');
      await loadFullThread();
    } else {
      showStatus('x2pdf: extracting article…');
      await delay(600);
    }

    showStatus('x2pdf: expanding truncated content…');
    await expandShowMore();

    showStatus('x2pdf: extracting content…');

    let allImageUrls = [];
    let html;

    if (articleMode) {
      const data = extractArticle();
      allImageUrls = [
        ...(data.coverImageUrls || []),
        ...(data.bodyImages || []),
      ];
      showStatus('x2pdf: downloading images…');
      const imageMap = await convertImages(allImageUrls);
      html = buildArticleHtml(data, imageMap);
    } else {
      const tweets = extractThreadTweets();
      allImageUrls = tweets.flatMap(t => t.imageUrls);
      showStatus('x2pdf: downloading images…');
      const imageMap = await convertImages(allImageUrls);
      html = buildThreadHtml(tweets, imageMap);
    }

    showStatus('x2pdf: opening PDF…');
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const newTab = window.open(url, '_blank');

    if (!newTab) {
      hideStatus();
      alert('x2pdf: Please allow popups for x.com, then click the extension button again.');
      window.__x2pdfRunning = false;
      return;
    }

    setTimeout(() => URL.revokeObjectURL(url), 90000);
    hideStatus();

  } catch (err) {
    console.error('[x2pdf]', err);
    hideStatus();
    alert(`x2pdf error: ${err.message}`);
  } finally {
    setTimeout(() => { delete window.__x2pdfRunning; }, 8000);
  }
})();
