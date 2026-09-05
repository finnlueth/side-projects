import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
const CHAT = '/chat/00000000-0000-4000-8000-0000000000ff';
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const nodeSel = (n) => `\`.ct-node[data-id="${String(n).padStart(8,'0')}-0000-4000-8000-000000000000"]\``;

const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 950, deviceScaleFactor: 2 });
page.on('pageerror', (e) => problems.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
await page.goto(`http://127.0.0.1:8765${CHAT}?theme=dark`, { waitUntil: 'networkidle0' });
await wait(800);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1200);

// --- one button, in the header, where the magnifier was ---------------------------
await page.evaluate(`${S}.querySelector(${nodeSel(2)}).click()`);
await wait(300);
const layout = await page.evaluate(`(() => {
  const head = ${S}.querySelector('.ct-detail-head');
  const goto = ${S}.querySelector('[data-detail="goto"]');
  const actions = [...${S}.querySelectorAll('.ct-detail-actions > *')].map(e => e.dataset.detail);
  return {
    onlyOneAction: ${S}.querySelectorAll('[data-detail="goto"]').length === 1,
    noMagnifier: !${S}.querySelector('[data-detail="locate"]'),
    noBodyButton: !${S}.querySelector('.ct-detail-body [data-detail="goto"]'),
    inHeader: head.contains(goto),
    firstInActions: actions[0] === 'goto',
    label: goto.textContent.trim(),
    hasIcon: !!goto.querySelector('svg'),
    oneLine: head.getBoundingClientRect().height < 50,
  };
})()`);
check('one action button, no magnifier, no body button',
  layout.onlyOneAction && layout.noMagnifier && layout.noBodyButton, layout);
check('it sits in the header where the magnifier was',
  layout.inHeader && layout.firstInActions && layout.oneLine, layout);
check('it carries the branch icon and a label', layout.hasIcon && layout.label.length > 5, layout);
check('on-branch message reads "Show in chat"', layout.label === 'Show in chat', layout.label);

// off-branch message keeps the original wording
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
check('off-branch message reads "Show this branch in the chat"', await page.evaluate(`
  ${S}.querySelector('[data-detail="goto"]').textContent.trim() === 'Show this branch in the chat'`),
  await page.evaluate(`${S}.querySelector('[data-detail="goto"]').textContent.trim()`));

// --- the one button does both jobs -------------------------------------------------
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(4000);
check('it switches branch for an off-branch message', await page.evaluate(`(() => ({
  variant: window.__harness.variant,
  toast: ${S}.querySelector('[data-role="toast"]').textContent }))()`)
  .then((v) => v.variant === 1 && v.toast.includes('Switched branch')),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// nodes 1-3 are on the shared prefix, so they are on screen whichever branch is showing
await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); });
await wait(250);
await page.evaluate(`${S}.querySelector(${nodeSel(1)}).click()`);
await wait(150);
await page.evaluate(`${S}.querySelector(${nodeSel(3)}).click()`);
await wait(250);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(5000);
check('and simply scrolls when it is already on the branch', await page.evaluate(`
  ${S}.querySelector('[data-role="toast"]').textContent === 'Scrolled to the message'`),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// double-click uses the same path
await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); });
await wait(250);
await page.evaluate(`(() => { const el = ${S}.querySelector(${nodeSel(2)});
  el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); })()`);
await wait(5000);
check('double-click runs the same action', await page.evaluate(`
  ${S}.querySelector('[data-role="toast"]').textContent.includes('Scrolled')`),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// --- showing a reply lands on the prompt that produced it ---------------------------
// JUMP_TO_PROMPT: a long answer scrolled to its own first line gives no clue what was asked,
// so the chat comes to rest on the question above it. Only the scrolling changes — the reply
// stays the message picked in the tree.
await page.evaluate(() => { const s = window.__harness.scroller;
  s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); });
await wait(300);
await page.evaluate(`${S}.querySelector(${nodeSel(9)}).click()`);   // one of Claude's replies
await wait(300);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(4000);
const landing = await page.evaluate(`(() => {
  const rows = [...document.querySelectorAll('[data-testid="transcript-row"]')];
  const at = (text) => {
    const row = rows.find((r) => r.textContent.includes(text));
    return row ? Math.round(row.getBoundingClientRect().top) : null;
  };
  return {
    prompt: at('Does the simple version ever overlap'),   // message 8, the question
    reply: at('Every node lands inside'),                 // message 9, its answer
    picked: (${S}.querySelector('.ct-node.is-selected') || {}).dataset?.id?.slice(0, 8),
  };
})()`);
check('showing a reply scrolls to the prompt above it',
  landing.prompt !== null && landing.prompt > -8 && landing.prompt < 260, landing);
check('and the reply is still the message picked in the tree',
  landing.picked === '00000009', landing);

await page.screenshot({ path: `${OUT}/d0-unified.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
