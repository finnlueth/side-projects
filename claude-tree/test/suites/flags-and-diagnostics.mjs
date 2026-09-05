import puppeteer from '../support/puppeteer.mjs';
const CHAT = '/chat/00000000-0000-4000-8000-0000000000ff';
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const nodeSel = (n) => `\`.ct-node[data-id="${String(n).padStart(8,'0')}-0000-4000-8000-000000000000"]\``;

const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 950, deviceScaleFactor: 2 });
const logs = [];
page.on('console', (m) => { logs.push(`${m.type()}:${m.text()}`);
  if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
page.on('pageerror', (e) => problems.push(e.message));
await page.goto(`http://127.0.0.1:8765${CHAT}?theme=dark`, { waitUntil: 'networkidle0' });
await wait(800);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1200);

// --- 1 · stats pills off, code intact --------------------------------------------
check('1 · summary pills are hidden', await page.evaluate(`(() => {
  const row = ${S}.querySelector('[data-role="stats"]');
  return row.hidden && ${S}.querySelectorAll('.ct-stat').length === 0; })()`));
check('1 · the tree itself is unaffected', await page.evaluate(`${S}.querySelectorAll('.ct-node').length >= 9`));

// --- 2 · toasts back on ----------------------------------------------------------
await page.evaluate(`${S}.querySelector(${nodeSel(1)}).click()`);
await wait(200);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(400);                       // check while it is still on screen
check('2 · toasts are visible again', await page.evaluate(`(() => {
  const t = ${S}.querySelector('[data-role="toast"]');
  return t.classList.contains('is-visible') && t.textContent.length > 0; })()`),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// --- 3 · diagnostics -------------------------------------------------------------
check('3 · logs what the API returned', logs.some((l) => /claude-tree\] \d+ messages, \d+ forks via/.test(l)),
  logs.filter((l) => l.includes('claude-tree')).slice(0, 2));

// a failing PUT must say why
await page.evaluate(() => {
  window.fetch = async (url, init) => (String(url).includes('/current_leaf_message_uuid')
    ? new Response('{}', { status: 403 })
    : new Response('{}', { status: 200 }));
});
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
await page.evaluate(() => {   // remove the in-page switcher so the API path is taken
  document.querySelectorAll('.switch').forEach((e) => e.remove());
  document.querySelectorAll('.row').forEach((r) => r.replaceWith(r.cloneNode(true)));
});
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(3000);
check('3 · a rejected branch switch names the reason', await page.evaluate(`
  ${S}.querySelector('[data-role="toast"]').textContent`).then((t) => t.includes('HTTP 403')),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// a conversation the page is branching but the API returned flat must be called out
const flat = await browser.newPage();
await flat.setViewport({ width: 1400, height: 900 });
const flatLogs = [];
flat.on('console', (m) => flatLogs.push(m.text()));
await flat.goto(`http://127.0.0.1:8765${CHAT}?theme=dark&flat=1`, { waitUntil: 'networkidle0' });
await flat.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await new Promise((r) => setTimeout(r, 1500));
check('3 · calls out a tree the API flattened',
  flatLogs.some((l) => l.includes('only sending the branch on screen')),
  flatLogs.filter((l) => l.includes('claude-tree')).slice(0, 1));
await flat.close();

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
