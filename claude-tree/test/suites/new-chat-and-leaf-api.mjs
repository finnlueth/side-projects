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

// ---- 2 · notifications off --------------------------------------------------------
await page.goto(`http://127.0.0.1:8765${CHAT}?theme=dark`, { waitUntil: 'networkidle0' });
await wait(800);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1200);
await page.evaluate(`${S}.querySelector(${nodeSel(1)}).click()`);
await wait(200);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(400);
check('2 · toasts report what happened', await page.evaluate(`(() => {
  const t = ${S}.querySelector('[data-role="toast"]');
  return t.classList.contains('is-visible') && t.textContent.includes('Scrolled'); })()`),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// ---- 1 · a new chat closes the pane and takes the button away ----------------------
check('1 · button is present inside a conversation', await page.evaluate(`
  !document.querySelector('.ct-toggle-host').hidden`));
await page.evaluate(() => history.pushState(null, '', '/new'));
await wait(900);
const onNew = await page.evaluate(`(() => ({
  buttonHidden: document.querySelector('.ct-toggle-host').hidden,
  paneHidden: document.querySelector('.ct-panel-host').hidden,
  paneOpen: window.__panel.open }))()`);
check('1 · starting a new chat closes the pane', onNew.paneHidden && !onNew.paneOpen, onNew);
check('1 · and removes the button from the page', onNew.buttonHidden, onNew);
await page.evaluate((c) => history.pushState(null, '', c), CHAT);
await wait(900);
check('1 · the button comes back in a conversation', await page.evaluate(`
  !document.querySelector('.ct-toggle-host').hidden`));

// ---- 3 · the API path when no in-page switcher exists ------------------------------
const page2 = await browser.newPage();
await page2.setViewport({ width: 1500, height: 950 });
page2.on('pageerror', (e) => problems.push(`[api] ${e.message}`));
await page2.goto(`http://127.0.0.1:8765${CHAT}?theme=dark&switcher=none`, { waitUntil: 'networkidle0' });
await wait(800);
await page2.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1200);
check('3 · no in-page switcher in this mode', await page2.evaluate(() => !document.querySelector('.switch')));

await page2.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
await page2.evaluate(() => { window.__beforeReload = true; });
await page2.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await page2.waitForNavigation({ waitUntil: 'networkidle0', timeout: 15000 }).catch(() => {});
await wait(1500);

const put = await page2.evaluate(() => JSON.parse(sessionStorage.getItem('ct-puts') || '[]'));
check('3 · issues the PUT claude.ai uses to move the branch',
  put.length === 1 && put[0].method === 'PUT' && put[0].url.includes('/current_leaf_message_uuid'), put[0] || null);
check('3 · sends the deepest message of that branch',
  put[0] && JSON.parse(put[0].body).current_leaf_message_uuid === '00000007-0000-4000-8000-000000000000',
  put[0] && put[0].body);
const afterReload = await page2.evaluate(`(() => ({
  reloaded: window.__beforeReload === undefined,
  open: window.__panel.open,
  hidden: document.querySelector('.ct-panel-host')?.hidden }))()`);
check('3 · reloads the page and brings the pane back open',
  afterReload.reloaded && afterReload.open === true && afterReload.hidden === false, afterReload);
check('3 · the one-shot resume flag is cleared', await page2.evaluate(() => !sessionStorage.getItem('ct-resume')));

await page2.screenshot({ path: `${OUT}/c0-api.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
