import puppeteer from '../support/puppeteer.mjs';
const CHAT = '/chat/00000000-0000-4000-8000-0000000000ff';
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const nodeSel = (n) => `\`.ct-node[data-id="${String(n).padStart(8,'0')}-0000-4000-8000-000000000000"]\``;

// --- 1 · colours ------------------------------------------------------------------
{
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 950 });
  page.on('pageerror', (e) => problems.push(e.message));
  await page.goto(`http://127.0.0.1:8765${CHAT}?theme=dark`, { waitUntil: 'networkidle0' });
  await wait(800);
  await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
  await wait(1200);
  const colours = await page.evaluate(`(() => {
    const human = ${S}.querySelector('.ct-node[data-sender="human"] .ct-node-who');
    const claude = ${S}.querySelector('.ct-node[data-sender="assistant"] .ct-node-who');
    return { human: getComputedStyle(human).color, claude: getComputedStyle(claude).color }; })()`);
  check('1 · Claude is the accent colour', colours.claude === 'rgb(217, 119, 87)', colours);
  check('1 · you are the slate colour', /^rgb\(1[0-9]{2}, 1[0-9]{2}, 1[0-9]{2}\)$/.test(colours.human)
    && colours.human !== colours.claude, colours);
  await page.evaluate(`${S}.querySelector(${nodeSel(1)}).click()`);
  await wait(250);
  check('1 · the drawer matches', await page.evaluate(`(() => {
    const who = ${S}.querySelector('.ct-detail-who');
    return who.dataset.sender === 'human' && getComputedStyle(who).color !== 'rgb(217, 119, 87)'; })()`));
  await page.close();
}

// --- 2 · a lagging read must not be reloaded onto ---------------------------------
for (const [label, lag] of [['no lag', 0], ['reads lag behind the write', 4], ['never settles', 99]]) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 950 });
  page.on('pageerror', (e) => problems.push(`[${label}] ${e.message}`));
  await page.goto(`http://127.0.0.1:8765${CHAT}?theme=dark&switcher=none&lag=${lag}`, { waitUntil: 'networkidle0' });
  await wait(800);
  await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
  await wait(1200);
  await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
  await wait(300);
  await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
  await page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 20000 }).catch(() => {});
  await wait(1500);

  const state = await page.evaluate(() => ({
    settledBeforeReload: sessionStorage.getItem('ct-settled') === '1',
    puts: JSON.parse(sessionStorage.getItem('ct-puts') || '[]').length,
    readsAtPut: Number(sessionStorage.getItem('ct-reads-at-put') || 0),
    open: window.__panel.open,
  }));
  if (lag === 99) {
    check('2 · gives up and reloads anyway if it never settles',
      !state.settledBeforeReload && state.puts === 1 && state.open === true, state);
  } else {
    check(`2 · ${label}: waits for the new branch to be readable before reloading`,
      state.settledBeforeReload && state.puts === 1 && state.open === true, state);
  }
  await page.close();
}

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
