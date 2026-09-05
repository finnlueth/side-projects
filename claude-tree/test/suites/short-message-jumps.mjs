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
page.on('pageerror', (e) => problems.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
await page.goto(`http://127.0.0.1:8765${CHAT}?theme=dark`, { waitUntil: 'networkidle0' });
await wait(800);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1400);

// --- 1 · jumping to the shortest possible turns -----------------------------------
const jump = async (n, text) => {
  await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); });
  await wait(250);
  await page.evaluate(`(() => { const sel = ${S}.querySelector('.ct-node.is-selected');
    if (sel && sel.dataset.id === ${nodeSel(n)}.slice(16, -2)) sel.click(); })()`).catch(() => {});
  await page.evaluate(`${S}.querySelector(${nodeSel(n)}).click()`);
  await wait(250);
  await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
  await wait(3000);
  return page.evaluate(`(() => ({
    toast: ${S}.querySelector('[data-role="toast"]').textContent,
    onScreen: [...document.querySelectorAll('[data-testid="transcript-row"]')].some(e =>
      e.textContent.includes(${JSON.stringify(text)}) && e.getBoundingClientRect().top < 400
      && e.getBoundingClientRect().bottom > 0),
  }))()`);
};

const twoChars = await jump(22, 'ok');
check('1 · jumps to a two-character message', twoChars.toast === 'Scrolled to the message' && twoChars.onScreen, twoChars);
const shortHuman = await jump(20, 'ok thanks');
check('1 · jumps to a short human message', shortHuman.toast === 'Scrolled to the message' && shortHuman.onScreen, shortHuman);
const shortClaude = await jump(21, 'Any time.');
check('1 · jumps to a short Claude message', shortClaude.toast === 'Scrolled to the message' && shortClaude.onScreen, shortClaude);

check('1 · short messages are marked while visible too', await page.evaluate(`(() => {
  const marked = [...${S}.querySelectorAll('.ct-node.is-current')].map(n => n.dataset.id.slice(0, 8));
  return marked.includes('00000022'); })()`),
  await page.evaluate(`[...${S}.querySelectorAll('.ct-node.is-current')].map(n => n.dataset.id.slice(0,8))`));

// --- 2 · the branch switch uses Claude's own controls, with no reload --------------
await page.evaluate(() => { window.__noReload = true; });
const before = await page.evaluate(() => window.__harness.variant);
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(8000);
const switched = await page.evaluate(`(() => ({
  stillSamePage: window.__noReload === true,
  variant: window.__harness.variant,
  puts: JSON.parse(sessionStorage.getItem('ct-puts') || '[]').length,
  toast: ${S}.querySelector('[data-role="toast"]').textContent,
  renders: [...document.querySelectorAll('[data-testid="transcript-row"]')]
    .some(e => e.textContent.includes('contour tracking')),
}))()`);
check('2 · switches through Claude\'s own control', switched.variant === 1 && switched.stillSamePage, { before, ...switched });
check('2 · the chat re-renders the new branch', switched.renders, switched);
check('2 · no API write or reload was needed', switched.puts === 0, switched);

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
