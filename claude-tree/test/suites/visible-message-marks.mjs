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
await wait(1400);

/** What the chat is showing vs what the tree has marked. */
const sample = async (scrollTop) => page.evaluate(`(async (top) => {
  const s = window.__harness.scroller;
  if (top !== null) { s.scrollTop = top; s.dispatchEvent(new Event('scroll')); }
  await new Promise(r => setTimeout(r, 250));
  const onScreen = [...document.querySelectorAll('.row')].filter(e => {
    const r = e.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.height;
  }).map(e => e.textContent.replace(/\\s+/g, ' ').trim().slice(0, 22));
  const marked = [...${S}.querySelectorAll('.ct-node.is-current')]
    .map(n => n.querySelector('.ct-node-text').textContent.replace(/\\s+/g, ' ').trim().slice(0, 22));
  return { onScreen, marked };
})(${scrollTop})`);

// a view holding several messages at once
const many = await sample(0);
check('every message on screen is marked', many.marked.length === many.onScreen.length
  && many.onScreen.every((t) => many.marked.some((m) => m.startsWith(t.slice(0, 15))
    || t.startsWith(m.slice(0, 15)))), many);
check('more than one is marked at a time', many.marked.length > 1, many);

// scrolling changes the set
const later = await sample(1500);
check('the set follows the scroll', later.marked.join('|') !== many.marked.join('|')
  && later.marked.length === later.onScreen.length, later);

// a view holding a single message marks only that one
const tall = await page.evaluate(`(async () => {
  const s = window.__harness.scroller, T = window.__harness.tops();
  s.scrollTop = T[3] + 200;                 // deep inside one long reply
  s.dispatchEvent(new Event('scroll'));
  await new Promise(r => setTimeout(r, 250));
  return { onScreen: [...document.querySelectorAll('.row')].filter(e => {
      const r = e.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.height; }).length,
    marked: ${S}.querySelectorAll('.ct-node.is-current').length };
})()`);
check('a single visible message marks only itself', tall.marked === tall.onScreen, tall);

// selection still wins over the visible outline
await page.evaluate(`(() => { const n = ${S}.querySelector('.ct-node.is-current'); n.click(); })()`);
await wait(400);
check('selection still overrides the visible outline', await page.evaluate(`(() => {
  const sel = ${S}.querySelector('.ct-node.is-selected');
  return sel.classList.contains('is-current')
    && getComputedStyle(sel).borderColor === 'rgb(217, 119, 87)'; })()`),
  await page.evaluate(`(() => { const sel = ${S}.querySelector('.ct-node.is-selected');
    return { current: sel.classList.contains('is-current'), border: getComputedStyle(sel).borderColor }; })()`));

// opening still lands on what the chat is showing
await page.evaluate(`${S}.querySelector('[data-action="close"]').click()`);
await wait(500);
await page.evaluate(() => { const s = window.__harness.scroller;
  s.scrollTop = window.__harness.tops()[5]; s.dispatchEvent(new Event('scroll')); });
await wait(300);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1500);
check('opening still centres on what the chat is showing', await page.evaluate(`(() => {
  const marks = [...${S}.querySelectorAll('.ct-node.is-current')];
  if (!marks.length) return false;
  const box = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
  const first = marks[0].getBoundingClientRect();
  return Math.abs((first.top + first.bottom) / 2 - (box.top + box.bottom) / 2) < 60; })()`),
  await page.evaluate(`${S}.querySelectorAll('.ct-node.is-current').length`));

await page.screenshot({ path: `${OUT}/e0-visible.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
