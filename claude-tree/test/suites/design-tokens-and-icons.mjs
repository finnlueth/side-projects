import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';

async function open(url, viewport = { width: 1400, height: 900, deviceScaleFactor: 2 }) {
  const page = await browser.newPage();
  await page.setViewport(viewport);
  page.on('pageerror', (e) => problems.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404') && !m.text().includes('nonexistent')) problems.push(m.text()); });
  await page.goto(url, { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 700));
  await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
  await new Promise((r) => setTimeout(r, 900));
  return page;
}

// ============ 7: tokens + icon font ============
let page = await open('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?iconfont=1&theme=dark');
check('7 · icons come from Claude\'s font, not our SVG', await page.evaluate(`(() => {
  const refresh = ${S}.querySelector('[data-action="refresh"] .ct-glyph');
  const close = ${S}.querySelector('[data-action="close"] .ct-glyph');
  return !!refresh && refresh.textContent === '\\ue11d' && !!close;
})()`));
check('7 · close glyph harvested from the live page', await page.evaluate(`(() => {
  const shown = ${S}.querySelector('[data-action="close"] .ct-glyph')?.textContent;
  const source = document.querySelector('[aria-label="Close"] [data-cds="Icon"]').textContent;
  return !!shown && shown === source;
})()`));

check('7 · colours, metrics and type resolve from claude.ai tokens', await page.evaluate(`(() => {
  const cs = getComputedStyle(${S}.querySelector('.ct-root'));
  const page = getComputedStyle(document.body).backgroundColor;
  return {
    raisedAbovePage: cs.backgroundColor !== page,
    canvasMatchesChat: getComputedStyle(${S}.querySelector('.ct-canvas')).backgroundColor === page,
    radius: cs.borderRadius === '12px',
    control: getComputedStyle(${S}.querySelector('.ct-icon-btn')).height === '32px',
    font: cs.fontFamily.includes('page-sans'),
    accent: getComputedStyle(${S}.querySelector('.ct-edge.is-path')).stroke === 'rgb(217, 119, 87)',
  };
})()`).then((v) => Object.values(v).every(Boolean)));

// live token change must flow through
check('7 · follows a live token change', await page.evaluate(`(async () => {
  document.documentElement.style.setProperty('--accent-brand', '210 70% 52%');
  await new Promise(r => setTimeout(r, 60));
  return getComputedStyle(${S}.querySelector('.ct-edge.is-path')).stroke;
})()`).then((v) => v !== 'rgb(217, 119, 87)'));
await page.evaluate(() => document.documentElement.style.removeProperty('--accent-brand'));

// ============ 5: no header ============
check('5 · title header gone, actions kept', await page.evaluate(`(() => {
  return !${S}.querySelector('.ct-head, .ct-title, .ct-subtitle')
      && !!${S}.querySelector('.ct-bar [data-action="refresh"]')
      && !!${S}.querySelector('.ct-bar [data-action="close"]'); })()`));

// ============ 8: no decorative bars ============
check('8 · node accent bars removed', await page.evaluate(`
  ${S}.querySelectorAll('.ct-node-bar').length === 0`));
check('8 · sender shown by fill, not decoration', await page.evaluate(`(() => {
  const h = ${S}.querySelector('.ct-node[data-sender="human"]');
  const a = ${S}.querySelector('.ct-node[data-sender="assistant"]');
  return getComputedStyle(h).backgroundColor !== getComputedStyle(a).backgroundColor; })()`));

await page.screenshot({ path: `${OUT}/50-dark.png` });
await page.close();

// ============ 6 + 9 ============
page = await open('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark');
// 6: jump lands on the top of the message
await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = s.scrollHeight; s.dispatchEvent(new Event('scroll')); });
await new Promise((r) => setTimeout(r, 300));
await page.evaluate(`${S}.querySelector('.ct-node').click()`);
await new Promise((r) => setTimeout(r, 200));
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await new Promise((r) => setTimeout(r, 3000));
check('6 · jump lands at the top of the message', await page.evaluate(() => {
  const s = window.__harness.scroller;
  const el = [...document.querySelectorAll('.row')].find((e) => e.textContent.includes('Can you explain'));
  if (!el) return false;
  const offset = el.getBoundingClientRect().top - s.getBoundingClientRect().top;
  return offset >= 0 && offset < 40;   // top-aligned, not centred
}), await page.evaluate(() => {
  const s = window.__harness.scroller;
  const el = [...document.querySelectorAll('.row')].find((e) => e.textContent.includes('Can you explain'));
  return el ? Math.round(el.getBoundingClientRect().top - s.getBoundingClientRect().top) : null;
}));

// 9: a short prompt keeps the mark across a long stretch of scrolling
await page.evaluate(`${S}.querySelector('[data-detail="close"]').click()`);
const shortMark = await page.evaluate(`(async () => {
  const s = window.__harness.scroller, T = window.__harness.tops();
  s.scrollTop = T[2] + s.getBoundingClientRect().top - innerHeight * 0.3 + 12;
  s.dispatchEvent(new Event('scroll'));
  await new Promise(r => setTimeout(r, 500));
  s.scrollTop = s.scrollTop; s.dispatchEvent(new Event('scroll'));  // cancel any smooth scroll
  await new Promise(r => setTimeout(r, 250));
  return [...${S}.querySelectorAll('.ct-node.is-current')]
    .map(n => n.textContent).join(' | ');
})()`);
check('9 · a short prompt is among the marked messages when on screen',
  Boolean(shortMark && shortMark.includes('recursive')), shortMark);

await page.screenshot({ path: `${OUT}/51-reading.png` });
await page.close();

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
