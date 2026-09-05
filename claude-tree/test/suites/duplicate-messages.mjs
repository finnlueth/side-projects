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

// --- identical messages must not collapse onto one another ------------------------
const dupes = await page.evaluate(`(async () => {
  const s = window.__harness.scroller, T = window.__harness.tops(), P = window.__harness.path();
  s.scrollTop = T[P.indexOf(22)] - 60;        // both "ok" messages on screen together
  s.dispatchEvent(new Event('scroll'));
  await new Promise(r => setTimeout(r, 350));
  const rows = [...document.querySelectorAll('[data-testid="transcript-row"]')];
  const onScreen = rows.filter(e => { const r = e.getBoundingClientRect();
    return r.bottom > 0 && r.top < innerHeight && r.height; });
  const marked = [...${S}.querySelectorAll('.ct-node.is-current')].map(n => n.dataset.id.slice(0, 8));
  return { onScreen: onScreen.length, marked,
    bothOks: marked.includes('00000022') && marked.includes('00000024'),
    okRowsVisible: onScreen.filter(e => e.textContent.startsWith('ok')).length };
})()`);
check('identical messages are marked separately', dupes.bothOks && dupes.marked.length === dupes.onScreen, dupes);

// jumping distinguishes them too
const jumpTo = async (n) => {
  await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); });
  await wait(250);
  await page.evaluate(`${S}.querySelector(${nodeSel(n)}).click()`);
  await wait(250);
  await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
  await wait(3500);
  return page.evaluate(`(() => {
    const s = window.__harness.scroller, T = window.__harness.tops(), P = window.__harness.path();
    const i = P.indexOf(${n});
    const row = [...document.querySelectorAll('[data-testid="transcript-row"]')]
      .find(e => Math.abs(parseFloat(e.style.top) - T[i]) < 2);
    const atMax = s.scrollTop >= s.scrollHeight - s.clientHeight - 2;
    const offset = row ? row.getBoundingClientRect().top - s.getBoundingClientRect().top : null;
    return { toast: ${S}.querySelector('[data-role="toast"]').textContent,
             atTop: row ? (Math.abs(offset) < 40 || (atMax && offset >= 0 && offset < s.clientHeight)) : false };
  })()`);
};
const first = await jumpTo(22);
check('jumps to the first of two identical messages', first.toast === 'Scrolled to the message' && first.atTop, first);
const second = await jumpTo(24);
check('jumps to the second of two identical messages', second.toast === 'Scrolled to the message' && second.atTop, second);

// --- every visible row is accounted for, none skipped -----------------------------
const coverage = await page.evaluate(`(async () => {
  const s = window.__harness.scroller;
  const out = [];
  for (const top of [0, 800, 1600, 2600, s.scrollHeight]) {
    s.scrollTop = top; s.dispatchEvent(new Event('scroll'));
    await new Promise(r => setTimeout(r, 220));
    const onScreen = [...document.querySelectorAll('[data-testid="transcript-row"]')]
      .filter(e => { const r = e.getBoundingClientRect();
        return r.bottom > 0 && r.top < innerHeight && r.height; }).length;
    out.push({ onScreen, marked: ${S}.querySelectorAll('.ct-node.is-current').length });
  }
  return out;
})()`);
check('no visible message is ever left unmarked',
  coverage.every((c) => c.marked === c.onScreen && c.onScreen > 0), coverage);

// --- branch switching -------------------------------------------------------------
await page.evaluate(() => { window.__noReload = true; });
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(8000);
const switched = await page.evaluate(`(() => ({
  noReload: window.__noReload === true,
  variant: window.__harness.variant,
  puts: JSON.parse(sessionStorage.getItem('ct-puts') || '[]').length,
  toast: ${S}.querySelector('[data-role="toast"]').textContent,
  renders: [...document.querySelectorAll('[data-testid="transcript-row"]')]
    .some(e => e.textContent.includes('contour tracking')),
}))()`);
check('switches branch through the page, no reload', switched.variant === 1 && switched.noReload
  && switched.puts === 0 && switched.renders, switched);

// and the marks follow the new branch once the tree reloads
await wait(2500);
check('marks follow onto the new branch', await page.evaluate(`(() => {
  const onScreen = [...document.querySelectorAll('[data-testid="transcript-row"]')]
    .filter(e => { const r = e.getBoundingClientRect();
      return r.bottom > 0 && r.top < innerHeight && r.height; }).length;
  return ${S}.querySelectorAll('.ct-node.is-current').length === onScreen; })()`),
  await page.evaluate(`${S}.querySelectorAll('.ct-node.is-current').length`));

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
