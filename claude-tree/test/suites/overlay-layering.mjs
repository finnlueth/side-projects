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
await wait(1200);

// --- 1 · a focus-taking find bar that is not a dialog -----------------------------
const find = await page.evaluate(`(async () => {
  const host = document.querySelector('.ct-panel-host');
  const before = getComputedStyle(host).zIndex;
  window.__harness.findBar(true);
  await new Promise(r => setTimeout(r, 250));
  const during = getComputedStyle(host).zIndex;
  const bar = document.getElementById('findbar');
  const r = bar.getBoundingClientRect();
  const hit = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
  const onTop = bar.contains(hit) || hit === bar;
  window.__harness.findBar(false);
  await new Promise(r => setTimeout(r, 250));
  return { before, during, onTop, after: getComputedStyle(host).zIndex,
           barZ: getComputedStyle(bar).zIndex };
})()`);
check('1 · the find bar renders over the pane', find.onTop && Number(find.during) < Number(find.barZ), find);
check('1 · the pane stays above the chat while it is open', Number(find.during) > 100, find);
check('1 · and returns to the front when it closes', find.after === find.before, find);

// dialogs still work
const dlg = await page.evaluate(`(async () => {
  const host = document.querySelector('.ct-panel-host');
  window.__harness.dialog(true);
  await new Promise(r => setTimeout(r, 250));
  const during = getComputedStyle(host).zIndex;
  window.__harness.dialog(false);
  await new Promise(r => setTimeout(r, 250));
  return { during, after: getComputedStyle(host).zIndex };
})()`);
check('1 · dialogs still take precedence', Number(dlg.during) < 9999 && dlg.after === '2147483000', dlg);

// --- 2 · no orange frame is ever painted on a message -----------------------------
await page.evaluate(`${S}.querySelector(${nodeSel(1)}).click()`);
await wait(200);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(2800);
const marks = await page.evaluate(() => {
  const styled = [...document.querySelectorAll('.row')].filter((e) => e.getAttribute('style') || '')
    .map((e) => e.getAttribute('style'))
    .filter((s) => /box-shadow|217, 119, 87/.test(s));
  return { styled, scrolled: [...document.querySelectorAll('.row')]
    .some((e) => e.textContent.includes('Can you explain') && e.getBoundingClientRect().top < 400) };
});
check('2 · no outline is left on any message', marks.styled.length === 0, marks.styled);
check('2 · jumping still scrolls to the message', marks.scrolled, marks);

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
