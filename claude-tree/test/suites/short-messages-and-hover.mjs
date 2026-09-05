import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
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
await page.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark&hoverswitch=1', { waitUntil: 'networkidle0' });
await wait(700);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1200);

// --- 5 · short messages, both senders, are indexed and highlighted ----------------
const shortMarks = await page.evaluate(`(async () => {
  const s = window.__harness.scroller, T = window.__harness.tops(), P = window.__harness.path();
  const out = {};
  for (const [label, msg] of [['human', 20], ['assistant', 21]]) {
    const i = P.indexOf(msg);
    // the scroller starts below the header, so its own offset counts toward the focus line
    const inset = s.getBoundingClientRect().top;
    s.scrollTop = T[i] + inset - innerHeight * 0.3 + 12;
    s.dispatchEvent(new Event('scroll'));
    await new Promise(r => setTimeout(r, 500));
    s.scrollTop = s.scrollTop; s.dispatchEvent(new Event('scroll'));
    await new Promise(r => setTimeout(r, 250));
    out[label] = [...${S}.querySelectorAll('.ct-node.is-current')]
      .map(n => n.dataset.id.slice(0, 8));
  }
  return out;
})()`);
check('5 · a short human message can be marked', shortMarks.human.includes('00000020'), shortMarks);
check('5 · a short Claude message can be marked', shortMarks.assistant.includes('00000021'), shortMarks);

// and they are reachable by find-in-chat too
await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); });
await wait(200);
await page.evaluate(`${S}.querySelector(${nodeSel(21)}).click()`);
await wait(200);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(2800);
check('5 · a short message can be jumped to', await page.evaluate(`
  ${S}.querySelector('[data-role="toast"]').textContent === 'Scrolled to the message'`),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// --- 6 · the mark follows a line near the top -------------------------------------
const topRule = await page.evaluate(`(async () => {
  const s = window.__harness.scroller, T = window.__harness.tops();
  s.scrollTop = T[3] + s.getBoundingClientRect().top - innerHeight * 0.3 + 200;
  s.dispatchEvent(new Event('scroll'));
  await new Promise(r => setTimeout(r, 500));
  s.scrollTop = s.scrollTop; s.dispatchEvent(new Event('scroll'));  // cancel any smooth scroll
  await new Promise(r => setTimeout(r, 250));
  const onScreen = [...document.querySelectorAll('.row')].filter(e => {
    const r = e.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.height; });
  const marks = [...${S}.querySelectorAll('.ct-node.is-current')];
  return { agrees: marks.length === onScreen.length && marks.length > 0,
           marks: marks.length, onScreen: onScreen.length };
})()`);
check('6 · the marked set matches what is on screen', topRule.agrees, topRule);
check('6 · marks match the messages on screen', await page.evaluate(`(() => {
  const onScreen = [...document.querySelectorAll('.row')].filter(e => {
    const r = e.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.height; }).length;
  return ${S}.querySelectorAll('.ct-node.is-current').length === onScreen; })()`),
  await page.evaluate(`${S}.querySelectorAll('.ct-node.is-current').length`));

// --- 7 · a hover-only switcher with nested markup is still found -------------------
const noSwitcherYet = await page.evaluate(() => !document.querySelector('.switch'));
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(4000);
const switched = await page.evaluate(`(() => ({
  variant: window.__harness.variant,
  toast: ${S}.querySelector('[data-role="toast"]').textContent }))()`);
check('7 · finds a switcher that only appears on hover', noSwitcherYet && switched.variant === 1
  && switched.toast .includes('Switched branch'), { noSwitcherYet, ...switched });

await page.screenshot({ path: `${OUT}/b0-final.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
