import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const nodeSel = (n) => `\`.ct-node[data-id="${String(n).padStart(8,'0')}-0000-4000-8000-000000000000"]\``;

const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 2 });
page.on('pageerror', (e) => problems.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
await page.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark', { waitUntil: 'networkidle0' });
await wait(700);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(900);

// --- borders are hairlines again, not solid light outlines -----------------------
const borders = await page.evaluate(`(() => {
  const plain = ${S}.querySelector('.ct-node:not(.is-current):not(.is-selected)');
  const alpha = (c) => { const m = /rgba?\\(([^)]+)\\)/.exec(c); const p = m[1].split(',').map(Number); return p.length > 3 ? p[3] : 1; };
  return { plain: getComputedStyle(plain).borderColor, plainAlpha: alpha(getComputedStyle(plain).borderColor),
           outlined: ${S}.querySelectorAll('.ct-node.is-current, .ct-node.is-selected').length,
           total: ${S}.querySelectorAll('.ct-node').length };
})()`);
check('cards use a hairline border, not a solid light outline', borders.plainAlpha < 0.35, borders);
check('the reading outline is a hairline apart from the marked set',
  borders.outlined >= 1 && borders.plainAlpha < 0.35, borders);

// --- no rules across the pane ----------------------------------------------------
check('no dividing rules above the canvas or drawer', await page.evaluate(`(() => {
  const w = (el) => getComputedStyle(el).borderTopWidth;
  return w(${S}.querySelector('.ct-canvas')) === '0px'; })()`));

// --- sender colours are back on the names ----------------------------------------
check('Claude takes the accent, you take the slate', await page.evaluate(`(() => {
  const h = ${S}.querySelector('.ct-node[data-sender="human"] .ct-node-who');
  const a = ${S}.querySelector('.ct-node[data-sender="assistant"] .ct-node-who');
  return getComputedStyle(h).color !== getComputedStyle(a).color
      && getComputedStyle(a).color === 'rgb(217, 119, 87)'; })()`));

// --- jumping across a branch ------------------------------------------------------
const before = await page.evaluate(() => window.__harness.variant);
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(200);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(4000);
const jump = await page.evaluate(`(() => ({
  toast: ${S}.querySelector('[data-role="toast"]').textContent,
  variant: window.__harness.variant,
  onScreen: [...document.querySelectorAll('.row')].some(e => e.textContent.includes('Tighter packing')),
}))()`);
check('jump reaches a message on another branch', jump.toast.includes('Switched branch') && jump.variant === 1 && jump.onScreen, { before, ...jump });

// --- and back the other way -------------------------------------------------------
await page.evaluate(`${S}.querySelector(${nodeSel(9)}).click()`);
await wait(200);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(4000);
check('and back to the other branch', await page.evaluate(`(() => {
  const t = ${S}.querySelector('[data-role="toast"]').textContent;
  return (t.includes('Switched branch') || t === 'Scrolled to the message') && window.__harness.variant === 0;
})()`), await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// --- dwell survived ---------------------------------------------------------------
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
check('a short prompt is among the marked messages when on screen',
  Boolean(shortMark && shortMark.includes('recursive')), shortMark);
check('marks match the messages on screen', await page.evaluate(`(() => {
  const onScreen = [...document.querySelectorAll('.row')].filter(e => {
    const r = e.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.height; }).length;
  return ${S}.querySelectorAll('.ct-node.is-current').length === onScreen; })()`),
  await page.evaluate(`${S}.querySelectorAll('.ct-node.is-current').length`));

await page.screenshot({ path: `${OUT}/70-fixed.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
