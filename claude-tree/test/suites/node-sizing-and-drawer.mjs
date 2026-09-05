import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 950, deviceScaleFactor: 2 });
page.on('pageerror', (e) => problems.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
await page.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark', { waitUntil: 'networkidle0' });
await wait(700);

// --- 4 · slide-in from the edge ---------------------------------------------------
const anim = await page.evaluate(`(async () => {
  const root = () => document.querySelector('.ct-panel-host')?.shadowRoot?.querySelector('.ct-root');
  const toggle = document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle');
  toggle.click();
  await new Promise(r => setTimeout(r, 1200));           // let it settle, then close
  const open = getComputedStyle(root()).translate;
  const transition = getComputedStyle(root()).transitionProperty;
  toggle.click();
  await new Promise(r => setTimeout(r, 600));            // settled closed, off to the side
  const closed = getComputedStyle(root()).translate;
  toggle.click();
  await new Promise(r => setTimeout(r, 1200));
  return { open, closed, transition, reopened: getComputedStyle(root()).translate };
})()`);
check('4 · pane slides in from the edge and settles at rest',
  anim.transition.includes('translate') && anim.open === '0px' && anim.reopened === '0px'
  && anim.closed !== '0px', anim);
await wait(400);

// --- 1 · 100% default, and no padding on short nodes ------------------------------
const sizing = await page.evaluate(`(() => {
  const nodes = [...${S}.querySelectorAll('.ct-node')];
  const heights = nodes.map(n => n.offsetHeight);
  const lines = nodes.map(n => {
    const t = n.querySelector('.ct-node-text');
    return +(t.offsetHeight / parseFloat(getComputedStyle(t).lineHeight)).toFixed(2);
  });
  const rows = {};
  for (const n of nodes) { const y = Math.round(parseFloat(n.style.top)); (rows[y] ||= []).push(n.offsetHeight); }
  return { zoom: ${S}.querySelector('[data-role="zoom"]').textContent,
           widths: [...new Set(nodes.map(n => n.offsetWidth))],
           minH: Math.min(...heights), maxH: Math.max(...heights),
           maxLines: Math.max(...lines), overFour: lines.filter(l => l > 4.05).length,
           rowCount: Object.keys(rows).length };
})()`);
check('1 · default zoom is 100%', sizing.zoom === '100%', sizing.zoom);
check('1 · short nodes are not padded to four lines', sizing.minH < sizing.maxH, sizing);
check('1 · four lines is a maximum', sizing.overFour === 0 && sizing.maxLines <= 4.05, sizing);
check('1 · nodes keep the fixed width', sizing.widths.length === 1 && sizing.widths[0] === 315, sizing.widths);

// no overlap on screen
check('1 · no nodes overlap after measuring', await page.evaluate(`(() => {
  const ns = [...${S}.querySelectorAll('.ct-node')].map(n => ({
    x: parseFloat(n.style.left), y: parseFloat(n.style.top), w: n.offsetWidth, h: n.offsetHeight }));
  for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
    const a = ns[i], b = ns[j];
    if (a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h) return false;
  }
  return true; })()`));

// --- 3 · resizable drawer ----------------------------------------------------------
await page.evaluate(`${S}.querySelector('.ct-node').click()`);
await wait(300);
const before = await page.evaluate(`${S}.querySelector('.ct-detail').getBoundingClientRect().height`);
const handle = await page.evaluate(`(() => { const r = ${S}.querySelector('.ct-detail-resize').getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()`);
await page.mouse.move(handle.x, handle.y);
await page.mouse.down();
await page.mouse.move(handle.x, handle.y - 200, { steps: 12 });
await page.mouse.up();
await wait(300);
const after = await page.evaluate(`${S}.querySelector('.ct-detail').getBoundingClientRect().height`);
check('3 · drawer can be dragged taller', after > before + 100, { before: Math.round(before), after: Math.round(after) });
check('3 · drawer has a default maximum', await page.evaluate(`(() => {
  const root = ${S}.querySelector('.ct-root').getBoundingClientRect().height;
  return ${S}.querySelector('.ct-detail').getBoundingClientRect().height <= root * 0.81; })()`));

// --- 2 · reading mark at the midpoint, short messages included --------------------
await page.evaluate(`${S}.querySelector('[data-detail="close"]').click()`);
const atFocus = await page.evaluate(`(async () => {
  const s = window.__harness.scroller, T = window.__harness.tops();
  s.scrollTop = T[2] + s.getBoundingClientRect().top - innerHeight * 0.3 + 12;
  s.dispatchEvent(new Event('scroll'));
  await new Promise(r => setTimeout(r, 500));
  s.scrollTop = s.scrollTop; s.dispatchEvent(new Event('scroll'));  // cancel any smooth scroll
  await new Promise(r => setTimeout(r, 250));
  return [...${S}.querySelectorAll('.ct-node.is-current')]
    .map(n => n.textContent).join(' | ');
})()`);
check('2 · the visible messages are marked',
  Boolean(atFocus && atFocus.includes('recursive')), atFocus);

check('2 · marks match the messages on screen', await page.evaluate(`(() => {
  const onScreen = [...document.querySelectorAll('.row')].filter(e => {
    const r = e.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.height; }).length;
  return ${S}.querySelectorAll('.ct-node.is-current').length === onScreen; })()`),
  await page.evaluate(`${S}.querySelectorAll('.ct-node.is-current').length`));

await page.screenshot({ path: `${OUT}/90-final.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
