import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
const browser = await puppeteer.launch();
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 2 });
const problems = [];
page.on('pageerror', (e) => problems.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
await page.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff', { waitUntil: 'networkidle0' });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
await wait(800);

const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const nodeSel = (n) => `\`.ct-node[data-id="${String(n).padStart(8,'0')}-0000-4000-8000-000000000000"]\``;
const checks = [];
const check = (name, pass, detail) => checks.push({ name, pass, detail });

// --- 4. icon spacing -------------------------------------------------------------
const icon = await page.evaluate(() => {
  const host = document.querySelector('.ct-toggle-host');
  const nb = document.querySelector('#grp button');
  const nr = nb.getBoundingClientRect();
  const br = host.shadowRoot.querySelector('.ct-toggle').getBoundingClientRect();
  return { sameBox: Math.round(nr.width) === Math.round(br.width) && Math.round(nr.height) === Math.round(br.height),
           sameRadius: getComputedStyle(nb).borderRadius === getComputedStyle(host.shadowRoot.querySelector('.ct-toggle')).borderRadius,
           margin: getComputedStyle(host).marginLeft + '/' + getComputedStyle(host).marginRight,
           gapLeft: Math.round(br.left - nr.right),
           gapRight: Math.round(document.querySelectorAll('#grp button')[1].getBoundingClientRect().left - br.right) };
});
check('icon matches neighbour box + radius', icon.sameBox && icon.sameRadius, icon);
check('icon spacing symmetric, no extra margin', icon.gapLeft === icon.gapRight && icon.margin === '0px/0px', icon);

await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(900);

// --- pane still docks and reflows ------------------------------------------------
const pane = await page.evaluate(() => {
  const r = document.querySelector('.ct-panel-host').getBoundingClientRect();
  const app = document.getElementById('app').getBoundingClientRect();
  return { top: Math.round(r.top), right: Math.round(innerWidth - r.right), overlap: app.right > r.left + 1,
           margin: getComputedStyle(document.body).marginRight, host: document.querySelector('.ct-panel-host').parentElement.tagName };
});
check('pane inset + page reflowed, no overlap', pane.top === 8 && pane.right === 8 && !pane.overlap && pane.host === 'HTML', pane);

// --- 1. jump to message ----------------------------------------------------------
await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = s.scrollHeight; s.dispatchEvent(new Event('scroll')); });
await wait(400);
const gone = await page.evaluate(() => ![...document.querySelectorAll('.row')].some((e) => e.textContent.includes('Can you explain')));
check('first message is virtualized out of the DOM', gone);

const locate = async (n) => {
  await page.evaluate(`${S}.querySelector(${nodeSel(n)}).click()`);
  await wait(200);
  await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
  await wait(2800);
  return page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`);
};
check('finds a far, unrendered message', (await locate(1)) === 'Scrolled to the message');
check('finds a markdown-formatted message', (await locate(4)) === 'Scrolled to the message');
check('reaches a message on another branch', (await locate(7)).includes('Switched branch'));

// --- 2. reading outline ----------------------------------------------------------
await page.evaluate(`${S}.querySelector('[data-detail="close"]').click()`);
await page.evaluate(() => { const s = window.__harness.scroller;
  s.scrollTop = window.__harness.tops()[2] - innerHeight * 0.28 + 60; s.dispatchEvent(new Event('scroll')); });
await wait(600);
const reading = await page.evaluate(`(() => {
  const marks = [...${S}.querySelectorAll('.ct-node.is-current')];
  if (!marks.length) return { count: 0 };
  return { count: marks.length,
           border: getComputedStyle(marks[0]).borderColor,
           matches: marks.some(m => m.textContent.includes('recursive')) };
})()`);
check('the visible messages are marked, including the one on screen', reading.count >= 1 && reading.matches, reading);

await page.evaluate(`${S}.querySelector('.ct-node.is-current').click()`);
await wait(400);
const prec = await page.evaluate(`(() => { const c = ${S}.querySelector('.ct-node.is-current');
  return { both: c.classList.contains('is-selected'), border: getComputedStyle(c).borderColor }; })()`);
check('selection outline overrides the reading outline', prec.both && prec.border === 'rgb(217, 119, 87)', prec);

// --- previews are rendered, not raw markdown -------------------------------------
const previews = await page.evaluate(`[...${S}.querySelectorAll('.ct-node-text')].map(e => e.textContent)`);
check('no raw markdown in node previews', !previews.some((t) => /\*\*|^##|\]\(/.test(t)), previews.slice(0, 3));

// --- 3. auto-update --------------------------------------------------------------
const before = await page.evaluate(`${S}.querySelectorAll('.ct-node').length`);
await page.evaluate(() => {
  window.__harness.addMessage(10, 9, 'human', 'One more question about contours.');
  window.__harness.addMessage(11, 10, 'assistant', 'Contours let sibling subtrees interleave.');
  const s = window.__harness.scroller;          // Claude follows a new exchange down
  s.scrollTop = s.scrollHeight; s.dispatchEvent(new Event('scroll'));
});
await wait(4500);
const after = await page.evaluate(`${S}.querySelectorAll('.ct-node').length`);
check('tree auto-updates after a new exchange', after === before + 2, { before, after });

// --- closing cleans up -----------------------------------------------------------
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(400);
const closed = await page.evaluate(() => ({ margin: getComputedStyle(document.body).marginRight,
  style: !!document.getElementById('claude-tree-pane-style'), classes: document.documentElement.className }));
check('closing restores the page', closed.margin === '0px' && !closed.style && !closed.classes, closed);

await page.screenshot({ path: `${OUT}/40-final.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.name}${c.pass ? '' : '  ' + JSON.stringify(c.detail)}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
