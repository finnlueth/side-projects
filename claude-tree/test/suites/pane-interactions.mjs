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
await page.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark', { waitUntil: 'networkidle0' });
await wait(700);

// park the chat mid-conversation before opening, so item 4 has something to centre on
await page.evaluate(() => { const s = window.__harness.scroller;
  s.scrollTop = window.__harness.tops()[3] - innerHeight * 0.5 + 300; s.dispatchEvent(new Event('scroll')); });
await wait(200);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1200);

// --- 4 · opens centred on the message the chat is showing -------------------------
check('4 · opens centred on the active message', await page.evaluate(`(() => {
  const cur = ${S}.querySelector('.ct-node.is-current');
  if (!cur) return false;
  const c = cur.getBoundingClientRect(), box = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
  return Math.abs((c.top + c.bottom) / 2 - (box.top + box.bottom) / 2) < 40
      && Math.abs((c.left + c.right) / 2 - (box.left + box.right) / 2) < 40;
})()`), await page.evaluate(`(() => { const c = ${S}.querySelector('.ct-node.is-current');
  return c ? c.textContent.trim().slice(0, 26) : null; })()`));

// --- 1 · pinch sensitivity --------------------------------------------------------
const pinch = await page.evaluate(`(async () => {
  const canvas = ${S}.querySelector('.ct-canvas');
  const box = canvas.getBoundingClientRect();
  const before = parseInt(${S}.querySelector('[data-role="zoom"]').textContent, 10);
  canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: -10, ctrlKey: true, bubbles: true, cancelable: true,
    clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 }));
  await new Promise(r => setTimeout(r, 80));
  return { before, after: parseInt(${S}.querySelector('[data-role="zoom"]').textContent, 10) };
})()`);
check('1 · one pinch step moves the zoom appreciably', pinch.after - pinch.before >= 8, pinch);

const wheelNotch = await page.evaluate(`(async () => {
  const canvas = ${S}.querySelector('.ct-canvas');
  const box = canvas.getBoundingClientRect();
  const before = parseInt(${S}.querySelector('[data-role="zoom"]').textContent, 10);
  canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: -120, ctrlKey: true, bubbles: true, cancelable: true,
    clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 }));
  await new Promise(r => setTimeout(r, 80));
  return { before, after: parseInt(${S}.querySelector('[data-role="zoom"]').textContent, 10) };
})()`);
check('1 · a mouse wheel notch stays controlled', wheelNotch.after / wheelNotch.before <= 1.3, wheelNotch);
await page.evaluate(`${S}.querySelector('[data-action="fit"]').click()`);
await wait(200);

// --- 3 · click the same node again to close the drawer ---------------------------
await page.evaluate(`${S}.querySelector(${nodeSel(2)}).click()`);
await wait(250);
const opened = await page.evaluate(`!${S}.querySelector('.ct-detail').hidden`);
await page.evaluate(`${S}.querySelector(${nodeSel(2)}).click()`);
await wait(250);
const closed = await page.evaluate(`${S}.querySelector('.ct-detail').hidden`);
check('3 · clicking the open message again closes the drawer', opened && closed, { opened, closed });

// --- 6 · double-click a node to jump to it ----------------------------------------
await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); });
await wait(200);
const zoomBefore = await page.evaluate(`${S}.querySelector('[data-role="zoom"]').textContent`);
await page.evaluate(`(() => { const el = ${S}.querySelector(${nodeSel(9)});
  el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); })()`);
await wait(3000);
const jumped = await page.evaluate(`(() => ({
  toast: ${S}.querySelector('[data-role="toast"]').textContent,
  onScreen: [...document.querySelectorAll('.row')].some(e => e.textContent.includes('Every node lands inside')),
  zoom: ${S}.querySelector('[data-role="zoom"]').textContent }))()`);
check('6 · double-clicking a node jumps the chat to it',
  jumped.onScreen && jumped.toast.includes('Scrolled') && jumped.zoom === zoomBefore, jumped);

// --- 2 · explicit branch switch from the drawer -----------------------------------
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
const hasButton = await page.evaluate(`!!${S}.querySelector('[data-detail="goto"]')`);
check('2 · off-branch messages offer a branch switch', hasButton);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(3500);
check('2 · the switch moves the chat onto that branch', await page.evaluate(`(() => ({
  variant: window.__harness.variant,
  toast: ${S}.querySelector('[data-role="toast"]').textContent }))()`).then((v) => {
    check.last = v; return v.variant === 1 && v.toast .includes('Switched branch'); }),
  await page.evaluate(`${S}.querySelector('[data-role="toast"]').textContent`));

// --- 5 · Claude's search comes out on top ----------------------------------------
const layering = await page.evaluate(`(async () => {
  const host = document.querySelector('.ct-panel-host');
  const before = getComputedStyle(host).zIndex;
  window.__harness.dialog(true);
  await new Promise(r => setTimeout(r, 250));
  const during = getComputedStyle(host).zIndex;
  const dialogZ = getComputedStyle(document.getElementById('dlg')).zIndex;
  // is the dialog actually the thing painted at its own centre?
  const dlg = document.getElementById('dlg');
  dlg.style.inset = '22% 5%';            // overlap the pane so the test is meaningful
  const r = dlg.getBoundingClientRect();
  const probeX = r.right - 40;           // inside both the dialog and the pane
  const hit = document.elementFromPoint(probeX, (r.top + r.bottom) / 2);
  const onTop = hit === dlg || dlg.contains(hit);
  window.__harness.dialog(false);
  await new Promise(r => setTimeout(r, 150));
  return { before, during, dialogZ, onTop, after: getComputedStyle(host).zIndex };
})()`);
check("5 · Claude's dialogs render over the pane", layering.onTop && Number(layering.during) < Number(layering.dialogZ), layering);
check('5 · the pane returns to the front afterwards', layering.after === layering.before, layering);

await page.screenshot({ path: `${OUT}/a0-final.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
