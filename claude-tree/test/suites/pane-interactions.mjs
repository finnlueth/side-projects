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

// --- 3 · clicking a message keeps it open -----------------------------------------
// It used to close again, which fought the double-click that shows it in the chat: the
// first of the two clicks closed the drawer the second one needed. The canvas clears the
// selection instead.
await page.evaluate(`${S}.querySelector(${nodeSel(2)}).click()`);
await wait(250);
const opened = await page.evaluate(`!${S}.querySelector('.ct-detail').hidden`);
await page.evaluate(`${S}.querySelector(${nodeSel(2)}).click()`);
await wait(250);
const stillOpen = await page.evaluate(`!${S}.querySelector('.ct-detail').hidden`);
check('3 · clicking the open message leaves it open', opened && stillOpen, { opened, stillOpen });

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

// --- 7 · the tree stays centred when the window is resized -------------------------
// The tree is drawn at a fixed offset inside the canvas, so a canvas that grows used to
// leave it pinned towards one edge with all the new space on the other side. Checked on a
// page of its own: with a message selected the pane also keeps the selection in view, which
// is a separate, deliberate movement.
{
  const fresh = await browser.newPage();
  await fresh.setViewport({ width: 1500, height: 950, deviceScaleFactor: 1 });
  fresh.on('pageerror', (e) => problems.push(e.message));
  await fresh.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark',
    { waitUntil: 'networkidle0' });
  await wait(700);
  await fresh.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot
    .querySelector('.ct-toggle').click());
  await wait(1200);

  const placement = `(() => {
    const canvas = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
    const box = ${S}.querySelector('.ct-node').getBoundingClientRect();
    return { dx: Math.round((box.left + box.width / 2) - (canvas.left + canvas.width / 2)),
             dy: Math.round((box.top + box.height / 2) - (canvas.top + canvas.height / 2)),
             w: Math.round(canvas.width), h: Math.round(canvas.height) };
  })()`;

  const framed = await fresh.evaluate(placement);
  await fresh.setViewport({ width: 1120, height: 700, deviceScaleFactor: 1 });
  await wait(700);
  const shorter = await fresh.evaluate(placement);
  // Narrow enough that the pane itself has to give way, so the canvas changes width too —
  // the direction in which the tree was left stranded against one edge.
  await fresh.setViewport({ width: 800, height: 700, deviceScaleFactor: 1 });
  await wait(700);
  const narrow = await fresh.evaluate(placement);
  await fresh.setViewport({ width: 1500, height: 950, deviceScaleFactor: 1 });
  await wait(700);
  const back = await fresh.evaluate(placement);

  const held = (a, b) => Math.abs(a.dx - b.dx) <= 6 && Math.abs(a.dy - b.dy) <= 6;
  check('7 · the canvas really did change height', shorter.h !== framed.h, { framed, shorter });
  check('7 · and width', narrow.w !== shorter.w, { shorter, narrow });
  check('7 · the tree keeps its place when the window gets shorter',
    held(framed, shorter), { framed, shorter });
  check('7 · and when the pane is squeezed narrower',
    held(shorter, narrow), { shorter, narrow });
  check('7 · and when the window grows back', held(framed, back), { framed, back });
  await fresh.screenshot({ path: `${OUT}/a1-resize.png` });
  await fresh.close();
}

// --- 8 · double-clicking the canvas brings the selected message back ----------------
// Panning around a large conversation loses sight of the message being worked with. A
// double-click on the canvas returns to it, rather than re-framing the tree from its root.
{
  const fresh = await browser.newPage();
  await fresh.setViewport({ width: 1500, height: 950, deviceScaleFactor: 1 });
  fresh.on('pageerror', (e) => problems.push(e.message));
  await fresh.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark',
    { waitUntil: 'networkidle0' });
  await wait(700);
  await fresh.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot
    .querySelector('.ct-toggle').click());
  await wait(1200);

  const offset = `(() => {
    const canvas = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
    const el = ${S}.querySelector('.ct-node.is-selected');
    if (!el) return null;
    const box = el.getBoundingClientRect();
    return { dx: Math.round((box.left + box.width / 2) - (canvas.left + canvas.width / 2)),
             dy: Math.round((box.top + box.height / 2) - (canvas.top + canvas.height / 2)) };
  })()`;
  await fresh.evaluate(`${S}.querySelector(${nodeSel(9)}).click()`);
  await wait(400);
  // The drawer lies over the bottom of the canvas, so only the strip above it can be clicked.
  const box = await fresh.evaluate(`(() => {
    const b = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
    const d = ${S}.querySelector('.ct-detail');
    const drawerTop = d && !d.hidden ? d.getBoundingClientRect().top : b.bottom;
    return { x: b.x, y: b.y, w: b.width, h: b.height, free: drawerTop - b.y }; })()`);

  // drag the canvas well away from it, staying clear of the drawer
  await fresh.mouse.move(box.x + box.w - 24, box.y + box.free - 24);
  await fresh.mouse.down();
  await fresh.mouse.move(box.x + 24, box.y + 24, { steps: 10 });
  await fresh.mouse.up();
  await wait(400);
  const panned = await fresh.evaluate(offset);

  await fresh.mouse.click(box.x + box.w - 16, box.y + box.free * 0.5, { clickCount: 2 });
  await wait(500);
  const returned = await fresh.evaluate(offset);

  check('8 · panning really did move the selection off centre',
    panned && (Math.abs(panned.dx) > 40 || Math.abs(panned.dy) > 40), panned);
  check('8 · double-clicking the canvas centres the selected message',
    returned && Math.abs(returned.dx) <= 8 && Math.abs(returned.dy) <= 8, returned);
  await fresh.screenshot({ path: `${OUT}/a2-recentre.png` });

  // --- 9 · the canvas clears the selection, a message does not -----------------------
  const selectedCount = `${S}.querySelectorAll('.ct-node.is-selected').length`;
  const drawerShut = `(() => { const d = ${S}.querySelector('.ct-detail'); return !d || d.hidden; })()`;

  await fresh.evaluate(`${S}.querySelector(${nodeSel(9)}).click()`);
  await wait(300);
  const one = await fresh.evaluate(selectedCount);
  await fresh.evaluate(`${S}.querySelector(${nodeSel(9)}).click()`);
  await wait(300);
  const two = await fresh.evaluate(selectedCount);
  check('9 · clicking a message keeps it selected', one === 1 && two === 1, { one, two });

  await fresh.mouse.click(box.x + box.w - 16, box.y + box.free * 0.5);
  await wait(700);   // past the pause that waits for a possible double-click
  const cleared = await fresh.evaluate(selectedCount);
  const shut = await fresh.evaluate(drawerShut);
  check('9 · one click on the canvas clears it', cleared === 0 && shut, { cleared, shut });

  // --- 10 · with nothing selected, the end of the current path -----------------------
  const endOffset = `(() => {
    const canvas = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
    const path = [...${S}.querySelectorAll('.ct-node.is-path')];
    const el = path[path.length - 1];
    if (!el) return null;
    const b = el.getBoundingClientRect();
    return { dx: Math.round((b.left + b.width / 2) - (canvas.left + canvas.width / 2)),
             dy: Math.round((b.top + b.height / 2) - (canvas.top + canvas.height / 2)) };
  })()`;
  await fresh.mouse.move(box.x + box.w - 24, box.y + box.free - 24);
  await fresh.mouse.down();
  await fresh.mouse.move(box.x + 24, box.y + 24, { steps: 10 });
  await fresh.mouse.up();
  await wait(400);
  const drifted = await fresh.evaluate(endOffset);
  await fresh.mouse.click(box.x + box.w - 16, box.y + box.free * 0.5, { clickCount: 2 });
  await wait(500);
  const atEnd = await fresh.evaluate(endOffset);
  check('10 · the pan moved the last message of the path off centre',
    drifted && (Math.abs(drifted.dx) > 40 || Math.abs(drifted.dy) > 40), drifted);
  check('10 · with nothing selected, double-click centres the end of the path',
    atEnd && Math.abs(atEnd.dx) <= 8 && Math.abs(atEnd.dy) <= 8, atEnd);

  // --- 11 · the drawer lies over the canvas -----------------------------------------
  // It used to be a sibling of the canvas in the same column, so its height came out of the
  // canvas: opening or closing it moved every message on screen.
  const whereIsNode = `(() => {
    const b = ${S}.querySelector(${nodeSel(2)}).getBoundingClientRect();
    return { top: Math.round(b.top), left: Math.round(b.left) };
  })()`;
  const shut0 = await fresh.evaluate(whereIsNode);
  await fresh.evaluate(`${S}.querySelector(${nodeSel(2)}).click()`);
  await wait(500);
  const open = await fresh.evaluate(whereIsNode);
  const covering = await fresh.evaluate(`(() => {
    const d = ${S}.querySelector('.ct-detail').getBoundingClientRect();
    const c = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
    return { overlaps: d.top < c.bottom - 4, height: Math.round(d.height) };
  })()`);
  await fresh.mouse.click(box.x + box.w - 16, box.y + 40);
  await wait(700);
  const shut1 = await fresh.evaluate(whereIsNode);

  check('11 · the drawer really is over the canvas, not below it',
    covering.overlaps && covering.height > 40, covering);
  check('11 · opening it leaves the tree where it was',
    Math.abs(open.top - shut0.top) <= 2 && Math.abs(open.left - shut0.left) <= 2, { shut0, open });
  check('11 · and so does closing it',
    Math.abs(shut1.top - shut0.top) <= 2 && Math.abs(shut1.left - shut0.left) <= 2, { shut0, shut1 });
  await fresh.close();
}

await page.screenshot({ path: `${OUT}/a0-final.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
