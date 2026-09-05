/**
 * The toolbar's follow toggle: with it on, the canvas keeps the messages the chat is showing
 * in view — centred sideways, and far enough down (or up) that the message leading the way is
 * on screen. Off by default, remembered between visits, and it never moves the canvas while
 * it is off.
 */
import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 950, deviceScaleFactor: 1 });
page.on('pageerror', (e) => problems.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
await page.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark',
  { waitUntil: 'networkidle0' });
await wait(700);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot
  .querySelector('.ct-toggle').click());
await wait(1200);

const toggle = `${S}.querySelector('[data-action="sticky"]')`;
const view = `(() => {
  const t = ${S}.querySelector('.ct-viewport').style.transform;
  const m = /translate\\(([-\\d.]+)px, ([-\\d.]+)px\\)/.exec(t);
  return m ? { x: Math.round(+m[1]), y: Math.round(+m[2]) } : null;
})()`;
/**
 * Where the messages on screen sit. Visibility is measured against the part of the canvas a
 * message can actually be seen in — the drawer is drawn over the bottom of it.
 */
const marked = `(() => {
  const canvas = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
  const drawer = ${S}.querySelector('.ct-detail');
  const floor = drawer && !drawer.hidden ? drawer.getBoundingClientRect().top : canvas.bottom;
  const els = [...${S}.querySelectorAll('.ct-node.is-current')];
  if (!els.length) return null;
  const boxes = els.map((e) => e.getBoundingClientRect());
  const highest = boxes.reduce((a, b) => (b.top < a.top ? b : a));
  const lowest = boxes.reduce((a, b) => (b.bottom > a.bottom ? b : a));
  const left = Math.min(...boxes.map((b) => b.left));
  const right = Math.max(...boxes.map((b) => b.right));
  const centred = (b) => Math.round((b.left + b.right) / 2 - (canvas.left + canvas.width / 2));
  const across = (b) => b.left >= canvas.left - 2 && b.right <= canvas.right + 2;
  return {
    count: els.length,
    spanWiderThanCanvas: (right - left) > canvas.width,
    highestVisible: highest.top >= canvas.top - 2 && highest.bottom <= floor + 2 && across(highest),
    lowestVisible: lowest.top >= canvas.top - 2 && lowest.bottom <= floor + 2 && across(lowest),
    highestDx: centred(highest),
    lowestDx: centred(lowest),
    lowestFromFloor: Math.round(floor - lowest.bottom),
  };
})()`;

/** Scroll the chat to a message and let the pane react. */
const scrollTo = async (index, settle = 900) => {
  await page.evaluate(`(() => {
    const h = window.__harness, s = h.scroller, T = h.tops();
    s.scrollTop = T[${index}] - 40;
    s.dispatchEvent(new Event('scroll'));
  })()`);
  await wait(settle);
};

// --- 1 · off by default, and the canvas stays put ---------------------------------
check('1 · the follow toggle is in the toolbar and starts off',
  await page.evaluate(`!!${toggle} && ${toggle}.getAttribute('aria-pressed') === 'false'`));

await scrollTo(1);
const restBefore = await page.evaluate(view);
await scrollTo(5);
const restAfter = await page.evaluate(view);
check('1 · with it off, scrolling the chat leaves the canvas alone',
  restBefore && restAfter && restBefore.x === restAfter.x && restBefore.y === restAfter.y,
  { restBefore, restAfter });

// --- 2 · on, the messages on screen are centred sideways ---------------------------
await page.evaluate(`${toggle}.click()`);
await wait(300);
check('2 · pressing it turns it on',
  await page.evaluate(`${toggle}.getAttribute('aria-pressed') === 'true'`));

await scrollTo(2);
await scrollTo(6);                       // downward, so the lowest is leading
const centred = await page.evaluate(marked);
check('2 · the message leading the way is centred sideways',
  centred && centred.count > 0 && Math.abs(centred.lowestDx) <= 12, centred);

// Across a fork the messages on screen can span further than the canvas is wide. Centring
// on the middle of that span puts both ends off the edge — the leading one included.
await scrollTo(2);
const upward = await page.evaluate(marked);
check('2 · and stays on canvas even when they span more than the canvas is wide',
  upward && upward.highestVisible && Math.abs(upward.highestDx) <= 12, upward);

// --- 3 · the leading message is brought onto the canvas ----------------------------
await scrollTo(7);
const down = await page.evaluate(marked);
check('3 · scrolling down brings the lowest of them onto the canvas',
  down && down.lowestVisible, down);

await scrollTo(3);
const up = await page.evaluate(marked);
check('3 · scrolling up brings the highest of them onto the canvas',
  up && up.highestVisible, up);

// The reported bug: scrolled all the way up, the first message stayed off the canvas and
// the messages below it were shown instead. The upward rule only ever looked at the top
// edge, so a leading message that was off the *bottom* was left there.
await page.evaluate(`(() => { const s = window.__harness.scroller;
  s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); })()`);
await wait(1000);
const first = await page.evaluate(marked);
check('3 · scrolling to the very first message brings it onto the canvas',
  first && first.highestVisible, first);

// Reached the same way a reader does: from further down, a step at a time, having only just
// turned following on — the case where there is no direction to go on yet.
await page.evaluate(`(() => { const h = window.__harness, s = h.scroller, T = h.tops();
  s.scrollTop = T[T.length - 1]; s.dispatchEvent(new Event('scroll')); })()`);
await wait(800);
await page.evaluate(`${toggle}.click()`); await wait(200);   // off
await page.evaluate(`${toggle}.click()`); await wait(400);   // and on again, direction forgotten
for (const i of [6, 4, 2, 0]) await scrollTo(i, 600);
await page.evaluate(`(() => { const s = window.__harness.scroller;
  s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); })()`);
await wait(900);
const walkedUp = await page.evaluate(marked);
check('3 · and when walked up to from further down, having just been switched on',
  walkedUp && walkedUp.highestVisible, walkedUp);

// --- 3c · scrolled up quickly, faster than the canvas can glide ---------------------
// Updates arrive sooner than a glide finishes, so each one has to be measured against where
// the canvas is heading. Measured against where it happens to be mid-glide, the follow reads
// a position it is only passing through as "already there" and stops early — finishing at
// the start of the conversation with the first message still off the canvas.
await page.evaluate(`(() => { const h = window.__harness, s = h.scroller, T = h.tops();
  s.scrollTop = T[T.length - 1]; s.dispatchEvent(new Event('scroll')); })()`);
await wait(900);
for (const i of [8, 6, 4, 2]) await scrollTo(i, 90);   // faster than the 260ms glide
await page.evaluate(`(() => { const s = window.__harness.scroller;
  s.scrollTop = 0; s.dispatchEvent(new Event('scroll')); })()`);
await wait(1400);
const rushed = await page.evaluate(marked);
check('3c · a quick scroll to the top still lands the first message on the canvas',
  rushed && rushed.highestVisible, rushed);

// --- 3b · and nothing moves while it is already there ------------------------------
// "Scrolling should only happen when a new off canvas message has to be moved onto it."
await scrollTo(4);
const settledView = await page.evaluate(view);
await page.evaluate(`(() => { const h = window.__harness, s = h.scroller;
  s.scrollTop += 12; s.dispatchEvent(new Event('scroll')); })()`);
await wait(700);
const nudgedView = await page.evaluate(view);
check('3b · a small scroll within the same messages leaves the canvas alone',
  settledView && nudgedView && settledView.x === nudgedView.x && settledView.y === nudgedView.y,
  { settledView, nudgedView });

// --- 4 · the canvas glides rather than jumping -------------------------------------
// Sampled while it is moving: a jump would already be at its destination.
await scrollTo(9, 0);
await wait(60);
const early = await page.evaluate(view);
await wait(900);
const settled = await page.evaluate(view);
check('4 · the canvas eases into place rather than snapping',
  early && settled && (Math.abs(early.x - settled.x) > 2 || Math.abs(early.y - settled.y) > 2),
  { early, settled });

// --- 6 · the open drawer is not somewhere a message can be put ---------------------
// The drawer is drawn over the bottom of the canvas, so a message moved into the strip
// behind it is not on screen at all.
await page.evaluate(`${S}.querySelector('.ct-node').click()`);   // opens the drawer
await wait(500);
const covered = await page.evaluate(`(() => {
  const c = ${S}.querySelector('.ct-canvas').getBoundingClientRect();
  const d = ${S}.querySelector('.ct-detail');
  return d && !d.hidden ? Math.round(c.bottom - d.getBoundingClientRect().top) : 0;
})()`);
await scrollTo(2, 800);
await scrollTo(8, 900);
const withDrawer = await page.evaluate(marked);
check('6 · the drawer really is covering part of the canvas', covered > 40, { covered });
check('6 · scrolling down keeps the message clear of the drawer',
  withDrawer && withDrawer.lowestVisible && withDrawer.lowestFromFloor >= -2, withDrawer);

// --- 5 · the setting is remembered -------------------------------------------------
await page.reload({ waitUntil: 'networkidle0' });
// Wait for the toggle to be back rather than guessing: under load the reload can still be
// settling, and evaluating into a frame that is on its way out fails as a detached frame.
await page.waitForFunction(
  () => !!document.querySelector('.ct-toggle-host')?.shadowRoot?.querySelector('.ct-toggle'),
  { timeout: 15000 });
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot
  .querySelector('.ct-toggle').click());
await page.waitForFunction(
  () => !!document.querySelector('.ct-panel-host')?.shadowRoot?.querySelector('[data-action="sticky"]'),
  { timeout: 15000 });
await wait(600);
check('5 · the setting survives a reload',
  await page.evaluate(`${toggle}.getAttribute('aria-pressed') === 'true'`));

await page.screenshot({ path: `${OUT}/sticky-final.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
