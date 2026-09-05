import puppeteer from '../support/puppeteer.mjs';
const CHAT = '/chat/00000000-0000-4000-8000-0000000000ff';
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 950, deviceScaleFactor: 2 });
page.on('pageerror', (e) => problems.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text()); });
await page.goto(`http://127.0.0.1:8765${CHAT}?theme=dark`, { waitUntil: 'networkidle0' });
await wait(800);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot.querySelector('.ct-toggle').click());
await wait(1400);

const marksVsScreen = () => page.evaluate(`(() => {
  const onScreen = [...document.querySelectorAll('[data-testid="transcript-row"]')]
    .filter(e => { const r = e.getBoundingClientRect();
      return r.bottom > 0 && r.top < innerHeight && r.height; }).length;
  return { onScreen, marked: ${S}.querySelectorAll('.ct-node.is-current').length };
})()`);

await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = s.scrollHeight; s.dispatchEvent(new Event('scroll')); });
await wait(400);
const settled = await marksVsScreen();
check('baseline: marks match the screen', settled.marked === settled.onScreen && settled.onScreen > 0, settled);

// Claude starts streaming a reply the tree does not know about yet
await page.evaluate(() => {
  window.__harness.startStreaming('Streaming a fresh answer that the tree has not seen yet.');
  const s = window.__harness.scroller, T = window.__harness.tops();
  s.scrollTop = T[T.length - 3];              // several known rows plus the streaming one
  s.dispatchEvent(new Event('scroll'));
});
await wait(700);
const streaming = await marksVsScreen();
check('highlights survive a message the tree has not seen',
  streaming.marked > 0 && streaming.onScreen > 1, streaming);
check('and the known messages are still marked correctly',
  streaming.marked === streaming.onScreen - 1 || streaming.marked === streaming.onScreen, streaming);

// once the reply lands and the tree refreshes, everything lines up again
await page.evaluate(() => window.__harness.finishStreaming());
await wait(4000);
await page.evaluate(() => { const s = window.__harness.scroller; s.scrollTop = s.scrollHeight; s.dispatchEvent(new Event('scroll')); });
await wait(600);
const after = await marksVsScreen();
check('and match again once the tree catches up', after.marked === after.onScreen && after.onScreen > 0, after);

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
