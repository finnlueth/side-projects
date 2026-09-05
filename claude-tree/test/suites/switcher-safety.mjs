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

// --- a fraction in a message is not a variant switcher ----------------------------
const math = await page.evaluate(`(async () => {
  const s = window.__harness.scroller;
  s.scrollTop = s.scrollHeight; s.dispatchEvent(new Event('scroll'));
  await new Promise(r => setTimeout(r, 300));
  const row = [...document.querySelectorAll('[data-testid="transcript-row"]')]
    .find(e => e.textContent.includes('fair die'));
  return { found: !!CT.chat.findVariantSwitcherForTest?.(row), fractions:
    [...row.querySelectorAll('span')].filter(x => /^\\d+\\s*\\/\\s*\\d+$/.test(x.textContent.trim())).length };
})()`).catch(() => ({ found: false, fractions: 0 }));
check('a fraction in a message is not mistaken for a switcher', math.fractions > 0, math);

// the real proof: switching must never press Copy or Retry
await page.evaluate(() => { window.__pressed = []; window.__noReload = true; });
await page.evaluate(`${S}.querySelector(${nodeSel(7)}).click()`);
await wait(300);
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(9000);
const outcome = await page.evaluate(`(() => ({
  pressed: window.__pressed || [],
  variant: window.__harness.variant,
  noReload: window.__noReload === true,
  puts: JSON.parse(sessionStorage.getItem('ct-puts') || '[]').length,
  toast: ${S}.querySelector('[data-role="toast"]').textContent,
  renders: [...document.querySelectorAll('[data-testid="transcript-row"]')]
    .some(e => e.textContent.includes('contour tracking')),
}))()`);
check('never presses Copy or Retry', !outcome.pressed.includes('action-bar-copy')
  && !outcome.pressed.includes('action-bar-retry'), outcome.pressed);

// --- switching works against controls that only listen for the pointer sequence ----
check('switches a pointer-driven control, in page, no reload',
  outcome.variant === 1 && outcome.noReload && outcome.puts === 0 && outcome.renders, outcome);
check('and says so', outcome.toast.includes('Switched branch'), outcome.toast);

// a bare click() would not have worked - prove the harness is actually strict
check('the harness control ignores a bare click()', await page.evaluate(`(async () => {
  const before = window.__harness.variant;
  const btn = document.querySelector('[data-testid="action-bar-previous-version"]:not([disabled])')
           || document.querySelector('[data-testid="action-bar-next-version"]:not([disabled])');
  if (!btn) return true;
  btn.click();
  await new Promise(r => setTimeout(r, 250));
  return window.__harness.variant === before; })()`));

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
