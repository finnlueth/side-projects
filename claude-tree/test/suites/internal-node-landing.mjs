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

// node 6 is INTERNAL on the other branch: 5 -> 6 -> 7, so it is not the leaf
await page.evaluate(`${S}.querySelector(${nodeSel(6)}).click()`);
await wait(300);
check('an internal off-branch node offers the switch', await page.evaluate(`
  ${S}.querySelector('[data-detail="goto"]').textContent.includes('Show this branch')`));
await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
await wait(9000);

const landed = await page.evaluate(`(() => {
  const s = window.__harness.scroller, P = window.__harness.path(), T = window.__harness.tops();
  const i = P.indexOf(6);
  const rows = [...document.querySelectorAll('[data-testid="transcript-row"]')];
  const target = rows.find(r => Math.abs(parseFloat(r.style.top) - T[i]) < 2);
  const first = rows.find(r => Math.abs(parseFloat(r.style.top) - T[0]) < 2);
  const top = s.getBoundingClientRect().top;
  return {
    onOtherBranch: window.__harness.variant === 1,
    branchHasTheNode: i >= 0,
    targetOffset: target ? Math.round(target.getBoundingClientRect().top - top) : null,
    firstMessageAtTop: first ? Math.abs(first.getBoundingClientRect().top - top) < 40 : false,
    scrollTop: Math.round(s.scrollTop),
    toast: ${S}.querySelector('[data-role="toast"]').textContent,
  };
})()`);
check('the chat switches to that branch', landed.onOtherBranch && landed.branchHasTheNode, landed);
check('and lands on the chosen message, not the top of the conversation',
  landed.targetOffset !== null && Math.abs(landed.targetOffset) < 60 && !landed.firstMessageAtTop, landed);
check('the pane says it scrolled there', landed.toast.includes('scrolled to the message'), landed.toast);

// the node is also selected, so the drawer shows what you jumped to
check('the message is selected in the tree', await page.evaluate(`
  ${S}.querySelector('.ct-node.is-selected')?.dataset.id.slice(0,8) === '00000006'`),
  await page.evaluate(`${S}.querySelector('.ct-node.is-selected')?.dataset.id.slice(0,8)`));

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
