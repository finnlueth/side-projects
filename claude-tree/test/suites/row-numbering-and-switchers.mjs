/**
 * The mechanics that make branch switching and highlighting reliable on claude.ai.
 *
 * Every check here stands for something that was observed failing against the real site:
 * a fork made by editing a prompt carries a different family of controls, the rendered rows
 * are numbered but not contiguous, a row can be on screen and still be unreachable because
 * Claude's header floats over it, and a control that changes nothing must not be reported
 * as a branch switch.
 */
import puppeteer from '../support/puppeteer.mjs';
const OUT = process.argv[2];
const browser = await puppeteer.launch();
const checks = []; const problems = [];
const check = (n, pass, d) => checks.push({ n, pass, d });
const S = 'document.querySelector(".ct-panel-host").shadowRoot';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const nodeSel = (n) => `.ct-node[data-id="${String(n).padStart(8, '0')}-0000-4000-8000-000000000000"]`;

const open = async (query) => {
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 950, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => problems.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text());
  });
  await page.goto(`http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark&${query}`,
    { waitUntil: 'networkidle0' });
  await wait(700);
  await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot
    .querySelector('.ct-toggle').click());
  await wait(1200);
  return page;
};

/** Ask the pane to go to a message, and report what the chat did. */
const goTo = async (page, id, ms = 6000) => {
  await page.evaluate(`${S}.querySelector('${nodeSel(id)}').click()`);
  await wait(300);
  await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
  await wait(ms);
  return page.evaluate(`(() => ({
    variant: window.__harness.variant,
    toast: ${S}.querySelector('[data-role="toast"]').textContent,
    onScreen: [...document.querySelectorAll('.row')].some((r) => {
      const b = r.getBoundingClientRect();
      return b.bottom > 0 && b.top < innerHeight; }) }))()`);
};

// --- 1 · a fork made by editing a prompt ------------------------------------------
// Claude puts these controls on the human row as user-message-*-version, and that row has
// no action bar at all. Matching only the action-bar family made every such fork look
// unswitchable, which is the most common kind of branch there is.
{
  const page = await open('promptfork=1&hoverswitch=1');
  const before = await page.evaluate(() => window.__harness.variant);
  const out = await goTo(page, 31);
  check('1 · switches a fork created by editing a prompt',
    before === 0 && out.variant === 1 && out.toast.includes('Switched branch'), out);
  await page.screenshot({ path: `${OUT}/rn-promptfork.png` });
  await page.close();
}

// --- 2 · rows numbered but not contiguous -----------------------------------------
// The transcript drops rows out of the middle of what it renders, so a message can be
// missing from between two rendered ones. Anything that treats the rendered numbers as an
// unbroken span looks straight past it.
{
  const page = await open('gaps=1');
  const gapped = await page.evaluate(() => {
    const seen = [...document.querySelectorAll('.row')]
      .map((r) => Number(r.dataset.index)).sort((a, b) => a - b);
    return { seen, contiguous: seen.every((n, i) => i === 0 || n === seen[i - 1] + 1) };
  });
  check('2 · the harness really does leave gaps', !gapped.contiguous, gapped);
  const out = await goTo(page, 24);
  check('2 · reaches a message missing from between rendered rows',
    out.toast.includes('Scrolled to the message'), out);
  await page.close();
}

// --- 2b · a message far above everything rendered -----------------------------------
// Reached from the very bottom of the conversation, which is the direction that used to go
// wrong: the scroll direction has to come from the row numbers, not from the interpolated
// offset, which by then sits behind the chat rather than ahead of it.
{
  const page = await open('gaps=1');
  await page.evaluate(() => {
    const s = window.__harness.scroller;
    s.scrollTop = s.scrollHeight;                     // start at the very bottom
    s.dispatchEvent(new Event('scroll'));
  });
  await wait(500);
  const far = await page.evaluate(() =>
    Math.min(...[...document.querySelectorAll('.row')].map((r) => Number(r.dataset.index))));
  const out = await goTo(page, 1);
  check('2b · reaches the first message from the bottom of a long chat',
    far > 0 && out.toast.includes('Scrolled to the message'), { startedBelowRow: far, ...out });
  await page.close();
}

// --- 3 · a fork row underneath Claude's floating header ----------------------------
// The row reports itself on screen, but the only part of it that is on screen sits under
// the header, so pointing at it lands on the header and Claude never mounts its controls.
{
  const page = await open('stickyheader=1&hoverswitch=1');
  const out = await goTo(page, 7);
  check('3 · switches when the fork row sits under the header',
    out.variant === 1 && out.toast.includes('Switched branch'), out);
  await page.screenshot({ path: `${OUT}/rn-sticky.png` });
  await page.close();
}

// --- 3b · landing on the message asked for, not the end of the branch ---------------
// Claude scrolls the transcript to the end of a branch as it renders it, and goes on doing
// so for a moment. A jump that scrolls once, while that is still happening, is quietly
// undone and the reader ends up on the branch's last message instead of the one they chose.
{
  const page = await open('hoverswitch=1&scrollonswitch=1');
  await page.evaluate(`${S}.querySelector('${nodeSel(6)}').click()`);   // mid-branch, not a leaf
  await wait(300);
  await page.evaluate(`${S}.querySelector('[data-detail="goto"]').click()`);
  await wait(6000);
  const where = await page.evaluate(`(() => {
    const s = window.__harness.scroller;
    const target = [...document.querySelectorAll('.row')].find((r) =>
      r.textContent.includes('What does contour tracking buy me'));
    return {
      variant: window.__harness.variant,
      atEnd: Math.abs(s.scrollTop - (s.scrollHeight - s.clientHeight)) < 60,
      targetOffset: target ? Math.round(target.getBoundingClientRect().top - s.getBoundingClientRect().top) : null,
      toast: ${S}.querySelector('[data-role="toast"]').textContent };
  })()`);
  check('3b · lands on the chosen message, not the end of the branch',
    where.variant === 1 && where.targetOffset !== null
    && where.targetOffset >= -8 && where.targetOffset < 120, where);
  await page.screenshot({ path: `${OUT}/rn-landing.png` });
  await page.close();
}

// --- 4 · the older, unnumbered markup ----------------------------------------------
// The row numbering is Claude's and could go away. Switching has to keep working without it.
{
  const page = await open('rownumbers=off&hoverswitch=1');
  const unnumbered = await page.evaluate(() =>
    [...document.querySelectorAll('.row')].every((r) => r.dataset.index === undefined));
  check('4 · the harness really does drop the numbering', unnumbered);
  const out = await goTo(page, 7);
  check('4 · still switches without row numbers',
    out.variant === 1 && out.toast.includes('Switched branch'), out);
  await page.close();
}

// --- 5 · a control that does nothing is not a branch switch -------------------------
// Reporting a switch that never happened leaves the reader looking at the same conversation
// while the tree insists it moved.
{
  const page = await open('hoverswitch=1&deadswitch=1');
  const out = await goTo(page, 7);
  check('5 · a dead control is not reported as a switch',
    out.variant === 0 && !out.toast.includes('Switched branch'), out);
  await page.close();
}

for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
