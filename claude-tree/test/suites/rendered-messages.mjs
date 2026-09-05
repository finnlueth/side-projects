/**
 * A message's own markup, drawn rather than stripped: maths as MathML, Markdown as Markdown,
 * pictures as pictures, and the SVG a message carries rebuilt from a whitelist.
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
await page.goto('http://127.0.0.1:8765/chat/00000000-0000-4000-8000-0000000000ff?theme=dark&rich=1',
  { waitUntil: 'networkidle0' });
await wait(700);
await page.evaluate(() => document.querySelector('.ct-toggle-host').shadowRoot
  .querySelector('.ct-toggle').click());
await wait(1200);

const nodeSel = (n) => `.ct-node[data-id="${String(n).padStart(8, '0')}-0000-4000-8000-000000000000"]`;

// --- 1 · the renderer itself -------------------------------------------------------
// Exercised directly so a failure names the construct rather than "the drawer looks wrong".
const rendered = await page.evaluate(`(() => {
  const r = CT.render;
  const out = {};
  out.inlineMath = r.inline('the value $p(\\\\omega) = 1/6$ holds');
  out.fraction = r.inline('$$\\\\frac{3}{6}$$');
  out.sum = r.inline('$\\\\sum_{i=1}^{n} x_i$');
  out.greekAndSets = r.inline('$A \\\\subseteq \\\\Omega$');
  out.unknownCommand = r.inline('$x \\\\nosuchcommand y + 1$');
  out.brokenStructure = r.inline('$\\\\frac{1}{$');
  out.emphasis = r.inline('**bold** and *slanted* and \`code\` and ~~gone~~');
  out.blocks = r.markdown('# Title\\n\\n- one\\n- two\\n\\n> quoted\\n\\n\`\`\`\\ncode()\\n\`\`\`');
  out.script = r.markdown('<img src=x onerror=alert(1)> and <script>alert(2)<\\/script>');
  // Escaped braces: a set written \\{ \\omega \\} is ordinary, and used to throw the whole
  // expression back to being printed as source.
  out.braces = r.inline('$p(\\\\omega) := P(\\\\{\\\\omega\\\\})$');
  out.measure = r.inline('$$\\\\text{Lap}_\\\\Omega(\\\\{\\\\omega\\\\}) = \\\\frac{1}{|\\\\Omega|}'
    + ' \\\\qquad\\\\text{and}\\\\qquad p_{\\\\text{Lap}_\\\\Omega}(\\\\omega) = \\\\frac{1}{|\\\\Omega|}$$');
  return out;
})()`);

check('1 · inline maths becomes MathML', rendered.inlineMath.includes('<math')
  && rendered.inlineMath.includes('ω') && !rendered.inlineMath.includes('\\omega'),
  rendered.inlineMath.slice(0, 90));
check('1 · fractions become mfrac', rendered.fraction.includes('<mfrac>'), rendered.fraction.slice(0, 70));
check('1 · a sum keeps its limits above and below',
  /<munder>|<mover>/.test(rendered.sum) && rendered.sum.includes('∑'), rendered.sum.slice(0, 80));
check('1 · greek and set symbols come through',
  rendered.greekAndSets.includes('⊆') && rendered.greekAndSets.includes('Ω'), rendered.greekAndSets);
// A gap in the symbol table costs one glyph, not the whole formula: an unlisted symbol used
// to send an entire line of maths back to being printed as LaTeX source.
check('1 · an unknown command still renders the rest of the formula',
  rendered.unknownCommand.includes('<math') && !rendered.unknownCommand.includes('ct-math-raw'),
  rendered.unknownCommand.slice(0, 90));
// A formula that cannot be parsed at all is still shown as written, rather than as nonsense.
check('1 · but a broken expression is shown as written',
  rendered.brokenStructure.includes('ct-math-raw'), rendered.brokenStructure);
check('1 · emphasis, code and strikethrough render',
  rendered.emphasis.includes('<strong>bold</strong>') && rendered.emphasis.includes('<em>slanted</em>')
  && rendered.emphasis.includes('<code>code</code>') && rendered.emphasis.includes('<del>gone</del>'),
  rendered.emphasis);
check('1 · headings, lists, quotes and fences render',
  /<h3>Title<\/h3>/.test(rendered.blocks) && /<ul><li>one<\/li><li>two<\/li><\/ul>/.test(rendered.blocks)
  && rendered.blocks.includes('<blockquote>quoted</blockquote>')
  && rendered.blocks.includes('<pre><code>code()</code></pre>'), rendered.blocks);
// Message text is written by whoever is in the conversation, so it is escaped, never trusted.
check('1 · markup in a message is escaped, not executed',
  !/<img|<script/i.test(rendered.script) && rendered.script.includes('&lt;img'), rendered.script);

check('1 · escaped braces render rather than falling back',
  rendered.braces.includes('<math') && !rendered.braces.includes('ct-math-raw')
  && rendered.braces.includes('{'), rendered.braces.slice(0, 110));
check('1 · a measure with \\text, subscripts and fractions renders',
  rendered.measure.includes('<mfrac>') && rendered.measure.includes('<mtext>Lap</mtext>')
  && !rendered.measure.includes('ct-math-raw'), rendered.measure.slice(0, 110));

// The formulas from a real conversation that used to fall back: a labelled arrow, a
// posterior with \left…\right and \mid, a contour integral, and accents.
const harder = await page.evaluate(`(() => {
  const r = CT.render;
  const one = (tex) => {
    const html = r.inline('$$' + tex + '$$');
    return { math: html.includes('<math'), fellBack: html.includes('ct-math-raw') };
  };
  return {
    clt: one('\\\\frac{1}{\\\\sqrt{n}}\\\\sum_{i=1}^{n}\\\\frac{X_i - \\\\mu}{\\\\sigma}'
      + ' \\\\xrightarrow{d} \\\\mathcal{N}(0,1)'),
    bayes: one('P\\\\left(\\\\theta \\\\mid D\\\\right) = \\\\frac{P(D \\\\mid \\\\theta)'
      + ' P(\\\\theta)}{\\\\int P(D \\\\mid \\\\theta) P(\\\\theta) d\\\\theta}'),
    contour: one('f(z_0) = \\\\frac{1}{2\\\\pi i}\\\\oint_C \\\\frac{f(z)}{z - z_0} dz'),
    accents: one('\\\\hat{\\\\mu} + \\\\bar{x} + \\\\vec{v}'),
  };
})()`);
for (const [name, got] of Object.entries(harder)) {
  check(`1 · ${name} renders as MathML rather than falling back`, got.math && !got.fellBack, got);
}

// --- 1a · the formulas from the screenshot ------------------------------------------
const real = await page.evaluate(`(() => {
  const r = CT.render;
  const one = (tex) => {
    const html = r.inline('$$' + tex + '$$');
    return { html, fellBack: html.includes('ct-math-raw'), text: html.replace(/<[^>]+>/g, '') };
  };
  return {
    parts: one('\\\\int_a^b u\\\\,dv = \\\\Big[uv\\\\Big]_a^b - \\\\int_a^b v\\\\,du'),
    svd: one('A = U \\\\Sigma V^{\\\\top}, \\\\qquad \\\\Sigma = \\\\operatorname{diag}'
      + '(\\\\sigma_1, \\\\ldots, \\\\sigma_r), \\\\quad \\\\sigma_1 \\\\ge \\\\cdots \\\\ge \\\\sigma_r > 0'),
    matrix: one('A = \\\\begin{pmatrix} 2 & 1 \\\\\\\\ 1 & 2 \\\\end{pmatrix}, \\\\qquad'
      + ' \\\\det(A - \\\\lambda I) = (2-\\\\lambda)^2 - 1'),
    eigen: one('v_1 = \\\\tfrac{1}{\\\\sqrt 2}(1,1)^{\\\\top}'),
    zeta: one('\\\\zeta(2) = \\\\sum_{n=1}^{\\\\infty} \\\\frac{1}{n^2} = \\\\frac{\\\\pi^2}{6}'),
    cauchy: one('|\\\\langle u, v \\\\rangle|^2 \\\\le \\\\langle u, u \\\\rangle'
      + ' \\\\cdot \\\\langle v, v \\\\rangle'),
    symdiff: one('A\\\\,\\\\triangle\\\\,B = (A\\\\setminus B)\\\\cup(B\\\\setminus A)'
      + '\\\\in\\\\mathcal{F}'),
  };
})()`);
for (const [name, got] of Object.entries(real)) {
  check(`1a · ${name} renders rather than falling back`, !got.fellBack, got.text.slice(0, 60));
}
// A closing bracket used to end the whole expression, dropping everything after it.
check('1a · a closing bracket does not truncate what follows',
  real.parts.text.includes('du') && (real.parts.html.match(/∫/g) || []).length === 2,
  real.parts.text);
check('1a · a matrix becomes a table with its brackets',
  (real.matrix.html.match(/<mtr>/g) || []).length === 2
  && (real.matrix.html.match(/<mtd>/g) || []).length === 4, real.matrix.text.slice(0, 50));
// Ordinary brackets used to grow to the height of whatever they stood beside.
// Symbols written without \left and \right, and a shape from set theory: both used to be
// missing from the table, and a missing entry cost the whole formula.
check('1a · angle brackets and shapes come out as symbols, not words',
  real.cauchy.text.includes('⟨') && real.cauchy.text.includes('⟩')
  && real.symdiff.text.includes('△') && real.symdiff.text.includes('∖'),
  { cauchy: real.cauchy.text, symdiff: real.symdiff.text });
check('1a · ordinary brackets are not stretched to the row',
  real.zeta.html.includes('stretchy="false"'), real.zeta.html.slice(0, 80));

// --- 1b · code blocks are coloured -------------------------------------------------
const coloured = await page.evaluate(`(() => {
  const r = CT.render;
  return {
    js: r.highlight('const x = 1; // note', 'js'),
    py: r.highlight('def f(n):  # note', 'python'),
    unknown: r.highlight('const x = 1;', 'brainfuck'),
    unsafe: r.highlight('<script>alert(1)</script>', 'js'),
  };
})()`);
check('1b · keywords, numbers and comments are picked out',
  coloured.js.includes('ct-tok-word') && coloured.js.includes('ct-tok-number')
  && coloured.js.includes('ct-tok-note'), coloured.js);
check('1b · and per language', coloured.py.includes('>def</span>')
  && coloured.py.includes('ct-tok-note'), coloured.py);
check('1b · an unknown language is left as plain text',
  !coloured.unknown.includes('ct-tok'), coloured.unknown);
check('1b · colouring never lets markup through',
  !/<script>/i.test(coloured.unsafe) && coloured.unsafe.includes('&lt;script&gt;'), coloured.unsafe);

// A fenced block has no room to be a block in a four-line box, but showing it as prose made
// it unreadable — it reads as code instead.
const previewCode = await page.evaluate(`CT.render.preview('How:\\n\\n\`\`\`python\\ndef f(n):\\n    return n + 1\\n\`\`\`\\n\\nDone.')`);
check('1b · code in a preview keeps its shape, coloured, like the drawer',
  previewCode.includes('ct-code') && previewCode.includes('>def</span>')
  && previewCode.includes('f(n)') && previewCode.includes('\n')
  && !previewCode.includes('\`\`\`'), previewCode);

// --- 2 · maths in the tree and the drawer ------------------------------------------
await page.evaluate(`${S}.querySelector('${nodeSel(25)}').click()`);
await wait(500);
const shown = await page.evaluate(`(() => ({
  nodeHasMath: !!${S}.querySelector('${nodeSel(25)} math'),
  nodeShowsBackslashes: /\\\\\\\\frac|\\\\\\\\omega/.test(${S}.querySelector('${nodeSel(25)}').textContent),
  drawerHasMath: !!${S}.querySelector('.ct-detail-text math'),
  drawerHasList: !!${S}.querySelector('.ct-detail-text li'),
  drawerHasCode: !!${S}.querySelector('.ct-detail-text pre code'),
}))()`);
check('2 · a formula in a node preview is drawn, not spelled out',
  shown.nodeHasMath && !shown.nodeShowsBackslashes, shown);
check('2 · and in the drawer', shown.drawerHasMath, shown);

await page.evaluate(`${S}.querySelector('${nodeSel(50)}').click()`);
await wait(500);
const full = await page.evaluate(`(() => ({
  math: !!${S}.querySelector('.ct-detail-text math'),
  heading: !!${S}.querySelector('.ct-detail-text h3'),
  list: ${S}.querySelectorAll('.ct-detail-text li').length,
  quote: !!${S}.querySelector('.ct-detail-text blockquote'),
  code: !!${S}.querySelector('.ct-detail-text pre code'),
  display: !!${S}.querySelector('.ct-detail-text math[display="block"]'),
}))()`);
check('2 · the drawer draws headings, lists, quotes and code blocks',
  full.heading && full.list === 2 && full.quote && full.code, full);
check('2 · display maths is set on its own line',
  full.math && full.display, full);

// --- 3 · pictures and drawings ------------------------------------------------------
await page.evaluate(`${S}.querySelector('${nodeSel(50)}').click()`);
await wait(600);
const media = await page.evaluate(`(() => {
  const shot = ${S}.querySelector('.ct-shot');
  const svg = ${S}.querySelector('.ct-detail-text .ct-svg, .ct-figure .ct-svg');
  return {
    shot: !!shot,
    shotSrc: shot ? shot.getAttribute('src').slice(0, 12) : null,
    svg: !!svg,
    svgKeptShape: svg ? svg.querySelectorAll('circle, rect, path').length : 0,
    svgDroppedScript: svg ? svg.querySelectorAll('script').length : 0,
    svgDroppedHandler: svg ? [...svg.querySelectorAll('*')].some((e) => e.hasAttribute('onload')) : null,
  };
})()`);
check('3 · a picture a message carries is shown as a picture',
  media.shot && media.shotSrc.startsWith('/api/'), media);
check('3 · an SVG a message contains is drawn', media.svg && media.svgKeptShape > 0, media);

// It used to be escaped into the paragraph *and* appended again underneath: the source and a
// duplicate of the drawing, both.
const svgPlacement = await page.evaluate(`(() => {
  const text = ${S}.querySelector('.ct-detail-text');
  return {
    drawings: text.querySelectorAll('.ct-svg').length,
    sourceShown: /<svg|<circle/i.test(text.textContent),
  };
})()`);
check('3 · drawn once, and its source is not printed as well',
  svgPlacement.drawings === 1 && !svgPlacement.sourceShown, svgPlacement);

// Code: fenced blocks keep their text and their line breaks, and can scroll sideways rather
// than stretching the pane.
const code = await page.evaluate(`(() => {
  const pre = ${S}.querySelector('.ct-detail-text pre');
  if (!pre) return null;
  const cs = getComputedStyle(pre);
  return { text: pre.textContent.trim(), wrap: getComputedStyle(pre.querySelector('code')).whiteSpace,
           scrolls: cs.overflowX, fits: pre.getBoundingClientRect().width
             <= ${S}.querySelector('.ct-detail').getBoundingClientRect().width };
})()`);
check('3 · a code block keeps its text and stays inside the pane',
  code && code.text.startsWith('Lap(A)') && code.scrolls === 'auto' && code.fits, code);
const inBlock = await page.evaluate(`(() => {
  const pre = ${S}.querySelector('.ct-detail-text pre');
  return { lang: pre.getAttribute('data-lang'), coloured: pre.querySelectorAll('.ct-tok-note').length };
})()`);
check('3 · and is coloured, and says what language it is',
  inBlock.lang === 'python' && inBlock.coloured === 1, inBlock);

// And a preview shows the code rather than a gap, without the fence rails.
const codePreview = await page.evaluate(`${S}.querySelector('${nodeSel(50)} .ct-node-text').textContent`);
check('3 · a preview keeps code and drops the fence rails',
  codePreview.includes('Lap(A)') && !codePreview.includes('\`\`\`') && !/<svg/i.test(codePreview),
  codePreview.slice(0, 90));
check('3 · and its scripts and handlers are left behind',
  media.svgDroppedScript === 0 && media.svgDroppedHandler === false, media);

await page.screenshot({ path: `${OUT}/rendered.png` });
for (const c of checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.n}${c.d !== undefined ? '  ' + JSON.stringify(c.d) : ''}`);
console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno page errors');
await browser.close();
