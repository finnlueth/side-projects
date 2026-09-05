/**
 * Turning a message's own text into something readable in the pane.
 *
 * The API hands over Markdown with LaTeX in it, so the pane used to strip the markup and show
 * what was left — which turns a formula into a line of backslashes and a list into a run-on
 * sentence. This renders it instead.
 *
 * Maths is emitted as MathML. Claude draws its own with KaTeX, but that is bundled rather than
 * exposed, and its stylesheet lives in the page where a shadow root cannot reach it, so
 * calling into it is not an option. MathML is what the browser already knows how to draw: no
 * library, no stylesheet, and it inherits the pane's own type. What it cannot express falls
 * back to the LaTeX itself, set as code, which still beats a line of backslashes.
 *
 * Everything here builds HTML out of escaped text. The one exception is SVG that a message
 * carries, which is parsed and rebuilt from a whitelist rather than trusted.
 */
(() => {
  'use strict';

  const CT = (globalThis.CT ||= {});

  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ESCAPES[c]);

  /* ------------------------------------------------------------------- maths --- */

  /** Commands that stand for a single character once the backslash is gone. */
  const SYMBOLS = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ',
    eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν',
    xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', upsilon: 'υ', phi: 'φ', varphi: 'φ',
    chi: 'χ', psi: 'ψ', omega: 'ω',
    Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ',
    Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
    times: '×', div: '÷', pm: '±', mp: '∓', cdot: '⋅', ast: '∗', star: '⋆',
    leq: '≤', le: '≤', geq: '≥', ge: '≥', neq: '≠', ne: '≠', approx: '≈', equiv: '≡',
    sim: '∼', simeq: '≃', cong: '≅', propto: '∝',
    in: '∈', notin: '∉', ni: '∋', subset: '⊂', subseteq: '⊆', supset: '⊃', supseteq: '⊇',
    cup: '∪', cap: '∩', setminus: '∖', emptyset: '∅', varnothing: '∅',
    infty: '∞', partial: '∂', nabla: '∇', forall: '∀', exists: '∃', neg: '¬',
    land: '∧', lor: '∨', wedge: '∧', vee: '∨',
    to: '→', rightarrow: '→', longrightarrow: '⟶', leftarrow: '←', Rightarrow: '⇒',
    Leftarrow: '⇐', leftrightarrow: '↔', Leftrightarrow: '⇔', mapsto: '↦',
    ldots: '…', cdots: '⋯', dots: '…', vdots: '⋮', ddots: '⋱',
    prime: '′', circ: '∘', bullet: '∙', oplus: '⊕', otimes: '⊗', perp: '⊥', angle: '∠',
    sum: '∑', prod: '∏', int: '∫', iint: '∬', oint: '∮', bigcup: '⋃', bigcap: '⋂',
    top: '⊤', bot: '⊥', dagger: '†', ddagger: '‡', sqcup: '⊔', sqcap: '⊓',
    lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉', backslash: '∖', ell: 'ℓ',
    Re: 'ℜ', Im: 'ℑ', aleph: 'ℵ', hbar: 'ℏ', deg: '°', degree: '°',
    // Brackets written without \left and \right, which is how most prose writes them.
    langle: '⟨', rangle: '⟩', lbrace: '{', rbrace: '}', lbrack: '[', rbrack: ']',
    vert: '|', Vert: '‖', lVert: '‖', rVert: '‖', lvert: '|', rvert: '|',
    // Shapes and the relations built on them.
    triangle: '△', bigtriangleup: '△', bigtriangledown: '▽', triangleq: '≜',
    square: '□', blacksquare: '■', diamond: '⋄', Diamond: '◊', bullet2: '•',
    // Comparisons beyond the basic four.
    ll: '≪', gg: '≫', prec: '≺', succ: '≻', preceq: '⪯', succeq: '⪰',
    leqslant: '⩽', geqslant: '⩾', subsetneq: '⊊', supsetneq: '⊋', sqsubseteq: '⊑',
    doteq: '≐', asymp: '≍', bowtie: '⋈', models: '⊨', vdash: '⊢', dashv: '⊣',
    parallel: '∥', nparallel: '∦', mid2: '∣', nmid: '∤', ncong: '≇', nsubseteq: '⊈',
    // More of the algebra of sets and operators.
    uplus: '⊎', amalg: '⨿', odot: '⊙', ominus: '⊖', oslash: '⊘', boxplus: '⊞',
    boxtimes: '⊠', ltimes: '⋉', rtimes: '⋊', intercal: '⊺', complement: '∁',
    // Reasoning, and the arrows that carry it.
    therefore: '∴', because: '∵', implies: '⟹', impliedby: '⟸', iff: '⟺',
    gets: '←', uparrow: '↑', downarrow: '↓', updownarrow: '↕', Uparrow: '⇑',
    Downarrow: '⇓', nearrow: '↗', searrow: '↘', swarrow: '↙', nwarrow: '↖',
    hookrightarrow: '↪', hookleftarrow: '↩', twoheadrightarrow: '↠', rightsquigarrow: '⇝',
    // Odds and ends that turn up in prose.
    nexists: '∄', wp: '℘', imath: 'ı', jmath: 'ȷ', flat: '♭', sharp: '♯',
    natural: '♮', clubsuit: '♣', diamondsuit: '♦', heartsuit: '♥', spadesuit: '♠',
    S: '§', P: '¶', copyright: '©', pounds: '£', yen: '¥', checkmark: '✓',
  };
  /** How many lines of a code block a tree box can show before it is cut off. */
  const PREVIEW_LINES = 4;
  /** Characters that MathML would otherwise grow to the height of whatever they sit beside. */
  const BRACKETS = '()[]{}|';
  /** Commands that set a word upright rather than in maths italic. */
  const FUNCTIONS = ['sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'lim', 'limsup', 'liminf', 'max', 'min',
    'sup', 'inf', 'det', 'dim', 'ker', 'deg', 'gcd', 'arg', 'Pr'];
  /** Alphabets, mapped to the MathML variant that draws them. */
  const ALPHABETS = {
    mathbb: 'double-struck', mathcal: 'script', mathfrak: 'fraktur', mathbf: 'bold',
    mathit: 'italic', mathrm: 'normal', mathsf: 'sans-serif', mathtt: 'monospace',
    boldsymbol: 'bold', bm: 'bold',
  };
  /** What \left and \right may be given, and the bracket each one draws. */
  const FENCES = {
    '(': '(', ')': ')', '[': '[', ']': ']', '\\{': '{', '\\}': '}',
    '\\langle': '⟨', '\\rangle': '⟩', '|': '|', '\\|': '‖', '.': '',
  };
  /**
   * Characters that LaTeX reserves and a message escapes to write literally.
   *
   * `\{` and `\}` are the common ones — a set written as `\{\omega\}` is ordinary in any
   * conversation about probability — and without them the whole expression fell back to being
   * printed as source.
   */
  const LITERALS = {
    '{': '{', '}': '}', '|': '|', $: '$', '%': '%', '&': '&', _: '_', '#': '#',
  };
  /** Marks that sit over the thing they belong to. */
  const ACCENTS = {
    hat: '^', widehat: '^', bar: '‾', overline: '‾', vec: '→', tilde: '~', widetilde: '~',
    dot: '˙', ddot: '¨', check: 'ˇ', breve: '˘', acute: '´', grave: '`',
  };
  /** Operators that take their scripts above and below rather than beside. */
  const LARGE = /[∑∏∫∬∮⋃⋂]/;

  /**
   * Read one LaTeX expression and write it as MathML.
   *
   * Deliberately partial: it covers what turns up in a conversation — fractions, powers,
   * roots, sums with limits, the Greek alphabet, set and relation symbols — and throws on
   * anything else, so the caller can fall back rather than print nonsense.
   */
  function mathml(latex, display) {
    const source = String(latex ?? '');
    let at = 0;

    const peek = () => source[at];
    const skipSpace = () => { while (source[at] === ' ') at += 1; };
    const command = () => {
      const match = /^\\([a-zA-Z]+|.)/.exec(source.slice(at));
      if (!match) return null;
      at += match[0].length;
      return match[1];
    };

    /** One atom: a group, a command, a number, or a single character. */
    function atom() {
      skipSpace();
      if (at >= source.length) return null;

      const ch = peek();
      if (ch === '{') { at += 1; return `<mrow>${list('}')}</mrow>`; }
      if (ch === '\\') return backslash();

      const number = /^\d+(?:[.,]\d+)?/.exec(source.slice(at));
      if (number) { at += number[0].length; return `<mn>${esc(number[0])}</mn>`; }

      if (/[A-Za-z]/.test(ch)) { at += 1; return `<mi>${esc(ch)}</mi>`; }

      at += 1;
      if (ch === '&') return '';
      /*
       * A closing bracket is a character like any other here. Treating it as the end of the
       * expression truncated everything after it — `[uv]_a^b - \int v\,du` stopped at `uv`,
       * because the `]` ended the whole run rather than just standing for itself. The only
       * thing that closes a group is `list` reaching the terminator it was asked for.
       *
       * Brackets are also told not to stretch: MathML grows them to the height of their row
       * by default, so an ordinary `f(x)` beside a fraction came out with parentheses as tall
       * as the fraction. Only the ones a message asked to stretch, with \left and \right, do.
       */
      return BRACKETS.includes(ch)
        ? `<mo stretchy="false">${esc(ch)}</mo>`
        : `<mo>${esc(ch)}</mo>`;
    }

    function backslash() {
      const name = command();
      if (name === null) throw new Error('stray backslash');

      if (name === 'frac' || name === 'dfrac' || name === 'tfrac') {
        return `<mfrac>${argument()}${argument()}</mfrac>`;
      }
      if (name === 'binom') {
        return `<mrow><mo>(</mo><mfrac linethickness="0">${argument()}${argument()}</mfrac>`
          + '<mo>)</mo></mrow>';
      }
      if (name === 'sqrt') {
        skipSpace();
        if (peek() === '[') {
          at += 1;
          const index = list(']');
          return `<mroot>${argument()}<mrow>${index}</mrow></mroot>`;
        }
        return `<msqrt>${argument()}</msqrt>`;
      }
      if (name === 'text' || name === 'textrm' || name === 'mbox' || name === 'operatorname') {
        return `<mtext>${esc(braced())}</mtext>`;
      }
      if (ALPHABETS[name]) return `<mi mathvariant="${ALPHABETS[name]}">${esc(braced())}</mi>`;
      if (name === 'left') {
        const open = fence();
        const inner = list('\\right');
        const close = fence();
        return `<mrow>${open ? `<mo stretchy="true">${esc(open)}</mo>` : ''}${inner}`
          + `${close ? `<mo stretchy="true">${esc(close)}</mo>` : ''}</mrow>`;
      }
      if (FUNCTIONS.includes(name)) return `<mi mathvariant="normal">${esc(name)}</mi>`;
      if (Object.hasOwn(SYMBOLS, name)) {
        const glyph = SYMBOLS[name];
        return /[A-Za-zΑ-Ωα-ω]/.test(glyph) ? `<mi>${esc(glyph)}</mi>` : `<mo>${esc(glyph)}</mo>`;
      }
      if (name === 'begin') return environment(braced());
      // Arrows that carry a label, as a limit or a convergence does.
      if (name === 'xrightarrow' || name === 'xleftarrow' || name === 'xrightharpoonup') {
        const arrow = name === 'xleftarrow' ? '←' : '→';
        return `<mover><mo stretchy="true">${arrow}</mo>${argument()}</mover>`;
      }
      if (name === 'overset' || name === 'stackrel') {
        const over = argument();
        return `<mover>${argument()}${over}</mover>`;
      }
      if (name === 'underset') {
        const under = argument();
        return `<munder>${argument()}${under}</munder>`;
      }
      if (Object.hasOwn(ACCENTS, name)) {
        return `<mover accent="true">${argument()}<mo>${esc(ACCENTS[name])}</mo></mover>`;
      }
      if (name === 'underline') return `<munder>${argument()}<mo>‾</mo></munder>`;
      // Size and layout hints that MathML works out for itself.
      if (/^(display|text|script|scriptscript)style$|^(bigg?|Bigg?)[lrm]?$|^n?o?limits$|^!$/.test(name)) {
        return '';
      }
      if (Object.hasOwn(LITERALS, name)) return `<mo>${esc(LITERALS[name])}</mo>`;
      if (name === 'mid') return '<mo>|</mo>';
      if (name === 'lvert' || name === 'rvert') return '<mo>|</mo>';
      if (name === 'lVert' || name === 'rVert') return '<mo>‖</mo>';
      if (name === 'colon') return '<mo>:</mo>';
      if (name === 'quad' || name === 'qquad') return '<mspace width="1em"/>';
      if (name === ',' || name === ';' || name === ':' || name === ' ' || name === '!') {
        return '<mspace width="0.2em"/>';
      }
      if (name === '\\') return '';

      /*
       * An unrecognised command is one glyph, not a broken formula.
       *
       * Throwing here sent the whole expression to the fallback, so a single symbol nobody
       * had listed — `\triangle`, `\langle` — cost the reader an entire line of rendered
       * maths and gave them LaTeX source instead. The table below can never be complete;
       * treating a gap in it as a formatting detail rather than a parse failure is what stops
       * that. Structural faults — an unclosed group, a fraction missing a half — still throw,
       * because what they would produce is not worth showing.
       */
      return `<mi mathvariant="normal">${esc(name)}</mi>`;
    }

    /**
     * A matrix, a set of cases, or aligned lines: rows split on `\\`, cells on `&`.
     *
     * Each cell is read on its own, so the parser does not need to know what a table is;
     * it only needs to know where the pieces are.
     */
    function environment(name) {
      if (name === 'array') { skipSpace(); if (peek() === '{') braced(); } // the column spec
      const closing = `\\end{${name}}`;
      const stop = source.indexOf(closing, at);
      if (stop < 0) throw new Error(`unclosed ${name}`);
      const inner = source.slice(at, stop);
      at = stop + closing.length;

      const rows = inner.split('\\\\').map((row) => row.trim()).filter(Boolean);
      if (!rows.length) throw new Error(`empty ${name}`);
      const table = `<mtable>${rows.map((row) => `<mtr>${row.split('&')
        .map((cell) => `<mtd>${fragment(cell)}</mtd>`).join('')}</mtr>`).join('')}</mtable>`;

      const around = MATRIX_FENCES[name];
      if (!around) return table;
      return `<mrow><mo stretchy="true">${esc(around[0])}</mo>${table}`
        + `${around[1] ? `<mo stretchy="true">${esc(around[1])}</mo>` : ''}</mrow>`;
    }

    /** The token after \left or \right: a bare bracket, or a command that draws one. */
    function fence() {
      skipSpace();
      const start = at;
      const token = source[at] === '\\' ? (command(), source.slice(start, at)) : source[at += 1, start];
      const written = source.slice(start, at);
      // Same reasoning as above: an unfamiliar bracket is drawn as written, not fatal.
      return FENCES[written] ?? written.replace(/^\\/, '');
    }

    /** The raw text of a {…} group, for the commands that take words rather than maths. */
    function braced() {
      skipSpace();
      if (source[at] !== '{') throw new Error('expected a group');
      at += 1;
      const start = at;
      let depth = 1;
      while (at < source.length && depth > 0) {
        if (source[at] === '{') depth += 1;
        else if (source[at] === '}') depth -= 1;
        at += 1;
      }
      if (depth) throw new Error('unclosed group');
      return source.slice(start, at - 1);
    }

    /** The next atom, wrapped so it can stand as a fraction's half or a root's body. */
    function argument() {
      const next = scripted();
      if (next === null) throw new Error('missing argument');
      return next.startsWith('<mrow>') ? next : `<mrow>${next}</mrow>`;
    }

    /** An atom together with any ^ and _ hanging off it. */
    function scripted() {
      let base = atom();
      if (base === null) return null;
      for (;;) {
        skipSpace();
        const mark = peek();
        if (mark !== '^' && mark !== '_') break;
        at += 1;
        const script = atom();
        if (script === null) throw new Error('missing script');
        const stacks = LARGE.test(base);
        if (mark === '^') base = stacks ? `<mover>${base}${script}</mover>` : `<msup>${base}${script}</msup>`;
        else base = stacks ? `<munder>${base}${script}</munder>` : `<msub>${base}${script}</msub>`;
      }
      return base;
    }

    function list(stop) {
      let out = '';
      for (;;) {
        skipSpace();
        if (at >= source.length) {
          if (stop) throw new Error('unclosed group');
          break;
        }
        if (stop === '}' && peek() === '}') { at += 1; break; }
        if (stop === ']' && peek() === ']') { at += 1; break; }
        if (stop === '\\right' && source.startsWith('\\right', at)) { at += 6; break; }
        const next = scripted();
        if (next === null) break;
        out += next;
      }
      return out;
    }

    const body = list(null);
    if (!body) throw new Error('empty');
    return `<math xmlns="http://www.w3.org/1998/Math/MathML"${display ? ' display="block"' : ''}>`
      + `<mrow>${body}</mrow></math>`;
  }

  /** The brackets each table environment draws around itself. */
  const MATRIX_FENCES = {
    pmatrix: ['(', ')'], bmatrix: ['[', ']'], Bmatrix: ['{', '}'],
    vmatrix: ['|', '|'], Vmatrix: ['‖', '‖'], cases: ['{', ''],
  };

  /** One expression rendered on its own, for a table cell. */
  function fragment(latex) {
    return mathml(latex, false)
      .replace(/^<math[^>]*><mrow>/, '')
      .replace(/<\/mrow><\/math>$/, '');
  }

  /** MathML where the LaTeX can be read, the LaTeX itself where it cannot. */
  function math(latex, display) {
    try {
      return mathml(latex, display);
    } catch {
      return `<code class="ct-math-raw">${esc(latex)}</code>`;
    }
  }

  /* --------------------------------------------------------------------- svg --- */

  const SVG_TAGS = new Set(['svg', 'g', 'path', 'rect', 'circle', 'ellipse', 'line', 'polyline',
    'polygon', 'text', 'tspan', 'defs', 'lineargradient', 'radialgradient', 'stop', 'title',
    'desc', 'symbol', 'marker', 'clippath', 'mask', 'pattern']);
  const SVG_ATTR = /^(d|fill|stroke|stroke-width|stroke-linecap|stroke-linejoin|stroke-dasharray|opacity|fill-opacity|stroke-opacity|x|y|x1|y1|x2|y2|cx|cy|r|rx|ry|width|height|points|transform|viewBox|preserveAspectRatio|offset|stop-color|stop-opacity|gradientUnits|gradientTransform|class|id|font-size|font-family|font-weight|text-anchor|dominant-baseline|dx|dy)$/;

  /**
   * Rebuild an SVG from a whitelist.
   *
   * A message is text from a conversation, so its markup is not trusted. Rather than filter
   * out what looks dangerous, this copies across only the elements and attributes named
   * above: scripts, event handlers, foreign objects and anything that could fetch are left
   * behind because they are never copied in the first place.
   */
  function safeSvg(markup) {
    let parsed;
    try {
      parsed = new DOMParser().parseFromString(String(markup ?? ''), 'image/svg+xml');
    } catch {
      return null;
    }
    if (parsed.querySelector('parsererror')) return null;
    const root = parsed.documentElement;
    if (!root || root.nodeName.toLowerCase() !== 'svg') return null;

    const copy = (source) => {
      const name = source.nodeName.toLowerCase();
      if (!SVG_TAGS.has(name)) return null;
      const made = document.createElementNS('http://www.w3.org/2000/svg', source.nodeName);
      for (const attr of source.attributes || []) {
        if (!SVG_ATTR.test(attr.name)) continue;
        // References may point within this drawing, never out of it.
        if (/url\s*\(\s*['"]?(?!#)/i.test(attr.value)) continue;
        made.setAttribute(attr.name, attr.value);
      }
      for (const child of source.childNodes) {
        if (child.nodeType === 3) made.appendChild(document.createTextNode(child.nodeValue));
        else if (child.nodeType === 1) {
          const kid = copy(child);
          if (kid) made.appendChild(kid);
        }
      }
      return made;
    };

    const clean = copy(root);
    if (!clean) return null;
    // Let the pane decide how big it is; keep the aspect ratio the drawing came with.
    clean.removeAttribute('width');
    clean.removeAttribute('height');
    clean.setAttribute('class', 'ct-svg');
    return clean.outerHTML;
  }

  /* --------------------------------------------------------------------- code --- */

  /**
   * Enough grammar to colour a code block, and no more.
   *
   * A real parser per language is not worth carrying in a side pane: comments, strings,
   * numbers and keywords are what make a block readable at a glance, and they can be told
   * apart with one pass. Anything unrecognised is left as plain text rather than guessed at.
   */
  const KEYWORDS = {
    js: 'const let var function return if else for while do class extends new await async import export from as of in typeof instanceof null undefined true false this try catch finally throw switch case break continue default yield delete void static get set',
    py: 'def class return if elif else for while import from as pass break continue lambda None True False and or not in is with try except finally raise yield global nonlocal assert async await del',
    sh: 'if then else elif fi for while do done case esac function return export local readonly source echo cd set unset trap exit',
    sql: 'select from where group by having order limit insert into values update set delete create table drop alter join left right inner outer on as and or not null distinct union',
    css: 'important media supports keyframes import font-face root',
    xml: '',
    json: 'true false null',
  };
  const DIALECTS = {
    js: 'js', javascript: 'js', jsx: 'js', ts: 'js', typescript: 'js', tsx: 'js', mjs: 'js',
    py: 'py', python: 'py', rb: 'py', ruby: 'py',
    sh: 'sh', bash: 'sh', shell: 'sh', zsh: 'sh', console: 'sh',
    sql: 'sql', css: 'css', scss: 'css',
    html: 'xml', xml: 'xml', svg: 'xml',
    json: 'json',
  };

  /** One pass over the source, emitting escaped text with the interesting parts wrapped. */
  function highlight(code, language) {
    const dialect = DIALECTS[String(language || '').toLowerCase()];
    const source = String(code ?? '');
    if (!dialect) return esc(source);

    const comment = dialect === 'py' || dialect === 'sh' ? /#[^\n]*/
      : dialect === 'sql' ? /--[^\n]*|\/\*[\s\S]*?\*\//
        : dialect === 'css' || dialect === 'xml' ? /\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->/
          : /\/\/[^\n]*|\/\*[\s\S]*?\*\//;
    const words = new Set((KEYWORDS[dialect] || '').split(' ').filter(Boolean));

    // Written as literals so the escaping stays readable, then joined into one pass.
    const strings = /`(?:\\.|[^`\\])*`|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"/;
    const scanner = new RegExp([
      `(?<comment>${comment.source})`,
      `(?<string>${strings.source})`,
      '(?<tag></?[A-Za-z][\\w:-]*)',
      '(?<number>\\b\\d[\\w.]*\\b)',
      '(?<word>[A-Za-z_$][\\w$-]*)',
    ].join('|'), 'g');

    let out = '';
    let at = 0;
    for (const match of source.matchAll(scanner)) {
      out += esc(source.slice(at, match.index));
      at = match.index + match[0].length;
      const { comment: c, string: str, tag, number, word } = match.groups;
      if (c !== undefined) out += `<span class="ct-tok-note">${esc(c)}</span>`;
      else if (str !== undefined) out += `<span class="ct-tok-text">${esc(str)}</span>`;
      else if (tag !== undefined && dialect === 'xml') out += `<span class="ct-tok-word">${esc(tag)}</span>`;
      else if (number !== undefined) out += `<span class="ct-tok-number">${esc(number)}</span>`;
      else if (word !== undefined && words.has(word)) out += `<span class="ct-tok-word">${esc(word)}</span>`;
      else out += esc(match[0]);
    }
    return out + esc(source.slice(at));
  }

  /* ---------------------------------------------------------------- markdown --- */

  /**
   * Inline markup.
   *
   * Maths and code are lifted out first and parked behind placeholders, so that a formula
   * full of underscores and asterisks cannot be mistaken for emphasis on the way past.
   */
  function inline(text) {
    const parked = [];
    const park = (html) => ` ${parked.push(html) - 1} `;

    let out = String(text ?? '')
      // A fenced block, kept as a block: the tree's boxes used to flatten code into one long
      // line of prose, which is unreadable — it should look the way it does in the drawer.
      .replace(/```([\w-]*)\n?([\s\S]*?)```/g, (_, tongue, code) => park(
        `<code class="ct-code"${tongue ? ` data-lang="${esc(tongue)}"` : ''}>`
        + `${highlight(code.replace(/\s+$/, ''), tongue)}</code>`))
      // A fence that was never closed — a message cut short, or a preview of one.
      .replace(/```([\w-]*)\n([\s\S]*)$/, (_, tongue, code) => park(
        `<code class="ct-code"${tongue ? ` data-lang="${esc(tongue)}"` : ''}>`
        + `${highlight(code.replace(/\s+$/, ''), tongue)}</code>`))
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => park(math(tex, true)))
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => park(math(tex, true)))
      .replace(/`([^`]+)`/g, (_, code) => park(`<code>${esc(code)}</code>`))
      .replace(/(^|[^\\$])\$([^$\n]+?)\$/g, (_, before, tex) => before + park(math(tex, false)))
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => park(math(tex, false)));

    out = esc(out)
      .replace(/\[([^\]]+)\]\(([^)\s]+)[^)]*\)/g, (_, label, href) => (/^https?:\/\//i.test(href)
        ? `<a href="${esc(href)}" target="_blank" rel="noreferrer noopener">${label}</a>`
        : label))
      .replace(/(\*\*|__)(?=\S)([\s\S]*?\S)\1/g, '<strong>$2</strong>')
      .replace(/(?<![\w*])(\*|_)(?=\S)([^*_\n]*?\S)\1(?![\w*])/g, '<em>$2</em>')
      .replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>');

    return out.replace(/ (\d+) /g, (_, i) => parked[Number(i)]);
  }

  /**
   * Block markup: headings, lists, quotes, fenced code and paragraphs.
   *
   * Small on purpose. It covers what a chat message actually contains, and anything it does
   * not recognise stays a paragraph rather than disappearing.
   */
  function markdown(text) {
    const lines = String(text ?? '').replace(/\r\n?/g, '\n').split('\n');
    const html = [];
    let open = null;

    const closeList = () => { if (open) { html.push(`</${open}>`); open = null; } };
    const starts = (line) => /^\s{0,3}(#{1,6}\s|>|```|[-*+]\s|\d+[.)]\s|(?:---+|\*\*\*+|___+)\s*$)/
      .test(line);

    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];

      const fence = /^\s{0,3}```(\w*)/.exec(line);
      if (fence) {
        closeList();
        const body = [];
        i += 1;
        while (i < lines.length && !/^\s{0,3}```/.test(lines[i])) { body.push(lines[i]); i += 1; }
        const code = body.join('\n');
        if (/^svg$/i.test(fence[1]) || /^\s*<svg[\s>]/i.test(code)) {
          const drawn = safeSvg(code);
          if (drawn) { html.push(`<figure class="ct-figure">${drawn}</figure>`); continue; }
        }
        const tongue = fence[1] ? ` data-lang="${esc(fence[1])}"` : '';
        html.push(`<pre${tongue}><code>${highlight(code, fence[1])}</code></pre>`);
        continue;
      }

      // A drawing written straight into the message, outside any fence. Rendered where it
      // stands: escaping it into the paragraph showed the source, and appending it separately
      // showed the drawing a second time underneath.
      if (/^\s{0,3}<svg[\s>]/i.test(line)) {
        closeList();
        const body = [line];
        while (i < lines.length && !/<\/svg>/i.test(lines[i])) { i += 1; if (i < lines.length) body.push(lines[i]); }
        const drawn = safeSvg(body.join('\n'));
        html.push(drawn ? `<figure class="ct-figure">${drawn}</figure>`
          : `<pre><code>${esc(body.join('\n'))}</code></pre>`);
        continue;
      }

      if (/^\s*$/.test(line)) { closeList(); continue; }

      const heading = /^\s{0,3}(#{1,6})\s+(.*)$/.exec(line);
      if (heading) {
        closeList();
        // Two levels down: a first-level heading in a side pane is shouting.
        html.push(`<h${Math.min(6, heading[1].length + 2)}>${inline(heading[2])}`
          + `</h${Math.min(6, heading[1].length + 2)}>`);
        continue;
      }

      const quote = /^\s{0,3}>\s?(.*)$/.exec(line);
      if (quote) { closeList(); html.push(`<blockquote>${inline(quote[1])}</blockquote>`); continue; }

      if (/^\s{0,3}(?:---+|\*\*\*+|___+)\s*$/.test(line)) { closeList(); html.push('<hr>'); continue; }

      const bullet = /^\s{0,3}[-*+]\s+(.*)$/.exec(line);
      const numbered = /^\s{0,3}\d+[.)]\s+(.*)$/.exec(line);
      if (bullet || numbered) {
        const want = bullet ? 'ul' : 'ol';
        if (open !== want) { closeList(); html.push(`<${want}>`); open = want; }
        html.push(`<li>${inline((bullet || numbered)[1])}</li>`);
        continue;
      }

      closeList();
      const paragraph = [line];
      while (i + 1 < lines.length && !/^\s*$/.test(lines[i + 1]) && !starts(lines[i + 1])) {
        i += 1;
        paragraph.push(lines[i]);
      }
      html.push(`<p>${inline(paragraph.join('\n')).replace(/\n/g, '<br>')}</p>`);
    }

    closeList();
    return html.join('');
  }

  /**
   * A message's own inline SVG, when it wrote one outside a code fence.
   * Returned separately so a drawing can be shown whole rather than inside a paragraph.
   */
  function drawings(text) {
    const found = [];
    const source = String(text ?? '');
    const pattern = /<svg[\s>][\s\S]*?<\/svg>/gi;
    for (const match of source.match(pattern) || []) {
      const drawn = safeSvg(match);
      if (drawn) found.push(drawn);
    }
    return found;
  }

  /** Thumbnails for the pictures a message carries. */
  function images(files) {
    const pictures = (files || [])
      .filter((file) => /image/i.test(file?.file_kind || file?.file_type || ''));
    const shots = pictures.map((file) => {
      const src = file.thumbnail_url || file.preview_url;
      // Same-origin paths only: the pane runs on claude.ai and these are its own API.
      if (!src || !src.startsWith('/')) return '';
      return `<img class="ct-shot" src="${esc(src)}" loading="lazy"`
        + ` alt="${esc(file.file_name || 'attachment')}">`;
    }).filter(Boolean);
    return shots.length ? `<div class="ct-shots">${shots.join('')}</div>` : '';
  }

  /**
   * A message reduced to one flowing line, with its inline markup drawn.
   *
   * For the tree's own boxes, which are four lines tall. Headings, bullets and quote marks
   * are dropped rather than rendered — a heading in a box that size is just a line of text,
   * and leaving the `##` in was worse than either. Fences lose their rails but keep their
   * contents, so a message that is only code still shows something.
   */
  function preview(text) {
    const flattened = String(text ?? '')
      .replace(/<svg[\s>][\s\S]*?<\/svg>/gi, ' ')
      // Code keeps its shape here too; only as many lines as the box can show are kept.
      .replace(/```([\w-]*)\n?([\s\S]*?)```/g, (_, tongue, code) => {
        const lines = code.replace(/\s+$/, '').split('\n');
        const kept = lines.slice(0, PREVIEW_LINES);
        if (lines.length > kept.length) kept.push('…');
        return `\n\`\`\`${tongue}\n${kept.join('\n')}\n\`\`\`\n`;
      })
      .replace(/^\s{0,3}#{1,6}\s+/gm, '')
      .replace(/^\s{0,3}>\s?/gm, '')
      .replace(/^\s{0,3}(?:[-*+]|\d+[.)])\s+/gm, '')
      .replace(/^\s{0,3}(?:---+|\*\*\*+|___+)\s*$/gm, ' ');
    return inline(flattened);
  }

  CT.render = { inline, preview, markdown, math, highlight, safeSvg, drawings, images, esc };
})();
