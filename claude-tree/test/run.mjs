#!/usr/bin/env node
/**
 * Runs the regression suites.
 *
 *   node test/run.mjs                 every suite
 *   node test/run.mjs duplicate       only suites whose name contains "duplicate"
 *
 * The browser suites drive the real extension sources against a harness that stands in for
 * claude.ai: it serves the conversation API, virtualises the transcript, mounts the action
 * bar on hover the way Claude does, and renders the same row attributes. They need Chrome.
 */
import { spawn } from 'node:child_process';
import { readdirSync, mkdirSync, copyFileSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import net from 'node:net';

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, '..');
const PORT = 8765;

const filter = process.argv[2] ?? '';
const suites = readdirSync(join(here, 'suites'))
  .filter((f) => f.endsWith('.mjs') && f.includes(filter))
  .sort();

const run = (cmd, args, opts = {}) => new Promise((resolve) => {
  const child = spawn(cmd, args, { cwd: repo, ...opts });
  let out = '';
  child.stdout?.on('data', (d) => { out += d; });
  child.stderr?.on('data', (d) => { out += d; });
  child.on('close', (code) => resolve({ code, out }));
});

const waitForPort = () => new Promise((resolve, reject) => {
  const started = Date.now();
  const tick = () => {
    const socket = net.connect(PORT, '127.0.0.1');
    socket.on('connect', () => { socket.end(); resolve(); });
    socket.on('error', () => {
      socket.destroy();
      if (Date.now() - started > 10000) reject(new Error('harness server did not start'));
      else setTimeout(tick, 100);
    });
  };
  tick();
});

// The harness is served from the repo root so the suites load the real src/ files.
copyFileSync(join(here, 'support', 'harness.html'), join(repo, '_check.html'));
const shots = join(here, '.shots');
mkdirSync(shots, { recursive: true });
const server = spawn('python3', [join(here, 'support', 'serve.py')], { cwd: repo, stdio: 'ignore' });

const cleanup = () => {
  server.kill();
  try { rmSync(join(repo, '_check.html')); } catch { /* already gone */ }
};
process.on('exit', cleanup);
process.on('SIGINT', () => { cleanup(); process.exit(130); });

await waitForPort();

let passed = 0;
let failedSuites = 0;
for (const suite of suites) {
  const name = suite.replace(/\.mjs$/, '');
  const { out } = await run('node', [join(here, 'suites', suite), shots]);
  const oks = (out.match(/^PASS/gm) || []).length;
  // A suite that reported nothing did not pass: it crashed before it could check anything,
  // which used to be printed as "ok (0)" and quietly subtracted from the total.
  const bad = (out.match(/^FAIL/gm) || []).length || /PROBLEMS/.test(out) || oks === 0;
  if (bad) {
    failedSuites += 1;
    console.log(`FAIL  ${name}${oks === 0 ? '  (reported nothing)' : ''}`);
    for (const line of out.split('\n').filter((l) => /^FAIL|^PROBLEMS|^\s+at |^\w*Error/.test(l))) {
      console.log(`        ${line}`);
    }
  } else {
    passed += oks;
    console.log(`ok    ${name}  (${oks})`);
  }
}

cleanup();
console.log(`\n${failedSuites ? `${failedSuites} suite(s) failed` : 'all suites green'} — ${passed} checks`);
process.exit(failedSuites ? 1 : 0);
