/**
 * Resolves puppeteer wherever it happens to be installed.
 *
 * The browser suites need a real Chrome. Puppeteer may be a local dependency, a global
 * install, or come bundled with a global CLI, so each is tried in turn rather than pinning
 * one machine's path.
 */
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';

const require = createRequire(import.meta.url);

function globalRoot() {
  try {
    return execSync('npm root -g', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return null;
  }
}

const candidates = ['puppeteer', 'puppeteer-core'];
const root = globalRoot();
if (root) {
  candidates.push(`${root}/puppeteer`, `${root}/puppeteer-cli/node_modules/puppeteer`);
}

let puppeteer = null;
const tried = [];
for (const candidate of candidates) {
  try {
    puppeteer = require(candidate);
    break;
  } catch {
    tried.push(candidate);
  }
}

if (!puppeteer) {
  throw new Error(
    'puppeteer not found — install it with `npm i -D puppeteer` (or globally).\n' +
    `Looked in: ${tried.join(', ')}`
  );
}

export default puppeteer.default ?? puppeteer;
