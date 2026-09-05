import fs from 'node:fs';
import vm from 'node:vm';

const ctx = { globalThis: null, console };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(new URL('../../src/model.js', import.meta.url), 'utf8'), ctx);
const { buildTree, computeLayout } = ctx.CT.model;

const ROOT = '00000000-0000-4000-8000-000000000000';
let t = 0;
const msg = (uuid, parent, sender, text) => ({
  uuid, parent_message_uuid: parent, sender,
  content: [{ type: 'text', text }],
  created_at: new Date(Date.UTC(2026, 0, 1, 0, 0, t++)).toISOString(),
  index: 0, attachments: [], files: [],
});

// u1 -> a1 -> u2 -> {a2, a2b(regenerate)}   and a2 -> u3 -> a3
const convo = {
  name: 'Branching chat',
  current_leaf_message_uuid: 'a3',
  chat_messages: [
    msg('u1', ROOT, 'human', 'Hello there'),
    msg('a1', 'u1', 'assistant', 'Hi! How can I help?'),
    msg('u2', 'a1', 'human', 'Explain trees'),
    msg('a2', 'u2', 'assistant', 'A tree is an acyclic graph.'),
    msg('a2b', 'u2', 'assistant', 'Trees are hierarchical structures.'),
    msg('u3', 'a2', 'human', 'More detail'),
    msg('a3', 'u3', 'assistant', 'Sure, here is more.'),
    msg('u4', 'a2b', 'human', 'A different follow-up'),
  ],
};

const tree = buildTree(convo);
const assert = (cond, label) => console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}`);

assert(tree.roots.length === 1 && tree.roots[0].id === 'u1', 'single root u1');
assert(tree.stats.messages === 8, `8 messages (got ${tree.stats.messages})`);
assert(tree.stats.forks === 1, `1 fork (got ${tree.stats.forks})`);
assert(tree.stats.branches === 2, `2 leaves/branches (got ${tree.stats.branches})`);
assert(tree.stats.depth === 6, `depth 6 (got ${tree.stats.depth})`);
assert(tree.leafId === 'a3', 'active leaf a3');
assert(tree.stats.pathLength === 6, `active path u1..a3 = 6 (got ${tree.stats.pathLength})`);
assert(!tree.nodes.get('a2b').onPath && tree.nodes.get('a2').onPath, 'off-path branch flagged');
const a2b = tree.nodes.get('a2b');
assert(a2b.siblingCount === 2 && a2b.siblingIndex === 1, `a2b is variant 2/2 (got ${a2b.siblingIndex + 1}/${a2b.siblingCount})`);

// Layout: variable node sizes, no overlap, rows aligned.
const heights = {};                       // deliberately uneven, like real previews
tree.order.forEach((n, i) => { heights[n.id] = [40, 96, 62, 40, 96][i % 5]; });
const sizeOf = (n) => heights[n.id];

for (const [name, conf] of Object.entries({
  vertical: { spreadOf: () => 315, depthOf: sizeOf, spreadGap: 24, depthGap: 34 },
  horizontal: { spreadOf: sizeOf, depthOf: () => 371, spreadGap: 18, depthGap: 60 },
})) {
  const extent = computeLayout(tree.roots, conf);

  const byDepth = new Map();
  let maxEdge = 0;
  for (const n of tree.order) {
    if (!byDepth.has(n.depth)) byDepth.set(n.depth, []);
    byDepth.get(n.depth).push(n);
    maxEdge = Math.max(maxEdge, n.s + n.sSize);
  }

  let ok = true;
  let aligned = true;
  for (const row of byDepth.values()) {
    row.sort((a, b) => a.s - b.s);
    for (let i = 1; i < row.length; i++) {
      if (row[i].s < row[i - 1].s + row[i - 1].sSize) ok = false;
    }
    if (row.some((n) => n.d !== row[0].d)) aligned = false;
  }
  assert(ok, `${name}: no overlaps at any depth, with uneven sizes`);
  assert(aligned, `${name}: every node on a depth shares its row`);
  assert(Math.abs(extent.spread - maxEdge) < 0.001, `${name}: extent.spread ${extent.spread} == far edge ${maxEdge}`);

  // a one-line node must not be padded out to the tallest node's height
  assert(tree.order.some((n) => n.dSize !== tree.order[0].dSize) || name === 'horizontal',
    `${name}: node sizes vary with content`);

  const fork = tree.nodes.get('u2');
  const kids = fork.children;
  const centre = (n) => n.s + n.sSize / 2;
  assert(Math.abs(centre(fork) - (centre(kids[0]) + centre(kids[1])) / 2) < 0.001,
    `${name}: fork centred over its children`);
}

// A node wider than its whole subtree must still not collide with the next subtree.
{
  const wide = (n) => (n.children.length === 0 ? 20 : 400);
  computeLayout(tree.roots, { spreadOf: wide, depthOf: () => 50, spreadGap: 10, depthGap: 10 });
  const byDepth = new Map();
  for (const n of tree.order) {
    if (!byDepth.has(n.depth)) byDepth.set(n.depth, []);
    byDepth.get(n.depth).push(n);
  }
  let ok = true;
  for (const row of byDepth.values()) {
    row.sort((a, b) => a.s - b.s);
    for (let i = 1; i < row.length; i++) if (row[i].s < row[i - 1].s + row[i - 1].sSize) ok = false;
  }
  assert(ok, 'oversized parents do not collide with neighbouring subtrees');
}

// A branch must end on a reply: claude.ai refuses any other message as the current leaf.
{
  const { deepestLeaf } = ctx.CT.model;
  const t2 = buildTree({ chat_messages: [
    msg('h1', ROOT, 'human', 'first ask'),
    msg('a1', 'h1', 'assistant', 'first reply'),
    msg('h2', 'a1', 'human', 'a follow-up that never got answered'),
  ]});
  const fromHuman = deepestLeaf(t2.nodes.get('h1'));
  assert(fromHuman && fromHuman.sender === 'assistant',
    `deepestLeaf backs up to a reply (got ${fromHuman && fromHuman.sender})`);
  assert(fromHuman.id === 'a1', 'and picks the nearest one inside the branch');

  const t3 = buildTree({ chat_messages: [msg('only', ROOT, 'human', 'no reply at all')] });
  assert(deepestLeaf(t3.nodes.get('only')) === null, 'a branch with no reply yields null');

  const t4 = buildTree({ chat_messages: [
    msg('x1', ROOT, 'human', 'ask'), msg('x2', 'x1', 'assistant', 'reply'),
  ]});
  assert(deepestLeaf(t4.nodes.get('x1')).id === 'x2', 'a normal branch still ends on its reply');
}

// Degenerate inputs must not throw.
assert(buildTree({}).stats.messages === 0, 'empty payload handled');
assert(buildTree({ chat_messages: [] }).roots.length === 0, 'empty message list handled');
const orphan = buildTree({ chat_messages: [msg('x', 'missing-parent', 'human', 'orphan')] });
assert(orphan.roots.length === 1, 'orphaned message becomes a root');
assert(computeLayout([], { spreadOf: () => 1, depthOf: () => 1, spreadGap: 1, depthGap: 1 }).spread === 0, 'empty layout');
