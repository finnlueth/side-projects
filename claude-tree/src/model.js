/**
 * Turns the flat `chat_messages` array from the API into a tree, and lays that tree out.
 *
 * Claude stores a conversation as a DAG-free tree: every message carries a
 * `parent_message_uuid`. Editing a prompt or regenerating a reply creates a *sibling*
 * rather than replacing anything, which is exactly what this extension visualises.
 */
(() => {
  'use strict';

  const CT = (globalThis.CT ||= {});

  const collapse = (text) => String(text ?? '').replace(/\s+/g, ' ').trim();

  function snippet(text, max = 140) {
    const flat = collapse(text);
    if (flat.length <= max) return flat;
    return `${flat.slice(0, max - 1).trimEnd()}…`;
  }

  /**
   * Strip markdown syntax for the one-line previews on tree nodes. The chat renders this
   * away, so leaving `**` and `##` in would show the reader something they never saw.
   * The detail drawer still shows the message exactly as it was sent.
   */
  function plainPreview(text) {
    return String(text ?? '')
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`([^`]*)`/g, '$1')
      .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/^\s{0,3}#{1,6}\s+/gm, '')
      .replace(/^\s{0,3}>\s?/gm, '')
      .replace(/^\s{0,3}(?:[-*+]|\d+\.)\s+/gm, '')
      .replace(/(\*\*|__)(.*?)\1/g, '$2')
      .replace(/(\*|_)([^*_\n]+)\1/g, '$2')
      .replace(/~~(.*?)~~/g, '$1');
  }

  function blocksOfType(blocks, type) {
    return blocks.filter((block) => block && block.type === type);
  }

  /** Pull display text, hidden reasoning and tool usage out of one API message. */
  function describeMessage(message) {
    const blocks = Array.isArray(message.content) ? message.content : [];

    let body = blocksOfType(blocks, 'text')
      .map((block) => block.text || '')
      .join('\n\n')
      .trim();
    if (!body && typeof message.text === 'string') body = message.text.trim();

    const thinking = blocksOfType(blocks, 'thinking')
      .map((block) => block.thinking || block.text || '')
      .join('\n\n')
      .trim();

    const tools = [];
    for (const block of blocks) {
      if (block?.type !== 'tool_use') continue;
      const name = block.name || 'tool';
      if (!tools.includes(name)) tools.push(name);
    }

    const attachments = [
      ...(Array.isArray(message.attachments) ? message.attachments : []),
      ...(Array.isArray(message.files) ? message.files : []),
    ];

    let preview = plainPreview(body);
    if (!preview && thinking) preview = plainPreview(thinking);
    if (!preview && tools.length) preview = `Used ${tools.join(', ')}`;
    if (!preview && attachments.length) preview = `${attachments.length} attachment${attachments.length === 1 ? '' : 's'}`;

    return { body, thinking, tools, attachments, preview };
  }

  function compareMessages(a, b) {
    const at = Date.parse(a.createdAt || '') || 0;
    const bt = Date.parse(b.createdAt || '') || 0;
    if (at !== bt) return at - bt;
    const ai = Number.isFinite(a.index) ? a.index : 0;
    const bi = Number.isFinite(b.index) ? b.index : 0;
    if (ai !== bi) return ai - bi;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  }

  /**
   * @param {object} conversation raw API payload
   * @returns {{roots: object[], nodes: Map<string, object>, order: object[], stats: object,
   *           title: string, leafId: string|null}}
   */
  function buildTree(conversation) {
    const messages = Array.isArray(conversation?.chat_messages) ? conversation.chat_messages : [];
    const nodes = new Map();

    for (const message of messages) {
      if (!message?.uuid) continue;
      const described = describeMessage(message);
      nodes.set(message.uuid, {
        id: message.uuid,
        parentId: message.parent_message_uuid || null,
        sender: message.sender === 'human' ? 'human' : 'assistant',
        createdAt: message.created_at || null,
        index: message.index,
        text: described.body,
        thinking: described.thinking,
        tools: described.tools,
        attachments: described.attachments.length,
        files: described.attachments,
        preview: described.preview,
        children: [],
        parent: null,
        depth: 0,
        siblingIndex: 0,
        siblingCount: 1,
        onPath: false,
      });
    }

    // A message whose parent is not in this payload starts a branch of its own. That covers
    // the sentinel uuid Claude gives first messages (00000000-0000-4000-8000-000000000000)
    // and, usefully, the sibling roots created by editing the very first prompt.
    const roots = [];
    for (const node of nodes.values()) {
      const parent = node.parentId ? nodes.get(node.parentId) : null;
      if (parent && parent !== node) {
        node.parent = parent;
        parent.children.push(node);
      } else {
        roots.push(node);
      }
    }

    roots.sort(compareMessages);
    for (const node of nodes.values()) node.children.sort(compareMessages);

    // Depth, sibling position, and a stable pre-order for keyboard navigation.
    const order = [];
    const walk = (node, depth) => {
      node.depth = depth;
      order.push(node);
      node.children.forEach((child, i) => {
        child.siblingIndex = i;
        child.siblingCount = node.children.length;
        walk(child, depth + 1);
      });
    };
    roots.forEach((root, i) => {
      root.siblingIndex = i;
      root.siblingCount = roots.length;
      walk(root, 0);
    });

    const leafId = resolveLeaf(conversation, nodes, order);
    for (let node = leafId ? nodes.get(leafId) : null; node; node = node.parent) {
      node.onPath = true;
    }

    const leaves = order.filter((node) => node.children.length === 0);
    const forks = order.filter((node) => node.children.length > 1);
    const stats = {
      messages: order.length,
      branches: Math.max(leaves.length, order.length ? 1 : 0),
      forks: forks.length + (roots.length > 1 ? 1 : 0),
      depth: order.reduce((max, node) => Math.max(max, node.depth + 1), 0),
      pathLength: order.filter((node) => node.onPath).length,
    };

    return {
      roots,
      nodes,
      order,
      leafId,
      stats,
      title: conversation?.name?.trim() || 'Untitled conversation',
      updatedAt: conversation?.updated_at || null,
    };
  }

  /** The message Claude currently considers "the end of the conversation". */
  function resolveLeaf(conversation, nodes, order) {
    const declared = conversation?.current_leaf_message_uuid;
    if (declared && nodes.has(declared)) return declared;

    let best = null;
    for (const node of order) {
      if (node.children.length) continue;
      if (!best) { best = node; continue; }
      const a = Date.parse(node.createdAt || '') || 0;
      const b = Date.parse(best.createdAt || '') || 0;
      if (a > b || (a === b && node.depth > best.depth)) best = node;
    }
    return best?.id ?? null;
  }

  /**
   * Tidy tree layout with variable node sizes.
   *
   * Leaves are placed one after another along the "spread" axis and every parent is centred
   * over its children. Along the "depth" axis each level is one row, as tall as the tallest
   * node on it, so a node showing one line does not force four lines of empty space on the
   * rest of its row.
   *
   * Nodes cannot overlap: a parent is clamped into its own subtree's span, and on the rare
   * occasion it is wider than that span the cursor is pushed out to reserve the difference.
   *
   * Coordinates are returned in an orientation-free (depth, spread) space; the caller maps
   * them onto x/y. Each node is annotated with `s`, `d` and its `sSize`/`dSize`.
   *
   * @param {object[]} roots
   * @param {{spreadOf: (node) => number, depthOf: (node) => number,
   *          spreadGap: number, depthGap: number}} options
   * @returns {{spread: number, depth: number}} the extent of the laid-out tree
   */
  function computeLayout(roots, { spreadOf, depthOf, spreadGap, depthGap }) {
    if (!roots.length) return { spread: 0, depth: 0 };

    // Row heights: every node at a given depth shares the tallest node's row.
    const rows = [];
    const measure = (node, depth) => {
      rows[depth] = Math.max(rows[depth] ?? 0, depthOf(node));
      for (const child of node.children) measure(child, depth + 1);
    };
    for (const root of roots) measure(root, 0);

    const rowOffset = [];
    let run = 0;
    for (let i = 0; i < rows.length; i++) {
      rowOffset[i] = run;
      run += rows[i] + depthGap;
    }

    let cursor = 0;
    let edge = 0;

    const place = (node, depth) => {
      node.d = rowOffset[depth];
      node.dSize = depthOf(node);
      node.sSize = spreadOf(node);

      if (!node.children.length) {
        node.s = cursor;
        cursor += node.sSize + spreadGap;
        edge = Math.max(edge, node.s + node.sSize);
        return { lo: node.s, hi: node.s + node.sSize };
      }

      let lo = Infinity;
      let hi = -Infinity;
      for (const child of node.children) {
        const span = place(child, depth + 1);
        lo = Math.min(lo, span.lo);
        hi = Math.max(hi, span.hi);
      }

      const centreOf = (child) => child.s + child.sSize / 2;
      const first = node.children[0];
      const last = node.children[node.children.length - 1];
      const centre = (centreOf(first) + centreOf(last)) / 2;
      node.s = Math.min(Math.max(centre - node.sSize / 2, lo), Math.max(lo, hi - node.sSize));

      const overflow = node.s + node.sSize - hi;
      if (overflow > 0) {
        cursor += overflow; // reserve the extra room before the next subtree starts
        hi = node.s + node.sSize;
      }
      edge = Math.max(edge, hi);
      return { lo, hi };
    };

    for (const root of roots) {
      place(root, 0);
      cursor += spreadGap; // extra air between disconnected roots
    }

    return { spread: edge, depth: run - depthGap };
  }

  /**
   * The message a branch should end at — its most recent reply.
   *
   * claude.ai will only accept an assistant message as a conversation's current leaf; asking
   * it to end on one of your own turns is refused outright ("Current leaf message is not an
   * assistant message in the conversation"). So the walk goes down to the newest descendant
   * and then back up to the nearest reply, which is still inside the branch that was chosen.
   *
   * @returns {?object} the message to end on, or null if the branch holds no reply at all
   */
  function deepestLeaf(node) {
    let current = node;
    while (current.children.length) current = current.children[current.children.length - 1];
    while (current && current.sender !== 'assistant') current = current.parent;
    return current;
  }

  /**
   * Every root-to-leaf path through the tree, the one the API calls active first.
   *
   * The chat decides which of these it is showing, not the API, so the caller offers all of
   * them and lets the page pick.
   */
  function branchPaths(tree) {
    const paths = [];
    for (const node of tree.order) {
      if (node.children.length) continue;
      const path = [];
      for (let n = node; n; n = n.parent) path.unshift(n);
      paths.push(path);
    }
    return paths.sort((a, b) => Number(b[b.length - 1].onPath) - Number(a[a.length - 1].onPath));
  }

  CT.model = { buildTree, computeLayout, snippet, deepestLeaf, branchPaths };
})();
