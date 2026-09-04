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

    let preview = body;
    if (!preview && thinking) preview = thinking;
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
   * Tidy tree layout.
   *
   * Leaves are placed left to right along the "spread" axis; every parent is centred over
   * its children. Because a node's spread coordinate always falls inside its own subtree's
   * leaf span, and leaf spans are separated by `spreadGap`, boxes can never overlap.
   *
   * Coordinates are returned in an orientation-free (depth, spread) space; the caller maps
   * them onto x/y.
   *
   * @returns {{spread: number, depth: number}} the extent of the laid-out tree
   */
  function computeLayout(roots, { spreadSize, spreadGap, depthSize, depthGap }) {
    const step = spreadSize + spreadGap;
    let cursor = 0;
    let lastLeaf = 0;
    let maxDepth = 0;

    const place = (node, depth) => {
      maxDepth = Math.max(maxDepth, depth);
      node.d = depth * (depthSize + depthGap);
      if (node.children.length === 0) {
        node.s = cursor;
        lastLeaf = cursor;
        cursor += step;
        return;
      }
      for (const child of node.children) place(child, depth + 1);
      const first = node.children[0];
      const last = node.children[node.children.length - 1];
      node.s = (first.s + last.s) / 2;
    };

    for (const root of roots) {
      place(root, 0);
      cursor += spreadGap; // extra air between disconnected roots
    }

    if (!roots.length) return { spread: 0, depth: 0 };
    return {
      spread: lastLeaf + spreadSize,
      depth: maxDepth * (depthSize + depthGap) + depthSize,
    };
  }

  CT.model = { buildTree, computeLayout, snippet, collapse };
})();
