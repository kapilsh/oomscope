// The caching allocator's structure as a graph, for the Structure view.
//
// The allocator does not keep one heap. Reserved memory is held in layers, and
// each layer narrows which free bytes a request can actually reach:
//
//   device      everything cudaMalloc has handed over
//   pool        the default pool, or a private one (CUDA graph, MemPool, ...)
//   free list   inside a pool, small (<= 1 MiB requests, 2 MiB segments) and
//               large blocks are kept apart, and a free block is only offered
//               to a request on the stream it was allocated on. So the real
//               unit of reuse is (pool, size class, stream).
//   segment     one cudaMalloc (or one expandable mapping)
//   block       a slice of a segment: live, pending free, or cached
//
// and every live block was put there by some line of code. Drawing that as
// columns whose heights are bytes makes the whole chain readable at once: a
// free list full of dark ribbons is memory nothing on that stream can reach.
//
// Every column sums to reserved, so the ribbons conserve bytes left to right.

import { blameFrame, frameKey, displayStack } from './frames.js'
import { DEFAULT_POOL } from './snapshot.js'

// Past these, a column is a smear rather than a set of things you can click.
// The overflow is lumped, not dropped, so the bytes still add up.
const MAX_SEGMENTS = 120
const MAX_SOURCES = 24

export const COLUMNS = ['device', 'pool', 'free list', 'segment', 'held by']

/** Free-list identity: the triple the allocator actually searches by. */
const flKey = (seg) => `${seg.poolKey}|${seg.type}|${seg.stream}`

/**
 * @param {object} view a device or a pool, as the other views receive it. A
 *   pool carries its device's id and full pool list, so either will do.
 */
export function buildStructure(view) {
  const nodes = new Map()
  const links = new Map()
  const add = (node) => {
    if (!nodes.has(node.id)) { nodes.set(node.id, { live: 0, free: 0, segments: [], ...node }) }
    return nodes.get(node.id)
  }
  const flow = (from, to, live, free, src) => {
    const id = `${from}>${to}`
    let l = links.get(id)
    if (!l) {
      l = { id, source: from, target: to, live: 0, free: 0, src: new Map() }
      links.set(id, l)
    }
    l.live += live
    l.free += free
    for (const [k, v] of src) { l.src.set(k, (l.src.get(k) ?? 0) + v) }
  }

  const pools = view.pool ? [view.pool] : view.pools.length ? view.pools : [{
    key: DEFAULT_POOL,
    poolId: [0, 0],
    kind: 'default',
    label: 'Default pool',
    origin: 'The caching allocator\'s ordinary pool: every allocation not routed anywhere else.',
    evidence: 'pool id (0, 0)',
  }]
  const poolOf = new Map(pools.map((p) => [p.key, p]))


  // Sources, in the same grouping the Allocations view uses, so a node here
  // and a row there are the same thing.
  const kept = new Set(view.blame.slice(0, MAX_SOURCES).map((g) => g.key))
  const blameOf = new Map(view.blame.map((g) => [g.key, g]))
  const sourceId = (key) => (kept.has(key) ? `src:${key}` : 'src:other')

  const shown = new Set(
    [...view.segments].sort((a, b) => b.totalSize - a.totalSize).slice(0, MAX_SEGMENTS),
  )

  add({ id: 'dev', col: 0, kind: 'device', label: `cuda:${view.id}` })

  for (const seg of view.segments) {
    const pool = poolOf.get(seg.poolKey) ?? pools[0]
    const poolId = `pool:${pool.key}`
    const fl = `fl:${flKey(seg)}`
    const segId = shown.has(seg) ? `seg:${seg.address}` : `segrest:${flKey(seg)}`

    add({ id: poolId, col: 1, kind: 'pool', pool, label: pool.label, sub: `(${pool.poolId.join(', ')})` })
    add({
      id: fl, col: 2, kind: 'freelist', pool,
      type: seg.type, stream: seg.stream, streamName: seg.streamName,
      label: `${seg.type}${seg.stream !== 0 ? ` · ${seg.streamName}` : ''}`,
      sub: pools.length > 1 ? pool.label : null,
    })
    const segNode = shown.has(seg)
      ? add({ id: segId, col: 3, kind: 'segment', seg, pool, label: null })
      : add({ id: segId, col: 3, kind: 'segrest', pool, label: null })
    segNode.segments.push(seg)
    nodes.get(fl).segments.push(seg)
    nodes.get(poolId).segments.push(seg)
    nodes.get('dev').segments.push(seg)

    // Down to the code: live bytes per source, then whatever is cached.
    const bySource = new Map()
    for (const b of seg.blocks) {
      if (!b.active) { continue }
      const key = frameKey(blameFrame(b.frames))
      bySource.set(key, (bySource.get(key) ?? 0) + b.size)
    }
    const free = seg.totalSize - seg.activeSize
    const live = seg.activeSize
    const srcIds = new Map()
    for (const [key, n] of bySource) {
      const id = sourceId(key)
      srcIds.set(id, (srcIds.get(id) ?? 0) + n)
    }

    for (const [id, n] of srcIds) {
      const key = id.slice(4)
      const g = blameOf.get(key)
      const node = id === 'src:other'
        ? add({ id, col: 4, kind: 'source', other: true, label: 'other sources' })
        : add({ id, col: 4, kind: 'source', blame: g, label: g?.label ?? key })
      node.segments.push(seg)
      flow(segId, id, n, 0, new Map([[id, n]]))
    }
    if (free > 0) {
      add({ id: 'free', col: 4, kind: 'free', label: 'cached, free' }).segments.push(seg)
      flow(segId, 'free', 0, free, new Map())
    }

    const src = new Map([...srcIds])
    flow(fl, segId, live, free, src)
    flow(poolId, fl, live, free, src)
    flow('dev', poolId, live, free, src)
  }

  // Node totals from the links, so a node is exactly as tall as what flows
  // through it.
  for (const l of links.values()) {
    const a = nodes.get(l.source)
    const b = nodes.get(l.target)
    if (a.col === 0) { a.live += l.live; a.free += l.free }
    b.live += l.live
    b.free += l.free
  }
  for (const n of nodes.values()) {
    n.value = n.live + n.free
    if (n.kind === 'segrest') { n.label = `${n.segments.length} smaller segments` }
    if (n.kind === 'source' && !n.other) { n.stack = displayStack(firstStack(n, n.id.slice(4))) }
  }

  return { nodes: [...nodes.values()], links: [...links.values()], pools }
}

/** A representative stack for a source node: the first live block it owns. */
function firstStack(node, key) {
  for (const seg of node.segments) {
    for (const b of seg.blocks) {
      if (b.active && frameKey(blameFrame(b.frames)) === key) { return b.frames }
    }
  }
  return []
}

/** Which source a block's bytes are attributed to, as a node id. */
export function blockSourceId(b, nodeIds) {
  const id = `src:${frameKey(blameFrame(b.frames))}`
  return nodeIds.has(id) ? id : 'src:other'
}

// ---- layout --------------------------------------------------------------

const MIN_H = 2

/**
 * Place the graph: x by column, y by bytes, every column filling the same
 * height so the ribbons between them conserve width.
 *
 * Order within a column follows the tree (a free list's segments sit together,
 * under their pool), and the last column is ordered by where its bytes come
 * from, which is what keeps the ribbons from crossing into a hairball.
 */
export function layoutStructure(graph, { xs, nodeW }) {
  const { nodes, links } = graph
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const cols = COLUMNS.map((_, c) => nodes.filter((n) => n.col === c))
  const out = (id) => links.filter((l) => l.source === id)

  // Tree order for the middle columns.
  const poolRank = new Map(graph.pools.map((p, i) => [p.key, i]))
  cols[1].sort((a, b) => poolRank.get(a.pool.key) - poolRank.get(b.pool.key))
  cols[2].sort((a, b) =>
    poolRank.get(a.pool.key) - poolRank.get(b.pool.key) ||
    (a.type === b.type ? 0 : a.type === 'large' ? -1 : 1) ||
    a.stream - b.stream)
  const flRank = new Map(cols[2].map((n, i) => [n.id, i]))
  const parentFl = new Map()
  for (const l of links) { if (byId.get(l.source).col === 2) { parentFl.set(l.target, l.source) } }
  cols[3].sort((a, b) =>
    flRank.get(parentFl.get(a.id)) - flRank.get(parentFl.get(b.id)) ||
    (a.kind === 'segrest') - (b.kind === 'segrest') ||
    (a.seg?.address ?? 0) - (b.seg?.address ?? 0))

  const nMax = Math.max(...cols.map((c) => c.length), 1)
  const height = Math.max(480, nMax * 6)
  const gapMin = (n) => (n > 40 ? 1 : n > 16 ? 2 : 6)
  const total = nodes[0]?.value || 1
  // One scale for every column, so a ribbon is the same width at both ends.
  // Each node is allowed MIN_H on top, so a sliver stays clickable.
  const k = Math.min(...cols.filter((c) => c.length).map((c) =>
    (height - (c.length - 1) * gapMin(c.length) - c.length * MIN_H) / total))

  const place = (c) => {
    const hs = c.map((n) => Math.max(MIN_H, n.value * k))
    const used = hs.reduce((a, b) => a + b, 0)
    const gap = c.length > 1 ? Math.min(28, (height - used) / (c.length - 1)) : 0
    let y = (height - used - gap * (c.length - 1)) / 2
    c.forEach((n, i) => {
      n.x0 = xs[n.col]
      n.x1 = n.x0 + nodeW
      n.y0 = y
      n.y1 = y + hs[i]
      y = n.y1 + gap
    })
    // How much vertical room each node's label has before the next node.
    c.forEach((n, i) => { n.room = (c[i + 1]?.y0 ?? height + 8) - n.y0 })
  }
  for (let c = 0; c < 4; c++) { place(cols[c]) }

  // Last column by the centre of mass of what feeds it; the two catch-alls
  // go underneath, free last, since it is the one everything drains into.
  const mid = (n) => (n.y0 + n.y1) / 2
  const bary = new Map()
  for (const n of cols[4]) {
    let w = 0, s = 0
    for (const l of links) {
      if (l.target !== n.id) { continue }
      const v = l.live + l.free
      w += v
      s += v * mid(byId.get(l.source))
    }
    bary.set(n.id, w ? s / w : 0)
  }
  const tail = (n) => (n.kind === 'free' ? 2 : n.other ? 1 : 0)
  cols[4].sort((a, b) => tail(a) - tail(b) || bary.get(a.id) - bary.get(b.id))
  place(cols[4])

  // Ribbon ends, stacked within each node in the order of what they connect
  // to, so neighbouring ribbons leave and arrive side by side.
  for (const n of nodes) {
    let y = n.y0
    for (const l of out(n.id).sort((a, b) => byId.get(a.target).y0 - byId.get(b.target).y0)) {
      l.w = (l.live + l.free) * k
      l.sy = y
      y += l.w
    }
  }
  for (const n of nodes) {
    let y = n.y0
    for (const l of links.filter((x) => x.target === n.id).sort((a, b) => byId.get(a.source).y0 - byId.get(b.source).y0)) {
      l.ty = y
      y += l.w
    }
  }

  return { nodes, links, byId, height, k }
}

/** Everything upstream and downstream of a node: the path its bytes take. */
export function lineage(graph, id) {
  const seen = new Set([id])
  const walk = (from, dir) => {
    for (const l of graph.links) {
      const [a, b] = dir === 'down' ? [l.source, l.target] : [l.target, l.source]
      if (a === from && !seen.has(b)) { seen.add(b); walk(b, dir) }
    }
  }
  walk(id, 'up')
  walk(id, 'down')
  return seen
}
