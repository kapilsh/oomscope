// The allocator's state at any event of the trace: every segment and every
// block in it, as they stood -- memory_viz's "Allocator State History".
//
// The snapshot is the one moment we know exactly. memory_viz gets any earlier
// moment by copying it and undoing every event after the one you clicked,
// which is a full replay per click. Here the undo runs once, back to the start
// of the trace, and records each block's and each mapped range's lifetime as
// it goes: a block is alive after event `start` and gone after event `end`.
// The state at event t is then the records alive at t, which is one pass over
// two typed arrays -- cheap enough to redo on every key press.
//
// Walking backwards from the snapshot rather than forwards from the first
// event matters for a ring-buffer trace that begins mid-run: what was alive
// before it began falls out as records nothing ended, instead of as frees of
// allocations we never saw.

import { BlockState } from './snapshot.js'

const BLOCK_ACTIONS = new Set(['alloc', 'free_requested', 'free_completed', 'annotate'])
const MAP_UNDO = new Set(['segment_alloc', 'segment_map'])
const UNMAP_UNDO = new Set(['segment_free', 'segment_unmap'])

// The allocator rounds a request up to 512 bytes. A block the snapshot never
// saw only has its requested size in the trace, so that is what its footprint
// is taken to be -- capped by whatever comes next, so it never overlaps.
const ROUND = 512

/**
 * @param {object[]} events trace events, in order
 * @param {object[]} segments the snapshot's segments, sorted by address
 */
export function buildHistory(events, segments) {
  const n = events.length
  const blocks = recordBlocks(events, segments, n)
  const pieces = recordPieces(events, segments, n)
  return { n, events, blocks, pieces, inconsistent: blocks.inconsistent + pieces.inconsistent }
}

// ---- blocks ----------------------------------------------------------------

function recordBlocks(events, segments, n) {
  const addr = [], req = [], size = [], start = [], end = [], pend = [], src = [], notes = []
  const live = new Map() // address -> record, alive "now" as we walk back
  let inconsistent = 0
  const rec = (a, r, sz, e, p, s) => {
    addr.push(a); req.push(r); size.push(sz); start.push(-1); end.push(e); pend.push(p); src.push(s); notes.push(null)
    return addr.length - 1
  }

  for (const seg of segments) {
    for (const b of seg.blocks) {
      if (!b.active) { continue }
      // A block still waiting on another stream has been pending since its
      // free_requested -- or since before the trace, if we never see one.
      const id = rec(b.address, b.requestedSize, b.size, n, b.state === BlockState.PENDING_FREE ? -1 : n, { block: b })
      live.set(b.address, id)
    }
  }

  for (let i = n - 1; i >= 0; i--) {
    const e = events[i]
    if (!BLOCK_ACTIONS.has(e.action)) { continue }
    const id = live.get(e.addr)
    switch (e.action) {
      case 'free_completed': {
        // Something is alive here after a free with no alloc between: the
        // trace lost that alloc. It began no earlier than this.
        if (id !== undefined) { start[id] = i; inconsistent++ }
        live.set(e.addr, rec(e.addr, e.size ?? 0, -1, i, i, { free: e }))
        break
      }
      case 'free_requested':
        // Only a record that can be pending: one the trace freed, or a block
        // the snapshot caught waiting.
        if (id !== undefined && (pend[id] === -1 || pend[id] === end[id])) { pend[id] = i }
        break
      case 'alloc':
        if (id !== undefined) {
          start[id] = i
          src[id].alloc = e
          live.delete(e.addr)
        } else {
          // Made, but neither freed later nor alive at the snapshot.
          inconsistent++
        }
        break
      case 'annotate':
        if (id !== undefined) {
          (notes[id] ??= []).unshift({ text: typeof e.user_metadata === 'string' ? e.user_metadata : '', frames: e.frames ?? [] })
        }
        break
      default: break
    }
  }

  const count = addr.length
  return {
    count,
    addr: Float64Array.from(addr),
    req: Float64Array.from(req),
    size: Float64Array.from(size), // exact footprint when the snapshot gave it, else -1
    start: Int32Array.from(start),
    end: Int32Array.from(end),
    pend: Int32Array.from(pend),
    src,
    notes,
    inconsistent,
  }
}

// ---- mapped ranges -----------------------------------------------------------

/**
 * Segments are ranges of mapped memory. An ordinary segment is mapped and
 * released whole; an expandable one grows and shrinks a range at a time, and
 * an unmap can take a slice out of the middle of what one map made. So the
 * records here are byte ranges -- pieces -- and undoing a map cuts whatever
 * piece covers it, leaving the parts outside as pieces of their own. Adjacent
 * pieces of an expandable segment are drawn as one segment again (layoutAt).
 */
function recordPieces(events, segments, n) {
  const P = { addr: [], size: [], start: [], end: [], stream: [], pool: [], exp: [], src: [] }
  let inconsistent = 0
  const piece = (a, sz, e, stream, pool, exp, s) => {
    P.addr.push(a); P.size.push(sz); P.start.push(-1); P.end.push(e)
    P.stream.push(stream); P.pool.push(pool); P.exp.push(exp); P.src.push(s)
    return P.addr.length - 1
  }
  // Alive pieces, sorted by address. Segments number in the hundreds, so a
  // sorted array with splices is the right structure.
  const alive = []
  const lowerBound = (a) => {
    let lo = 0, hi = alive.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (P.addr[alive[mid]] + P.size[alive[mid]] <= a) { lo = mid + 1 } else { hi = mid }
    }
    return lo
  }

  for (const seg of segments) {
    alive.push(piece(seg.address, seg.totalSize, n, seg.stream, seg.poolKey, seg.isExpandable, { segment: seg }))
  }

  for (let i = n - 1; i >= 0; i--) {
    const e = events[i]
    const a = e.addr ?? 0
    const b = a + (e.size ?? 0)
    if (MAP_UNDO.has(e.action)) {
      // Before event i, [a, b) was not mapped: whatever covers it began here.
      let k = lowerBound(a)
      let hit = false
      const keep = []
      while (k < alive.length && P.addr[alive[k]] < b) {
        const id = alive[k]
        const pa = P.addr[id], pb = pa + P.size[id]
        P.start[id] = i
        hit = true
        // The parts outside [a, b) were mapped before this and stay so.
        if (pa < a) { keep.push(piece(pa, a - pa, i, P.stream[id], P.pool[id], P.exp[id], { ...P.src[id] })) }
        if (pb > b) { keep.push(piece(b, pb - b, i, P.stream[id], P.pool[id], P.exp[id], { ...P.src[id] })) }
        P.src[id].map = e // the event that made it: its stack is the segment's
        alive.splice(k, 1)
      }
      if (keep.length) { alive.splice(k, 0, ...keep) }
      if (!hit) { inconsistent++ }
    } else if (UNMAP_UNDO.has(e.action)) {
      // Before event i, [a, b) was mapped.
      const k = lowerBound(a)
      const next = alive[k]
      if (next !== undefined && P.addr[next] < b) { inconsistent++; continue }
      const prev = k > 0 ? alive[k - 1] : undefined
      // A slice unmapped from an expandable segment belongs to it; take the
      // pool from the event when torch stamped one, else from its neighbour.
      const nb = prev !== undefined && P.addr[prev] + P.size[prev] === a ? prev : next !== undefined && P.addr[next] === b ? next : undefined
      const pool = Array.isArray(e.pool_id) ? `${e.pool_id[0]},${e.pool_id[1]}` : nb !== undefined ? P.pool[nb] : '0,0'
      const exp = e.action === 'segment_unmap' || (nb !== undefined && P.exp[nb])
      alive.splice(k, 0, piece(a, b - a, i, e.stream ?? 0, pool, exp, { free: e }))
    }
  }

  return {
    count: P.addr.length,
    addr: Float64Array.from(P.addr),
    size: Float64Array.from(P.size),
    start: Int32Array.from(P.start),
    end: Int32Array.from(P.end),
    stream: P.stream,
    pool: P.pool,
    exp: P.exp,
    src: P.src,
    inconsistent,
  }
}

// ---- the state at one event -------------------------------------------------

/**
 * The segment map after event t (t = n - 1 is the snapshot), shaped like the
 * snapshot's own segments so the segment-map rows render it unchanged: each
 * segment's blocks tile it, live blocks and the free space between them.
 *
 * @param {object} h from buildHistory
 * @param {number} t event index
 * @param {(stream: number) => string} streamName
 */
export function layoutAt(h, t, streamName = () => '') {
  const P = h.pieces
  const ids = []
  for (let i = 0; i < P.count; i++) {
    if (P.start[i] <= t && t < P.end[i]) { ids.push(i) }
  }
  ids.sort((x, y) => P.addr[x] - P.addr[y])

  // Merge the pieces of one expandable segment back into it.
  const segs = []
  for (const id of ids) {
    const last = segs[segs.length - 1]
    if (last && P.exp[id] && last.isExpandable && last.address + last.totalSize === P.addr[id] &&
        last.stream === P.stream[id] && last.poolKey === P.pool[id]) {
      last.totalSize += P.size[id]
      last.pieces.push(id)
      continue
    }
    segs.push({
      address: P.addr[id],
      totalSize: P.size[id],
      stream: P.stream[id],
      poolKey: P.pool[id],
      isExpandable: P.exp[id],
      pieces: [id],
    })
  }

  const B = h.blocks
  const live = []
  for (let i = 0; i < B.count; i++) {
    if (B.start[i] <= t && t < B.end[i]) { live.push(i) }
  }
  live.sort((x, y) => B.addr[x] - B.addr[y])

  let k = 0
  let orphans = 0
  let reserved = 0, allocated = 0, freeBytes = 0, largestFree = 0, freeBlocks = 0, liveBlocks = 0
  const out = segs.map((g) => {
    const lo = g.address, hi = g.address + g.totalSize
    while (k < live.length && B.addr[live[k]] < lo) { k++; orphans++ }
    const src = segSource(P, g.pieces)
    const tiles = []
    let cursor = lo
    const free = (a, b) => {
      if (b <= a) { return }
      tiles.push({
        index: tiles.length, address: a, offset: a - lo, size: b - a, requestedSize: 0,
        state: BlockState.INACTIVE, active: false, frames: [], forwardFrames: [], metadata: '', annotations: [],
      })
      freeBytes += b - a
      freeBlocks++
      if (b - a > largestFree) { largestFree = b - a }
    }
    let active = 0, requested = 0, allocatedSize = 0
    while (k < live.length && B.addr[live[k]] < hi) {
      const id = live[k++]
      const a = B.addr[id]
      const nextA = k < live.length ? Math.min(B.addr[live[k]], hi) : hi
      const exact = B.size[id] >= 0
      const sz = exact ? B.size[id] : Math.max(B.req[id], Math.min(nextA - a, roundUp(B.req[id])))
      free(cursor, a)
      const pending = B.pend[id] <= t
      tiles.push(blockTile(B, id, tiles.length, lo, sz, exact, pending))
      active += sz
      requested += B.req[id]
      if (!pending) { allocatedSize += sz }
      liveBlocks++
      cursor = Math.max(cursor, a + sz)
    }
    free(cursor, hi)
    reserved += g.totalSize
    allocated += requested
    return {
      device: 0,
      address: lo,
      totalSize: g.totalSize,
      allocatedSize,
      activeSize: active,
      requestedSize: requested,
      stream: g.stream,
      streamName: streamName(g.stream),
      type: src.segment?.type ?? (!g.isExpandable && g.totalSize <= 2 * 1024 * 1024 ? 'small' : 'large'),
      poolKey: g.poolKey,
      isExpandable: g.isExpandable,
      frames: src.frames,
      forwardFrames: [],
      metadata: src.metadata,
      allocEvent: src.event,
      blocks: tiles,
      unaccounted: 0,
      allocIndex: -1,
      madeAt: Math.min(...g.pieces.map((p) => P.start[p])),
    }
  })
  orphans += live.length - k

  return {
    t,
    segments: out,
    stats: { reserved, allocated, freeBytes, largestFree, freeBlocks, liveBlocks, segments: out.length },
    orphans,
  }
}

function roundUp(x) {
  return x <= 0 ? ROUND : Math.ceil(x / ROUND) * ROUND
}

/** A segment's stack, metadata and making event, from the first of its pieces that has any. */
function segSource(P, pieces) {
  for (const p of pieces) {
    const s = P.src[p]
    const ev = s.map ?? s.free ?? s.segment?.allocEvent ?? null
    const frames = s.segment?.frames?.length ? s.segment.frames : ev?.frames ?? []
    if (frames.length || ev || s.segment) {
      return {
        segment: s.segment ?? null,
        frames,
        metadata: s.segment?.metadata || (typeof ev?.user_metadata === 'string' ? ev.user_metadata : ''),
        event: ev,
      }
    }
  }
  return { segment: null, frames: [], metadata: '', event: null }
}

function blockTile(B, id, index, segAddr, size, exact, pending) {
  const s = B.src[id]
  const b = s.block
  const ev = s.alloc ?? s.free ?? null
  // The alloc's stack is where the tensor was made; a block whose alloc the
  // trace lost only has the stack its free was recorded with.
  const frames = b ? b.frames : (s.alloc ?? s.free).frames ?? []
  return {
    index,
    address: B.addr[id],
    offset: B.addr[id] - segAddr,
    size,
    sizeExact: exact,
    requestedSize: B.req[id],
    state: pending ? BlockState.PENDING_FREE : BlockState.ALLOCATED,
    active: true,
    frames,
    forwardFrames: b?.forwardFrames ?? [],
    metadata: b?.metadata || (typeof ev?.user_metadata === 'string' ? ev.user_metadata : ''),
    annotations: B.notes[id] ?? [],
    allocEvent: s.alloc ?? b?.allocEvent ?? null,
    start: B.start[id],
    end: B.end[id],
    framesFromFree: !b && !s.alloc,
  }
}
