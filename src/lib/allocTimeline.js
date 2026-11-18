// Every allocation over the trace, stacked -- the "active memory timeline" of
// torch's own memory_viz, built to stay interactive on traces with a million
// events.
//
// The picture: each allocation is a band whose height is its size. A new one
// lands on top of the stack; when one is freed, everything above it slides
// down to close the gap. So at any instant the stack is the live set, ordered
// by when each block was allocated, and the top of it is allocated memory.
//
// memory_viz draws that literally: one polygon per allocation, with a vertex
// pair every time something below it is freed. That is O(frees x live) vertices
// and SVG nodes, which is why it caps itself at the 15k largest allocations
// and lumps the rest into one band, and why it still crawls on a big trace.
//
// Here the stacking order is the thing that makes it cheap. Because a band
// only ever slides down, never past another, the live set's order is fixed:
// give every allocation a slot by allocation order and its offset at any
// instant is the sum of the live slots below it. A Fenwick tree over slots
// keeps that sum, and answers the inverse -- "which allocation is at height
// v" -- in O(log n). Drawing a frame is then a sweep over the events in view
// (one tree update each) and, per pixel column, one lookup per band tall
// enough to cover a pixel centre. The cost tracks the pixels on screen, not
// how many allocations are alive under them, so nothing is summarised away.

// Tableau 10, the categorical palette memory_viz uses, so the two read alike.
const PALETTE = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
  '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
]

/** As little-endian RGBA words, ready to drop into an ImageData buffer. */
export const PALETTE_RGBA = Uint32Array.from(PALETTE, (hex) => {
  const v = parseInt(hex.slice(1), 16)
  return ((255 << 24) | ((v & 0xff) << 16) | (v & 0xff00) | (v >> 16)) >>> 0
})
export const PALETTE_CSS = PALETTE

const KINDS = {
  blocks: {
    add: new Set(['alloc']),
    // free_requested is the tensor dying; the block is still the allocator's
    // until free_completed, which is also what the allocated curve counts.
    remove: new Set(['free_completed']),
  },
  segments: {
    add: new Set(['segment_alloc', 'segment_map']),
    remove: new Set(['segment_free', 'segment_unmap']),
  },
}

/**
 * Replay a trace into stackable elements.
 *
 * @param {object[]} events trace events, in order
 * @param {object[]} segments the snapshot's segments, sorted by address
 * @param {'blocks'|'segments'} mode tensors, or the segments holding them
 *
 * A ring-buffer trace starts mid-run, so some of what it shows was allocated
 * before it began: a free with no alloc, or a block alive at the snapshot that
 * the trace never saw made. Both become elements alive from the start, which
 * puts the curve at its true height rather than shifting it up by a guess.
 * They go at the bottom of the stack -- the ones alive throughout lowest, then
 * the rest in reverse order of their free -- so they never make anything
 * above them slide.
 */
export function buildAllocModel(events, segments, mode = 'blocks') {
  const n = events.length
  const { add, remove } = KINDS[mode]
  const start = []
  const end = []
  const size = []
  const src = []
  const live = new Map() // addr -> element

  const push = (s, e, sz, from) => {
    start.push(s); end.push(e); size.push(sz); src.push(from)
    return start.length - 1
  }

  for (let i = 0; i < n; i++) {
    const e = events[i]
    const a = e.action
    if (add.has(a)) {
      // Two allocs at one address with no free between them: the trace lost
      // the free. The first must be gone by now.
      const prev = live.get(e.addr)
      if (prev !== undefined) { end[prev] = i }
      live.set(e.addr, push(i, n, e.size ?? 0, e))
    } else if (remove.has(a)) {
      const id = live.get(e.addr)
      if (id !== undefined) {
        end[id] = i
        live.delete(e.addr)
      } else {
        push(-1, i, e.size ?? 0, e)
      }
    }
  }

  // What the snapshot holds that the trace never saw allocated.
  if (mode === 'blocks') {
    for (const seg of segments) {
      for (const b of seg.blocks) {
        if (b.active && !live.has(b.address)) {
          push(-1, n, b.requestedSize, { block: b, segment: seg })
        }
      }
    }
  } else {
    // Expandable segments are mapped piecewise, so a snapshot segment can be
    // covered by several live maps; only the part no map explains predates
    // the trace.
    const covered = new Float64Array(segments.length)
    for (const [address, id] of live) {
      const k = findSegment(segments, address)
      if (k >= 0) { covered[k] += size[id] }
    }
    segments.forEach((seg, k) => {
      const rest = seg.totalSize - covered[k]
      if (rest > 0) { push(-1, n, rest, { segment: seg }) }
    })
  }

  // Slot order: alive-throughout first, then pre-trace blocks by descending
  // free, then the trace's own allocations in the order they were made.
  const count = start.length
  const order = new Int32Array(count)
  for (let i = 0; i < count; i++) { order[i] = i }
  order.sort((x, y) => {
    const px = start[x] < 0, py = start[y] < 0
    if (px !== py) { return px ? -1 : 1 }
    if (px) { return end[y] - end[x] || x - y }
    return start[x] - start[y]
  })

  const m = {
    mode,
    n,
    count,
    start: new Int32Array(count),
    end: new Int32Array(count),
    size: new Float64Array(count),
    src: new Array(count),
    // At most one band appears and one disappears per event (a lost free
    // makes an alloc do both), so two flat arrays index the whole replay.
    addAt: new Int32Array(n).fill(-1),
    removeAt: new Int32Array(n).fill(-1),
    // Stack height after each event, and before the first.
    totals: new Float64Array(n),
    base: 0,
    peak: 0,
    events,
    colorCache: new Map(),
  }
  for (let slot = 0; slot < count; slot++) {
    const id = order[slot]
    m.start[slot] = start[id]
    m.end[slot] = end[id]
    m.size[slot] = size[id]
    m.src[slot] = src[id]
    if (start[id] >= 0) { m.addAt[start[id]] = slot }
    if (end[id] < n) { m.removeAt[end[id]] = slot }
    if (start[id] < 0) { m.base += size[id] }
  }

  let total = m.base
  let peak = total
  for (let i = 0; i < n; i++) {
    const r = m.removeAt[i]
    if (r >= 0) { total -= m.size[r] }
    const a = m.addAt[i]
    if (a >= 0) { total += m.size[a] }
    m.totals[i] = total
    if (total > peak) { peak = total }
  }
  m.peak = peak
  return m
}

function findSegment(segments, address) {
  let lo = 0, hi = segments.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const s = segments[mid]
    if (address < s.address) { hi = mid - 1 } else if (address >= s.address + s.totalSize) { lo = mid + 1 } else { return mid }
  }
  return -1
}

/**
 * One palette index per slot. By allocation, neighbours in the stack differ
 * the way memory_viz colours them; by source, every block from one line of
 * code shares a colour, so a leak reads as one hue climbing.
 *
 * @param {(frames: object[]) => string} keyOf grouping key for a stack
 */
export function slotColors(m, by, keyOf) {
  const hit = m.colorCache.get(by)
  if (hit) { return hit }
  const out = new Uint32Array(m.count)
  if (by === 'source') {
    const idx = new Map()
    for (let s = 0; s < m.count; s++) {
      const key = keyOf(framesOf(m.src[s]))
      let k = idx.get(key)
      if (k === undefined) { k = idx.size; idx.set(key, k) }
      out[s] = PALETTE_RGBA[k % PALETTE_RGBA.length]
    }
  } else {
    for (let s = 0; s < m.count; s++) { out[s] = PALETTE_RGBA[s % PALETTE_RGBA.length] }
  }
  m.colorCache.set(by, out)
  return out
}

function framesOf(src) {
  return src.frames ?? src.block?.frames ?? src.segment?.frames ?? []
}

/** Everything the detail panel says about one band. */
export function describe(m, slot) {
  const src = m.src[slot]
  const start = m.start[slot]
  const end = m.end[slot]
  const ev = (i) => (i >= 0 && i < m.n ? m.events[i] : null)
  const from = src.block ? 'snapshot' : src.segment ? 'snapshot' : start < 0 ? 'free' : 'trace'
  return {
    slot,
    size: m.size[slot],
    address: src.addr ?? src.block?.address ?? src.segment?.address ?? null,
    stream: src.stream ?? src.segment?.stream ?? null,
    start,
    end,
    alive: end >= m.n,
    startUs: ev(start)?.time_us ?? null,
    endUs: ev(end)?.time_us ?? null,
    frames: framesOf(src),
    metadata: (typeof src.user_metadata === 'string' && src.user_metadata) || src.block?.metadata || src.segment?.metadata || '',
    from,
    segment: src.segment ?? null,
  }
}

// ---- drawing -------------------------------------------------------------

/** Prefix sums over slots: live sizes in, "what is at height v" out. */
class Fenwick {
  constructor(n) {
    this.n = n
    this.t = new Float64Array(n + 1)
    let top = 1
    while (top * 2 <= n) { top *= 2 }
    this.top = top
  }

  /** O(n) build from a dense array of slot values. */
  load(vals) {
    const t = this.t, n = this.n
    t[0] = 0
    t.set(vals, 1)
    for (let i = 1; i <= n; i++) {
      const j = i + (i & -i)
      if (j <= n) { t[j] += t[i] }
    }
  }

  add(slot, d) {
    const t = this.t, n = this.n
    for (let i = slot + 1; i <= n; i += i & -i) { t[i] += d }
  }

  /** Sum of the slots below `slot`: where that band's bottom sits. */
  below(slot) {
    const t = this.t
    let s = 0
    for (let i = slot; i > 0; i -= i & -i) { s += t[i] }
    return s
  }
}

/**
 * Rasterise one frame of the stack into an RGBA buffer, plus a slot id per
 * pixel so hovering is a lookup rather than a search.
 *
 * Each pixel column is a range of events. It is drawn at the event in its
 * range where the stack is tallest, so zooming out never hides a peak --
 * the same max-keeping rule the summary curve downsamples by.
 *
 * @param {object} m model from buildAllocModel
 * @param {object} o
 * @param {number} o.x0 first event in view (fractional)
 * @param {number} o.x1 end of view, exclusive
 * @param {number} o.width columns, device pixels
 * @param {number} o.height rows, device pixels
 * @param {number} o.yMax bytes at the top row
 * @param {Uint32Array} o.colors per-slot RGBA from slotColors
 * @param {Float64Array} [o.overlay] a second series (reserved, say) to
 *   reduce per column alongside the stack
 * @param {object} [o.buf] buffers from a previous call, reused when the size matches
 */
export function rasterize(m, o) {
  const { x0, x1, width: W, height: H, yMax, colors, overlay } = o
  let buf = o.buf
  if (!buf || buf.width !== W || buf.height !== H) {
    buf = {
      width: W,
      height: H,
      pix: new Uint32Array(W * H),
      ids: new Int32Array(W * H),
      sample: new Int32Array(W), // event each column was drawn at
      stackTop: new Float64Array(W),
      overlayMax: new Float64Array(W),
      vals: new Float64Array(m.count),
      tree: null,
    }
  }
  if (buf.vals.length !== m.count) { buf.vals = new Float64Array(m.count) }
  const { pix, ids, sample, stackTop, overlayMax } = buf
  pix.fill(0)
  ids.fill(0)
  sample.fill(-1)
  stackTop.fill(0)
  overlayMax.fill(0)
  const n = m.n
  if (n === 0 || m.count === 0 || W === 0 || H === 0) { return buf }

  const span = (x1 - x0) / W
  const first = Math.max(0, Math.min(n - 1, Math.floor(x0)))

  // The live set just before `first`, loaded in one O(n) pass rather than by
  // replaying everything up to it -- so panning deep into a long trace costs
  // the same as looking at its start.
  const vals = buf.vals
  const { start, end, size, totals, addAt, removeAt } = m
  for (let s = 0; s < m.count; s++) {
    vals[s] = start[s] < first && end[s] >= first ? size[s] : 0
  }
  if (!buf.tree || buf.tree.n !== m.count) { buf.tree = new Fenwick(m.count) }
  const tree = buf.tree
  tree.load(vals)
  const t = tree.t, tn = tree.n, ttop = tree.top
  let total = first > 0 ? totals[first - 1] : m.base
  let next = first // next event to apply
  let dirty = 0 // lowest height changed since the previous column

  const rowsPerByte = H / yMax
  for (let c = 0; c < W; c++) {
    const lo = x0 + c * span
    if (lo >= n) { break }
    let a = Math.floor(lo)
    let b = Math.ceil(lo + span)
    if (a < 0) { a = 0 }
    if (b > n) { b = n }
    if (b <= a) { b = a + 1 }

    let at = a, best = totals[a], ov = overlay ? overlay[a] : 0
    for (let i = a + 1; i < b; i++) {
      if (totals[i] > best) { best = totals[i]; at = i }
      if (overlay && overlay[i] > ov) { ov = overlay[i] }
    }
    if (at < next) { at = next - 1 } // columns narrower than an event share it
    // Track the lowest height any band moved at since the last column. Below
    // it the column is the same as the one before, so it is copied rather than
    // looked up again -- and most of a real trace's churn is near the top of
    // the stack, with weights and optimizer state sitting still underneath.
    for (; next <= at; next++) {
      const r = removeAt[next]
      if (r >= 0) {
        const lo = tree.below(r)
        if (lo < dirty) { dirty = lo }
        tree.add(r, -size[r]); total -= size[r]
      }
      const ad = addAt[next]
      if (ad >= 0) {
        const lo = tree.below(ad)
        if (lo < dirty) { dirty = lo }
        tree.add(ad, size[ad]); total += size[ad]
      }
    }
    sample[c] = at
    stackTop[c] = total
    overlayMax[c] = ov

    let r = 0
    if (c > 0) {
      // Rows whose centre is below `dirty` are covered by bands that did not
      // move; take them from the previous column.
      let keep = dirty === Infinity ? H : Math.ceil(dirty * rowsPerByte - 0.5)
      if (keep > H) { keep = H }
      for (let p = (H - 1) * W + c; r < keep; r++, p -= W) {
        pix[p] = pix[p - 1]
        ids[p] = ids[p - 1]
      }
    }
    dirty = Infinity

    // Walk up the rest a band at a time: find what covers this row's centre,
    // fill every row it covers, jump past it.
    while (r < H) {
      const v = (r + 0.5) / rowsPerByte
      if (v >= total) { break }
      let pos = 0, rem = v
      for (let step = ttop; step > 0; step >>= 1) {
        const np = pos + step
        if (np <= tn && t[np] <= rem) { pos = np; rem -= t[np] }
      }
      if (pos >= m.count) { break } // float slack past the top
      const top = v - rem + size[pos]
      let rEnd = Math.ceil(top * rowsPerByte - 0.5)
      if (rEnd <= r) { rEnd = r + 1 }
      if (rEnd > H) { rEnd = H }
      const color = colors[pos]
      const id = pos + 1
      for (let k = r, p = (H - 1 - r) * W + c; k < rEnd; k++, p -= W) {
        pix[p] = color
        ids[p] = id
      }
      r = rEnd
    }
  }
  return buf
}

/** The highest of `series` over events [a, b), for fitting the y axis. */
export function rangeMax(series, a, b) {
  let mx = 0
  for (let i = Math.max(0, a); i < Math.min(series.length, b); i++) {
    if (series[i] > mx) { mx = series[i] }
  }
  return mx
}
