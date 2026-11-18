import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

import { unpickle } from '../src/lib/unpickle.js'
import { parseSnapshot } from '../src/lib/snapshot.js'
import { buildAllocModel, rasterize, slotColors } from '../src/lib/allocTimeline.js'

const here = dirname(fileURLToPath(import.meta.url))
const SAMPLES = join(here, '..', 'public', 'samples')

const seg = (address, totalSize, blocks = []) => ({ address, totalSize, stream: 0, blocks, frames: [] })
const blk = (address, size, active = true) => ({ address, size, requestedSize: size, active, frames: [] })

/**
 * The stack after event i the slow way: live slots in slot order, each with
 * the offset of everything live below it. What the rasteriser must agree with.
 */
function stackAt(m, i) {
  const out = []
  let off = 0
  for (let s = 0; s < m.count; s++) {
    if (m.start[s] <= i && m.end[s] > i && m.size[s] > 0) {
      out.push({ s, lo: off, hi: off + m.size[s] })
      off += m.size[s]
    }
  }
  return out
}

/** Pixel ids by brute force, for the events the rasteriser chose per column. */
function checkRaster(m, opts) {
  const colors = slotColors(m, 'alloc')
  const buf = rasterize(m, { ...opts, colors })
  const { width: W, height: H, yMax } = opts
  for (let c = 0; c < W; c++) {
    const i = buf.sample[c]
    if (i < 0) { continue }
    const stack = stackAt(m, i)
    for (let r = 0; r < H; r++) {
      const v = (r + 0.5) * yMax / H
      const hit = stack.find((b) => v >= b.lo && v < b.hi)
      const got = buf.ids[(H - 1 - r) * W + c]
      assert.equal(got, hit ? hit.s + 1 : 0, `column ${c} (event ${i}) row ${r}`)
    }
  }
  return buf
}

test('frees out of order close the gap, and the picture matches a brute-force stack', () => {
  const ev = [
    { action: 'alloc', addr: 1, size: 100 },
    { action: 'alloc', addr: 2, size: 50 },
    { action: 'alloc', addr: 3, size: 25 },
    { action: 'free_requested', addr: 1, size: 100 },
    { action: 'free_completed', addr: 1, size: 100 }, // bottom freed: 2 and 3 slide down
    { action: 'alloc', addr: 1, size: 10 },
    { action: 'free_completed', addr: 3, size: 25 },
  ]
  const m = buildAllocModel(ev, [], 'blocks')
  assert.deepEqual([...m.totals], [100, 150, 175, 175, 75, 85, 60])
  assert.equal(m.peak, 175)
  for (const [x0, x1, width] of [[0, 7, 7], [0, 7, 40], [0, 7, 3], [2.5, 5.5, 11]]) {
    checkRaster(m, { x0, x1, width, height: 64, yMax: 200 })
  }
})

test('a ring-buffer trace is rebuilt from the frees it lost the alloc for and the snapshot', () => {
  const ev = [
    { action: 'free_completed', addr: 9, size: 40 }, // allocated before the trace
    { action: 'alloc', addr: 1, size: 10 },
    { action: 'free_completed', addr: 8, size: 30 },
  ]
  const segments = [seg(0, 1000, [blk(1, 10), blk(500, 200), blk(700, 300, false)])]
  const m = buildAllocModel(ev, segments, 'blocks')
  // 200 alive throughout (snapshot only), 40 and 30 alive from the start.
  assert.equal(m.base, 270)
  assert.deepEqual([...m.totals], [230, 240, 210])
  // Bottom of the stack never moves: the snapshot-only block, then the free
  // that comes last, then the one that comes first.
  assert.deepEqual([...m.size.slice(0, 3)], [200, 30, 40])
  checkRaster(m, { x0: 0, x1: 3, width: 9, height: 50, yMax: 300 })
})

test('segments: piecewise maps of an expandable segment are not double counted', () => {
  const ev = [
    { action: 'segment_map', addr: 0, size: 100 },
    { action: 'segment_map', addr: 100, size: 100 },
  ]
  const m = buildAllocModel(ev, [seg(0, 300)], 'segments')
  assert.equal(m.base, 100) // the third 100 was mapped before the trace
  assert.deepEqual([...m.totals], [200, 300])
})

test('zoomed-in columns narrower than an event, and panning deep into the trace', () => {
  const ev = []
  // Interleaved lifetimes, so plenty of bands slide.
  for (let i = 0; i < 400; i++) {
    ev.push({ action: 'alloc', addr: i, size: 1 + ((i * 37) % 23) })
    if (i % 3 === 2) { ev.push({ action: 'free_completed', addr: i - 2 - (i % 2), size: 0 }) }
  }
  const sizes = new Map(ev.filter((e) => e.action === 'alloc').map((e) => [e.addr, e.size]))
  for (const e of ev) { if (e.action === 'free_completed') { e.size = sizes.get(e.addr) } }
  const m = buildAllocModel(ev, [], 'blocks')
  checkRaster(m, { x0: 0, x1: m.n, width: 97, height: 120, yMax: m.peak })
  checkRaster(m, { x0: 300.25, x1: 310.75, width: 50, height: 120, yMax: m.peak })
  checkRaster(m, { x0: m.n - 20, x1: m.n, width: 33, height: 80, yMax: m.peak * 0.5 })
})

// On every shipped sample, the stack at the end of the trace must be exactly
// what the snapshot holds: its live blocks, or its segments.
for (const f of readdirSync(SAMPLES).filter((x) => x.endsWith('.pickle'))) {
  test(`end of the stack is the snapshot: ${f}`, () => {
    const model = parseSnapshot(unpickle(readFileSync(join(SAMPLES, f))))
    for (const d of model.devices) {
      if (!d.timeline.hasTrace) { continue }
      const b = buildAllocModel(d.timeline.events, d.segments, 'blocks')
      const s = buildAllocModel(d.timeline.events, d.segments, 'segments')
      const live = d.segments.reduce((n, sg) => n + sg.blocks.filter((x) => x.active).reduce((k, x) => k + x.requestedSize, 0), 0)
      assert.equal(b.totals[b.n - 1], live, 'live block bytes')
      assert.equal(s.totals[s.n - 1], d.stats.reserved, 'reserved bytes')
      checkRaster(b, { x0: 0, x1: b.n, width: 60, height: 40, yMax: Math.max(b.peak, 1) })
    }
  })
}
