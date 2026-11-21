import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

import { unpickle } from '../src/lib/unpickle.js'
import { parseSnapshot, BlockState } from '../src/lib/snapshot.js'
import { buildAllocModel } from '../src/lib/allocTimeline.js'
import { buildHistory, layoutAt } from '../src/lib/history.js'

const here = dirname(fileURLToPath(import.meta.url))
const SAMPLES = join(here, '..', 'public', 'samples')

const seg = (address, totalSize, blocks = [], extra = {}) => ({
  address, totalSize, stream: 0, poolKey: '0,0', isExpandable: false, frames: [], blocks, ...extra,
})
const blk = (address, size, state = BlockState.ALLOCATED) => ({
  address, size, requestedSize: size, state, active: state !== BlockState.INACTIVE, frames: [],
})
const live = (L) => L.segments.flatMap((s) => s.blocks.filter((b) => b.active).map((b) => b.address))

test('stepping back undoes allocs and frees, and the end is the snapshot', () => {
  const ev = [
    { action: 'segment_alloc', addr: 0, size: 4096 },
    { action: 'alloc', addr: 0, size: 1024 },
    { action: 'alloc', addr: 1024, size: 1024 },
    { action: 'free_requested', addr: 0, size: 1024 },
    { action: 'free_completed', addr: 0, size: 1024 },
    { action: 'alloc', addr: 2048, size: 512 },
    { action: 'free_requested', addr: 1024, size: 1024 }, // waiting on a stream at the snapshot
  ]
  const segments = [seg(0, 4096, [blk(0, 1024, BlockState.INACTIVE), blk(1024, 1024, BlockState.PENDING_FREE), blk(2048, 512), blk(2560, 1536, BlockState.INACTIVE)])]
  const h = buildHistory(ev, segments)
  assert.equal(h.inconsistent, 0)
  assert.deepEqual(layoutAt(h, 0).segments.map((s) => s.blocks.length), [1]) // one free tile
  assert.deepEqual(live(layoutAt(h, 2)), [0, 1024])
  assert.deepEqual(live(layoutAt(h, 4)), [1024])
  const end = layoutAt(h, 6)
  assert.deepEqual(live(end), [1024, 2048])
  assert.equal(end.segments[0].blocks.find((b) => b.address === 1024).state, BlockState.PENDING_FREE)
  assert.equal(layoutAt(h, 5).segments[0].blocks.find((b) => b.address === 1024).state, BlockState.ALLOCATED)
  assert.equal(end.stats.reserved, 4096)
  assert.equal(end.stats.largestFree, 1536)
})

test('an expandable segment unmapped in the middle splits, and its pieces merge back', () => {
  const MB = 1 << 20
  const ev = [
    { action: 'segment_map', addr: 0, size: 6 * MB },
    { action: 'segment_unmap', addr: 2 * MB, size: 2 * MB },
    { action: 'segment_map', addr: 2 * MB, size: MB },
  ]
  // At the snapshot: [0, 3MB) mapped and [4MB, 6MB) mapped, as two segments.
  const exp = { isExpandable: true }
  const h = buildHistory(ev, [seg(0, 3 * MB, [], exp), seg(4 * MB, 2 * MB, [], exp)])
  assert.equal(h.inconsistent, 0)
  const sizes = (t) => layoutAt(h, t).segments.map((s) => [s.address / MB, s.totalSize / MB])
  assert.deepEqual(sizes(0), [[0, 6]])
  assert.deepEqual(sizes(1), [[0, 2], [4, 2]])
  assert.deepEqual(sizes(2), [[0, 3], [4, 2]])
})

test('a wrapped trace: what was alive before it began comes from frees and the snapshot', () => {
  const ev = [
    { action: 'free_completed', addr: 512, size: 512 },
    { action: 'alloc', addr: 1024, size: 100 },
  ]
  const h = buildHistory(ev, [seg(0, 4096, [blk(0, 512), blk(512, 512, BlockState.INACTIVE), blk(1024, 512)])])
  assert.equal(h.inconsistent, 0)
  assert.deepEqual(live(layoutAt(h, -1)), [0, 512])
  assert.deepEqual(live(layoutAt(h, 0)), [0])
  assert.deepEqual(live(layoutAt(h, 1)), [0, 1024])
})

// On every shipped sample: the state at each event adds up to what the
// timeline's own replay says, and at the end it is the snapshot.
for (const f of readdirSync(SAMPLES).filter((x) => x.endsWith('.pickle'))) {
  test(`history agrees with the timeline and ends at the snapshot: ${f}`, () => {
    const model = parseSnapshot(unpickle(readFileSync(join(SAMPLES, f))))
    for (const d of [...model.devices, ...model.devices.flatMap((x) => x.pools)]) {
      const ev = d.timeline.events
      if (!d.timeline.hasTrace) { continue }
      const h = buildHistory(ev, d.segments)
      const b = buildAllocModel(ev, d.segments, 'blocks')
      const s = buildAllocModel(ev, d.segments, 'segments')
      const step = Math.max(1, Math.floor(ev.length / 200))
      for (let t = 0; t < ev.length; t += step) {
        const L = layoutAt(h, t)
        assert.equal(L.stats.allocated, b.totals[t], `allocated at ${t}`)
        assert.equal(L.stats.reserved, s.totals[t], `reserved at ${t}`)
        assert.equal(L.orphans, 0, `blocks outside any segment at ${t}`)
        for (const sg of L.segments) {
          let at = sg.address
          for (const x of sg.blocks) { assert.equal(x.address, at, `tiles contiguous at ${t}`); assert.ok(x.size > 0); at += x.size }
          assert.equal(at, sg.address + sg.totalSize, `tiles cover the segment at ${t}`)
        }
      }
      const end = layoutAt(h, ev.length - 1)
      const want = d.segments.flatMap((sg) => sg.blocks.filter((x) => x.active).map((x) => x.address)).sort((x, y) => x - y)
      assert.deepEqual(live(end), want, 'live blocks at the snapshot')
      assert.equal(end.stats.reserved, d.stats.reserved)
      assert.equal(h.inconsistent, 0, 'events that do not reconcile')
    }
  })
}
