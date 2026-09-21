// Parse every snapshot in snapshots/ with the app's own reader and print what
// each one exercises.
//
//     node scripts/check_snapshots.mjs [dir]
//
// This is the cheap half of testing the viewer: it proves each file decodes and
// what states it puts the model into, so you know which one to open when you
// want to look at a particular view. It does not render anything -- that is
// scripts/smoke.mjs.

import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

import { unpickle } from '../src/lib/unpickle.js'
import { parseSnapshot } from '../src/lib/snapshot.js'
import { bytes, pct } from '../src/lib/format.js'

const dir = process.argv[2] ?? 'public/samples'
const files = readdirSync(dir).filter((f) => f.endsWith('.pickle')).sort()

if (files.length === 0) {
  console.error(`no .pickle files in ${dir}/ -- run scripts/make_test_snapshots.py first`)
  process.exit(1)
}

let failed = 0
for (const f of files) {
  const raw = readFileSync(join(dir, f))
  let model
  const t0 = performance.now()
  try {
    model = parseSnapshot(unpickle(raw))
  } catch (err) {
    failed++
    console.log(`\n${f}\n  FAILED: ${err.message}`)
    continue
  }
  const ms = performance.now() - t0

  console.log(`\n${f}  (${(raw.length / 1024).toFixed(0)} KiB, parsed ${ms.toFixed(0)} ms)`)
  for (const d of model.devices) {
    const s = d.stats
    const tl = d.timeline
    console.log(
      `  cuda:${d.id}  reserved ${bytes(s.reserved).padStart(9)}` +
      `  live ${bytes(s.active).padStart(9)}` +
      `  util ${pct(s.utilisation).padStart(4)}` +
      `  frag ${pct(s.fragmentation).padStart(4)}`,
    )
    console.log(
      `          segments ${String(s.segmentCount).padStart(3)}` +
      ` (${s.largeSegments} large, ${s.smallSegments} small` +
      `${s.expandableSegments ? `, ${s.expandableSegments} expandable` : ''})` +
      `  blocks ${s.activeBlocks} live / ${s.freeBlocks} free` +
      `  largest free ${bytes(s.largestFreeBlock)}`,
    )
    console.log(
      `          trace ${tl.hasTrace ? `${tl.points.length} events` : 'NONE'}` +
      `${tl.truncated ? ' (TRUNCATED)' : ''}` +
      `${tl.oomEvents.length ? `  OOM x${tl.oomEvents.length}` : ''}` +
      `  blame groups ${d.blame.length}`,
    )
    if (d.blame.length) {
      const top = d.blame[0]
      console.log(`          top source: ${bytes(top.bytes)} over ${top.count} blocks -- ${top.label}`)
    }
  }
}

console.log(`\n${files.length - failed}/${files.length} parsed`)
if (failed) { process.exitCode = 1 }
