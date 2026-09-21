// Regenerate src/samples.stats.json from the files in public/samples/.
//
//     node scripts/make_samples_manifest.mjs
//
// The chooser on the landing page shows each sample's headline numbers so you
// can pick the one you want without opening all of them. Those numbers are
// measured here rather than typed into the prose, so they cannot drift away
// from the files after a regeneration.

import { readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { unpickle } from '../src/lib/unpickle.js'
import { parseSnapshot } from '../src/lib/snapshot.js'

const DIR = 'public/samples'
const OUT = 'src/samples.stats.json'

const stats = {}
for (const file of readdirSync(DIR).filter((f) => f.endsWith('.pickle')).sort()) {
  const raw = readFileSync(join(DIR, file))
  const model = parseSnapshot(unpickle(raw))
  const d = model.devices[0]
  stats[file] = {
    fileBytes: raw.length,
    devices: model.devices.length,
    // Sum across devices so a multi-device sample reports its whole footprint.
    reserved: model.devices.reduce((n, x) => n + x.stats.reserved, 0),
    active: model.devices.reduce((n, x) => n + x.stats.active, 0),
    segments: model.devices.reduce((n, x) => n + x.stats.segmentCount, 0),
    utilisation: d ? d.stats.utilisation : 0,
    fragmentation: d ? d.stats.fragmentation : 0,
    traceEvents: model.devices.reduce((n, x) => n + x.timeline.points.length, 0),
    oomEvents: model.devices.reduce((n, x) => n + x.timeline.oomEvents.length, 0),
    truncated: model.devices.some((x) => x.timeline.truncated),
    hasTrace: model.hasAnyTrace,
  }
  console.log(`  ${file}`)
}

writeFileSync(OUT, `${JSON.stringify(stats, null, 2)}\n`)
console.log(`\nwrote ${OUT} for ${Object.keys(stats).length} samples`)
