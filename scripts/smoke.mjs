// Render every view against the demo snapshot and fail on anything thrown.
//
// A build succeeding only proves the modules parse. This proves each view can
// actually render a real snapshot -- the class of bug (a missing field, a bad
// map over an empty array, a divide by a zero-size segment) that otherwise
// shows up as a blank page after deploy.
//
//     node scripts/smoke.mjs
//
// It renders to a string, so there is no DOM and no browser: enough to catch a
// crash, not a substitute for looking at the thing.

import { readFileSync, writeFileSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { build } from 'esbuild'

const ROOT = process.cwd()
const SRC = join(ROOT, 'src')
// Build inside the repo so that react / react-dom, which we deliberately leave
// unbundled, resolve through the repo's own node_modules at import time.
const WORK = join(ROOT, 'node_modules', '.oomscope-smoke')

const ENTRY = `
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement as h } from 'react'

import { unpickle } from '${SRC}/lib/unpickle.js'
import { parseSnapshot } from '${SRC}/lib/snapshot.js'
import Overview from '${SRC}/components/Overview.jsx'
import Segments from '${SRC}/components/Segments.jsx'
import Allocations from '${SRC}/components/Allocations.jsx'
import Timeline from '${SRC}/components/Timeline.jsx'

export function run(bytes) {
  const model = parseSnapshot(unpickle(bytes))
  const device = model.devices[0]
  const out = {}
  for (const [name, C, props] of [
    ['overview', Overview, { device, model }],
    ['segments', Segments, { device }],
    ['allocations', Allocations, { device }],
    ['timeline', Timeline, { device }],
  ]) {
    out[name] = renderToStaticMarkup(h(C, props)).length
  }
  return { out, model }
}
`

try {
  mkdirSync(WORK, { recursive: true })
  const entryPath = join(WORK, 'entry.jsx')
  const outPath = join(WORK, 'bundle.mjs')
  writeFileSync(entryPath, ENTRY)

  await build({
    entryPoints: [entryPath],
    bundle: true,
    format: 'esm',
    platform: 'node',
    outfile: outPath,
    absWorkingDir: ROOT,
    jsx: 'automatic',
    logLevel: 'error',
    // react-dom/server is CommonJS and calls require() at load time, which a
    // bundled ESM output cannot do. Node loads it fine unbundled.
    external: ['react', 'react-dom', 'react/jsx-runtime', 'react-dom/server'],
  })

  const { run } = await import(`file://${outPath}?t=${Date.now()}`)
  const { out, model } = run(readFileSync(join(ROOT, 'public', 'demo.pickle')))

  const d = model.devices[0]
  console.log(
    `snapshot: ${d.segments.length} segments, ${d.timeline.points.length} trace events, ` +
    `${d.timeline.oomEvents.length} OOM, ${d.blame.length} blame groups`,
  )

  let bad = 0
  for (const [name, len] of Object.entries(out)) {
    // A view that renders a few hundred bytes rendered its empty state, which
    // against this snapshot means something silently produced nothing.
    const ok = len > 500
    if (!ok) { bad++ }
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(12)} ${len.toLocaleString()} bytes of markup`)
  }
  if (bad > 0) {
    console.error(`\n${bad} view(s) rendered suspiciously little`)
    process.exitCode = 1
  } else {
    console.log('\nall views rendered')
  }
} finally {
  rmSync(WORK, { recursive: true, force: true })
}
